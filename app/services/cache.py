# Redis cache for LLM responses, in two layers:
#
# 1. Exact-match (make_key/get/set): a hash of the system prompt, the
#    (normalized) user message, and the generation knobs (model, max_tokens).
#    It is NOT just "the question": the same transcription asked with a
#    different model, or a different max_tokens budget, can legitimately
#    produce a different answer, so each of those must be part of the key or
#    we'd serve a stale/wrong response. Cheap (no network call) and catches
#    exact repeats plus trivial formatting differences (extra spaces, casing).
#
# 2. Semantic (make_bucket_key/get_semantic/set_semantic): on an exact-match
#    miss, an embedding-similarity lookup that catches paraphrases and typos
#    the exact layer can't. Entries are scoped to a "bucket" keyed by system
#    prompt + model (+ response_model for structured calls) : the same
#    discriminators as the exact key, minus the user message itself : so a
#    prompt-version or model change can't leak a stale semantic hit across
#    incompatible responses. Within a bucket, similarity search is a plain
#    brute-force cosine comparison against every stored embedding: buckets are
#    capped at SEMANTIC_CACHE_MAX_ENTRIES entries, so this stays cheap without
#    needing a vector index (e.g. RediSearch/redisvl) as extra infra.
#
# Callers are expected to check the exact cache first, then the semantic one,
# and : critically : only write a response to either cache *after* it has
# passed whatever output validation applies (Pydantic model_validators for
# structured calls). Caching an invalid/hallucinated response would otherwise
# serve it to every future request that resembles the one that produced it,
# for the entire TTL.

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import redis
import structlog

log = structlog.get_logger()


def normalize_text(text: str) -> str:
    """Collapse whitespace/casing so trivial formatting differences still hit the cache."""
    return re.sub(r"\s+", " ", text.strip().lower())


class EstimationCache:
    """Thin wrapper around redis-py with deterministic keying and TTL."""

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "EstimationCache":
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        response_model: str | None = None,
    ) -> str:
        payload = {
            "system_prompt": system_prompt,
            "user_message": normalize_text(user_message),
            "model": model,
            "max_tokens": max_tokens,
        }
        if response_model is not None:
            # Only added for structured calls, so plain-text cache keys
            # (response_model=None) hash identically to before this field existed.
            payload["response_model"] = response_model
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"estimation:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            log.warning("cache_get_failed", error=str(exc))
            return None
        if cached:
            log.info("cache_hit", key_prefix=key[:24])
            return json.loads(cached)
        log.info("cache_miss", key_prefix=key[:24])
        return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        try:
            self.redis.setex(key, self.ttl, json.dumps(response))
            log.info("cache_stored", key_prefix=key[:24], ttl=self.ttl)
        except redis.RedisError as exc:
            log.warning("cache_set_failed", error=str(exc))

    @staticmethod
    def make_bucket_key(*, system_prompt: str, model: str, response_model: str | None = None) -> str:
        """Same discriminators as make_key, minus the user message: this is
        the scope within which semantic similarity search is allowed to match."""
        payload = {"system_prompt": system_prompt, "model": model}
        if response_model is not None:
            payload["response_model"] = response_model
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"semcache:{digest}"

    def get_semantic(
        self, bucket_key: str, query_embedding: list[float], threshold: float
    ) -> dict[str, Any] | None:
        try:
            raw_entries = self.redis.lrange(bucket_key, 0, -1)
        except redis.RedisError as exc:
            log.warning("semantic_cache_get_failed", error=str(exc))
            return None

        best_response: dict[str, Any] | None = None
        best_similarity = 0.0
        for raw in raw_entries:
            try:
                entry = json.loads(raw)
                similarity = _cosine_similarity(query_embedding, entry["embedding"])
            except (TypeError, ValueError, KeyError):
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = entry["response"]

        if best_response is not None and best_similarity >= threshold:
            log.info("semantic_cache_hit", bucket_prefix=bucket_key[:24], similarity=round(best_similarity, 4))
            return best_response
        log.info("semantic_cache_miss", bucket_prefix=bucket_key[:24], best_similarity=round(best_similarity, 4))
        return None

    def set_semantic(
        self,
        bucket_key: str,
        embedding: list[float],
        response: dict[str, Any],
        max_entries: int,
    ) -> None:
        try:
            entry = json.dumps({"embedding": embedding, "response": response})
            pipe = self.redis.pipeline()
            pipe.rpush(bucket_key, entry)
            # Keep only the most recent max_entries so brute-force similarity
            # search over a bucket stays bounded regardless of traffic volume.
            pipe.ltrim(bucket_key, -max_entries, -1)
            pipe.expire(bucket_key, self.ttl)
            pipe.execute()
            log.info("semantic_cache_stored", bucket_prefix=bucket_key[:24], ttl=self.ttl)
        except redis.RedisError as exc:
            log.warning("semantic_cache_set_failed", error=str(exc))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
