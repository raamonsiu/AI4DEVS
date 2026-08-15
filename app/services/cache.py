# Exact-match Redis cache for LLM responses.
#
# The cache key is a hash of the system prompt, the (normalized) user message,
# and the generation knobs (model, max_tokens). It is NOT just "the question":
# the same transcription asked with a different model, or a different
# max_tokens budget, can legitimately produce a different answer, so each of
# those must be part of the key or we'd serve a stale/wrong response.
#
# The user message is normalized (trimmed, lowercased, whitespace collapsed)
# before hashing so trivial differences : extra spaces, casing, a stray
# newline : still hit the cache. This is a cheap improvement over pure exact
# match; it does not catch typos or paraphrases. A semantic cache (embedding
# similarity) would catch those, at the cost of a similarity-threshold tuning
# problem and an extra embedding call per request : left as a future step.

from __future__ import annotations

import hashlib
import json
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
    def make_key(*, system_prompt: str, user_message: str, model: str, max_tokens: int) -> str:
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": normalize_text(user_message),
                "model": model,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
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
