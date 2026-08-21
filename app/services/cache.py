"""Exact-match Redis cache for LLM responses.

The cache key is a SHA-256 over the inputs that can legitimately change the
answer. It is NOT just "the question": the same description asked with a
different model, prompt version or ``max_tokens`` budget can produce a
different answer, so each of those must be part of the key or we'd serve a
stale/wrong response. Any change to the prompt templates also implicitly
invalidates the cache, because the rendered system prompt text is part of the
key (for ``make_key``) or the prompt version is (for the structured pipeline's
key in ``app/services/estimation.py``).

This layer is cheap (no network call beyond Redis) and catches exact repeats.
Paraphrases and typos are caught by the semantic layer in
``app/cache/semantic.py``, which runs only after this one misses.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import redis
import structlog

log = structlog.get_logger()


def normalize_text(text: str) -> str:
    """Collapse whitespace/casing so trivial formatting differences still hit
    the cache. A cheap improvement over pure exact match; it does not catch
    typos or paraphrases : that's what the semantic layer is for."""
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
    ) -> str:
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
