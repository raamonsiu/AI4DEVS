# Embedding client for the semantic cache.
#
# OpenAI-only, mirroring guardrails.py's moderation client: if no
# OPENAI_API_KEY is configured (e.g. an Anthropic-only deployment), embed_text
# returns None and callers fall back to the exact-match cache alone rather
# than failing the request.
#
# No cross-provider fallback (unlike LLMWrapper's OpenAI->Anthropic router):
# embeddings from different models aren't comparable by cosine similarity at
# all (different vector spaces/dimensions), so there is nothing sensible to
# "fall over" to within the same semantic-cache bucket, and Anthropic doesn't
# offer an embeddings API to fall over to in the first place. Instead, any
# failure here (timeout, rate limit, ...) degrades to a semantic-cache miss :
# the request still proceeds via the exact-match cache and the LLM call,
# which is the actually-important resilience property.

from functools import lru_cache

import structlog
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from app.config import get_settings

log = structlog.get_logger()


@lru_cache
def _embeddings_client() -> OpenAI | None:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_text(text: str) -> list[float] | None:
    client = _embeddings_client()
    if client is None:
        return None
    settings = get_settings()
    try:
        response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
    except (APIError, APITimeoutError, RateLimitError) as exc:
        log.warning("embedding_call_failed", error=str(exc))
        return None
    return response.data[0].embedding
