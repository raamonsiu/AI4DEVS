# Embedding client for the semantic cache.
#
# OpenAI-only, mirroring guardrails.py's moderation client: if no
# OPENAI_API_KEY is configured (e.g. an Anthropic-only deployment), embed_text
# returns None and callers fall back to the exact-match cache alone rather
# than failing the request.

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


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
    response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
    return response.data[0].embedding
