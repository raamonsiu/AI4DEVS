# LiteLLM-backed wrapper: one call site for every LLM interaction in the
# estimator, adding provider fallback, an exact-match cache, cost tracking,
# and structured logging on top of the raw provider call.
#
# Fallback strategy
# ------------------
# A single provider/model with no plan B means an expired key, a rate limit,
# or an outage takes the whole feature down. The Router below is configured
# with two deployments under the same logical "estimator" model name — primary
# first, fallback second — plus its own retry budget. LiteLLM's Router already
# implements the escalation this project wants:
#   1. Try the primary model.
#   2. On a transient error, retry the primary model up to LLM_RETRIES times.
#   3. If it still fails (or the error is non-retryable, e.g. no credits),
#      fall over to the fallback deployment.
# Different failure classes (rate limit vs. auth vs. timeout) are handled
# internally by LiteLLM's retry/fallback policy rather than reimplemented here.
#
# Cache
# -----
# Every call checks the cache first (see cache.py for the key derivation) and
# populates it after a successful response. Streaming responses are cached as
# a single joined string and replayed as one chunk on a hit, so the UX stays
# consistent whether or not the answer was cached.

from __future__ import annotations

import time
from typing import Any, Iterator

import structlog
from litellm import Router

from app.constants import MODELS_PRICING
from app.services.cache import EstimationCache

log = structlog.get_logger()


def _normalise_model_name(model: str) -> str:
    """Strip provider prefixes like ``anthropic/`` that LiteLLM may emit."""
    return model.split("/", 1)[1] if "/" in model else model


def _provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3")):
        return "openai"
    return "unknown"


def _lookup_pricing(model: str) -> dict[str, float]:
    """Match a (possibly versioned, e.g. gpt-4o-mini-2024-07-18) model name
    against the known pricing table by longest matching prefix."""
    name = _normalise_model_name(model)
    if name in MODELS_PRICING:
        return MODELS_PRICING[name]
    matches = [key for key in MODELS_PRICING if name.startswith(key)]
    if matches:
        return MODELS_PRICING[max(matches, key=len)]
    return {"input": 0.0, "output": 0.0}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _lookup_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def _extract_delta(chunk: Any) -> str:
    """Pull the text delta out of a LiteLLM streaming chunk."""
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return ""
    return getattr(delta, "content", None) or ""


class LLMWrapper:
    """Unified LLM client with cache, provider fallback, and cost tracking."""

    def __init__(
        self,
        *,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        primary_model: str,
        fallback_model: str,
        timeout: int,
        num_retries: int,
        cache: EstimationCache,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.num_retries = num_retries
        self.cache = cache

        self.router = Router(
            model_list=[
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": primary_model,
                        "api_key": openai_api_key,
                        "timeout": timeout,
                    },
                },
                {
                    "model_name": "estimator",
                    "litellm_params": {
                        "model": fallback_model,
                        "api_key": anthropic_api_key,
                        "timeout": timeout,
                    },
                },
            ],
            fallbacks=[{"estimator": ["estimator"]}],
            num_retries=num_retries,
        )

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Blocking call. Returns estimation text plus usage/cost/cache metadata."""
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.primary_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_call_started", mode="blocking", primary_model=self.primary_model)
        t0 = time.perf_counter()
        try:
            response = self.router.completion(
                model="estimator", messages=messages, max_tokens=max_tokens
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        result = self._normalise_response(response, latency_ms=latency_ms)
        log.info(
            "llm_call_completed",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
        )
        self.cache.set(cache_key, result)
        return {**result, "cache_hit": False}

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4000,
        meta_holder: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive. A cache hit replays the cached text
        as a single chunk so the client UX stays consistent either way.

        If ``meta_holder`` is passed, it is populated in place with
        ``cache_hit``, ``cost_usd``, ``model`` and ``provider`` once the
        generator is exhausted — callers that need that metadata (e.g. the SSE
        router, to emit a final event) read it after the ``for`` loop ends.
        """
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=self.primary_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            log.info("stream_cache_hit", chars=len(cached.get("estimation", "")))
            if meta_holder is not None:
                meta_holder.update(
                    cache_hit=True,
                    cost_usd=cached.get("cost_usd", 0.0),
                    model=cached.get("model", self.primary_model),
                    provider=cached.get("provider", _provider_from_model(self.primary_model)),
                )
            yield cached.get("estimation", "")
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_stream_started", primary_model=self.primary_model)
        t0 = time.perf_counter()
        full_text: list[str] = []
        usage_chunk: Any = None
        model_seen: str | None = None
        try:
            response = self.router.completion(
                model="estimator",
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in response:
                delta = _extract_delta(chunk)
                if delta:
                    full_text.append(delta)
                    yield delta
                # The final chunk of the stream carries cumulative usage
                # (OpenAI/Anthropic both do this via LiteLLM's stream_options).
                if getattr(chunk, "usage", None):
                    usage_chunk = chunk.usage
                if getattr(chunk, "model", None):
                    model_seen = chunk.model
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_stream_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        rendered = "".join(full_text)
        model = _normalise_model_name(model_seen or self.primary_model)
        provider = _provider_from_model(model)
        input_tokens = getattr(usage_chunk, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_chunk, "completion_tokens", 0) or 0
        cost_usd = _estimate_cost(model, input_tokens, output_tokens)
        log.info(
            "llm_stream_completed",
            latency_ms=latency_ms,
            chars=len(rendered),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

        self.cache.set(
            cache_key,
            {
                "estimation": rendered,
                "model": model,
                "provider": provider,
                "finish_reason": "stop",
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
        )

        if meta_holder is not None:
            meta_holder.update(cache_hit=False, cost_usd=cost_usd, model=model, provider=provider)

    @staticmethod
    def _normalise_response(response: Any, *, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        finish_reason = (choice.finish_reason or "stop").lower()
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        model = _normalise_model_name(response.model)
        return {
            "estimation": choice.message.content or "",
            "model": model,
            "provider": _provider_from_model(model),
            "finish_reason": finish_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "latency_ms": latency_ms,
            "cost_usd": _estimate_cost(model, input_tokens, output_tokens),
        }
