"""Pipeline orchestrator. Glue between guardrails, caches, prompt rendering and
the LLM wrapper. The router holds none of this logic — its only job is to
translate HTTP errors.

Pipeline:

    1. Input guardrails (moderation + prompt injection + PII heuristics)
    2. Exact-match cache lookup  → return cached=True on hit
    3. Semantic cache lookup     → return cached=True on hit
    4. Render the versioned prompt
    5. LLM call via Instructor with response_model=EstimationResult
    6. Output guardrail (enforce_scope_response, filter policy)
    7. Write to BOTH caches (exact + semantic)
    8. Return EstimationResponse with cached=False

Order rationale: guardrails go before any cache because a malicious or PII
description should never be served from cache : skipping them to save 50 ms
would let an attacker replay a cached answer without passing moderation. The
exact-match cache goes before the semantic cache because it's the cheapest (no
embedding call). Both cache writes happen AFTER output validation so a failed
or hallucinated estimation is never persisted and replayed for the whole TTL.
"""

from __future__ import annotations

import hashlib
import json

import structlog

from app.cache.semantic import EstimationSemanticCache
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts import render_estimation_prompt
from app.schemas.estimation import (
    CallMeta,
    EstimationDraft,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
)
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()


def _exact_cache_key(request: EstimationRequest, prompt_version: str, model: str) -> str:
    """Deterministic SHA-256 key over the typed request + prompt_version + model."""
    payload = json.dumps(
        {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "prompt_version": prompt_version,
            "model": model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:{digest}"


class EstimationService:
    """Single entry point for the structured estimation pipeline."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        exact_cache: EstimationCache,
        semantic_cache: EstimationSemanticCache | None = None,
        openai_client=None,
        prompt_version: str = "v1",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        # 1. Input guardrails — raises InputGuardrailViolation on rejection.
        check_input(request.description, openai_client=self.openai_client)

        # 2. Exact-match cache lookup.
        cache_key = _exact_cache_key(request, self.prompt_version, self.llm_wrapper.primary_model)
        cached = self.exact_cache.get(cache_key)
        if cached:
            log.info("estimation_cache_hit", kind="exact", key_prefix=cache_key[:24])
            return EstimationResponse(
                result=EstimationResult.model_validate(cached["result"]),
                prompt_version=self.prompt_version,
                cached=True,
                # Nothing was spent on this call; model/provider describe
                # whichever deployment originally produced the cached answer.
                meta=CallMeta(**{**cached.get("meta", {}), "cost_usd": 0.0, "latency_ms": 0}),
            )

        # 3. Semantic cache lookup.
        if self.semantic_cache is not None:
            semantic_hit = self.semantic_cache.lookup(request, self.prompt_version)
            if semantic_hit is not None:
                log.info("estimation_cache_hit", kind="semantic")
                return EstimationResponse(
                    result=semantic_hit,
                    prompt_version=self.prompt_version,
                    cached=True,
                    meta=CallMeta(),
                )

        # 4. Render the versioned prompt.
        system_prompt, user_message = render_estimation_prompt(
            request, version=self.prompt_version
        )

        # 5. LLM call with Instructor + Pydantic validators (re-prompts on failure).
        #    The model fills an EstimationDraft (no totals); the totals are
        #    summed here in Python : see EstimationDraft's docstring for the
        #    measured reason why asking the LLM for them does not work.
        draft, meta = self.llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationDraft,
        )
        result = EstimationResult.from_draft(draft)
        log.info(
            "estimation_generated",
            prompt_version=self.prompt_version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            **meta,
        )

        # 6. Output guardrail (filter): normalises low-confidence answers.
        result = enforce_scope_response(result)

        # 7. Cache the validated payload only (never persist failed validations).
        self.exact_cache.set(
            cache_key,
            {"result": result.model_dump(mode="json"), "meta": meta},
        )
        if self.semantic_cache is not None:
            self.semantic_cache.store(request, result, self.prompt_version)

        # 8. Return.
        return EstimationResponse(
            result=result,
            prompt_version=self.prompt_version,
            cached=False,
            meta=CallMeta(**meta),
        )
