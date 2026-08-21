"""Free-text estimation endpoints — the original transcription-based flow.

These sit alongside the structured ``POST /api/v1/estimate``: same guardrails,
same wrapper, same exact-match cache, but the input is a raw meeting
transcription and the output is Markdown rather than a validated schema.

- ``POST /api/v1/estimate/text``   blocking, plus the static structural
  evaluation of the generated Markdown (see ``services/evaluation.py``).
- ``POST /api/v1/estimate/stream`` the same thing token by token over SSE,
  which is what the Streamlit chat consumes.

Note on paths: the blocking variant used to live at ``POST /api/v1/estimate``.
That path now serves the structured pipeline, so this one moved to
``/estimate/text``. The streaming path is unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper, get_openai_client
from app.guardrails.input import InputGuardrailViolation, check_input
from app.schemas.estimations import (
    EstimationRequest,
    EstimationResponse,
    StreamEstimationRequest,
)
from app.services.llm_service import build_system_prompt, generate_estimation
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations-text"])


def _guard(transcription: str) -> None:
    """Run the shared input guardrails, mapping a violation to a 400 whose body
    matches the structured endpoint's shape (``{reason, message}``) so clients
    can handle both the same way."""
    try:
        check_input(transcription, openai_client=get_openai_client())
    except InputGuardrailViolation as exc:
        log.info("text_estimation_blocked", reason=exc.reason, message=exc.message)
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc


@router.post("/estimate/text", response_model=EstimationResponse)
async def estimate_text(request: EstimationRequest) -> dict:
    """Blocking free-text estimation with a static structural evaluation."""
    _guard(request.transcription)
    return generate_estimation(request.transcription)


@router.post("/estimate/stream")
async def estimate_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a free-text estimation token by token via Server-Sent Events."""
    _guard(request.transcription)

    system_prompt = build_system_prompt()

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        meta_holder: dict = {}
        chunks = wrapper.complete_stream(
            system_prompt=system_prompt,
            user_message=request.transcription,
            meta_holder=meta_holder,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None

        try:
            while True:
                # complete_stream is a sync generator; pulling it in an executor
                # keeps the event loop free while waiting on the provider.
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "meta", "data": json.dumps(meta_holder)}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:
            log.error("stream_failed", error_type=type(exc).__name__, error=str(exc)[:200])
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())
