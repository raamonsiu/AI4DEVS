import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.dependencies import get_llm_wrapper
from app.schemas.estimations import EstimationRequest, EstimationResponse, StreamEstimationRequest
from app.services.llm_service import build_system_prompt, generate_estimation
from app.services.llm_wrapper import LLMWrapper

router = APIRouter(prefix="/api/v1", tags=["estimations"])

@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest):
    result = generate_estimation(request.transcription)
    return result

@router.post("/estimate/stream")
async def estimate_stream(
    request: StreamEstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EventSourceResponse:
    """Stream a software estimation token by token via Server-Sent Events."""
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
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "meta", "data": json.dumps(meta_holder)}
            yield {"event": "done", "data": "[DONE]"}
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())

