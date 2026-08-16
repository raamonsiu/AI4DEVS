import structlog
from fastapi import APIRouter, Depends, HTTPException
from instructor.v2.core.errors import InstructorError

from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationDraft, EstimationRequest, EstimationResponse, EstimationResult
from app.services.guardrails import InputModerationError, validate_input
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

router = APIRouter(prefix="/api/v2", tags=["estimations-v2"])

PROMPT_VERSION = "v1"

@router.post("/estimate", response_model=EstimationResponse)
def estimate(
    request: EstimationRequest,
    wrapper: LLMWrapper = Depends(get_llm_wrapper),
) -> EstimationResponse:
    try:
        validate_input(request.description)
    except InputModerationError as exc:
        log.warning("estimation_v2_input_rejected", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    system, user = render_estimation_prompt(request)
    try:
        outcome = wrapper.complete_structured(
            system_prompt=system,
            user_message=user,
            response_model=EstimationDraft,
        )
    except InstructorError as exc:
        log.error("estimation_v2_endpoint_error", error=str(exc))
        raise HTTPException(
            status_code=502,
            detail="The model could not produce a valid estimation after retrying.",
        ) from exc

    draft: EstimationDraft = outcome["result"]
    # Totals are summed here, not asked of the LLM: small models are
    # unreliable at keeping a separately-declared total consistent with the
    # phases across retries (see app/schemas/estimation.py::EstimationDraft).
    result = EstimationResult(
        summary=draft.summary,
        total_duration_weeks=sum(p.duration_weeks for p in draft.phases),
        total_cost_eur=sum(p.cost_eur for p in draft.phases),
        confidence_pct=draft.confidence_pct,
        phases=draft.phases,
    )
    return EstimationResponse(result=result, prompt_version=PROMPT_VERSION)
