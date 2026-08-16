# Input guardrail: reject flagged content or obvious prompt-injection
# attempts before the description ever reaches the prompt template.
#
# This is the "exception" policy: on failure the request is rejected
# outright (HTTP 400 at the router). No retry, no degradation : unlike the
# output-side guardrails, there is nothing to salvage from an input that is
# actively hostile.

from functools import lru_cache

import instructor
import structlog
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import get_settings

log = structlog.get_logger()

_PROMPT_INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all instructions",
    "you are now",
    "system prompt",
    "</project_description>",
]

_TOPIC_CHECK_SYSTEM_PROMPT = (
    "You decide whether a user message is describing a software development "
    "project (an app, website, system, integration, etc.) that could plausibly "
    "be estimated in hours/cost/duration. Recipes, personal stories, general "
    "knowledge questions, or anything unrelated to commissioning a software "
    "project are NOT in scope, even if long and well-written."
)


class InputModerationError(Exception):
    """Raised when the input fails moderation, looks like a prompt injection
    attempt, or isn't a software project description at all."""


class _TopicCheck(BaseModel):
    is_software_project_description: bool = Field(
        ..., description="True only if the text describes a software project to estimate."
    )


@lru_cache
def _moderation_client() -> OpenAI | None:
    """Moderation is OpenAI-specific; return None if no OpenAI key is configured
    (e.g. an Anthropic-only deployment) so the pattern check still runs alone."""
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _check_prompt_injection(description: str) -> None:
    lowered = description.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in lowered:
            log.warning("prompt_injection_pattern_matched", pattern=pattern)
            raise InputModerationError(
                "Your message couldn't be processed because it looks like it's "
                "trying to instruct the AI system directly rather than describe "
                "a project. Please rephrase your request."
            )


def _check_moderation(description: str) -> None:
    client = _moderation_client()
    if client is None:
        log.warning("moderation_skipped_no_openai_key")
        return

    moderation = client.moderations.create(input=description)
    if moderation.results[0].flagged:
        log.warning("moderation_flagged", categories=moderation.results[0].categories.model_dump())
        raise InputModerationError(
            "Your message was flagged by our content moderation system and "
            "can't be processed. Please rephrase your request."
        )


def _check_topic(description: str) -> None:
    """Reject descriptions that aren't software project requests at all :
    e.g. a recipe or a trivia question : before any estimation prompt runs.
    This is a real (cheap) LLM call, not a keyword heuristic: topic is too
    fuzzy a signal for pattern matching to catch reliably.
    """
    client = _moderation_client()
    if client is None:
        log.warning("topic_check_skipped_no_openai_key")
        return

    topic_client = instructor.from_openai(client)
    check = topic_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=_TopicCheck,
        max_tokens=50,
        messages=[
            {"role": "system", "content": _TOPIC_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
    )
    if not check.is_software_project_description:
        log.warning("topic_check_rejected", description_preview=description[:80])
        raise InputModerationError(
            "This doesn't look like a software project description. This tool "
            "only estimates software development projects : please describe "
            "the app, system, or feature you'd like estimated."
        )


def validate_input(description: str) -> None:
    """Raise InputModerationError with a message meant to be shown to the end
    user as-is (it becomes the HTTPException detail at the router). The
    technical reason is logged separately, not leaked to the client."""
    _check_prompt_injection(description)
    _check_topic(description)
    _check_moderation(description)
