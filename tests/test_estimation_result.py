"""The two business rules the LLM cannot break. When one of these validators
raises, Instructor re-prompts the model with the ValueError message."""

import pytest
from pydantic import ValidationError

from app.schemas import OUT_OF_SCOPE_PREFIX, EstimationResult

PHASES = [
    {"name": "Discovery", "duration_weeks": 1, "cost_eur": 2_500, "summary": "Workshops and scoping."},
    {"name": "Implementation", "duration_weeks": 5, "cost_eur": 14_500, "summary": "Build the core features."},
]


def test_totals_matching_phases_are_accepted():
    result = EstimationResult(
        summary="A mid-sized internal tool for equipment loans.",
        confidence_pct=70,
        phases=PHASES,
        total_duration_weeks=6,
        total_cost_eur=17_000,
    )
    assert result.total_cost_eur == sum(p["cost_eur"] for p in PHASES)


def test_total_cost_must_match_sum_of_phases():
    with pytest.raises(ValidationError, match="does not match total_cost_eur"):
        EstimationResult(
            summary="A mid-sized internal tool for equipment loans.",
            confidence_pct=70,
            phases=PHASES,
            total_duration_weeks=6,
            total_cost_eur=99_000,
        )


def test_low_confidence_requires_out_of_scope_prefix():
    with pytest.raises(ValidationError, match="Out of scope"):
        EstimationResult(
            summary="A vague idea that was estimated anyway.",
            confidence_pct=10,
            phases=PHASES,
            total_duration_weeks=6,
            total_cost_eur=17_000,
        )


def test_low_confidence_is_allowed_when_declared():
    result = EstimationResult(
        summary=f"{OUT_OF_SCOPE_PREFIX} the description is a recipe, not a software project.",
        confidence_pct=0,
        phases=[
            {
                "name": "Not estimated",
                "duration_weeks": 1,
                "cost_eur": 0,
                "summary": "Cannot be sized without more information.",
            }
        ],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
    assert result.confidence_pct == 0
