import pytest
from pydantic import ValidationError

from app.schemas import EstimationResult, Phase


def test_estimation_result_total_cost_must_match_phases():
    with pytest.raises(ValidationError):
        EstimationResult(
            summary="Test",
            total_duration_weeks=10,
            total_cost_eur=10000,
            confidence_pct=80,
            phases=[
                Phase(name="Design", duration_weeks=4, cost_eur=4000,
                      confidence_pct=90, assumptions=[]),
                Phase(name="Build", duration_weeks=6, cost_eur=8000,
                      confidence_pct=70, assumptions=[]),
            ],
        )
        # 4000 + 8000 = 12000, but total says 10000 -> should fail
        # the model_validator we defined above
