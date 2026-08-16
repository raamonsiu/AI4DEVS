from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """An actual past project the client wants used as a calibration anchor,
    on top of the model's own few-shot examples."""

    name: str
    description: str
    actual_duration_weeks: int = Field(ge=1, le=104)
    actual_cost_eur: int = Field(ge=0)


class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None


class Phase(BaseModel):
    name: str
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0)
    confidence_pct: int = Field(ge=0, le=100)
    assumptions: list[str]


def _check_low_confidence_explicit(summary: str, confidence_pct: int) -> None:
    """Shared rule: a confidence below 30% must be self-declared out of scope,
    never silently produced alongside a normal-looking estimate."""
    if confidence_pct < 30 and not summary.startswith("Out of scope:"):
        raise ValueError("Confidence below 30% requires an explicit out-of-scope summary")


class EstimationDraft(BaseModel):
    """What the LLM actually fills in. No total_* fields on purpose: small
    models are unreliable at keeping a separately-declared total consistent
    with the sum of several phases across retries. Totals are computed from
    ``phases`` in Python instead : see EstimationResult.

    This is also the model Instructor validates and retries against, so the
    low-confidence rule must live here (not only on EstimationResult) for the
    retry loop to actually catch a violation.
    """

    summary: str
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase]

    @model_validator(mode="after")
    def low_confidence_must_be_explicit(self):
        _check_low_confidence_explicit(self.summary, self.confidence_pct)
        return self


class EstimationResult(BaseModel):
    summary: str
    total_duration_weeks: int = Field(ge=0)
    total_cost_eur: int = Field(ge=0)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase]

    @model_validator(mode="after")
    def total_must_match_sum_of_phases(self):
        sum_weeks = sum(p.duration_weeks for p in self.phases)
        sum_cost = sum(p.cost_eur for p in self.phases)
        if abs(sum_weeks - self.total_duration_weeks) > 1:
            raise ValueError("total_duration_weeks does not match phases")
        if self.total_cost_eur == 0:
            if sum_cost != 0:
                raise ValueError("total_cost_eur does not match phases")
        elif abs(sum_cost - self.total_cost_eur) / self.total_cost_eur > 0.05:
            raise ValueError("total_cost_eur does not match phases")
        return self

    @model_validator(mode="after")
    def low_confidence_must_be_explicit(self):
        _check_low_confidence_explicit(self.summary, self.confidence_pct)
        return self


class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str
