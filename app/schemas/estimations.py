from pydantic import BaseModel, Field

class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=50,
        description="Client meet transcription summary."
    )

class StreamEstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=50,
        description="Client meet transcription summary."
    )

class EvaluationResult(BaseModel):
    has_title: bool
    has_breakdown_table: bool
    has_total_hours: bool
    has_total_cost: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: float | None
    sum_task_hours: float | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_task_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float = Field(..., description="Structural quality score between 0 and 1")
    issues: list[str]

class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    tokens_used: int = Field(
        ...,
        description="Total tokens used (input + output)"
    )
    estimated_cost: float = Field(
        ...,
        description="Estimated cost in USD"
    )
    cache_hit: bool = Field(
        default=False,
        description="True when the response was served from the Redis cache"
    )
    evaluation: EvaluationResult = Field(
        ...,
        description="Static structural evaluation of the generated estimation"
    )