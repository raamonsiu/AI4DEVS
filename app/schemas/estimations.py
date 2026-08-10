from pydantic import BaseModel, Field

class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=50,
        description="Client meet transcription summary."
    )

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