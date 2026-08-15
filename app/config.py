from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "DEBUG"

    # LiteLLM wrapper: tries PRIMARY_MODEL first, falls back to FALLBACK_MODEL
    # (same or different provider) on failure.
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2

    # Redis exact-match cache
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400

    # Base URL the Streamlit client uses to reach this API
    ESTIMATOR_API_BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

    @model_validator(mode="after")
    def validate_at_least_one_api_key(self) -> "Settings":
        """LiteLLM's Router may dispatch to either provider on fallback, so at
        least one API key must be configured or every call would fail with no
        plan B."""
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set"
            )
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()