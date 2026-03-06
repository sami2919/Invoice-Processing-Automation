"""App settings via pydantic-settings. Reads from env vars / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Grok / xAI
    xai_api_key: str
    xai_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-3-fast"
    grok_mini_model: str = "grok-3-mini-fast"

    # approval thresholds (USD)
    auto_approve_threshold: float = 10000.0
    manager_approval_threshold: float = 10000.0
    director_approval_threshold: float = 25000.0

    # risk score bands
    high_risk_threshold: int = 70
    medium_risk_threshold: int = 30

    # extraction
    max_extraction_retries: int = 3
    min_confidence_threshold: float = 0.7

    # db
    db_path: str = "inventory.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
