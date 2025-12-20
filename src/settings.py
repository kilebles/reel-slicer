"""Application settings using Pydantic."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    
    API_KEY: str
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    MAX_TOKENS: int = 4096

    CHUNK_SIZE: int = 4000  


settings = Settings()
