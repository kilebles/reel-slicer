"""Application settings using Pydantic."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    API_KEY: str
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"
    MAX_TOKENS: int = 4096

    CHUNK_SIZE: int = 4000
    SEGMENT_START_OFFSET: float = 0  # 2.5
    SEGMENT_END_OFFSET: float = 0  # 0.64


settings = Settings()
