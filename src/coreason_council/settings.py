from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed by pydantic-settings.
    Reads from environment variables and .env file.
    """

    openai_api_key: str | None = None
    presets_file: str = "src/coreason_council/resources/presets.yaml"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
