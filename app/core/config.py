from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "sqlite:///./dev.db"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    documents_dir: str = "documents"
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 480


@lru_cache
def get_settings() -> Settings:
    return Settings()
