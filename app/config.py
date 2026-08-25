from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings.

    The API key is optional at import time so tests, retrieval tooling, and the
    deterministic order path can run without an LLM credential. Policy answer
    generation will safely fall back when no key is configured.
    """

    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-20b"

    app_env: str = "development"
    log_level: str = "INFO"

    knowledge_base_dir: str = "knowledge-base"
    orders_file: str = "data/orders.json"
    index_path: Path = PROJECT_ROOT / "indexes"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def knowledge_base_path(self) -> Path:
        return PROJECT_ROOT / self.knowledge_base_dir

    @property
    def orders_path(self) -> Path:
        return PROJECT_ROOT / self.orders_file


settings = Settings()
