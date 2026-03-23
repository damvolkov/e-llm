from pathlib import Path
from typing import ClassVar

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(extra="ignore")

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = Path("/data")
    DEBUG: bool = True
    GUI_HOST: str = "0.0.0.0"
    GUI_PORT: int = 8080
    LLAMACPP_URL: str = "http://127.0.0.1:45150"

    @computed_field
    @property
    def config_path(self) -> Path:
        return self.DATA_DIR / "config" / "config.yaml"

    @computed_field
    @property
    def models_path(self) -> Path:
        return self.DATA_DIR / "models"

    @computed_field
    @property
    def cache_path(self) -> Path:
        return self.DATA_DIR / "cache"


settings = Settings()
