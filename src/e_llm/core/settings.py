"""Application settings — YAML + env, typed, with ClassVar constants."""

from pathlib import Path
from typing import ClassVar

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings.main import PydanticBaseSettingsSource


def _read_pyproject(base_dir: Path) -> dict:
    path = base_dir / "pyproject.toml"
    if not path.exists():
        return {}
    try:
        import tomllib

        return tomllib.loads(path.read_text())
    except Exception:
        return {}


def _get_version(base_dir: Path) -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=base_dir,
        )
        if result.returncode == 0:
            return result.stdout.strip().lstrip("v")
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("e-llm")
    except Exception:
        return "0.0.0"


class Settings(BaseSettings):
    """Unified settings — env vars override YAML values."""

    model_config = SettingsConfigDict(
        yaml_file="data/config/config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    ##### CLASS-LEVEL CONSTANTS #####

    BASE_DIR: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT: ClassVar[dict] = _read_pyproject(BASE_DIR)
    API_NAME: ClassVar[str] = PROJECT.get("project", {}).get("name", "e-llm")
    API_DESCRIPTION: ClassVar[str] = PROJECT.get("project", {}).get("description", "")
    API_VERSION: ClassVar[str] = _get_version(BASE_DIR)
    ASSETS_PATH: ClassVar[Path] = BASE_DIR / "assets"
    LOGO_PATH: ClassVar[Path] = ASSETS_PATH / "e-llm-landscape-front.svg"
    ICON_PATH: ClassVar[Path] = ASSETS_PATH / "e-llm-icon-front.svg"
    LOGO_HEIGHT: ClassVar[int] = 160

    LLAMA_SERVER_BIN: ClassVar[str] = "/app/llama-server"
    LLAMA_INTERNAL_HOST: ClassVar[str] = "127.0.0.1"
    LLAMA_INTERNAL_PORT: ClassVar[int] = 45151
    HEALTH_POLL_INTERVAL: ClassVar[float] = 3.0

    ##### RUNTIME SETTINGS (env + YAML) #####

    DEBUG: bool = True
    GUI_HOST: str = "0.0.0.0"
    GUI_PORT: int = 8080
    LLAMACPP_URL: str = "http://127.0.0.1:45151"
    DATA_DIR: Path = BASE_DIR / "data"

    @computed_field
    @property
    def models_path(self) -> Path:
        return self.DATA_DIR / "models"

    @computed_field
    @property
    def config_path(self) -> Path:
        return self.DATA_DIR / "config" / "config.yaml"

    @computed_field
    @property
    def profiles_path(self) -> Path:
        return self.DATA_DIR / "config" / "profiles"

    @computed_field
    @property
    def data_path(self) -> Path:
        return self.DATA_DIR

    @computed_field
    @property
    def api_url(self) -> str:
        return f"http://{self.GUI_HOST}:{self.GUI_PORT}"

    @classmethod
    def settings_customise_sources(  # noqa: N805
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Env vars > YAML > defaults."""
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


settings = Settings()
