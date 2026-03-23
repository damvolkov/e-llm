"""Tests for application settings."""

from pathlib import Path

from e_llm.core.settings import Settings

##### SETTINGS DEFAULTS #####


async def test_settings_defaults() -> None:
    s = Settings()
    assert s.DEBUG is True
    assert s.GUI_HOST == "0.0.0.0"
    assert s.GUI_PORT == 8080
    assert s.LLAMACPP_URL == "http://127.0.0.1:45150"


async def test_settings_computed_paths() -> None:
    s = Settings(DATA_DIR=Path("/tmp/test-ellm"))
    assert s.config_path == Path("/tmp/test-ellm/config/config.yaml")
    assert s.models_path == Path("/tmp/test-ellm/models")
    assert s.cache_path == Path("/tmp/test-ellm/cache")


async def test_settings_base_dir_is_path() -> None:
    assert isinstance(Settings.BASE_DIR, Path)
