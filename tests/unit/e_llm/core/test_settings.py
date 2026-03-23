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


async def test_settings_data_paths() -> None:
    s = Settings(DATA_DIR=Path("/tmp/test-ellm"))
    assert s.models_path == Path("/tmp/test-ellm/models")
    assert s.config_path == Path("/tmp/test-ellm/config/config.yaml")
    assert s.data_path == Path("/tmp/test-ellm")


async def test_settings_classvar_constants() -> None:
    assert isinstance(Settings.BASE_DIR, Path)
    assert isinstance(Settings.ASSETS_PATH, Path)
    assert Settings.API_NAME == "e-llm"
    assert isinstance(Settings.API_VERSION, str)


async def test_settings_computed_api_url() -> None:
    s = Settings()
    assert s.api_url == f"http://{s.GUI_HOST}:{s.GUI_PORT}"
