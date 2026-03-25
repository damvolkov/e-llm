"""Tests for ServerController — lifecycle control with resource gating."""

from pathlib import Path
from types import SimpleNamespace

import pynvml
import pytest
from pytest_mock import MockerFixture

from e_llm.models.server import ServerConfig
from e_llm.operational.controller import _VRAM_BUSY_PCT, ResourceCheck, ServerController
from e_llm.operational.server import ServerManager

##### FIXTURES #####


@pytest.fixture
def manager(tmp_data_dir: Path) -> ServerManager:
    return ServerManager(tmp_data_dir / "models")


@pytest.fixture
def controller(manager: ServerManager) -> ServerController:
    return ServerController(manager)


##### PROPERTIES #####


async def test_controller_enabled_by_default(controller: ServerController) -> None:
    assert controller.enabled is True


async def test_controller_exposes_manager(controller: ServerController, manager: ServerManager) -> None:
    assert controller.manager is manager


##### CHECK RESOURCES #####


async def test_check_resources_no_gpu(controller: ServerController, mocker: MockerFixture) -> None:
    """No GPU available — returns available=True for CPU mode."""
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlInit",
        side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED),
    )
    check = controller.check_resources()
    assert check.available is True
    assert check.vram_used_pct == 0.0
    assert "CPU" in check.reason


async def test_check_resources_vram_free(controller: ServerController, mocker: MockerFixture) -> None:
    """GPU with low VRAM usage — available."""
    mocker.patch("e_llm.operational.controller.pynvml.nvmlInit")
    mocker.patch("e_llm.operational.controller.pynvml.nvmlShutdown")
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlDeviceGetHandleByIndex",
        return_value="handle",
    )
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlDeviceGetMemoryInfo",
        return_value=SimpleNamespace(used=1024**3, total=8 * 1024**3),
    )
    check = controller.check_resources()
    assert check.available is True
    assert check.vram_used_pct == pytest.approx(12.5)


async def test_check_resources_vram_busy(controller: ServerController, mocker: MockerFixture) -> None:
    """GPU with high VRAM usage — blocked."""
    mocker.patch("e_llm.operational.controller.pynvml.nvmlInit")
    mocker.patch("e_llm.operational.controller.pynvml.nvmlShutdown")
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlDeviceGetHandleByIndex",
        return_value="handle",
    )
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlDeviceGetMemoryInfo",
        return_value=SimpleNamespace(used=7 * 1024**3, total=8 * 1024**3),
    )
    check = controller.check_resources()
    assert check.available is False
    assert check.vram_used_pct > _VRAM_BUSY_PCT


@pytest.mark.parametrize(
    ("used_gb", "total_gb", "expected_available"),
    [(3, 8, True), (4, 8, True), (5, 8, False), (0, 8, True)],
    ids=["below-threshold", "at-threshold", "above-threshold", "empty"],
)
async def test_check_resources_threshold_boundary(
    controller: ServerController,
    mocker: MockerFixture,
    used_gb: int,
    total_gb: int,
    expected_available: bool,
) -> None:
    mocker.patch("e_llm.operational.controller.pynvml.nvmlInit")
    mocker.patch("e_llm.operational.controller.pynvml.nvmlShutdown")
    mocker.patch("e_llm.operational.controller.pynvml.nvmlDeviceGetHandleByIndex", return_value="h")
    mocker.patch(
        "e_llm.operational.controller.pynvml.nvmlDeviceGetMemoryInfo",
        return_value=SimpleNamespace(used=used_gb * 1024**3, total=total_gb * 1024**3),
    )
    check = controller.check_resources()
    assert check.available is expected_available


##### ENABLE #####


async def test_enable_starts_server(
    controller: ServerController,
    sample_model_file: Path,
    sample_config_yaml: Path,
    mocker: MockerFixture,
) -> None:
    controller._enabled = False
    mocker.patch.object(ServerController, "check_resources", return_value=ResourceCheck(True, 10.0, "ok"))
    mocker.patch(
        "e_llm.operational.controller.ServerConfig.from_yaml",
        return_value=ServerConfig.from_yaml(sample_config_yaml),
    )
    mock_start = mocker.patch.object(ServerManager, "start", return_value=True)
    check = await controller.enable()
    assert check.available is True
    assert controller.enabled is True
    mock_start.assert_called_once()


async def test_enable_blocked_by_resources(controller: ServerController, mocker: MockerFixture) -> None:
    controller._enabled = False
    mocker.patch.object(
        ServerController,
        "check_resources",
        return_value=ResourceCheck(False, 90.0, "VRAM busy"),
    )
    check = await controller.enable()
    assert check.available is False
    assert controller.enabled is False


##### DISABLE #####


async def test_disable_stops_server(controller: ServerController, mocker: MockerFixture) -> None:
    mock_stop = mocker.patch.object(ServerManager, "stop")
    await controller.disable()
    assert controller.enabled is False
    mock_stop.assert_called_once()


##### TOGGLE #####


async def test_toggle_from_enabled_disables(controller: ServerController, mocker: MockerFixture) -> None:
    mocker.patch.object(ServerManager, "stop")
    result = await controller.toggle()
    assert result is None
    assert controller.enabled is False


async def test_toggle_from_disabled_enables(
    controller: ServerController,
    sample_model_file: Path,
    sample_config_yaml: Path,
    mocker: MockerFixture,
) -> None:
    controller._enabled = False
    mocker.patch.object(ServerController, "check_resources", return_value=ResourceCheck(True, 5.0, "ok"))
    mocker.patch(
        "e_llm.operational.controller.ServerConfig.from_yaml",
        return_value=ServerConfig.from_yaml(sample_config_yaml),
    )
    mocker.patch.object(ServerManager, "start", return_value=True)
    result = await controller.toggle()
    assert result is not None
    assert result.available is True
    assert controller.enabled is True


##### RESTART #####


async def test_restart_success(
    controller: ServerController,
    sample_model_file: Path,
    sample_config_yaml: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(ServerController, "check_resources", return_value=ResourceCheck(True, 5.0, "ok"))
    mocker.patch.object(ServerManager, "stop")
    mocker.patch.object(ServerManager, "start", return_value=True)
    from e_llm.models.server import ServerConfig

    config = ServerConfig.from_yaml(sample_config_yaml)
    result = await controller.restart(config)
    assert result is True
    assert controller.enabled is True


async def test_restart_blocked_by_resources(controller: ServerController, mocker: MockerFixture) -> None:
    mocker.patch.object(ServerController, "check_resources", return_value=ResourceCheck(False, 85.0, "busy"))
    mocker.patch.object(ServerManager, "stop")
    from e_llm.models.server import ServerConfig

    result = await controller.restart(ServerConfig())
    assert result is False
    assert controller.enabled is False


##### RESOURCE CHECK DATACLASS #####


async def test_resource_check_frozen() -> None:
    check = ResourceCheck(available=True, vram_used_pct=0.0, reason="ok")
    with pytest.raises(AttributeError):
        check.available = False  # type: ignore[misc]
