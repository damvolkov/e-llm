"""Tests for ServerManager."""

from pathlib import Path
from unittest.mock import patch

import pytest

from e_llm.models.server import CacheSpec, ComputeSpec, ModelSpec, ServerConfig
from e_llm.operational.server import ServerManager

##### FIND MODEL #####


async def test_find_model_by_config_path(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(path="test.gguf"))
    found = mgr.find_model(config)
    assert found is not None
    assert found.name == "test.gguf"


async def test_find_model_auto_detect(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(path=""))
    found = mgr.find_model(config)
    assert found is not None
    assert found.suffix == ".gguf"


async def test_find_model_none_when_empty(tmp_data_dir: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(path=""))
    assert mgr.find_model(config) is None


async def test_find_model_absolute_path(tmp_path: Path) -> None:
    model = tmp_path / "absolute.gguf"
    model.write_bytes(b"\x00")
    mgr = ServerManager(tmp_path)
    config = ServerConfig(model=ModelSpec(path=str(model)))
    assert mgr.find_model(config) == model


async def test_find_model_missing_absolute(tmp_path: Path) -> None:
    mgr = ServerManager(tmp_path)
    config = ServerConfig(model=ModelSpec(path="/nonexistent/model.gguf"))
    assert mgr.find_model(config) is None


##### PROPERTIES #####


async def test_manager_not_running_by_default(tmp_path: Path) -> None:
    mgr = ServerManager(tmp_path)
    assert mgr.is_running is False
    assert mgr.pid is None


##### START / STOP #####


async def test_start_returns_false_without_model(tmp_path: Path) -> None:
    mgr = ServerManager(tmp_path)
    config = ServerConfig(model=ModelSpec(path="nonexistent.gguf"))
    result = await mgr.start(config)
    assert result is False
    assert mgr.is_running is False


async def test_start_launches_process(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(path="test.gguf"))
    with patch("e_llm.operational.server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.pid = 12345
        result = await mgr.start(config)
        assert result is True
        assert mgr.is_running is True
        assert mgr.pid == 12345


async def test_stop_on_non_running_is_noop(tmp_path: Path) -> None:
    mgr = ServerManager(tmp_path)
    await mgr.stop()
    assert mgr.is_running is False


async def test_restart_calls_stop_then_start(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(path="test.gguf"))
    with patch("e_llm.operational.server.subprocess.Popen") as mock_popen:
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.pid = 999
        result = await mgr.restart(config)
        assert result is True


##### BUILD COMMAND #####


async def test_build_command_basic(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(
        server=ServerConfig().server,
        model=ModelSpec(path="test.gguf"),
        compute=ServerConfig().compute,
    )
    cmd = mgr._sm_build_command(config, sample_model_file)
    assert cmd[0] == "/app/llama-server"
    assert "--model" in cmd
    assert str(sample_model_file) in cmd
    assert "--jinja" in cmd


async def test_build_command_no_flash_attn(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(compute=ComputeSpec(flash_attn=False, fit=False))
    cmd = mgr._sm_build_command(config, sample_model_file)
    assert "--flash-attn" not in cmd
    assert "--fit" not in cmd


@pytest.mark.parametrize(
    ("flag", "field", "value"),
    [
        ("--no-kv-offload", "no_kv_offload", True),
        ("--mlock", "mlock", True),
        ("--no-mmap", "no_mmap", True),
    ],
    ids=["no-kv-offload", "mlock", "no-mmap"],
)
async def test_build_command_boolean_flags(
    tmp_data_dir: Path, sample_model_file: Path, flag: str, field: str, value: bool
) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    if field == "no_kv_offload":
        config = ServerConfig(cache=CacheSpec(**{field: value}))
    else:
        config = ServerConfig(compute=ComputeSpec(**{field: value}))
    cmd = mgr._sm_build_command(config, sample_model_file)
    assert flag in cmd


async def test_build_command_gpu_layers(tmp_data_dir: Path, sample_model_file: Path) -> None:
    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(model=ModelSpec(n_gpu_layers=32))
    cmd = mgr._sm_build_command(config, sample_model_file)
    assert "--n-gpu-layers" in cmd
    idx = cmd.index("--n-gpu-layers")
    assert cmd[idx + 1] == "32"


async def test_build_command_chat_template(tmp_data_dir: Path, sample_model_file: Path) -> None:
    from e_llm.models.server import TemplateSpec

    mgr = ServerManager(tmp_data_dir / "models")
    config = ServerConfig(template=TemplateSpec(chat_template="custom"))
    cmd = mgr._sm_build_command(config, sample_model_file)
    assert "--chat-template" in cmd
    idx = cmd.index("--chat-template")
    assert cmd[idx + 1] == "custom"
