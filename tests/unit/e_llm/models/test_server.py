"""Tests for ServerConfig and spec models."""

from pathlib import Path

import pytest
import yaml

from e_llm.models.server import (
    CacheSpec,
    ComputeSpec,
    ContextSpec,
    ModelSpec,
    SamplingSpec,
    ServerConfig,
    ServerSpec,
    TemplateSpec,
)

##### SPEC DEFAULTS #####


async def test_server_spec_defaults() -> None:
    spec = ServerSpec()
    assert spec.host == "0.0.0.0"
    assert spec.port == 45150
    assert spec.alias == "default"


async def test_model_spec_defaults() -> None:
    spec = ModelSpec()
    assert spec.path == ""
    assert spec.n_gpu_layers == -1


async def test_context_spec_defaults() -> None:
    spec = ContextSpec()
    assert spec.ctx_size == 8192
    assert spec.parallel == 1


async def test_cache_spec_defaults() -> None:
    spec = CacheSpec()
    assert spec.type_k == "f16"
    assert spec.no_kv_offload is False


async def test_compute_spec_defaults() -> None:
    spec = ComputeSpec()
    assert spec.flash_attn is True
    assert spec.fit is True
    assert spec.mlock is True


async def test_sampling_spec_defaults() -> None:
    spec = SamplingSpec()
    assert spec.temp == 0.7
    assert spec.top_p == 0.95


async def test_template_spec_defaults() -> None:
    spec = TemplateSpec()
    assert spec.jinja is True
    assert spec.chat_template == ""


##### SERVER CONFIG #####


async def test_server_config_defaults() -> None:
    config = ServerConfig()
    assert config.server.host == "0.0.0.0"
    assert config.model.path == ""
    assert config.context.ctx_size == 8192


async def test_server_config_from_yaml(sample_config_yaml: Path) -> None:
    config = ServerConfig.from_yaml(sample_config_yaml)
    assert config.server.alias == "test"
    assert config.model.path == "test.gguf"
    assert config.context.ctx_size == 2048


async def test_server_config_from_yaml_missing_file(tmp_path: Path) -> None:
    config = ServerConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert config.server.host == "0.0.0.0"


async def test_server_config_from_yaml_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    config = ServerConfig.from_yaml(empty)
    assert config.server.host == "0.0.0.0"


async def test_server_config_to_yaml(tmp_path: Path) -> None:
    config = ServerConfig(
        server=ServerSpec(alias="saved"),
        model=ModelSpec(path="model.gguf"),
    )
    path = tmp_path / "out.yaml"
    config.to_yaml(path)
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert data["server"]["alias"] == "saved"
    assert data["model"]["path"] == "model.gguf"


async def test_server_config_roundtrip(tmp_path: Path) -> None:
    original = ServerConfig(
        context=ContextSpec(ctx_size=32768, parallel=4),
        sampling=SamplingSpec(temp=0.3),
    )
    path = tmp_path / "rt.yaml"
    original.to_yaml(path)
    loaded = ServerConfig.from_yaml(path)
    assert loaded.context.ctx_size == 32768
    assert loaded.context.parallel == 4
    assert loaded.sampling.temp == pytest.approx(0.3)


async def test_server_config_to_yaml_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "config.yaml"
    ServerConfig().to_yaml(path)
    assert path.exists()
