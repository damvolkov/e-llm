"""Unit test fixtures shared across all unit tests."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory with config/models/cache subdirs."""
    for sub in ("config", "models", "cache"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def sample_config_yaml(tmp_data_dir: Path) -> Path:
    """Write a minimal config.yaml and return its path."""
    config_path = tmp_data_dir / "config" / "config.yaml"
    config_path.write_text(
        "server:\n  host: '0.0.0.0'\n  port: 45150\n  alias: test\n"
        "model:\n  path: test.gguf\n  n_gpu_layers: -1\n"
        "context:\n  ctx_size: 2048\n  parallel: 1\n  batch_size: 512\n  ubatch_size: 128\n"
        "cache:\n  type_k: f16\n  type_v: f16\n  no_kv_offload: false\n  defrag_thold: 0.1\n"
        "compute:\n  threads: 2\n  threads_batch: 2\n  flash_attn: false\n  fit: false\n  mlock: false\n  no_mmap: false\n"
        "sampling:\n  temp: 0.7\n  top_p: 0.95\n  top_k: 40\n  min_p: 0.05\n  repeat_penalty: 1.0\n"
        "template:\n  jinja: true\n  no_context_shift: false\n  chat_template: ''\n"
    )
    return config_path


@pytest.fixture
def sample_model_file(tmp_data_dir: Path) -> Path:
    """Create a fake .gguf file in the models dir."""
    model = tmp_data_dir / "models" / "test.gguf"
    model.write_bytes(b"\x00" * 1024)
    return model


@pytest.fixture
def mock_adapter() -> AsyncMock:
    """Mock LlamaCppAdapter with default responses."""
    adapter = AsyncMock()
    adapter.get_health.return_value = {"status": "ok"}
    adapter.get_models.return_value = [{"id": "test"}]
    return adapter
