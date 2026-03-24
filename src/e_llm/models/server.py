from pathlib import Path

import yaml
from pydantic import BaseModel

##### SERVER BINDING #####


class ServerSpec(BaseModel):
    """llama.cpp server binding."""

    host: str = "0.0.0.0"
    port: int = 45150
    alias: str = "default"


##### MODEL #####


class ModelSpec(BaseModel):
    """Model file and GPU layer configuration."""

    path: str = ""
    n_gpu_layers: int = -1


##### CONTEXT #####


class ContextSpec(BaseModel):
    """Context window and batching."""

    ctx_size: int = 8192
    parallel: int = 1
    batch_size: int = 2048
    ubatch_size: int = 512


##### KV CACHE #####


class CacheSpec(BaseModel):
    """KV cache parameters."""

    type_k: str = "f16"
    type_v: str = "f16"
    no_kv_offload: bool = False
    defrag_thold: float = 0.1


##### COMPUTE #####


class ComputeSpec(BaseModel):
    """Compute and threading."""

    threads: int = 4
    threads_batch: int = 4
    flash_attn: bool = True
    fit: bool = True
    mlock: bool = True
    no_mmap: bool = False


##### SAMPLING #####


class SamplingSpec(BaseModel):
    """Sampling parameters."""

    temp: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.0


##### TEMPLATE #####


class TemplateSpec(BaseModel):
    """Template and formatting."""

    jinja: bool = True
    no_context_shift: bool = False
    chat_template: str = ""


##### AGGREGATE CONFIG #####


class ServerConfig(BaseModel):
    """Full llama.cpp server configuration persisted as YAML."""

    server: ServerSpec = ServerSpec()
    model: ModelSpec = ModelSpec()
    context: ContextSpec = ContextSpec()
    cache: CacheSpec = CacheSpec()
    compute: ComputeSpec = ComputeSpec()
    sampling: SamplingSpec = SamplingSpec()
    template: TemplateSpec = TemplateSpec()

    @classmethod
    def from_yaml(cls, path: Path) -> "ServerConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data) if data else cls()

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(), default_flow_style=False, sort_keys=False))
