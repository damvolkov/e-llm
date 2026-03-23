"""llama-server process lifecycle management."""

import asyncio
import signal
import subprocess
from pathlib import Path

from e_llm.core.settings import settings as st
from e_llm.models.server import ServerConfig

##### BOOLEAN FLAGS — config path → cli flag #####

_TOGGLE_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("compute.flash_attn", "--flash-attn", "on"),
    ("compute.fit", "--fit", "on"),
    ("cache.no_kv_offload", "--no-kv-offload", ""),
    ("compute.mlock", "--mlock", ""),
    ("compute.no_mmap", "--no-mmap", ""),
    ("template.jinja", "--jinja", ""),
    ("template.no_context_shift", "--no-context-shift", ""),
)


def _resolve_flag(config: ServerConfig, path: str) -> bool:
    """Resolve a dotted config path to its boolean value."""
    section, field = path.split(".")
    return bool(getattr(getattr(config, section), field))


class ServerManager:
    """Manage the llama-server process lifecycle."""

    __slots__ = ("_process", "_models_dir")

    def __init__(self, models_dir: Path) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._models_dir = models_dir

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self.is_running else None

    def find_model(self, config: ServerConfig) -> Path | None:
        """Resolve model path from config or auto-detect first .gguf."""
        if config.model.path:
            for candidate in (Path(config.model.path), self._models_dir / config.model.path):
                if candidate.exists():
                    return candidate

        gguf_files = sorted(self._models_dir.glob("*.gguf"))
        return gguf_files[0] if gguf_files else None

    async def start(self, config: ServerConfig) -> bool:
        """Start llama-server. Returns True on success."""
        if self.is_running:
            await self.stop()

        if not (model_path := self.find_model(config)):
            return False

        self._process = subprocess.Popen(
            self._sm_build_command(config, model_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True

    async def stop(self) -> None:
        """Gracefully stop the running server."""
        if not self.is_running:
            return
        self._process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(self._process.wait, timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            await asyncio.to_thread(self._process.wait, timeout=5)
        self._process = None

    async def restart(self, config: ServerConfig) -> bool:
        """Stop and restart with new config."""
        await self.stop()
        return await self.start(config)

    def _sm_build_command(self, config: ServerConfig, model: Path) -> list[str]:
        """Build the llama-server CLI from config — declarative flag mapping."""
        cmd = [
            st.LLAMA_SERVER_BIN,
            "--model",
            str(model),
            "--alias",
            config.server.alias,
            "--host",
            st.LLAMA_INTERNAL_HOST,
            "--port",
            str(st.LLAMA_INTERNAL_PORT),
            "--ctx-size",
            str(config.context.ctx_size),
            "--parallel",
            str(config.context.parallel),
            "--batch-size",
            str(config.context.batch_size),
            "--ubatch-size",
            str(config.context.ubatch_size),
            "--cache-type-k",
            config.cache.type_k,
            "--cache-type-v",
            config.cache.type_v,
            "--threads",
            str(config.compute.threads),
            "--threads-batch",
            str(config.compute.threads_batch),
            "--temp",
            str(config.sampling.temp),
            "--top-p",
            str(config.sampling.top_p),
            "--top-k",
            str(config.sampling.top_k),
            "--min-p",
            str(config.sampling.min_p),
            "--repeat-penalty",
            str(config.sampling.repeat_penalty),
        ]

        # Boolean toggle flags — declarative
        for path, flag, value in _TOGGLE_FLAGS:
            if _resolve_flag(config, path):
                cmd.append(flag)
                if value:
                    cmd.append(value)

        # Optional value flags
        if config.template.chat_template:
            cmd.extend(["--chat-template", config.template.chat_template])
        if config.model.n_gpu_layers != -1:
            cmd.extend(["--n-gpu-layers", str(config.model.n_gpu_layers)])
        if config.cache.defrag_thold > 0:
            cmd.extend(["--defrag-thold", str(config.cache.defrag_thold)])

        return cmd
