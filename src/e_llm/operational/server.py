import asyncio
import signal
import subprocess
from pathlib import Path

from e_llm.models.server import ServerConfig

_LLAMA_SERVER_BIN = "/app/llama-server"
_INTERNAL_HOST = "127.0.0.1"
_INTERNAL_PORT = 45150


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
            candidate = Path(config.model.path)
            if candidate.exists():
                return candidate
            candidate = self._models_dir / config.model.path
            if candidate.exists():
                return candidate

        gguf_files = sorted(self._models_dir.glob("*.gguf"))
        return gguf_files[0] if gguf_files else None

    async def start(self, config: ServerConfig) -> bool:
        """Start llama-server with the given config. Returns True on success."""
        if self.is_running:
            await self.stop()

        model_path = self.find_model(config)
        if not model_path:
            return False

        cmd = self._sm_build_command(config, model_path)
        self._process = subprocess.Popen(
            cmd,
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
        """Build the llama-server command line from config."""
        cmd = [
            _LLAMA_SERVER_BIN,
            "--model",
            str(model),
            "--alias",
            config.server.alias,
            "--host",
            _INTERNAL_HOST,
            "--port",
            str(_INTERNAL_PORT),
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

        if config.compute.flash_attn:
            cmd.extend(["--flash-attn", "on"])

        if config.compute.fit:
            cmd.extend(["--fit", "on"])

        if config.cache.no_kv_offload:
            cmd.append("--no-kv-offload")

        if config.compute.mlock:
            cmd.append("--mlock")

        if config.compute.no_mmap:
            cmd.append("--no-mmap")

        if config.template.jinja:
            cmd.append("--jinja")

        if config.template.no_context_shift:
            cmd.append("--no-context-shift")

        if config.template.chat_template:
            cmd.extend(["--chat-template", config.template.chat_template])

        if config.model.n_gpu_layers != -1:
            cmd.extend(["--n-gpu-layers", str(config.model.n_gpu_layers)])

        if config.cache.defrag_thold > 0:
            cmd.extend(["--defrag-thold", str(config.cache.defrag_thold)])

        return cmd
