"""Server lifecycle controller — safe start/stop with resource gating."""

import asyncio
import contextlib
from dataclasses import dataclass

import pynvml

from e_llm.core.logger import logger
from e_llm.core.settings import settings as st
from e_llm.models.server import ServerConfig
from e_llm.operational.server import ServerManager

_VRAM_BUSY_PCT = 50.0


@dataclass(frozen=True, slots=True)
class ResourceCheck:
    """Result of a GPU resource availability check."""

    available: bool
    vram_used_pct: float
    reason: str


class ServerController:
    """Controls server lifecycle with user intent and resource gating.

    Wraps ServerManager with an enabled/disabled toggle and GPU resource
    checks before starting.  The health poller respects ``enabled`` so
    disabling the server keeps it stopped until the user re-enables it.
    """

    __slots__ = ("_manager", "_enabled")

    def __init__(self, manager: ServerManager) -> None:
        self._manager = manager
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def manager(self) -> ServerManager:
        return self._manager

    def check_resources(self) -> ResourceCheck:
        """Check GPU VRAM before starting. Safe no-op when no GPU is present."""
        nvml_started = False
        try:
            pynvml.nvmlInit()
            nvml_started = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_pct = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total else 0.0
        except pynvml.NVMLError:
            return ResourceCheck(available=True, vram_used_pct=0.0, reason="No GPU — CPU mode")
        finally:
            if nvml_started:
                with contextlib.suppress(pynvml.NVMLError):
                    pynvml.nvmlShutdown()

        if vram_pct > _VRAM_BUSY_PCT:
            return ResourceCheck(
                available=False,
                vram_used_pct=vram_pct,
                reason=f"VRAM {vram_pct:.0f}% used — other processes active",
            )
        return ResourceCheck(available=True, vram_used_pct=vram_pct, reason="Resources available")

    async def enable(self) -> ResourceCheck:
        """Enable and start server. Returns ResourceCheck (.available may be False)."""
        check = await asyncio.to_thread(self.check_resources)
        if not check.available:
            logger.info("start blocked", step="WARN", reason=check.reason)
            return check
        self._enabled = True
        config = ServerConfig.from_yaml(st.config_path)
        if self._manager.find_model(config):
            await self._manager.start(config)
            logger.info("server enabled", step="OK")
        return check

    async def disable(self) -> None:
        """Disable and stop server."""
        self._enabled = False
        await self._manager.stop()
        logger.info("server disabled", step="STOP")

    async def toggle(self) -> ResourceCheck | None:
        """Toggle server state. Returns ResourceCheck when enabling, None when disabling."""
        if self._enabled:
            await self.disable()
            return None
        return await self.enable()

    async def restart(self, config: ServerConfig) -> bool:
        """Re-enable and restart with new config. Returns True on success."""
        self._enabled = True
        await self._manager.stop()
        check = await asyncio.to_thread(self.check_resources)
        if not check.available:
            self._enabled = False
            logger.info("restart blocked", step="WARN", reason=check.reason)
            return False
        started = await self._manager.start(config)
        if not started:
            logger.info("restart failed — no model", step="WARN")
        return started
