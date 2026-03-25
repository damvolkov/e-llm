"""Health check state resolution — maps server status to UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from e_llm.core.settings import settings as st
from e_llm.models.server import ServerConfig

if TYPE_CHECKING:
    from e_llm.core.state import State


@dataclass(frozen=True, slots=True)
class HealthState:
    """Visual state for the health indicator."""

    color: str
    pulsing: bool
    label: str
    tooltip: str


async def resolve_health(s: State) -> HealthState:
    """Evaluate server status and return the appropriate visual state."""
    ctrl = s.controller
    mgr = ctrl.manager

    if not ctrl.enabled:
        return HealthState("grey", False, "Disabled", "Server disabled by user.\nClick power button to enable.")

    if not mgr.is_running:
        config = ServerConfig.from_yaml(st.config_path)
        if not mgr.find_model(config):
            return HealthState("red", False, "Stopped", "No model loaded.\nDownload one in Configuration → Models.")
        check = await ctrl.enable()
        if not check.available:
            return HealthState("red", False, "Blocked", check.reason)
        return HealthState("orange", True, "Starting...", "Process launched — loading model")

    health_data = await s.adapter.get_health() or {}
    status = health_data.get("status", "")

    match status:
        case "ok":
            model = str(health_data.get("model_path", ""))
            name = model.rsplit("/", 1)[-1] if model else "unknown"
            return HealthState("green", False, "Ready", f"Model: {name}\nPID: {mgr.pid}")
        case "loading model":
            return HealthState("orange", True, "Loading...", f"PID: {mgr.pid}\nLoading model into memory")
        case _:
            return HealthState("orange", True, "Starting...", f"PID: {mgr.pid}\nWaiting for server")
