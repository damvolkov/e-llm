"""Application state container for dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from e_llm.adapters.huggingface import HuggingFaceAdapter
    from e_llm.adapters.llamacpp import LlamaCppAdapter
    from e_llm.models.system import SystemInfo
    from e_llm.operational.controller import ServerController
    from e_llm.operational.server import ServerManager


class State:
    """Mutable state container — holds adapters, managers, and hardware info."""

    __slots__ = ("adapter", "hf_adapter", "server_manager", "controller", "system_info")

    adapter: LlamaCppAdapter
    hf_adapter: HuggingFaceAdapter
    server_manager: ServerManager
    controller: ServerController
    system_info: SystemInfo | None
