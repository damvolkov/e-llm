"""e-llm — NiceGUI application entrypoint."""

from pathlib import Path

from nicegui import app, ui

from e_llm.adapters.llamacpp import LlamaCppAdapter
from e_llm.core.settings import settings as st
from e_llm.models.server import ServerConfig
from e_llm.operational.server import ServerManager
from e_llm.pages.config import create as create_config
from e_llm.pages.test import create as create_test

_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "e-llm-landscape-front.svg"

_CSS = """
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
:root {
    --bg: #0a0e14; --surface: #111820; --border: #1a3a4a;
    --accent: #2196f3; --accent-dim: #1a6e8a; --text: #c8d6d8; --text-dim: #6b7f82;
}
body { background: var(--bg) !important; color: var(--text) !important; }
.q-card { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }
.q-header { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
.q-tab-panels, .q-tab-panel { background: transparent !important; }
.q-tab-panel { padding: 16px 0 !important; }
.q-expansion-item { border-radius: 10px !important; }
.q-field--outlined .q-field__control { border-radius: 10px !important; }
.q-btn { border-radius: 10px !important; }
.q-separator { background: var(--border) !important; opacity: .35 !important; }
.q-badge { border-radius: 6px !important; }
.q-linear-progress { border-radius: 6px !important; }
.q-chat-message { max-width: 85% !important; }
.ellm-logo { height: 32px; }
"""

##### SINGLETONS #####

adapter = LlamaCppAdapter(st.LLAMACPP_URL)
server_manager = ServerManager(st.models_path)


##### HEALTH #####


@app.get("/health")
async def health() -> dict[str, object]:
    """Container health probe."""
    server_health = await adapter.get_health()
    return {
        "status": "ok",
        "server": {
            "running": server_manager.is_running,
            "pid": server_manager.pid,
            "healthy": server_health is not None,
        },
    }


##### LIFECYCLE #####


@app.on_startup
async def _on_startup() -> None:
    config = ServerConfig.from_yaml(st.config_path)
    if server_manager.find_model(config):
        started = await server_manager.start(config)
        print(f"[e-llm] llama-server {'started' if started else 'failed'}")
    else:
        print("[e-llm] no model found — download via GUI")


@app.on_shutdown
async def _on_shutdown() -> None:
    await server_manager.stop()


##### GUI #####


def _load_logo() -> str:
    if _LOGO_PATH.exists():
        return _LOGO_PATH.read_text()
    return '<span style="color:var(--accent);font-size:24px;font-weight:700">e-llm</span>'


@ui.page("/")
async def index() -> None:
    """Main page — Configuration (default) + Test tab."""
    ui.dark_mode(True)
    ui.add_head_html(f"<style>{_CSS}</style>")

    with ui.header().classes("items-center justify-between px-6 py-2"):
        ui.html(_load_logo()).classes("ellm-logo")

        with ui.tabs().classes("self-center") as tabs:
            ui.tab("config", icon="tune", label="Configuration")
            ui.tab("test", icon="science", label="Test")

        with ui.row().classes("items-center gap-2"):
            health_dot = ui.icon("circle", color="red").classes("text-lg")
            health_label = ui.label("Stopped").classes("text-caption").style("color: var(--text-dim)")

    with (
        ui.column().classes("w-full max-w-7xl mx-auto px-4 py-2 flex-grow"),
        ui.tab_panels(tabs, value="config").classes("w-full"),
    ):
        with ui.tab_panel("config"):
            create_config(server_manager, adapter)
        with ui.tab_panel("test"):
            create_test(adapter, server_manager)

    async def _check_health() -> None:
        running = server_manager.is_running
        health_data = await adapter.get_health() if running else None
        status = health_data.get("status", "") if health_data else ""

        if not running:
            config = ServerConfig.from_yaml(st.config_path)
            if server_manager.find_model(config):
                await server_manager.start(config)
                _set_state("orange", True, "Starting...", "Process launched — loading model")
                return
            _set_state("red", False, "Stopped", "No model loaded.\nDownload one in Configuration → Models.")
        elif status == "ok":
            model = health_data.get("model_path", "")
            name = model.rsplit("/", 1)[-1] if model else "unknown"
            _set_state("green", False, "Ready", f"Model: {name}\nPID: {server_manager.pid}")
        elif status == "loading model":
            _set_state("orange", True, "Loading...", f"PID: {server_manager.pid}\nLoading model into memory")
        else:
            _set_state("orange", True, "Starting...", f"PID: {server_manager.pid}\nWaiting for server")

    def _set_state(color: str, pulsing: bool, label: str, tip: str) -> None:
        health_dot.props(f'color="{color}"')
        health_dot.style("animation: pulse 1.4s ease-in-out infinite" if pulsing else "")
        health_label.text = label
        health_dot.tooltip(tip)

    ui.timer(3.0, _check_health)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="e-llm",
        host=st.GUI_HOST,
        port=st.GUI_PORT,
        reload=st.DEBUG,
        dark=True,
        favicon="🤖",
    )
