"""e-llm — NiceGUI application entrypoint."""

from nicegui import app, ui

from e_llm.adapters.huggingface import HuggingFaceAdapter
from e_llm.adapters.llamacpp import LlamaCppAdapter
from e_llm.core.health import resolve_health
from e_llm.core.logger import logger
from e_llm.core.settings import settings as st
from e_llm.core.state import State
from e_llm.models.server import ServerConfig
from e_llm.operational.server import ServerManager
from e_llm.operational.system import SystemEvaluator
from e_llm.pages.config import create as create_config
from e_llm.pages.test import create as create_test

state = State()

_THEME_CSS = (st.ASSETS_PATH / "theme.css").read_text() if (st.ASSETS_PATH / "theme.css").exists() else ""
_FAVICON_LINK = '<link rel="icon" type="image/svg+xml" href="/assets/e-llm-icon-front.svg">'
_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet">'
)


def _load_logo() -> str:
    """Load landscape SVG with explicit height, fallback to text."""
    if st.LOGO_PATH.exists():
        return st.LOGO_PATH.read_text().replace("<svg ", f'<svg height="{st.LOGO_HEIGHT}" ', 1)
    return f'<span style="color:var(--accent);font-size:24px;font-weight:700">{st.API_NAME}</span>'


##### SERVER — HEALTH + LIFECYCLE #####


@app.get("/health")
async def get_health() -> dict[str, object]:
    """Container health probe."""
    server_health = await state.adapter.get_health()
    return {
        "status": "ok",
        "server": {
            "running": state.server_manager.is_running,
            "pid": state.server_manager.pid,
            "healthy": server_health is not None,
        },
    }


@app.on_startup
async def on_startup() -> None:
    """Initialize adapters, evaluate hardware, auto-start server."""
    state.adapter = LlamaCppAdapter(st.LLAMACPP_URL)
    state.hf_adapter = HuggingFaceAdapter()
    state.server_manager = ServerManager(st.models_path)
    state.system_info = await SystemEvaluator(st.data_path).evaluate()

    logger.info("adapters initialized", step="START", url=st.LLAMACPP_URL)
    logger.info(
        "system evaluated",
        step="OK",
        cpu=state.system_info.cpu.model,
        ram=f"{state.system_info.ram.total_gb}GB",
        gpu=state.system_info.gpu.name if state.system_info.gpu else "none",
    )

    config = ServerConfig.from_yaml(st.config_path)
    if not (model := state.server_manager.find_model(config)):
        logger.info("no model found — download via GUI", step="WARN")
        return
    started = await state.server_manager.start(config)
    logger.info("llama-server", step="OK" if started else "ERROR", model=model.name)


@app.on_shutdown
async def on_shutdown() -> None:
    """Graceful shutdown."""
    await state.server_manager.stop()
    logger.info("shutdown complete", step="STOP")


##### CLIENT — GUI #####


@ui.page("/")
async def index() -> None:
    """Main page — Configuration (default) + Test tab."""
    ui.dark_mode(True)
    ui.add_head_html(_FONT_LINK)
    ui.add_head_html(f"<style>{_THEME_CSS}</style>")
    ui.add_head_html(_FAVICON_LINK)

    with (
        ui.header().classes("items-center justify-center px-6 py-3 no-wrap"),
        ui.column().classes("w-full max-w-7xl items-center gap-2"),
    ):
        ui.html(_load_logo())

        with ui.row().classes("w-full items-center justify-center gap-4 mt-1"):
            with (
                ui.tabs()
                .props("inline-label no-caps indicator-color=primary")
                .classes("self-center text-subtitle1") as tabs
            ):
                ui.tab("config", icon="tune", label="Configuration")
                ui.tab("test", icon="science", label="Test")

            with ui.row().classes("items-center gap-2 absolute-right pr-6"):
                health_dot = ui.icon("circle", color="red").classes("text-lg")
                health_label = ui.label("Stopped").classes("text-caption").style("color: var(--text-dim)")

    ui.separator().classes("ellm-divider")

    with (
        ui.column().classes("w-full max-w-7xl mx-auto px-4 py-4 flex-grow"),
        ui.tab_panels(tabs, value="config").classes("w-full"),
    ):
        with ui.tab_panel("config"):
            create_config(state)
        with ui.tab_panel("test"):
            create_test(state)

    async def _poll_health() -> None:
        hs = await resolve_health(state)
        health_dot.props(f'color="{hs.color}"')
        health_dot.style("animation: pulse 1.4s ease-in-out infinite" if hs.pulsing else "")
        health_label.text = hs.label
        health_dot.tooltip(hs.tooltip)

    ui.timer(st.HEALTH_POLL_INTERVAL, _poll_health)


##### STATIC + RUN #####

app.add_static_files("/assets", str(st.ASSETS_PATH))

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=st.API_NAME,
        host=st.GUI_HOST,
        port=st.GUI_PORT,
        reload=st.DEBUG,
        dark=True,
        favicon="/assets/e-llm-icon-front.svg",
    )
