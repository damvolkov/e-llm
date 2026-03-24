"""e-llm — NiceGUI application entrypoint."""

import asyncio
from collections.abc import Iterable

from nicegui import app, ui

from e_llm.adapters.huggingface import HuggingFaceAdapter
from e_llm.adapters.llamacpp import LlamaCppAdapter
from e_llm.core.health import resolve_health
from e_llm.core.logger import logger
from e_llm.core.settings import settings as st
from e_llm.core.state import State
from e_llm.models.server import ServerConfig
from e_llm.operational.monitor import SystemMonitor
from e_llm.operational.server import ServerManager
from e_llm.operational.system import SystemEvaluator
from e_llm.pages.config import create as create_config
from e_llm.pages.test import create as create_test

state = State()
monitor = SystemMonitor()

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
    monitor.shutdown()
    logger.info("shutdown complete", step="STOP")


##### MONITOR HELPERS #####


_AREA_ALPHA: dict[str, str] = {
    "#4caf50": "rgba(76,175,80,0.10)",
    "#ff9800": "rgba(255,152,0,0.10)",
    "#f44336": "rgba(244,67,54,0.10)",
}


def _update_row(
    row: dict,
    pct: float,
    history: Iterable[float],
    custom_label: str | None = None,
) -> None:
    """Update a single monitor row — label, bar, sparkline."""
    row["pct"].text = custom_label or f"{pct:.0f}%"
    row["bar"].value = pct / 100.0
    color = "#4caf50" if pct < 60 else "#ff9800" if pct < 85 else "#f44336"
    row["bar"].props(f'color="{color}"')
    series = row["spark"].options["series"][0]
    series["data"] = list(history)
    series["lineStyle"]["color"] = color
    series["areaStyle"]["color"] = _AREA_ALPHA.get(color, "rgba(33,150,243,0.10)")
    row["spark"].update()


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

            with (
                ui.card()
                .classes("absolute-right mr-6")
                .style(
                    "padding: 8px 14px; min-width: 220px;"
                    "background: var(--surface) !important;"
                    "border: 1px solid var(--border) !important;"
                    "border-radius: 10px !important;"
                ),
            ):
                with ui.row().classes("items-center gap-2 mb-1"):
                    health_dot = ui.icon("circle", color="red").classes("text-sm")
                    health_label = (
                        ui.label("Stopped")
                        .classes("text-caption text-weight-medium")
                        .style("color: var(--text-dim); font-size: 11px")
                    )

                mon_rows: dict[str, dict] = {}
                for key, _icon, label in [
                    ("vram", "memory", "VRAM"),
                    ("gpu", "developer_board", "GPU"),
                    ("cpu", "memory", "CPU"),
                    ("ram", "storage", "RAM"),
                ]:
                    with ui.row().classes("w-full items-center gap-2").style("height: 18px"):
                        ui.label(label).style("font-size: 9px; color: var(--text-dim); width: 30px; font-weight: 600")
                        pct_label = ui.label("—").style(
                            "font-size: 9px; color: var(--text); width: 32px; text-align: right"
                        )
                        bar = (
                            ui.linear_progress(value=0, show_value=False)
                            .style("height: 4px; flex: 1; border-radius: 2px")
                            .props("instant-feedback")
                        )
                        spark = ui.echart(
                            {
                                "grid": {"top": 0, "bottom": 0, "left": 0, "right": 0},
                                "xAxis": {"show": False, "type": "category"},
                                "yAxis": {"show": False, "type": "value", "min": 0, "max": 100},
                                "series": [
                                    {
                                        "type": "line",
                                        "data": [],
                                        "smooth": True,
                                        "symbol": "none",
                                        "lineStyle": {"width": 1.2, "color": "#2196f3"},
                                        "areaStyle": {"color": "rgba(33,150,243,0.10)"},
                                    }
                                ],
                            }
                        ).style("width: 50px; height: 16px")
                    mon_rows[key] = {"pct": pct_label, "bar": bar, "spark": spark}

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

    async def _poll_monitor() -> None:
        snap = await asyncio.to_thread(monitor.poll)
        _update_row(mon_rows["cpu"], snap.cpu_pct, monitor.cpu_history)
        _update_row(mon_rows["ram"], snap.ram_pct, monitor.ram_history)
        if snap.gpu_available:
            _update_row(mon_rows["gpu"], snap.gpu_util_pct, monitor.gpu_util_history)
            _update_row(
                mon_rows["vram"],
                snap.vram_pct,
                monitor.vram_history,
                f"{snap.vram_used_mb // 1024}/{snap.vram_total_mb // 1024}G",
            )
        else:
            for key in ("gpu", "vram"):
                mon_rows[key]["pct"].text = "n/a"

    ui.timer(st.HEALTH_POLL_INTERVAL, _poll_health)
    ui.timer(2.0, _poll_monitor)


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
