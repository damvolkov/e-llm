"""Bench tab — configure, run llama-bench, visualize throughput results."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from e_llm.core.settings import settings as st
from e_llm.core.state import State
from e_llm.models.bench import BenchConfig, BenchPoint
from e_llm.operational.bench import run_bench

_CACHE_TYPES: list[str] = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
_PROMPT_SIZES: tuple[int, ...] = (128, 256, 512, 1024, 2048)
_GEN_SIZES: tuple[int, ...] = (64, 128, 256, 512)

_BAR_COLORS: dict[str, str] = {"pp": "#42a5f5", "tg": "#66bb6a"}


def _find_models(models_path: Path) -> list[str]:
    return sorted(str(p.relative_to(models_path)) for p in models_path.rglob("*.gguf"))


def _build_chart(points: list[BenchPoint]) -> dict:
    colors = [_BAR_COLORS.get(p.test[:2], "#ab47bc") for p in points]
    return {
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c} t/s"},
        "xAxis": {
            "type": "category",
            "data": [p.test for p in points],
            "axisLabel": {"color": "#aaa", "fontSize": 11},
        },
        "yAxis": {
            "type": "value",
            "name": "tokens / sec",
            "nameTextStyle": {"color": "#aaa", "fontSize": 10},
            "axisLabel": {"color": "#aaa", "fontSize": 10},
            "splitLine": {"lineStyle": {"color": "#333"}},
        },
        "series": [
            {
                "type": "bar",
                "data": [
                    {"value": round(p.avg_ts, 1), "itemStyle": {"color": c}}
                    for p, c in zip(points, colors, strict=False)
                ],
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": "{c}",
                    "color": "#eee",
                    "fontSize": 11,
                },
            }
        ],
    }


def create(s: State) -> None:
    """Build bench page — configure, run, visualize."""
    models = _find_models(st.models_path)

    ##### CONFIG CARD #####

    with ui.card().classes("w-full").style("background: var(--surface); border: 1px solid var(--border)"):
        ui.label("Benchmark Configuration").classes("text-subtitle1 text-weight-bold mb-3")

        with ui.row().classes("w-full gap-4 flex-wrap items-end"):
            model_select = (
                ui.select(
                    options=models,
                    value=models[0] if models else None,
                    label="Model",
                )
                .classes("flex-1")
                .style("min-width: 280px")
            )
            reps = (
                ui.number("Repetitions", value=3, min=1, max=10, precision=0)
                .classes("w-32")
                .tooltip("Number of times each test is repeated")
            )

        with ui.row().classes("w-full gap-6 flex-wrap items-center mt-3"):
            gpu_layers = ui.number("GPU layers", value=-1, min=-1, max=200, precision=0).classes("w-28")
            flash_attn = ui.checkbox("Flash attn", value=True)
            no_mmap = ui.checkbox("No mmap", value=True)
            type_k = ui.select(_CACHE_TYPES, value="q8_0", label="Cache K").classes("w-32")
            type_v = ui.select(_CACHE_TYPES, value="q4_0", label="Cache V").classes("w-32")
            threads = ui.number("Threads", value=20, min=1, max=64, precision=0).classes("w-24")
            threads_batch = ui.number("Batch thr.", value=32, min=1, max=64, precision=0).classes("w-24")

        with ui.row().classes("w-full gap-8 flex-wrap mt-3"):
            with ui.column().classes("gap-1"):
                ui.label("Prompt tokens (pp)").style("font-size: 11px; font-weight: 600; color: var(--text-dim)")
                with ui.row().classes("gap-2"):
                    prompt_checks = {n: ui.checkbox(str(n), value=(n == 512)) for n in _PROMPT_SIZES}

            with ui.column().classes("gap-1"):
                ui.label("Generate tokens (tg)").style("font-size: 11px; font-weight: 600; color: var(--text-dim)")
                with ui.row().classes("gap-2"):
                    gen_checks = {n: ui.checkbox(str(n), value=(n == 128)) for n in _GEN_SIZES}

        with ui.row().classes("w-full items-center gap-3 mt-4"):
            run_btn = ui.button("Run Benchmark", icon="speed").props('color="primary"').classes("px-6")
            status_label = ui.label("").style("font-size: 11px; color: var(--text-dim)")

    ##### LOG CARD #####

    log_card = ui.card().classes("w-full mt-3").style("background: var(--surface); border: 1px solid var(--border)")
    log_card.set_visibility(False)
    with log_card:
        with ui.row().classes("w-full items-center justify-between mb-2"):
            ui.label("Execution Log").classes("text-subtitle2 text-weight-bold")
            ui.badge("Server paused during run", color="orange").style("font-size: 10px")
        log_output = (
            ui.log(max_lines=300)
            .classes("w-full")
            .style(
                "height: 180px; font-size: 10px; font-family: 'JetBrains Mono', monospace;"
                "background: #0d0d0d; border-radius: 6px; padding: 8px"
            )
        )

    ##### RESULTS CARD #####

    results_card = ui.card().classes("w-full mt-3").style("background: var(--surface); border: 1px solid var(--border)")
    results_card.set_visibility(False)
    with results_card:
        ui.label("Results").classes("text-subtitle2 text-weight-bold mb-2")
        results_table = (
            ui.table(
                columns=[
                    {"name": "test", "label": "Test", "field": "test", "align": "left"},
                    {"name": "avg_ts", "label": "Avg t/s", "field": "avg_ts", "align": "right"},
                    {"name": "sd_ts", "label": "± SD", "field": "sd_ts", "align": "right"},
                ],
                rows=[],
            )
            .classes("w-full")
            .props("dense flat")
        )
        results_chart = ui.echart({}).classes("w-full").style("height: 280px; margin-top: 12px")

    ##### RUN HANDLER #####

    async def _run() -> None:
        model_name = model_select.value
        if not model_name:
            ui.notify("Select a model first", type="warning")
            return

        model_path = st.models_path / model_name
        if not model_path.exists():
            ui.notify(f"Model not found: {model_name}", type="negative")
            return

        n_prompt = tuple(n for n, cb in prompt_checks.items() if cb.value)
        n_gen = tuple(n for n, cb in gen_checks.items() if cb.value)
        if not n_prompt and not n_gen:
            ui.notify("Select at least one prompt or gen token count", type="warning")
            return

        config = BenchConfig(
            model_path=model_path,
            n_gpu_layers=int(gpu_layers.value),
            flash_attn=flash_attn.value,
            type_k=type_k.value,
            type_v=type_v.value,
            n_prompt=n_prompt if n_prompt else (0,),
            n_gen=n_gen if n_gen else (0,),
            repetitions=int(reps.value),
            threads=int(threads.value),
            threads_batch=int(threads_batch.value),
            no_mmap=no_mmap.value,
        )

        run_btn.props(add="loading")
        run_btn.disable()
        log_card.set_visibility(True)
        log_output.clear()
        status_label.set_text("Stopping server…")

        await s.controller.disable()
        status_label.set_text("Running benchmark…")

        result = await run_bench(config, on_log=lambda line: log_output.push(line))

        status_label.set_text("Restarting server…")
        await s.controller.enable()

        run_btn.props(remove="loading")
        run_btn.enable()

        if result is None:
            status_label.set_text("Benchmark failed — see log")
            ui.notify("Benchmark failed", type="negative")
            return

        status_label.set_text(f"Done — {len(result.points)} tests · {result.model}")

        results_table.rows = [
            {"test": p.test, "avg_ts": f"{p.avg_ts:.1f}", "sd_ts": f"±{p.sd_ts:.1f}"} for p in result.points
        ]
        results_table.update()

        if result.points:
            results_chart.options.clear()
            results_chart.options.update(_build_chart(list(result.points)))
            results_chart.update()

        results_card.set_visibility(True)
        ui.notify(f"Benchmark complete — {len(result.points)} tests", type="positive")

    run_btn.on_click(_run)
