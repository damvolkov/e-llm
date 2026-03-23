"""Configuration page — system info, model management, server parameters."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from e_llm.adapters.huggingface import hf_adapter
from e_llm.core.settings import settings as st
from e_llm.models.server import (
    CacheSpec,
    ComputeSpec,
    ContextSpec,
    ModelSpec,
    SamplingSpec,
    ServerConfig,
    ServerSpec,
    TemplateSpec,
)
from e_llm.operational.models import GGUFFile, list_quants, search_models
from e_llm.operational.system import SystemEvaluator

if TYPE_CHECKING:
    from e_llm.adapters.llamacpp import LlamaCppAdapter
    from e_llm.operational.server import ServerManager

_evaluator = SystemEvaluator(st.DATA_DIR)
_CACHE_TYPES = ["f16", "f32", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]


def _list_available_models() -> dict[str, str]:
    models_dir = st.models_path
    models_dir.mkdir(parents=True, exist_ok=True)
    return {p.name: p.name for p in sorted(models_dir.glob("*.gguf"))}


def create(manager: ServerManager, adapter: LlamaCppAdapter) -> None:
    """Build unified configuration page."""
    config = ServerConfig.from_yaml(st.config_path)

    ##### SYSTEM INFORMATION #####

    with ui.expansion("System Information", icon="computer").classes("w-full").props("header-class=text-h6"):
        info_container = ui.column().classes("w-full gap-0")

        async def _refresh_system() -> None:
            info_container.clear()
            with info_container:
                ui.spinner(size="sm")
            info = await _evaluator.evaluate()
            info_container.clear()
            rows: list[tuple[str, str, str]] = [
                ("CPU", "memory", info.cpu.model),
                ("Cores", "developer_board", f"{info.cpu.cores_physical}P / {info.cpu.cores_logical}L"),
                ("Frequency", "speed", f"{info.cpu.frequency_mhz:.0f} MHz"),
                ("RAM Total", "storage", f"{info.ram.total_gb} GB"),
                ("RAM Free", "check_circle", f"{info.ram.available_gb} GB ({100 - info.ram.usage_pct:.0f}% free)"),
                ("Disk", "hard_drive", f"{info.disk.free_gb} GB free / {info.disk.total_gb} GB total"),
            ]
            if info.gpu:
                rows.extend(
                    [
                        ("GPU", "videocam", info.gpu.name),
                        ("VRAM", "memory", f"{info.gpu.vram_free_mb} MB free / {info.gpu.vram_total_mb} MB total"),
                        ("Driver", "settings", f"{info.gpu.driver_version} (CUDA {info.gpu.cuda_version})"),
                    ]
                )
            else:
                rows.append(("GPU", "videocam_off", "Not detected"))
            with info_container:
                for label, icon, value in rows:
                    with ui.row().classes("w-full items-center gap-4 py-1"):
                        ui.icon(icon).classes("text-grey text-lg")
                        ui.label(label).classes("w-32 text-weight-medium")
                        ui.label(value).classes("font-mono text-sm")
                    ui.separator()

        ui.timer(0.1, _refresh_system, once=True)

    ##### MODELS — SEARCH & MANAGE #####

    with ui.expansion("Models", icon="folder", value=True).classes("w-full").props("header-class=text-h6"):
        _debounce_timer: dict[str, ui.timer | None] = {"t": None}

        search_input = (
            ui.input(
                placeholder="Search HuggingFace... (e.g. 'qwen3 coder', 'olmoe 1b')",
            )
            .props("outlined dense clearable")
            .classes("w-full")
        )

        search_spinner = ui.spinner(size="lg").classes("self-center mt-4")
        search_spinner.visible = False

        results_container = ui.column().classes("w-full gap-1 mt-2")
        quants_container = ui.column().classes("w-full gap-0 mt-2")

        dl_status = ui.label("").classes("text-caption mt-1")
        dl_progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        dl_progress.visible = False

        ui.separator().classes("my-3")
        ui.label("Downloaded Models").classes("text-subtitle1 text-weight-medium")
        models_container = ui.column().classes("w-full gap-0")

        async def _do_search() -> None:
            query = search_input.value
            if not query or len(query.strip()) < 2:
                results_container.clear()
                quants_container.clear()
                search_spinner.visible = False
                return
            search_spinner.visible = True
            results_container.clear()
            quants_container.clear()
            try:
                results = await asyncio.to_thread(search_models, query.strip(), limit=5)
            except Exception:
                search_spinner.visible = False
                with results_container:
                    ui.label("Search failed").classes("text-negative text-sm")
                return
            search_spinner.visible = False
            results_container.clear()
            if not results:
                with results_container:
                    ui.label("No GGUF models found").classes("text-grey text-sm")
                return
            with results_container:
                for result in results:
                    with ui.card().classes("w-full cursor-pointer").on("click", lambda _e, r=result: _select_repo(r)):
                        with ui.row().classes("w-full items-center gap-2"):
                            ui.icon("inventory_2").classes("text-lg").style("color: var(--ellm-accent)")
                            ui.label(result.repo_id).classes("text-weight-medium text-sm flex-grow")
                            ui.label(f"{result.downloads:,} ⬇").classes("text-grey text-xs")
                            ui.label(f"{result.likes} ❤").classes("text-grey text-xs")
                        if result.quants:
                            with ui.row().classes("gap-1 mt-1 flex-wrap"):
                                for q in result.quants[:8]:
                                    ui.badge(q).props("outline").classes("text-xs").style(
                                        "color: var(--ellm-accent); border-color: var(--ellm-accent-dim)"
                                    )

        def _on_value_change() -> None:
            if _debounce_timer["t"]:
                _debounce_timer["t"].deactivate()
            _debounce_timer["t"] = ui.timer(0.6, _do_search, once=True)

        search_input.on_value_change(lambda _e: _on_value_change())

        async def _select_repo(result: object) -> None:
            quants_container.clear()
            with quants_container:
                ui.spinner(size="sm")
            try:
                quants = await asyncio.to_thread(list_quants, result.repo_id)
            except Exception:
                quants_container.clear()
                with quants_container:
                    ui.label("Failed to fetch quant list").classes("text-negative text-sm")
                return
            quants_container.clear()
            with quants_container:
                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.icon("folder_open").style("color: var(--ellm-accent)")
                    ui.label(result.repo_id).classes("text-subtitle2")
                    ui.label(f"{len(quants)} files").classes("text-grey text-xs")
                for f in quants:
                    already = (st.models_path / f.filename).exists()
                    with ui.row().classes("w-full items-center py-1 gap-2"):
                        ui.label(f.filename).classes("font-mono text-xs flex-grow")
                        if f.quant:
                            ui.badge(f.quant).props("outline").classes("text-xs").style(
                                "color: var(--ellm-accent); border-color: var(--ellm-accent-dim)"
                            )
                        ui.label(f"{f.size_gb} GB").classes("text-grey text-xs")
                        if already:
                            ui.icon("check_circle", color="green").classes("text-sm").tooltip("Downloaded")
                        else:
                            ui.button(
                                icon="download",
                                on_click=lambda _e, r=result.repo_id, gf=f: _download(r, gf),
                            ).props("flat round dense size=sm").style("color: var(--ellm-accent)")
                    ui.separator()

        async def _download(repo: str, file: GGUFFile) -> None:
            dest = st.models_path / file.filename
            if dest.exists():
                dl_status.text = "Already downloaded"
                return
            dl_progress.visible = True
            dl_progress.value = 0
            dl_status.text = f"Downloading {file.filename}..."

            def _on_progress(downloaded: int, total: int) -> None:
                if total > 0:
                    dl_progress.value = downloaded / total
                    pct = downloaded / total * 100
                    dl_status.text = (
                        f"Downloading... {pct:.0f}% ({downloaded / (1024**3):.1f}/{total / (1024**3):.1f} GB)"
                    )

            try:
                await hf_adapter.download_model(repo, file.filename, dest, _on_progress)
                dl_status.text = f"Downloaded {file.filename}"
                dl_progress.value = 1.0
                _refresh_models()
                _refresh_model_dropdown()
            except Exception as exc:
                dl_status.text = f"Download failed: {exc}"
                dest.unlink(missing_ok=True)
            finally:
                dl_progress.visible = False

        def _refresh_models() -> None:
            models_container.clear()
            models_dir = st.models_path
            models_dir.mkdir(parents=True, exist_ok=True)
            gguf_files = sorted(models_dir.glob("*.gguf"))
            if not gguf_files:
                with models_container:
                    ui.label("No models downloaded yet.").classes("text-grey py-2")
                return
            with models_container:
                for mp in gguf_files:
                    size_gb = mp.stat().st_size / (1024**3)
                    with ui.row().classes("w-full items-center justify-between py-1"):
                        ui.icon("description").classes("text-grey")
                        ui.label(mp.name).classes("flex-grow font-mono text-sm")
                        ui.label(f"{size_gb:.2f} GB").classes("text-grey text-sm")
                        ui.button(
                            icon="delete",
                            color="negative",
                            on_click=lambda _e, p=mp: _delete_model(p),
                        ).props("flat round dense")
                    ui.separator()

        def _delete_model(path: Path) -> None:
            path.unlink(missing_ok=True)
            _refresh_models()
            _refresh_model_dropdown()

        _refresh_models()

    ##### SERVER CONFIGURATION #####

    with ui.expansion("Server Configuration", icon="tune").classes("w-full").props("header-class=text-h6"):
        # Model select
        with ui.expansion("Model", icon="smart_toy", value=True).classes("w-full"):
            available = _list_available_models()
            current_model = config.model.path if config.model.path in available else None
            mdl_select = (
                ui.select(
                    options=available,
                    label="Model file",
                    value=current_model,
                )
                .props("outlined dense")
                .classes("w-full")
            )
            mdl_ngl = ui.number(
                label="GPU Layers (-1 = all)",
                value=config.model.n_gpu_layers,
            ).props("outlined dense")

        def _refresh_model_dropdown() -> None:
            mdl_select.options = _list_available_models()
            mdl_select.update()

        # Server
        with ui.expansion("Server", icon="dns").classes("w-full"):
            srv_host = ui.input(label="Host", value=config.server.host).props("outlined dense")
            srv_port = ui.number(label="Port", value=config.server.port, min=1024, max=65535).props("outlined dense")
            srv_alias = ui.input(label="Model Alias", value=config.server.alias).props("outlined dense")

        # Context
        with ui.expansion("Context", icon="data_array").classes("w-full"):
            ctx_size = ui.number(
                label="Context Size", value=config.context.ctx_size, min=512, max=131072, step=1024
            ).props("outlined dense")
            ctx_parallel = ui.number(label="Parallel Sequences", value=config.context.parallel, min=1, max=64).props(
                "outlined dense"
            )
            ctx_batch = ui.number(
                label="Batch Size", value=config.context.batch_size, min=32, max=16384, step=256
            ).props("outlined dense")
            ctx_ubatch = ui.number(
                label="Micro Batch Size", value=config.context.ubatch_size, min=32, max=8192, step=128
            ).props("outlined dense")

        # Cache
        with ui.expansion("KV Cache", icon="cached").classes("w-full"):
            cache_k = ui.select(_CACHE_TYPES, label="Key Cache Type", value=config.cache.type_k).props("outlined dense")
            cache_v = ui.select(_CACHE_TYPES, label="Value Cache Type", value=config.cache.type_v).props(
                "outlined dense"
            )
            cache_no_offload = ui.switch("No KV Offload (keep on CPU)", value=config.cache.no_kv_offload)
            cache_defrag = ui.number(
                label="Defrag Threshold", value=config.cache.defrag_thold, min=0, max=1, step=0.05
            ).props("outlined dense")

        # Compute
        with ui.expansion("Compute", icon="memory").classes("w-full"):
            cmp_threads = ui.number(label="Threads", value=config.compute.threads, min=1, max=256).props(
                "outlined dense"
            )
            cmp_threads_batch = ui.number(
                label="Threads (Batch)", value=config.compute.threads_batch, min=1, max=256
            ).props("outlined dense")
            cmp_flash = ui.switch("Flash Attention", value=config.compute.flash_attn)
            cmp_fit = ui.switch("Fit Model to VRAM", value=config.compute.fit)
            cmp_mlock = ui.switch("Memory Lock (mlock)", value=config.compute.mlock)
            cmp_no_mmap = ui.switch("Disable Memory Map (no-mmap)", value=config.compute.no_mmap)

        # Sampling
        with ui.expansion("Sampling", icon="casino").classes("w-full"):
            smp_temp = ui.number(label="Temperature", value=config.sampling.temp, min=0, max=2, step=0.05).props(
                "outlined dense"
            )
            smp_top_p = ui.number(label="Top P", value=config.sampling.top_p, min=0, max=1, step=0.01).props(
                "outlined dense"
            )
            smp_top_k = ui.number(label="Top K", value=config.sampling.top_k, min=0, max=200).props("outlined dense")
            smp_min_p = ui.number(label="Min P", value=config.sampling.min_p, min=0, max=1, step=0.01).props(
                "outlined dense"
            )
            smp_repeat = ui.number(
                label="Repeat Penalty", value=config.sampling.repeat_penalty, min=0, max=3, step=0.05
            ).props("outlined dense")

        # Template
        with ui.expansion("Template", icon="code").classes("w-full"):
            tpl_jinja = ui.switch("Jinja Templates", value=config.template.jinja)
            tpl_no_shift = ui.switch("No Context Shift", value=config.template.no_context_shift)
            tpl_chat = (
                ui.input(label="Chat Template Override", value=config.template.chat_template)
                .props("outlined dense")
                .classes("w-full")
            )

        ui.separator().classes("my-4")

        with ui.row().classes("gap-2"):
            save_btn = (
                ui.button("Save & Apply", icon="rocket_launch")
                .props("unelevated")
                .style("background: var(--ellm-accent) !important")
            )
            reset_btn = (
                ui.button("Reset to Defaults", icon="restart_alt").props("flat").style("color: var(--ellm-accent)")
            )
        save_status = ui.label("").classes("text-caption")

        def _build_config() -> ServerConfig:
            return ServerConfig(
                server=ServerSpec(
                    host=srv_host.value or "0.0.0.0",
                    port=int(srv_port.value or 45150),
                    alias=srv_alias.value or "default",
                ),
                model=ModelSpec(
                    path=mdl_select.value or "", n_gpu_layers=int(mdl_ngl.value if mdl_ngl.value is not None else -1)
                ),
                context=ContextSpec(
                    ctx_size=int(ctx_size.value or 8192),
                    parallel=int(ctx_parallel.value or 1),
                    batch_size=int(ctx_batch.value or 2048),
                    ubatch_size=int(ctx_ubatch.value or 512),
                ),
                cache=CacheSpec(
                    type_k=cache_k.value or "f16",
                    type_v=cache_v.value or "f16",
                    no_kv_offload=cache_no_offload.value,
                    defrag_thold=float(cache_defrag.value or 0.1),
                ),
                compute=ComputeSpec(
                    threads=int(cmp_threads.value or 4),
                    threads_batch=int(cmp_threads_batch.value or 4),
                    flash_attn=cmp_flash.value,
                    fit=cmp_fit.value,
                    mlock=cmp_mlock.value,
                    no_mmap=cmp_no_mmap.value,
                ),
                sampling=SamplingSpec(
                    temp=float(smp_temp.value or 0.7),
                    top_p=float(smp_top_p.value or 0.95),
                    top_k=int(smp_top_k.value or 40),
                    min_p=float(smp_min_p.value or 0.05),
                    repeat_penalty=float(smp_repeat.value or 1.0),
                ),
                template=TemplateSpec(
                    jinja=tpl_jinja.value, no_context_shift=tpl_no_shift.value, chat_template=tpl_chat.value or ""
                ),
            )

        async def _save_and_apply() -> None:
            if not mdl_select.value:
                save_status.text = "Select a model first"
                return
            new_config = _build_config()
            new_config.to_yaml(st.config_path)
            save_status.text = "Saved — restarting server..."
            save_btn.props(add="loading")
            started = await manager.restart(new_config)
            save_btn.props(remove="loading")
            save_status.text = (
                "Server launched — check header indicator" if started else "Failed to start — check model"
            )

        def _reset_config() -> None:
            ServerConfig().to_yaml(st.config_path)
            save_status.text = "Reset to defaults — reload page to see changes"

        save_btn.on_click(_save_and_apply)
        reset_btn.on_click(_reset_config)
