"""llama-bench subprocess runner — async streaming + JSON result parsing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import orjson

from e_llm.core.logger import logger
from e_llm.core.settings import settings as st
from e_llm.models.bench import BenchConfig, BenchPoint, BenchRun


def _build_command(config: BenchConfig) -> list[str]:
    cmd = [
        st.LLAMA_BENCH_BIN,
        "--model",
        str(config.model_path),
        "--n-gpu-layers",
        str(config.n_gpu_layers),
        "--flash-attn",
        "1" if config.flash_attn else "0",
        "--cache-type-k",
        config.type_k,
        "--cache-type-v",
        config.type_v,
        "--threads",
        str(config.threads),
        "--threads-batch",
        str(config.threads_batch),
        "--no-mmap",
        "1" if config.no_mmap else "0",
        "--repetitions",
        str(config.repetitions),
        "--output",
        "json",
    ]
    if config.n_prompt:
        cmd.extend(["--n-prompt", ",".join(str(p) for p in config.n_prompt)])
    if config.n_gen:
        cmd.extend(["--n-gen", ",".join(str(g) for g in config.n_gen)])
    return cmd


async def run_bench(
    config: BenchConfig,
    on_log: Callable[[str], None] | None = None,
) -> BenchRun | None:
    """Run llama-bench, stream stderr to on_log, return parsed results."""
    cmd = _build_command(config)
    logger.info("bench start", model=config.model_path.name, cmd=" ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.error("🔴 LLAMA_BENCH_NOT_FOUND", bin=st.LLAMA_BENCH_BIN)
        if on_log:
            on_log(f"ERROR: llama-bench not found at {st.LLAMA_BENCH_BIN}")
        return None

    async def _drain_stderr() -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            if on_log:
                on_log(raw.decode(errors="replace").rstrip())

    assert proc.stdout is not None
    stderr_task = asyncio.create_task(_drain_stderr())
    stdout_data = await proc.stdout.read()
    await stderr_task
    await proc.wait()

    if proc.returncode != 0:
        logger.error("bench failed", returncode=proc.returncode)
        if on_log:
            on_log(f"ERROR: exit code {proc.returncode}")
        return None

    try:
        raw_results: list[dict] = orjson.loads(stdout_data)
    except Exception as exc:
        logger.error("bench json parse failed", error=str(exc), raw=stdout_data[:200].decode())
        if on_log:
            on_log(f"ERROR: failed to parse JSON output — {exc}")
        return None

    points = tuple(
        BenchPoint(test=r["test"], avg_ts=r.get("avg_ts", 0.0), sd_ts=r.get("sd_ts", 0.0)) for r in raw_results
    )
    run = BenchRun(
        model=config.model_path.name,
        config=config,
        points=points,
        timestamp=datetime.now(UTC).isoformat(),
    )
    logger.info("bench complete", model=run.model, n_points=len(points))
    return run
