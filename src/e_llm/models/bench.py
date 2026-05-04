"""Bench data models — config, result point, and run container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """Parameters for a single llama-bench invocation."""

    model_path: Path
    n_gpu_layers: int = -1
    flash_attn: bool = True
    type_k: str = "q8_0"
    type_v: str = "q4_0"
    n_prompt: tuple[int, ...] = (512,)
    n_gen: tuple[int, ...] = (128,)
    repetitions: int = 3
    threads: int = 20
    threads_batch: int = 32
    no_mmap: bool = True


@dataclass(frozen=True, slots=True)
class BenchPoint:
    """Single test result: one (test, t/s) measurement."""

    test: str  # e.g. "pp512", "tg128"
    avg_ts: float  # mean tokens/sec
    sd_ts: float  # std dev tokens/sec


@dataclass(frozen=True, slots=True)
class BenchRun:
    """Complete bench run — all test points for one config."""

    model: str  # filename only
    config: BenchConfig
    points: tuple[BenchPoint, ...]
    timestamp: str  # ISO 8601
