import asyncio
import contextlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

##### DATA MODELS #####


@dataclass(frozen=True, slots=True)
class CpuInfo:
    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: float


@dataclass(frozen=True, slots=True)
class RamInfo:
    total_gb: float
    available_gb: float
    used_gb: float
    usage_pct: float


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    driver_version: str
    cuda_version: str


@dataclass(frozen=True, slots=True)
class DiskInfo:
    path: str
    total_gb: float
    free_gb: float
    usage_pct: float


@dataclass(frozen=True, slots=True)
class SystemInfo:
    cpu: CpuInfo
    ram: RamInfo
    gpu: GpuInfo | None
    disk: DiskInfo


##### EVALUATOR #####


class SystemEvaluator:
    """Evaluate host machine hardware capabilities in parallel."""

    __slots__ = ("_data_path",)

    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path

    async def evaluate(self) -> SystemInfo:
        """Run all hardware evaluations concurrently."""
        async with asyncio.TaskGroup() as tg:
            cpu_task = tg.create_task(asyncio.to_thread(self._se_eval_cpu))
            ram_task = tg.create_task(asyncio.to_thread(self._se_eval_ram))
            gpu_task = tg.create_task(asyncio.to_thread(self._se_eval_gpu))
            disk_task = tg.create_task(asyncio.to_thread(self._se_eval_disk))

        return SystemInfo(
            cpu=cpu_task.result(),
            ram=ram_task.result(),
            gpu=gpu_task.result(),
            disk=disk_task.result(),
        )

    def _se_eval_cpu(self) -> CpuInfo:
        freq = psutil.cpu_freq()
        model = "Unknown"
        with contextlib.suppress(OSError), open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        return CpuInfo(
            model=model,
            cores_physical=psutil.cpu_count(logical=False) or 0,
            cores_logical=psutil.cpu_count(logical=True) or 0,
            frequency_mhz=freq.current if freq else 0.0,
        )

    def _se_eval_ram(self) -> RamInfo:
        mem = psutil.virtual_memory()
        return RamInfo(
            total_gb=round(mem.total / (1024**3), 1),
            available_gb=round(mem.available / (1024**3), 1),
            used_gb=round(mem.used / (1024**3), 1),
            usage_pct=mem.percent,
        )

    def _se_eval_gpu(self) -> GpuInfo | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 5:
                return None

            cuda_version = ""
            full = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in full.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    break

            return GpuInfo(
                name=parts[0],
                vram_total_mb=int(float(parts[1])),
                vram_used_mb=int(float(parts[2])),
                vram_free_mb=int(float(parts[3])),
                driver_version=parts[4],
                cuda_version=cuda_version,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            return None

    def _se_eval_disk(self) -> DiskInfo:
        path = self._data_path if self._data_path.exists() else Path("/")
        usage = shutil.disk_usage(str(path))
        return DiskInfo(
            path=str(path),
            total_gb=round(usage.total / (1024**3), 1),
            free_gb=round(usage.free / (1024**3), 1),
            usage_pct=round((usage.used / usage.total) * 100, 1),
        )
