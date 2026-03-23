"""Tests for SystemEvaluator."""

from pathlib import Path

from e_llm.models.system import CpuInfo, DiskInfo, RamInfo, SystemInfo
from e_llm.operational.system import SystemEvaluator

##### DATA MODELS #####


async def test_cpu_info_frozen() -> None:
    info = CpuInfo(model="Test", n_cores_physical=4, n_cores_logical=8, frequency_mhz=3600.0)
    assert info.model == "Test"
    assert info.n_cores_logical == 8


async def test_ram_info_frozen() -> None:
    info = RamInfo(total_gb=32.0, available_gb=16.0, used_gb=16.0, usage_pct=50.0)
    assert info.total_gb == 32.0


async def test_disk_info_frozen() -> None:
    info = DiskInfo(path="/data", total_gb=500.0, free_gb=200.0, usage_pct=60.0)
    assert info.free_gb == 200.0


async def test_gpu_info_optional() -> None:
    info = SystemInfo(
        cpu=CpuInfo(model="Test", n_cores_physical=4, n_cores_logical=8, frequency_mhz=3600.0),
        ram=RamInfo(total_gb=32.0, available_gb=16.0, used_gb=16.0, usage_pct=50.0),
        gpu=None,
        disk=DiskInfo(path="/", total_gb=500.0, free_gb=200.0, usage_pct=60.0),
    )
    assert info.gpu is None


##### EVALUATOR #####


async def test_evaluator_runs(tmp_path: Path) -> None:
    evaluator = SystemEvaluator(tmp_path)
    info = await evaluator.evaluate()
    assert info.cpu.n_cores_logical > 0
    assert info.ram.total_gb > 0
    assert info.disk.total_gb > 0
