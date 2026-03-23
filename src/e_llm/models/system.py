"""Structured system hardware information — used by SystemEvaluator and Tuner agent."""

from pydantic import BaseModel, Field


class CpuInfo(BaseModel):
    """CPU hardware details."""

    model: str = Field(description="CPU model name (e.g. 'Intel Core i9-14900K')")
    n_cores_physical: int = Field(description="Physical CPU cores")
    n_cores_logical: int = Field(description="Logical CPU cores (threads)")
    frequency_mhz: float = Field(description="Current CPU frequency in MHz")


class RamInfo(BaseModel):
    """System RAM details."""

    total_gb: float = Field(description="Total RAM in GB")
    available_gb: float = Field(description="Available RAM in GB")
    used_gb: float = Field(description="Used RAM in GB")
    usage_pct: float = Field(description="RAM usage percentage")


class GpuInfo(BaseModel):
    """NVIDIA GPU details."""

    name: str = Field(description="GPU model name (e.g. 'NVIDIA GeForce RTX 4090')")
    vram_total_mb: int = Field(description="Total VRAM in MB")
    vram_used_mb: int = Field(description="Used VRAM in MB")
    vram_free_mb: int = Field(description="Free VRAM in MB")
    driver_version: str = Field(description="NVIDIA driver version")
    cuda_version: str = Field(description="CUDA version")


class DiskInfo(BaseModel):
    """Disk storage details for the data directory."""

    path: str = Field(description="Monitored path")
    total_gb: float = Field(description="Total disk space in GB")
    free_gb: float = Field(description="Free disk space in GB")
    usage_pct: float = Field(description="Disk usage percentage")


class SystemInfo(BaseModel):
    """Full hardware profile of the host machine."""

    cpu: CpuInfo
    ram: RamInfo
    gpu: GpuInfo | None = Field(default=None, description="GPU info, None if no NVIDIA GPU detected")
    disk: DiskInfo
