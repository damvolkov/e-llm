"""Download state models — persistent download tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class DownloadStatus(StrEnum):
    """Download lifecycle states."""

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class DownloadState:
    """Persistent download state — serialized to JSON."""

    task_id: str
    repo: str
    filename: str
    dest: str
    total_bytes: int
    downloaded_bytes: int
    status: DownloadStatus
    started_at: str
    updated_at: str
    error: str | None = None

    @property
    def progress_pct(self) -> float:
        """Download progress as percentage (0-100)."""
        if self.total_bytes == 0:
            return 0.0
        return round((self.downloaded_bytes / self.total_bytes) * 100, 1)

    @property
    def size_gb(self) -> float:
        """Total size in GB."""
        return round(self.total_bytes / (1024**3), 2)

    @property
    def downloaded_gb(self) -> float:
        """Downloaded size in GB."""
        return round(self.downloaded_bytes / (1024**3), 2)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "filename": self.filename,
            "dest": self.dest,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DownloadState:
        """Deserialize from dict."""
        return cls(
            task_id=data["task_id"],
            repo=data["repo"],
            filename=data["filename"],
            dest=data["dest"],
            total_bytes=data["total_bytes"],
            downloaded_bytes=data["downloaded_bytes"],
            status=DownloadStatus(data["status"]),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            error=data.get("error"),
        )

    @classmethod
    def create(
        cls,
        task_id: str,
        repo: str,
        filename: str,
        dest: Path,
        total_bytes: int,
    ) -> DownloadState:
        """Create new download state."""
        now = datetime.now(UTC).isoformat()
        return cls(
            task_id=task_id,
            repo=repo,
            filename=filename,
            dest=str(dest),
            total_bytes=total_bytes,
            downloaded_bytes=0,
            status=DownloadStatus.QUEUED,
            started_at=now,
            updated_at=now,
        )

    def update_progress(self, downloaded: int) -> None:
        """Update download progress."""
        self.downloaded_bytes = downloaded
        self.updated_at = datetime.now(UTC).isoformat()

    def mark_completed(self) -> None:
        """Mark download as completed."""
        self.status = DownloadStatus.COMPLETED
        self.downloaded_bytes = self.total_bytes
        self.updated_at = datetime.now(UTC).isoformat()

    def mark_failed(self, error: str) -> None:
        """Mark download as failed."""
        self.status = DownloadStatus.FAILED
        self.error = error
        self.updated_at = datetime.now(UTC).isoformat()

    def mark_cancelled(self) -> None:
        """Mark download as cancelled."""
        self.status = DownloadStatus.CANCELLED
        self.updated_at = datetime.now(UTC).isoformat()
