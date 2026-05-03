"""Download manager — background download orchestration with persistence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from e_llm.models.download import DownloadState, DownloadStatus

if TYPE_CHECKING:
    from e_llm.adapters.huggingface import HuggingFaceAdapter


class DownloadManager:
    """Manage background downloads with persistent state."""

    __slots__ = ("hf_adapter", "state_file", "downloads", "tasks")

    def __init__(self, hf_adapter: HuggingFaceAdapter, state_file: Path) -> None:
        self.hf_adapter = hf_adapter
        self.state_file = state_file
        self.downloads: dict[str, DownloadState] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.load_state()

    def load_state(self) -> None:
        """Load download state from JSON file."""
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            self.downloads = {
                task_id: DownloadState.from_dict(state_dict)
                for task_id, state_dict in data.get("downloads", {}).items()
            }
        except (json.JSONDecodeError, KeyError):
            self.downloads = {}

    def save_state(self) -> None:
        """Persist download state to JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"downloads": {task_id: state.to_dict() for task_id, state in self.downloads.items()}}
        self.state_file.write_text(json.dumps(data, indent=2))

    def get_status(self, task_id: str) -> DownloadState | None:
        """Get download status by task ID."""
        return self.downloads.get(task_id)

    def list_downloads(self, status: DownloadStatus | None = None) -> list[DownloadState]:
        """List all downloads, optionally filtered by status."""
        if status is None:
            return list(self.downloads.values())
        return [state for state in self.downloads.values() if state.status == status]

    def list_active(self) -> list[DownloadState]:
        """List active downloads (queued or downloading)."""
        return [
            state
            for state in self.downloads.values()
            if state.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING)
        ]

    async def start_download(
        self,
        repo: str,
        filename: str,
        dest: Path,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> str:
        """Start background download. Returns task_id."""
        # Check if file already exists
        if dest.exists():
            raise FileExistsError(f"File already exists: {dest}")

        # Validate model exists and get size
        exists, total_bytes = await self.hf_adapter.validate_model(repo, filename)
        if not exists:
            raise ValueError(f"Model not found: {repo}/{filename}")

        # Create download state
        task_id = str(uuid.uuid4())
        state = DownloadState.create(task_id, repo, filename, dest, total_bytes)
        self.downloads[task_id] = state
        self.save_state()

        # Start background task
        task = asyncio.create_task(self.execute_download(task_id, on_progress))
        self.tasks[task_id] = task

        return task_id

    async def execute_download(
        self,
        task_id: str,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Execute download in background — updates state and persists progress."""
        state = self.downloads.get(task_id)
        if not state:
            return

        dest = Path(state.dest)
        state.status = DownloadStatus.DOWNLOADING
        self.save_state()

        def progress_callback(downloaded: int, total: int) -> None:
            state.update_progress(downloaded)
            # Persist every 100MB to avoid excessive I/O
            if downloaded % (100 * 1024 * 1024) < (1024 * 1024):
                self.save_state()
            if on_progress:
                on_progress(task_id, downloaded, total)

        try:
            await self.hf_adapter.download_model(
                state.repo,
                state.filename,
                dest,
                progress_callback,
            )
            state.mark_completed()
            self.save_state()
        except asyncio.CancelledError:
            state.mark_cancelled()
            dest.unlink(missing_ok=True)
            self.save_state()
            raise
        except Exception as exc:
            state.mark_failed(str(exc))
            dest.unlink(missing_ok=True)
            self.save_state()
        finally:
            self.tasks.pop(task_id, None)

    async def cancel_download(self, task_id: str) -> None:
        """Cancel active download."""
        task = self.tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def cleanup_failed(self) -> None:
        """Remove partial files from failed downloads."""
        for state in self.list_downloads(DownloadStatus.FAILED):
            dest = Path(state.dest)
            dest.unlink(missing_ok=True)

    async def cleanup_cancelled(self) -> None:
        """Remove partial files from cancelled downloads."""
        for state in self.list_downloads(DownloadStatus.CANCELLED):
            dest = Path(state.dest)
            dest.unlink(missing_ok=True)
