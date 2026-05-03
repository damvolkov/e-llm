"""Tests for download models."""

from pathlib import Path

from e_llm.models.download import DownloadState, DownloadStatus


def test_download_state_create() -> None:
    """Test creating a new download state."""
    state = DownloadState.create(
        task_id="test-123",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    assert state.task_id == "test-123"
    assert state.repo == "test/repo"
    assert state.filename == "model.gguf"
    assert state.dest == "/tmp/model.gguf"
    assert state.total_bytes == 1000
    assert state.downloaded_bytes == 0
    assert state.status == DownloadStatus.QUEUED
    assert state.error is None


def test_download_state_progress_pct() -> None:
    """Test progress percentage calculation."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    state.downloaded_bytes = 250
    assert state.progress_pct == 25.0

    state.downloaded_bytes = 500
    assert state.progress_pct == 50.0

    state.downloaded_bytes = 1000
    assert state.progress_pct == 100.0


def test_download_state_progress_pct_zero_total() -> None:
    """Test progress percentage when total is zero."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=0,
    )
    assert state.progress_pct == 0.0


def test_download_state_size_gb() -> None:
    """Test size conversion to GB."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=5 * 1024**3,  # 5 GB
    )
    assert state.size_gb == 5.0

    state.downloaded_bytes = int(2.5 * 1024**3)  # 2.5 GB
    assert state.downloaded_gb == 2.5


def test_download_state_update_progress() -> None:
    """Test updating download progress."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    before = state.updated_at
    state.update_progress(500)
    assert state.downloaded_bytes == 500
    assert state.updated_at > before


def test_download_state_mark_completed() -> None:
    """Test marking download as completed."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    state.downloaded_bytes = 800
    state.mark_completed()
    assert state.status == DownloadStatus.COMPLETED
    assert state.downloaded_bytes == 1000


def test_download_state_mark_failed() -> None:
    """Test marking download as failed."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    state.mark_failed("Connection timeout")
    assert state.status == DownloadStatus.FAILED
    assert state.error == "Connection timeout"


def test_download_state_mark_cancelled() -> None:
    """Test marking download as cancelled."""
    state = DownloadState.create(
        task_id="test",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    state.mark_cancelled()
    assert state.status == DownloadStatus.CANCELLED


def test_download_state_to_dict() -> None:
    """Test serialization to dict."""
    state = DownloadState.create(
        task_id="test-123",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    state.downloaded_bytes = 500
    data = state.to_dict()
    assert data["task_id"] == "test-123"
    assert data["repo"] == "test/repo"
    assert data["filename"] == "model.gguf"
    assert data["dest"] == "/tmp/model.gguf"
    assert data["total_bytes"] == 1000
    assert data["downloaded_bytes"] == 500
    assert data["status"] == DownloadStatus.QUEUED
    assert data["error"] is None


def test_download_state_from_dict() -> None:
    """Test deserialization from dict."""
    data = {
        "task_id": "test-123",
        "repo": "test/repo",
        "filename": "model.gguf",
        "dest": "/tmp/model.gguf",
        "total_bytes": 1000,
        "downloaded_bytes": 500,
        "status": "downloading",
        "started_at": "2026-05-03T16:00:00Z",
        "updated_at": "2026-05-03T16:05:00Z",
        "error": None,
    }
    state = DownloadState.from_dict(data)
    assert state.task_id == "test-123"
    assert state.repo == "test/repo"
    assert state.filename == "model.gguf"
    assert state.dest == "/tmp/model.gguf"
    assert state.total_bytes == 1000
    assert state.downloaded_bytes == 500
    assert state.status == DownloadStatus.DOWNLOADING
    assert state.error is None


def test_download_state_roundtrip() -> None:
    """Test serialization roundtrip."""
    original = DownloadState.create(
        task_id="test-123",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    original.downloaded_bytes = 500
    original.status = DownloadStatus.DOWNLOADING

    data = original.to_dict()
    restored = DownloadState.from_dict(data)

    assert restored.task_id == original.task_id
    assert restored.repo == original.repo
    assert restored.filename == original.filename
    assert restored.dest == original.dest
    assert restored.total_bytes == original.total_bytes
    assert restored.downloaded_bytes == original.downloaded_bytes
    assert restored.status == original.status
