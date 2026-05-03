"""Tests for DownloadManager."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from e_llm.models.download import DownloadStatus
from e_llm.operational.downloads import DownloadManager


@pytest.fixture
def mock_hf_adapter() -> AsyncMock:
    """Mock HuggingFaceAdapter."""
    adapter = AsyncMock()
    adapter.validate_model.return_value = (True, 1000)
    adapter.download_model = AsyncMock()
    return adapter


@pytest.fixture
def download_manager(tmp_path: Path, mock_hf_adapter: AsyncMock) -> DownloadManager:
    """Create DownloadManager with temp state file."""
    state_file = tmp_path / "downloads.json"
    return DownloadManager(mock_hf_adapter, state_file)


##### INITIALIZATION #####


def test_init_creates_empty_state(tmp_path: Path, mock_hf_adapter: AsyncMock) -> None:
    """Test initialization with no existing state file."""
    state_file = tmp_path / "downloads.json"
    mgr = DownloadManager(mock_hf_adapter, state_file)
    assert mgr.downloads == {}
    assert mgr.tasks == {}


def test_load_state_from_file(tmp_path: Path, mock_hf_adapter: AsyncMock) -> None:
    """Test loading existing state from JSON file."""
    state_file = tmp_path / "downloads.json"
    state_data = {
        "downloads": {
            "task-1": {
                "task_id": "task-1",
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
        }
    }
    state_file.write_text(json.dumps(state_data))

    mgr = DownloadManager(mock_hf_adapter, state_file)
    assert len(mgr.downloads) == 1
    assert "task-1" in mgr.downloads
    assert mgr.downloads["task-1"].filename == "model.gguf"
    assert mgr.downloads["task-1"].downloaded_bytes == 500


def test_load_state_handles_corrupt_json(tmp_path: Path, mock_hf_adapter: AsyncMock) -> None:
    """Test loading state handles corrupt JSON gracefully."""
    state_file = tmp_path / "downloads.json"
    state_file.write_text("{ invalid json")

    mgr = DownloadManager(mock_hf_adapter, state_file)
    assert mgr.downloads == {}


##### SAVE STATE #####


def test_save_state_creates_file(tmp_path: Path, mock_hf_adapter: AsyncMock) -> None:
    """Test saving state creates JSON file."""
    state_file = tmp_path / "downloads.json"
    mgr = DownloadManager(mock_hf_adapter, state_file)
    mgr.save_state()

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert "downloads" in data
    assert data["downloads"] == {}


def test_save_state_persists_downloads(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test saving state persists download data."""
    from e_llm.models.download import DownloadState

    state = DownloadState.create(
        task_id="test-123",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    download_manager.downloads["test-123"] = state
    download_manager.save_state()

    data = json.loads(download_manager.state_file.read_text())
    assert "test-123" in data["downloads"]
    assert data["downloads"]["test-123"]["filename"] == "model.gguf"


##### GET STATUS #####


def test_get_status_returns_state(download_manager: DownloadManager) -> None:
    """Test getting download status."""
    from e_llm.models.download import DownloadState

    state = DownloadState.create(
        task_id="test-123",
        repo="test/repo",
        filename="model.gguf",
        dest=Path("/tmp/model.gguf"),
        total_bytes=1000,
    )
    download_manager.downloads["test-123"] = state

    result = download_manager.get_status("test-123")
    assert result is not None
    assert result.task_id == "test-123"


def test_get_status_returns_none_when_not_found(download_manager: DownloadManager) -> None:
    """Test getting status for non-existent download."""
    result = download_manager.get_status("nonexistent")
    assert result is None


##### LIST DOWNLOADS #####


def test_list_downloads_all(download_manager: DownloadManager) -> None:
    """Test listing all downloads."""
    from e_llm.models.download import DownloadState

    state1 = DownloadState.create("task-1", "repo1", "file1.gguf", Path("/tmp/file1.gguf"), 1000)
    state2 = DownloadState.create("task-2", "repo2", "file2.gguf", Path("/tmp/file2.gguf"), 2000)
    state2.status = DownloadStatus.COMPLETED

    download_manager.downloads["task-1"] = state1
    download_manager.downloads["task-2"] = state2

    all_downloads = download_manager.list_downloads()
    assert len(all_downloads) == 2


def test_list_downloads_by_status(download_manager: DownloadManager) -> None:
    """Test listing downloads filtered by status."""
    from e_llm.models.download import DownloadState

    state1 = DownloadState.create("task-1", "repo1", "file1.gguf", Path("/tmp/file1.gguf"), 1000)
    state2 = DownloadState.create("task-2", "repo2", "file2.gguf", Path("/tmp/file2.gguf"), 2000)
    state2.status = DownloadStatus.COMPLETED

    download_manager.downloads["task-1"] = state1
    download_manager.downloads["task-2"] = state2

    completed = download_manager.list_downloads(DownloadStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].task_id == "task-2"


def test_list_active_downloads(download_manager: DownloadManager) -> None:
    """Test listing active downloads."""
    from e_llm.models.download import DownloadState

    state1 = DownloadState.create("task-1", "repo1", "file1.gguf", Path("/tmp/file1.gguf"), 1000)
    state1.status = DownloadStatus.DOWNLOADING

    state2 = DownloadState.create("task-2", "repo2", "file2.gguf", Path("/tmp/file2.gguf"), 2000)
    state2.status = DownloadStatus.COMPLETED

    state3 = DownloadState.create("task-3", "repo3", "file3.gguf", Path("/tmp/file3.gguf"), 3000)
    state3.status = DownloadStatus.QUEUED

    download_manager.downloads["task-1"] = state1
    download_manager.downloads["task-2"] = state2
    download_manager.downloads["task-3"] = state3

    active = download_manager.list_active()
    assert len(active) == 2
    assert all(dl.status in (DownloadStatus.QUEUED, DownloadStatus.DOWNLOADING) for dl in active)


##### START DOWNLOAD #####


@pytest.mark.asyncio
async def test_start_download_creates_task(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test starting a download creates background task."""
    dest = tmp_path / "model.gguf"

    task_id = await download_manager.start_download("test/repo", "model.gguf", dest)

    assert task_id in download_manager.downloads
    assert task_id in download_manager.tasks
    assert download_manager.downloads[task_id].status == DownloadStatus.QUEUED


@pytest.mark.asyncio
async def test_start_download_validates_model(
    download_manager: DownloadManager, tmp_path: Path, mock_hf_adapter: AsyncMock
) -> None:
    """Test starting download validates model exists."""
    dest = tmp_path / "model.gguf"
    mock_hf_adapter.validate_model.return_value = (True, 5000)

    await download_manager.start_download("test/repo", "model.gguf", dest)

    mock_hf_adapter.validate_model.assert_called_once_with("test/repo", "model.gguf")


@pytest.mark.asyncio
async def test_start_download_fails_if_file_exists(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test starting download fails if file already exists."""
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        await download_manager.start_download("test/repo", "model.gguf", dest)


@pytest.mark.asyncio
async def test_start_download_fails_if_model_not_found(
    download_manager: DownloadManager, tmp_path: Path, mock_hf_adapter: AsyncMock
) -> None:
    """Test starting download fails if model doesn't exist."""
    dest = tmp_path / "model.gguf"
    mock_hf_adapter.validate_model.return_value = (False, 0)

    with pytest.raises(ValueError, match="Model not found"):
        await download_manager.start_download("test/repo", "model.gguf", dest)


##### CANCEL DOWNLOAD #####


@pytest.mark.asyncio
async def test_cancel_download_cancels_task(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test cancelling download cancels the task."""
    dest = tmp_path / "model.gguf"

    # Start download
    task_id = await download_manager.start_download("test/repo", "model.gguf", dest)

    # Cancel it
    await download_manager.cancel_download(task_id)

    # Wait a bit for task to finish cancelling
    await asyncio.sleep(0.1)

    # Task should be cancelled (may still be in dict until cleanup)
    task = download_manager.tasks.get(task_id)
    if task:
        assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_cancel_download_no_op_if_not_found(download_manager: DownloadManager) -> None:
    """Test cancelling non-existent download is no-op."""
    await download_manager.cancel_download("nonexistent")  # Should not raise


##### CLEANUP #####


@pytest.mark.asyncio
async def test_cleanup_failed_removes_partial_files(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test cleanup removes partial files from failed downloads."""
    from e_llm.models.download import DownloadState

    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"partial")

    state = DownloadState.create("task-1", "test/repo", "model.gguf", dest, 1000)
    state.mark_failed("Connection error")
    download_manager.downloads["task-1"] = state

    await download_manager.cleanup_failed()

    assert not dest.exists()


@pytest.mark.asyncio
async def test_cleanup_cancelled_removes_partial_files(download_manager: DownloadManager, tmp_path: Path) -> None:
    """Test cleanup removes partial files from cancelled downloads."""
    from e_llm.models.download import DownloadState

    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"partial")

    state = DownloadState.create("task-1", "test/repo", "model.gguf", dest, 1000)
    state.mark_cancelled()
    download_manager.downloads["task-1"] = state

    await download_manager.cleanup_cancelled()

    assert not dest.exists()


##### EXECUTE DOWNLOAD #####


@pytest.mark.asyncio
async def test_execute_download_marks_completed_on_success(
    download_manager: DownloadManager, tmp_path: Path, mock_hf_adapter: AsyncMock
) -> None:
    """Test execute_download marks state as completed on success."""
    from e_llm.models.download import DownloadState

    dest = tmp_path / "model.gguf"
    state = DownloadState.create("task-1", "test/repo", "model.gguf", dest, 1000)
    download_manager.downloads["task-1"] = state

    # Mock successful download
    async def mock_download(repo: str, filename: str, dest: Path, on_progress) -> None:
        dest.write_bytes(b"downloaded")
        on_progress(1000, 1000)

    mock_hf_adapter.download_model.side_effect = mock_download

    await download_manager.execute_download("task-1")

    assert download_manager.downloads["task-1"].status == DownloadStatus.COMPLETED
    assert download_manager.downloads["task-1"].downloaded_bytes == 1000


@pytest.mark.asyncio
async def test_execute_download_marks_failed_on_error(
    download_manager: DownloadManager, tmp_path: Path, mock_hf_adapter: AsyncMock
) -> None:
    """Test execute_download marks state as failed on error."""
    from e_llm.models.download import DownloadState

    dest = tmp_path / "model.gguf"
    state = DownloadState.create("task-1", "test/repo", "model.gguf", dest, 1000)
    download_manager.downloads["task-1"] = state

    # Mock failed download
    mock_hf_adapter.download_model.side_effect = Exception("Connection timeout")

    await download_manager.execute_download("task-1")

    assert download_manager.downloads["task-1"].status == DownloadStatus.FAILED
    error = download_manager.downloads["task-1"].error
    assert error is not None and "Connection timeout" in error


@pytest.mark.asyncio
async def test_execute_download_cleans_partial_file_on_error(
    download_manager: DownloadManager, tmp_path: Path, mock_hf_adapter: AsyncMock
) -> None:
    """Test execute_download removes partial file on error."""
    from e_llm.models.download import DownloadState

    dest = tmp_path / "model.gguf"
    state = DownloadState.create("task-1", "test/repo", "model.gguf", dest, 1000)
    download_manager.downloads["task-1"] = state

    # Mock download that creates partial file then fails
    async def mock_download_fail(repo: str, filename: str, dest: Path, on_progress) -> None:
        dest.write_bytes(b"partial")
        raise Exception("Network error")

    mock_hf_adapter.download_model.side_effect = mock_download_fail

    await download_manager.execute_download("task-1")

    assert not dest.exists()
