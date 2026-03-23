from collections.abc import Callable
from pathlib import Path

import aiofiles
import httpx

_HF_BASE = "https://huggingface.co"


class HuggingFaceAdapter:
    """Download and validate GGUF models from HuggingFace."""

    __slots__ = ("_client",)

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )

    async def validate_model(self, repo: str, filename: str) -> tuple[bool, int]:
        """Check if model exists on HuggingFace. Returns (exists, size_bytes)."""
        url = f"{_HF_BASE}/{repo}/resolve/main/{filename}"
        try:
            response = await self._client.head(url)
            if response.status_code == 200:
                size = int(response.headers.get("content-length", 0))
                return True, size
            return False, 0
        except (httpx.ConnectError, httpx.TimeoutException):
            return False, 0

    async def download_model(
        self,
        repo: str,
        filename: str,
        dest: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Download model file with optional progress callback."""
        url = f"{_HF_BASE}/{repo}/resolve/main/{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=30.0),
                follow_redirects=True,
            ) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            async with aiofiles.open(dest, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)


hf_adapter = HuggingFaceAdapter()
