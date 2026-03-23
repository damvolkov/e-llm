"""Integration test fixtures — requires a running e-llm container."""

from collections.abc import AsyncGenerator

import httpx
import pytest


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the running e-llm server."""
    return "http://localhost:45100"


@pytest.fixture(scope="session")
async def http_client(base_url: str) -> AsyncGenerator[httpx.AsyncClient]:
    """Shared async HTTP client for integration tests."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client
