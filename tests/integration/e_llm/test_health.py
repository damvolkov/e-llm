"""Integration tests for health endpoint."""

import httpx
import pytest


@pytest.mark.slow
async def test_health_returns_200(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "server" in data


@pytest.mark.slow
async def test_health_server_running(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/health")
    server = response.json()["server"]
    assert server["running"] is True
    assert server["pid"] is not None
    assert server["healthy"] is True
