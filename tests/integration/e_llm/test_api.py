"""Integration tests for llama.cpp API endpoints via nginx proxy."""

import httpx
import orjson
import pytest


@pytest.mark.slow
async def test_v1_models_returns_list(http_client: httpx.AsyncClient) -> None:
    response = await http_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0


@pytest.mark.slow
async def test_v1_chat_completions_non_streaming(http_client: httpx.AsyncClient) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
        "stream": False,
    }
    response = await http_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"]


@pytest.mark.slow
async def test_v1_chat_completions_streaming(http_client: httpx.AsyncClient) -> None:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Count to 3."}],
        "max_tokens": 32,
        "stream": True,
    }
    chunks: list[str] = []
    async with http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = orjson.loads(data)
            if delta := chunk["choices"][0]["delta"].get("content"):
                chunks.append(delta)
    assert len(chunks) > 0


@pytest.mark.slow
async def test_llama_health_via_nginx(http_client: httpx.AsyncClient) -> None:
    """Test that /health on llama-server is reachable via the GUI health endpoint."""
    response = await http_client.get("/health")
    assert response.json()["server"]["healthy"] is True
