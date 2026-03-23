"""Tests for LlamaCppAdapter."""

import httpx
import orjson
import pytest
from pytest_httpserver import HTTPServer

from e_llm.adapters.llamacpp import LlamaCppAdapter

##### INIT #####


async def test_adapter_creates_client() -> None:
    adapter = LlamaCppAdapter("http://localhost:9999")
    assert adapter._client is not None
    assert isinstance(adapter._client, httpx.AsyncClient)


##### HEALTH #####


async def test_get_health_returns_none_on_connect_error() -> None:
    adapter = LlamaCppAdapter("http://localhost:1")
    result = await adapter.get_health()
    assert result is None


async def test_get_health_returns_dict(httpserver: HTTPServer) -> None:
    httpserver.expect_request("/health").respond_with_data(
        orjson.dumps({"status": "ok"}), content_type="application/json"
    )
    adapter = LlamaCppAdapter(httpserver.url_for(""))
    result = await adapter.get_health()
    assert result == {"status": "ok"}


async def test_get_health_loading_model(httpserver: HTTPServer) -> None:
    httpserver.expect_request("/health").respond_with_data(
        orjson.dumps({"status": "loading model"}), content_type="application/json"
    )
    adapter = LlamaCppAdapter(httpserver.url_for(""))
    result = await adapter.get_health()
    assert result["status"] == "loading model"


##### MODELS #####


async def test_get_models_returns_empty_on_connect_error() -> None:
    adapter = LlamaCppAdapter("http://localhost:1")
    result = await adapter.get_models()
    assert result == []


async def test_get_models_returns_list(httpserver: HTTPServer) -> None:
    body = {"data": [{"id": "test-model", "object": "model"}]}
    httpserver.expect_request("/v1/models").respond_with_data(orjson.dumps(body), content_type="application/json")
    adapter = LlamaCppAdapter(httpserver.url_for(""))
    result = await adapter.get_models()
    assert len(result) == 1
    assert result[0]["id"] == "test-model"


##### STREAM COMPLETION #####


async def test_stream_completion_yields_tokens(httpserver: HTTPServer) -> None:
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    httpserver.expect_request("/v1/chat/completions").respond_with_data(sse_body, content_type="text/event-stream")
    adapter = LlamaCppAdapter(httpserver.url_for(""))
    tokens: list[str] = []
    async for token in adapter.stream_completion(
        messages=[{"role": "user", "content": "hi"}],
    ):
        tokens.append(token)
    assert tokens == ["Hello", " world"]


async def test_stream_completion_raises_on_http_error(httpserver: HTTPServer) -> None:
    httpserver.expect_request("/v1/chat/completions").respond_with_data("error", status=500)
    adapter = LlamaCppAdapter(httpserver.url_for(""))
    with pytest.raises(httpx.HTTPStatusError):
        async for _ in adapter.stream_completion(
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
