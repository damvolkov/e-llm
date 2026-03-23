from collections.abc import AsyncGenerator

import httpx
import orjson


class LlamaCppAdapter:
    """Async client for the llama.cpp OpenAI-compatible API."""

    __slots__ = ("_client",)

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def get_health(self) -> dict[str, str] | None:
        """Check server health. Returns status dict or None if unreachable."""
        try:
            response = await self._client.get("/health")
            return orjson.loads(response.content)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout):
            return None

    async def get_models(self) -> list[dict[str, str]]:
        """List loaded models from the server."""
        try:
            response = await self._client.get("/v1/models")
            data = orjson.loads(response.content)
            return data.get("data", [])
        except (httpx.ConnectError, httpx.TimeoutException):
            return []

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str]:
        """Stream chat completion tokens via SSE."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = orjson.loads(data)
                if (delta := chunk["choices"][0]["delta"].get("content")) is not None:
                    yield delta
