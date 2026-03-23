"""Pydantic AI agents — Pinger (connectivity) and Tuner (config recommendation)."""

from __future__ import annotations

import time
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from e_llm.models.agent import PingResult, TunerInput, TunerOutput
from e_llm.operational.models import search_models

_TUNER_SYSTEM = """\
You are an expert llama.cpp server configuration advisor. Your job is to recommend
the optimal llama.cpp server configuration for a given machine's hardware profile.

## Your expertise

- GGUF model quantization formats (Q2_K through Q8_0, F16, IQ variants, UD variants)
- Hybrid GPU/CPU inference: how to split layers between GPU VRAM and system RAM
- KV cache quantization and offloading strategies
- Threading optimization for batch and prompt processing
- Context window sizing relative to available memory
- MoE (Mixture of Experts) model characteristics and their hybrid affinity

## Key principles

1. **Maximize GPU utilization**: fill VRAM with as many layers as fit, use flash attention
2. **Leverage CPU for overflow**: MoE models are excellent for hybrid — only active experts
   need GPU, dormant ones can live in RAM
3. **KV cache strategy**: if RAM is abundant but VRAM is tight, offload KV cache to CPU
   with quantized types (q8_0 or q4_0) to save VRAM for model layers
4. **Threading**: set threads to physical cores (not logical) for prompt processing,
   logical cores for batch processing
5. **Context size**: scale with available memory — 8K minimum, 32K-64K for large RAM systems
6. **Batch size**: larger is faster but uses more memory — 2048 is a good default
7. **Model selection**: pick the largest quantization that fits in total memory
   (VRAM + RAM headroom), prefer Q4_K_M or Q4_K_XL for quality/size balance

## Constraints

- The config MUST be valid for llama.cpp server (all field names match the ServerConfig schema)
- The model path should be a real GGUF filename (use the search tool to verify availability)
- Be conservative with memory estimates — leave 2GB VRAM and 4GB RAM headroom
- If no GPU is available, use CPU-only config with mlock enabled

## Output

Return a complete ServerConfig with reasoning. If you find a good model via search,
include it as model_suggestion with the full repo_id and filename.
"""

##### PROVIDER REGISTRY #####

_PROVIDER_DEFAULTS: dict[str, dict[str, str | list[str]]] = {
    "openai": {
        "url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"],
    },
    "anthropic": {
        "url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-20250514",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-20250514", "claude-3-5-haiku-20241022"],
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    },
}


def get_provider_url(provider: str) -> str:
    """Default base URL for a provider."""
    return str(_PROVIDER_DEFAULTS.get(provider, {}).get("url", ""))


def get_provider_default_model(provider: str) -> str:
    """Default model name for a provider."""
    return str(_PROVIDER_DEFAULTS.get(provider, {}).get("default_model", "gpt-4o-mini"))


def get_provider_models(provider: str) -> list[str]:
    """Available models for a provider."""
    models = _PROVIDER_DEFAULTS.get(provider, {}).get("models", [])
    return list(models) if isinstance(models, list) else []


##### MODEL BUILDER #####


def _build_model(provider: str, model_name: str, base_url: str, api_key: str) -> Any:
    """Instantiate a pydantic-ai Model object for the given provider + model."""
    match provider:
        case "openai":
            prov = OpenAIProvider(api_key=api_key, base_url=base_url or None)
            return OpenAIChatModel(model_name, provider=prov)
        case "anthropic":
            prov = AnthropicProvider(api_key=api_key)
            return AnthropicModel(model_name, provider=prov)
        case "google":
            prov = GoogleProvider(api_key=api_key)
            return GoogleModel(model_name, provider=prov)
        case _:
            prov = OpenAIProvider(api_key=api_key, base_url=base_url or None)
            return OpenAIChatModel(model_name, provider=prov)


##### TOOLS #####


def search_gguf_models(query: str) -> str:
    """Search HuggingFace for GGUF models. Returns top results with quant info."""
    results = search_models(query, limit=5)
    if not results:
        return "No GGUF models found for this query."
    lines = []
    for r in results:
        quants = ", ".join(r.quants[:6]) if r.quants else "no quants listed"
        lines.append(f"- {r.repo_id} ({r.downloads:,} downloads) [{quants}]")
    return "\n".join(lines)


##### PINGER #####


async def run_ping(provider: str, model_name: str, base_url: str, api_key: str) -> PingResult:
    """Ping a model provider — validates API key AND model availability."""
    full_id = f"{provider}:{model_name}"
    try:
        model = _build_model(provider, model_name, base_url, api_key)
        pinger: Agent[None, str] = Agent(model=model, output_type=str)
        start = time.monotonic()
        await pinger.run("Reply with 'pong'")
        latency = (time.monotonic() - start) * 1000
        return PingResult(ok=True, model=full_id, latency_ms=round(latency, 1))
    except Exception as exc:
        msg = str(exc)
        if "model" in msg.lower() and ("not found" in msg.lower() or "does not exist" in msg.lower()):
            return PingResult(ok=False, model=full_id, error=f"Model not found: {model_name}")
        if "auth" in msg.lower() or "api key" in msg.lower() or "401" in msg:
            return PingResult(ok=False, model=full_id, error="Invalid API key")
        return PingResult(ok=False, model=full_id, error=msg[:120])


##### TUNER #####


def build_tuner(provider: str, model_name: str, base_url: str, api_key: str) -> Agent[None, TunerOutput]:
    """Create a Tuner agent with the correct provider and model."""
    model = _build_model(provider, model_name, base_url, api_key)
    return Agent(
        model=model,
        system_prompt=_TUNER_SYSTEM,
        output_type=TunerOutput,
        tools=[search_gguf_models],
    )


async def run_tuner(
    provider: str,
    model_name: str,
    base_url: str,
    api_key: str,
    input_data: TunerInput,
) -> TunerOutput:
    """Ping first (precondition), then execute the Tuner agent."""
    ping = await run_ping(provider, model_name, base_url, api_key)
    if not ping.ok:
        msg = f"Provider check failed: {ping.error}"
        raise ConnectionError(msg)

    tuner = build_tuner(provider, model_name, base_url, api_key)
    prompt = f"Hardware profile:\n{input_data.system.model_dump_json(indent=2)}"
    if input_data.additional_prompt:
        prompt += f"\n\nUser context:\n{input_data.additional_prompt}"
    result = await tuner.run(prompt)
    return result.output
