"""Agent input/output models for Tuner and Pinger."""

from pydantic import BaseModel, Field

from e_llm.models.server import ServerConfig
from e_llm.models.system import SystemInfo


class TunerInput(BaseModel):
    """Input to the Tuner agent — hardware profile + optional user context."""

    system: SystemInfo = Field(description="Hardware profile of the host machine")
    additional_prompt: str = Field(
        default="",
        description="Optional user-provided context or constraints for the recommendation",
    )


class TunerOutput(BaseModel):
    """Output from the Tuner agent — recommended llama.cpp server configuration."""

    config: ServerConfig = Field(description="Recommended llama.cpp server configuration")
    reasoning: str = Field(description="Explanation of why this configuration was chosen")
    model_suggestion: str = Field(
        default="",
        description="Suggested GGUF model repo and quant if applicable",
    )


class PingResult(BaseModel):
    """Result of a provider connectivity test."""

    ok: bool = Field(description="Whether the provider responded successfully")
    model: str = Field(description="Model identifier tested")
    latency_ms: float = Field(default=0.0, description="Response latency in milliseconds")
    error: str = Field(default="", description="Error message if failed")
