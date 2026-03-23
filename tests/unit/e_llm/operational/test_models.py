"""Tests for model discovery (HuggingFace search + quant extraction)."""

import dataclasses

import pytest

from e_llm.operational.models import GGUFFile, ModelResult, _extract_quant

##### QUANT EXTRACTION #####


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model-Q4_K_M.gguf", "Q4_K_M"),
        ("model-Q8_0.gguf", "Q8_0"),
        ("model-IQ4_NL.gguf", "IQ4_NL"),
        ("model-F16.gguf", "F16"),
        ("model-BF16.gguf", "BF16"),
        ("model-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("model-q4_k_m.gguf", "Q4_K_M"),
        ("model-q8_0.gguf", "Q8_0"),
        ("model.gguf", None),
        ("readme.md", None),
    ],
    ids=[
        "Q4_K_M",
        "Q8_0",
        "IQ4_NL",
        "F16",
        "BF16",
        "UD-Q4_K_XL",
        "lowercase-q4",
        "lowercase-q8",
        "no-quant",
        "non-gguf",
    ],
)
async def test_extract_quant(filename: str, expected: str | None) -> None:
    assert _extract_quant(filename) == expected


##### GGUF FILE #####


async def test_gguf_file_size_gb() -> None:
    f = GGUFFile(filename="model.gguf", size_bytes=4_294_967_296, quant="Q4_K_M")
    assert f.size_gb == pytest.approx(4.0, abs=0.01)


async def test_gguf_file_frozen() -> None:
    f = GGUFFile(filename="model.gguf", size_bytes=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.filename = "other.gguf"  # type: ignore[misc]


##### MODEL RESULT #####


async def test_model_result_quants() -> None:
    result = ModelResult(
        repo_id="test/repo",
        downloads=100,
        likes=10,
        files=(
            GGUFFile(filename="a-Q4_K_M.gguf", size_bytes=1000, quant="Q4_K_M"),
            GGUFFile(filename="a-Q8_0.gguf", size_bytes=2000, quant="Q8_0"),
            GGUFFile(filename="a-Q4_K_M-split.gguf", size_bytes=500, quant="Q4_K_M"),
        ),
    )
    assert result.quants == ["Q4_K_M", "Q8_0"]


async def test_model_result_total_size() -> None:
    result = ModelResult(
        repo_id="test/repo",
        downloads=0,
        likes=0,
        files=(
            GGUFFile(filename="a.gguf", size_bytes=1_073_741_824),
            GGUFFile(filename="b.gguf", size_bytes=1_073_741_824),
        ),
    )
    assert result.total_size_gb == pytest.approx(2.0, abs=0.01)
