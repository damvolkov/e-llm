"""Model discovery for e-llm. Search HuggingFace Hub, filter GGUF quants."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

from huggingface_hub import HfApi

if TYPE_CHECKING:
    from collections.abc import Iterator

_QUANT_PATTERN = re.compile(
    r"(UD-)?"
    r"("
    r"Q[2-8]_K(?:_[A-Z]+)?"
    r"|Q[2-8]_[0-9]"
    r"|IQ[1-4]_[A-Z]+"
    r"|F(?:16|32)|BF16|FP8"
    r"|MXFP4(?:_MOE)?"
    r")",
    re.IGNORECASE,
)
_GB = 1_073_741_824


@dataclass(slots=True, frozen=True)
class GGUFFile:
    """Single GGUF file within a repository."""

    filename: str
    size_bytes: int
    quant: str | None = None

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / _GB, 2)


@dataclass(slots=True, frozen=True)
class ModelResult:
    """A discovered model repo with its GGUF files."""

    repo_id: str
    downloads: int
    likes: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    files: tuple[GGUFFile, ...] = field(default_factory=tuple)

    @property
    def quants(self) -> list[str]:
        return sorted({f.quant for f in self.files if f.quant})

    @property
    def total_size_gb(self) -> float:
        return round(sum(f.size_bytes for f in self.files) / _GB, 2)


def _extract_quant(filename: str) -> str | None:
    """Extract quantization type from filename."""
    match _QUANT_PATTERN.search(filename):
        case None:
            return None
        case m:
            prefix = (m.group(1) or "").upper()
            return f"{prefix}{m.group(2).upper()}"


@cache
def _api() -> HfApi:
    return HfApi()


def search_models(
    query: str,
    *,
    limit: int = 5,
    gguf_only: bool = True,
) -> list[ModelResult]:
    """Search HuggingFace Hub for GGUF models by partial name."""
    api = _api()
    models = api.list_models(
        search=query,
        filter="gguf" if gguf_only else None,
        sort="downloads",
        limit=limit,
        expand=["siblings", "tags", "downloads", "likes"],
    )
    return [
        ModelResult(
            repo_id=m.id,
            downloads=m.downloads or 0,
            likes=m.likes or 0,
            tags=tuple(m.tags or []),
            files=tuple(_extract_gguf_files(m.siblings or [])),
        )
        for m in models
    ]


def _extract_gguf_files(siblings: list) -> Iterator[GGUFFile]:
    for s in siblings:
        if not s.rfilename.endswith(".gguf"):
            continue
        yield GGUFFile(
            filename=s.rfilename,
            size_bytes=s.size or 0,
            quant=_extract_quant(s.rfilename),
        )


def list_quants(repo_id: str) -> list[GGUFFile]:
    """List all GGUF quants available in a repo, sorted by size ascending."""
    api = _api()
    tree = api.list_repo_tree(repo_id, recursive=True)
    return sorted(
        (
            GGUFFile(
                filename=str(f.rfilename),
                size_bytes=getattr(f, "size", 0) or 0,
                quant=_extract_quant(str(f.rfilename)),
            )
            for f in tree
            if hasattr(f, "rfilename") and str(f.rfilename).endswith(".gguf")
        ),
        key=lambda f: f.size_bytes,
    )
