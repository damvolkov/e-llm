# Future: TurboQuant KV Cache Integration

Google Research published TurboQuant at ICLR 2026 (March 24, 2026) — a training-free,
data-oblivious vector quantization algorithm that compresses the KV cache to 3-4 bits
per element with negligible quality loss. 4-6x KV cache memory reduction.

## How it works

Two-stage compression pipeline:

1. **FWHT rotation** (Fast Walsh-Hadamard Transform) — makes vector coordinates uniform
2. **PolarQuant** — scalar quantization with precomputed codebook on rotated coordinates

Key insight: K (keys) controls attention routing via softmax, so asymmetric config
`-ctk q8_0 -ctv turbo3` preserves quality — keys at high precision, values compressed.

## Available implementations

| Fork | Status | Notes |
|------|--------|-------|
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | Working CUDA kernels | turbo3/turbo4, custom FA kernels, norm correction, 98.8% of q8_0 prefill speed |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Integrates spiritbuun + extras | block_size=128 optimization, HIP/ROCm support |
| [ik_llama.cpp #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) | Issue open | Not yet integrated |
| [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) | Discussion | Not merged upstream |

Google has NOT released official code. All implementations are community-built from the paper.

## Impact for e-llm (RTX 4090 + 192GB RAM + Qwen3-Coder-Next 128K)

With `-ctk q8_0 -ctv turbo3` at 128K context:

- KV cache compresses ~4.6x → frees VRAM for more model layers on GPU → more tok/s
- For coding agent workloads (generation-heavy, prefill-light), combining ik_llama.cpp
  (fused MoE, +90% generation speed) + TurboQuant is the optimal setup

## Changes required in e-llm

| File | Change | Risk |
|------|--------|------|
| `Dockerfile` | Multi-stage build from TurboQuant fork instead of `ghcr.io/ggml-org/llama.cpp:server-cuda` | HIGH — build time +10min, fork dependency |
| `src/e_llm/pages/config.py` | Add `turbo3`, `turbo4` to `_CACHE_TYPES` | LOW |
| `src/e_llm/operational/agents.py` | Update `_TUNER_SYSTEM` with TurboQuant knowledge | LOW |
| `tests/` | New parametrize cases for turbo cache types | LOW |

No changes needed in `models/server.py` (str fields) or `operational/server.py` (pass-through).

## Priority order

1. Update to llama.cpp HEAD (PR #19375 merged) — +38% tok/s
2. Test ik_llama.cpp fork — +90% tok/s in generation
3. TurboQuant via spiritbuun fork — 128K+ context viable
4. Migrate to upstream when merged

## Risks

- **Fork dependency**: spiritbuun may abandon. Mitigation: turboquant_plus has broader community.
- **CI build time**: CUDA compilation is slow. Mitigation: Docker layer caching or prebuilt base image.
- **Custom FA kernels**: Less tested than upstream. Monitor stability.
- **Upstream merge**: When llama.cpp merges TurboQuant, switch back to official image.

## References

- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Tom's Hardware coverage](https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss)
- [DEV Community guide](https://dev.to/arshtechpro/turboquant-what-developers-need-to-know-about-googles-kv-cache-compression-eeg)
- [VentureBeat](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
