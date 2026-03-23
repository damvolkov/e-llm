# e-llm

Self-contained LLM inference server — **llama.cpp + NiceGUI + nginx** in a single Docker container.

Configure, download, and run GGUF models (including hybrid GPU/CPU and MoE) through a clean web interface. The same port serves both the GUI and an OpenAI-compatible API.

## Quick Start

```bash
docker compose up -d
# Open http://localhost:45100
```

GPU required (NVIDIA + CUDA). First run will build the image (~2 min).

## What It Does

| Feature | Description |
|---------|-------------|
| **GUI** | NiceGUI dashboard — system info, model search/download, server config, test chat |
| **API** | OpenAI-compatible at `/v1/chat/completions`, `/v1/models` |
| **Health** | `/health` endpoint for monitoring |
| **Config** | YAML-based (`data/config/config.yaml`), editable from GUI |
| **Models** | Search HuggingFace, download GGUF, auto-detect quants |
| **Hybrid** | Full llama.cpp parameter control — GPU layers, KV cache offload, threading |

## Architecture

Single container, three internal processes:

```
:80 (nginx)
├── /           → NiceGUI     :8080
├── /v1/*       → llama-server :45150
└── /health     → NiceGUI     :8080
```

NiceGUI manages the llama-server subprocess lifecycle (start/stop/restart via `ServerManager`).

## Development

```bash
make install     # uv sync + pre-commit hooks
make dev         # NiceGUI with hot-reload on :8080
make check       # lint + type + test
```

## Docker

```bash
make docker-up       # Build + start on :45100
make docker-down     # Stop
make log             # Tail logs
```

Volumes:
- `./data/config/` — YAML configuration
- `./data/models/` — Downloaded GGUF files
- `./data/cache/` — llama.cpp KV cache

## Configuration

Edit `data/config/config.yaml` or use the GUI (Configuration → Server Configuration → Save & Apply).

Key parameters: model path, GPU layers, context size, threads, KV cache type, sampling, flash attention, fit-to-VRAM.

## API

```bash
# Chat completion
curl http://localhost:45100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'

# List models
curl http://localhost:45100/v1/models

# Health
curl http://localhost:45100/health
```

## Project Structure

```
src/e_llm/
├── main.py                 # NiceGUI entrypoint, lifecycle, DI
├── core/settings.py        # pydantic-settings
├── models/server.py        # ServerConfig (YAML bidirectional)
├── adapters/
│   ├── llamacpp.py         # httpx async client
│   └── huggingface.py      # Model download
├── operational/
│   ├── server.py           # ServerManager (subprocess lifecycle)
│   ├── system.py           # Hardware evaluator
│   └── models.py           # HF Hub search + quant extraction
└── pages/
    ├── config.py           # Configuration tab (system, models, params)
    └── test.py             # Test chat tab
```

## License

MIT
