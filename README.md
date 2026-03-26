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
| **Monitor** | Live VRAM/GPU/CPU/RAM sparklines, health indicator, power button to start/stop the server |
| **API** | OpenAI-compatible at `/v1/chat/completions`, `/v1/models` |
| **Health** | `/health` endpoint for monitoring |
| **Config** | YAML-based (`data/config/config.yaml`), editable from GUI |
| **Models** | Search HuggingFace, download GGUF, auto-detect quants |
| **Hybrid** | Full llama.cpp parameter control — GPU layers, KV cache offload, threading |

## Monitor

<p align="center">
  <img src="src/assets/monitor.png" alt="Live Monitor" width="320"/>
</p>

The header includes a real-time monitoring panel that stays visible on every page:

| Element | Description |
|---------|-------------|
| **Health indicator** | Color-coded dot — green (healthy), orange (loading), red (error), gray (disabled) |
| **Power button** | Toggle the llama-server on/off in real time. Checks available VRAM before starting (>50% free required) |
| **VRAM** | Live VRAM consumption (used/total GB) with historical sparkline |
| **GPU** | GPU utilization % with sparkline |
| **CPU** | CPU usage % with sparkline |
| **RAM** | RAM usage % with sparkline |

Sparklines are color-coded: **green** (<60%), **orange** (60–85%), **red** (>85%). Metrics poll every 2 seconds, health every 3 seconds.

### Server Control

- **Enable** — validates GPU resources are available, then starts llama-server with the current YAML config.
- **Disable** — gracefully stops the server (SIGTERM → SIGKILL fallback) and marks it as disabled.
- **Toggle** — power button cycles between enabled/disabled states with resource gating.

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
│   ├── controller.py       # ServerController (enable/disable + resource gating)
│   ├── monitor.py          # SystemMonitor (live CPU/RAM/GPU/VRAM metrics)
│   ├── system.py           # Hardware evaluator
│   └── models.py           # HF Hub search + quant extraction
└── pages/
    ├── config.py           # Configuration tab (system, models, params)
    └── test.py             # Test chat tab
```

## License

MIT
