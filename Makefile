PROJECT ?= e-llm
PACKAGE ?= src/e_llm
PORT ?= 45100

BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[1;32m
CYAN   := \033[0;36m

export PYTHONPATH := $(CURDIR)/src

.PHONY: help install sync lock lint type test test-integration check dev docker-build docker-up docker-down log clean

help:
	@echo "$(BOLD)$(CYAN)e-llm$(RESET) — LLM inference server (llama.cpp + NiceGUI)"
	@echo ""
	@echo "$(BOLD)Setup:$(RESET)"
	@echo "  $(GREEN)make install$(RESET)          Install deps + pre-commit hooks"
	@echo "  $(GREEN)make sync$(RESET)             Sync dependencies from lockfile"
	@echo ""
	@echo "$(BOLD)Quality:$(RESET)"
	@echo "  $(GREEN)make lint$(RESET)             Ruff check + format"
	@echo "  $(GREEN)make type$(RESET)             ty type checker"
	@echo "  $(GREEN)make test$(RESET)             Unit tests (parallel, >90% coverage)"
	@echo "  $(GREEN)make check$(RESET)            lint + type + test"
	@echo ""
	@echo "$(BOLD)Development:$(RESET)"
	@echo "  $(GREEN)make dev$(RESET)              Run NiceGUI locally (reload)"
	@echo ""
	@echo "$(BOLD)Docker:$(RESET)"
	@echo "  $(GREEN)make docker-up$(RESET)        Build + start (GPU, :$(PORT))"
	@echo "  $(GREEN)make docker-down$(RESET)      Stop"
	@echo "  $(GREEN)make docker-build$(RESET)     Build image only"
	@echo "  $(GREEN)make log$(RESET)              Tail container logs"
	@echo ""
	@echo "$(BOLD)Cleanup:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)            Remove caches and build artifacts"


# Setup

install:
	@echo "$(GREEN)[1/3] Syncing Python dependencies$(RESET)"
	@uv sync --dev --quiet
	@echo "$(GREEN)[2/3] Installing pre-commit hooks$(RESET)"
	@uv run pre-commit install > /dev/null
	@echo "$(GREEN)[3/3] Done$(RESET)"

sync:
	@uv sync --dev

lock:
	@uv lock


# Quality

lint:
	@uv run ruff check --fix $(PACKAGE) tests/
	@uv run ruff format $(PACKAGE) tests/

type:
	@uv run ty check

test:
	@uv run pytest tests/unit -n auto -v --cov --cov-report=term-missing

test-integration:
	@uv run pytest tests/integration -v -m slow

check: lint type test


# Development

dev:
	@echo "$(CYAN)=== NiceGUI: http://localhost:8080 [reload] ===$(RESET)"
	@LLAMACPP_URL=http://127.0.0.1:45150 DATA_DIR=./data uv run python src/e_llm/main.py


# Docker

docker-build:
	@echo "$(CYAN)=== Building Docker image ===$(RESET)"
	@docker compose build
	@echo "$(GREEN)=== Build complete ===$(RESET)"

docker-up: docker-build
	@docker compose up -d
	@echo "$(GREEN)=== Running at http://localhost:$(PORT) ===$(RESET)"

docker-down:
	@docker compose down

log:
	@docker compose logs -f


# Cleanup

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ *.egg-info/
	@echo "$(GREEN)=== Clean ===$(RESET)"
