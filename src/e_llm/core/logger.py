"""Structured logging with colored step labels."""

import logging
import sys
from datetime import UTC, datetime

import structlog

_STEP_COLORS: dict[str, str] = {
    "START": "\033[38;2;100;200;100m",
    "STOP": "\033[38;2;200;100;100m",
    "OK": "\033[38;2;100;200;255m",
    "MODEL": "\033[38;2;180;140;255m",
    "DOWNLOAD": "\033[38;2;100;180;255m",
    "SERVER": "\033[38;2;255;200;100m",
    "HEALTH": "\033[38;2;100;255;200m",
    "ERROR": "\033[38;2;255;80;80m",
    "WARN": "\033[38;2;255;180;60m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_LEVEL_COLORS = {
    "error": "\033[38;2;255;80;80m",
    "critical": "\033[38;2;255;80;80m",
    "warning": "\033[38;2;255;180;60m",
}


class ColorRenderer:
    """Render structured logs as: HH:MM:SS [STEP] message key=value."""

    def __call__(self, _logger: object, _name: str, event_dict: dict) -> str:
        ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
        step = event_dict.pop("step", "")
        event = event_dict.pop("event", "")
        level = event_dict.pop("level", "info")

        step_color = _STEP_COLORS.get(step, _DIM)
        level_color = _LEVEL_COLORS.get(level, "")

        parts = [f"{_DIM}{ts}{_RESET}"]
        if step:
            parts.append(f"{step_color}[{step}]{_RESET}")
        if level_color:
            parts.append(f"{level_color}{event}{_RESET}")
        else:
            parts.append(event)

        extras = " ".join(f"{_DIM}{k}={v}{_RESET}" for k, v in event_dict.items() if k not in {"timestamp"})
        if extras:
            parts.append(extras)

        return " ".join(parts)


def configure_logging() -> None:
    """Set up structlog with colored renderer."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.set_exc_info,
            ColorRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    for name in ("watchfiles", "httpx", "httpcore", "nicegui", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


configure_logging()

logger = structlog.get_logger("ellm")
