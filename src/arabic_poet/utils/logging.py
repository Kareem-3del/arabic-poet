"""Structured logging: human-readable console via rich, JSON lines for run logs."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = RichHandler(show_path=False, rich_tracebacks=True, console=None)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("arabic_poet")
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True
    if not sys.stderr.isatty():
        handler.setLevel(level.upper())


def add_run_log(path: Path) -> logging.Handler:
    """Attach a JSON-lines file handler for a single run; caller removes it when done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logging.getLogger("arabic_poet").addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger("arabic_poet").removeHandler(handler)
    handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("arabic_poet") else f"arabic_poet.{name}")
