"""Centralised logging configuration — reads settings from rules.yaml."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → Invoice_Auditor/
_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"

_LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(module)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger from rules.yaml.

    Format : [timestamp][level][module] message
    Outputs: console + rotating file handler (when enable_audit_log is true).
    """
    try:
        with open(_RULES_PATH, encoding="utf-8") as fh:
            rules = yaml.safe_load(fh) or {}
    except Exception:
        rules = {}

    log_cfg = rules.get("logging", {})
    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "./logs/invoice_auditor.log")

    log_path = (_PROJECT_ROOT / log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)

    if log_cfg.get("enable_audit_log", True):
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
