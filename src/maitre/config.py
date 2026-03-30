"""Configuration loading and merging."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

log = logging.getLogger(__name__)

_DEFAULTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "defaults.toml"
_USER_CONFIG_DIR = Path.home() / ".config" / "maitre"
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_defaults() -> dict[str, Any]:
    if _DEFAULTS_PATH.exists():
        return tomllib.loads(_DEFAULTS_PATH.read_text())
    return {}


def load_user_config() -> dict[str, Any]:
    if _USER_CONFIG_PATH.exists():
        return tomllib.loads(_USER_CONFIG_PATH.read_text())
    return {}


def load_config(extra_path: Path | None = None) -> dict[str, Any]:
    """Load and merge configuration: defaults -> user -> optional extra file."""
    cfg = load_defaults()
    cfg = _deep_merge(cfg, load_user_config())
    if extra_path and extra_path.exists():
        cfg = _deep_merge(cfg, tomllib.loads(extra_path.read_text()))
    return cfg


def save_user_config(cfg: dict[str, Any]) -> Path:
    """Write *cfg* to the user config file, creating parent dirs if needed."""
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_bytes(tomli_w.dumps(cfg).encode())
    log.info("Saved user config to %s", _USER_CONFIG_PATH)
    return _USER_CONFIG_PATH


def user_config_path() -> Path:
    return _USER_CONFIG_PATH
