"""Shared HTTP client helpers for CLI commands that talk to a running server."""

from __future__ import annotations

import os
import sys

import httpx

DEFAULT_URL = "http://localhost:8741"


def base_url() -> str:
    return os.environ.get("MAITRE_URL", DEFAULT_URL)


def get(path: str) -> dict | list:
    url = f"{base_url()}{path}"
    try:
        resp = httpx.get(url, timeout=30)
    except httpx.ConnectError:
        print(f"[error] Cannot connect to maitre at {base_url()}. Is the server running?", file=sys.stderr)
        raise SystemExit(1)
    resp.raise_for_status()
    return resp.json()


def post(path: str, json: dict | None = None) -> dict | list:
    url = f"{base_url()}{path}"
    try:
        resp = httpx.post(url, json=json, timeout=300)  # model loads can be slow
    except httpx.ConnectError:
        print(f"[error] Cannot connect to maitre at {base_url()}. Is the server running?", file=sys.stderr)
        raise SystemExit(1)
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text)
        print(f"[error] {resp.status_code}: {detail}", file=sys.stderr)
        raise SystemExit(1)
    return resp.json()
