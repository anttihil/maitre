"""Integration tests for the system API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from maitre.cache import ModelCache
from maitre.registry import Registry
from maitre.server import create_app, state
from tests.conftest import StubProvider, StubVRAMManager


@pytest.fixture
def client():
    app = create_app()

    stub_provider = StubProvider()
    providers = {"stub": stub_provider}
    stub_vram = StubVRAMManager(total_mb=8000, reserved_mb=512)

    cfg = {
        "providers": {"stub": {"enabled": True}},
        "models": [],
        "server": {"host": "0.0.0.0", "port": 8741},
        "vram": {"backend": "stub", "reserved_mb": 512, "device_index": 0},
        "cache": {"max_loaded_models": 10},
    }

    state.config = cfg
    state.vram = stub_vram
    state.registry = Registry(providers, cfg)
    state.cache = ModelCache(providers=providers, vram_manager=stub_vram, max_loaded=10)

    return TestClient(app)


def test_health(client):
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_vram(client):
    resp = client.get("/api/v1/system/vram")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_mb" in data
    assert "device_name" in data


def test_vram_budget(client):
    resp = client.get("/api/v1/system/vram/budget")
    assert resp.status_code == 200
    data = resp.json()
    assert "device" in data
    assert "allocated_by_models_mb" in data
    assert data["allocated_by_models_mb"] == 0


def test_config(client):
    resp = client.get("/api/v1/system/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "server" in data
