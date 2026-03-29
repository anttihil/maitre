"""Integration tests for the models API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from maitre.cache import ModelCache
from maitre.models import ModelSpec
from maitre.registry import Registry
from maitre.server import create_app, state
from tests.conftest import StubProvider, StubVRAMManager


@pytest.fixture
def client():
    """Create a test client with stub provider and VRAM."""
    app = create_app()

    # Replace state internals with stubs
    stub_provider = StubProvider()
    providers = {"stub": stub_provider}
    stub_vram = StubVRAMManager(total_mb=8000, reserved_mb=512)

    cfg = {
        "providers": {"stub": {"enabled": True}},
        "models": [
            {"name": "test-model", "provider": "stub", "vram_mb": 2000, "tags": ["test"]},
        ],
        "server": {"host": "0.0.0.0", "port": 8741},
        "vram": {"backend": "stub", "reserved_mb": 512, "device_index": 0},
        "cache": {"max_loaded_models": 10},
    }

    state.config = cfg
    state.vram = stub_vram
    state.registry = Registry(providers, cfg)
    state.cache = ModelCache(providers=providers, vram_manager=stub_vram, max_loaded=10)

    return TestClient(app)


def test_list_models(client):
    resp = client.get("/api/v1/models/")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 1
    assert models[0]["name"] == "test-model"
    assert models[0]["status"] == "unloaded"


def test_load_and_status(client):
    resp = client.post("/api/v1/models/test-model/load")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"

    resp = client.get("/api/v1/models/test-model/status")
    assert resp.status_code == 200
    assert resp.json()["loaded"] is True


def test_load_and_unload(client):
    client.post("/api/v1/models/test-model/load")
    resp = client.post("/api/v1/models/test-model/unload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"


def test_load_nonexistent(client):
    resp = client.post("/api/v1/models/nope/load")
    assert resp.status_code == 404


def test_list_after_load(client):
    client.post("/api/v1/models/test-model/load")
    resp = client.get("/api/v1/models/")
    models = resp.json()
    assert models[0]["status"] == "loaded"
