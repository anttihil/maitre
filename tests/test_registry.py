"""Tests for the model/provider registry."""

from __future__ import annotations

from maitre.models import ModelSpec
from maitre.registry import Registry


def _make_config(models=None):
    return {
        "providers": {"stub": {"enabled": True}},
        "models": models or [],
    }


def test_load_models_from_config(stub_provider):
    cfg = _make_config(
        models=[
            {"name": "m1", "provider": "stub", "vram_mb": 1000, "tags": ["llm"]},
            {"name": "m2", "provider": "stub", "vram_mb": 2000, "tags": ["stt"]},
        ]
    )
    reg = Registry({"stub": stub_provider}, cfg)
    assert len(reg.list_models()) == 2
    assert reg.get_model("m1") is not None
    assert reg.get_model("m1").vram_mb == 1000


def test_filter_by_provider(stub_provider):
    cfg = _make_config(
        models=[
            {"name": "m1", "provider": "stub", "vram_mb": 1000},
            {"name": "m2", "provider": "other", "vram_mb": 2000},
        ]
    )
    reg = Registry({"stub": stub_provider}, cfg)
    assert len(reg.list_models(provider="stub")) == 1
    assert len(reg.list_models(provider="other")) == 1


def test_filter_by_tag(stub_provider):
    cfg = _make_config(
        models=[
            {"name": "m1", "provider": "stub", "vram_mb": 1000, "tags": ["llm", "chat"]},
            {"name": "m2", "provider": "stub", "vram_mb": 2000, "tags": ["stt"]},
        ]
    )
    reg = Registry({"stub": stub_provider}, cfg)
    assert len(reg.list_models(tag="llm")) == 1
    assert len(reg.list_models(tag="stt")) == 1
    assert len(reg.list_models(tag="ocr")) == 0


def test_add_and_remove_model(stub_provider):
    reg = Registry({"stub": stub_provider}, _make_config())
    spec = ModelSpec(name="dynamic", provider="stub", vram_mb=500)
    reg.add_model(spec)
    assert reg.get_model("dynamic") is not None
    removed = reg.remove_model("dynamic")
    assert removed is not None
    assert reg.get_model("dynamic") is None


def test_list_providers(stub_provider):
    reg = Registry({"stub": stub_provider}, _make_config())
    infos = reg.list_providers()
    assert len(infos) == 1
    assert infos[0].name == "stub"
    assert infos[0].installed is True
    assert infos[0].enabled is True
