"""Tests for VRAMManager (using the stub backend)."""

from __future__ import annotations

from maitre.vram import VRAMManager


def test_stub_backend():
    """Without an NVIDIA GPU, the stub backend should be used."""
    mgr = VRAMManager(device_index=0, reserved_mb=512, backend="stub")
    assert mgr.has_gpu is False
    status = mgr.query_device()
    assert status.device_name == "none"
    assert status.total_mb == 0


def test_auto_fallback_to_stub():
    """On a machine without pynvml or nvidia-smi, auto should fall back to stub."""
    mgr = VRAMManager(backend="auto")
    # We don't know what backend is available in CI, but it shouldn't crash
    status = mgr.query_device()
    assert status is not None
