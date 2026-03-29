"""Shared fixtures for tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from maitre.models import ModelSpec
from maitre.providers.base import BaseProvider


class StubProvider(BaseProvider):
    """A test-only provider that tracks load/unload calls."""

    def __init__(self) -> None:
        self.loaded: dict[str, Any] = {}
        self._load_mock = AsyncMock(side_effect=self._do_load)
        self._unload_mock = AsyncMock(side_effect=self._do_unload)

    @property
    def name(self) -> str:
        return "stub"

    @property
    def display_name(self) -> str:
        return "Stub Provider"

    def is_installed(self) -> bool:
        return True

    def install_instructions(self) -> str:
        return "Already installed (stub)"

    async def _do_load(self, spec: ModelSpec) -> str:
        handle = f"handle:{spec.name}"
        self.loaded[spec.name] = handle
        return handle

    async def _do_unload(self, handle: Any) -> None:
        name = handle.replace("handle:", "")
        self.loaded.pop(name, None)

    async def load_model(self, spec: ModelSpec) -> Any:
        return await self._load_mock(spec)

    async def unload_model(self, handle: Any) -> None:
        return await self._unload_mock(handle)


class StubVRAMManager:
    """A test-only VRAM manager with configurable capacity."""

    def __init__(self, total_mb: int = 8000, reserved_mb: int = 512) -> None:
        self.total_mb = total_mb
        self.reserved_mb = reserved_mb
        self.has_gpu = True

    def can_fit(self, model_vram_mb: int, allocated_mb: int) -> bool:
        remaining = self.total_mb - self.reserved_mb - allocated_mb
        return model_vram_mb <= remaining

    def query_device(self):  # noqa: ANN201
        from maitre.models import VRAMStatus

        used = 0
        free = self.total_mb
        return VRAMStatus(
            total_mb=self.total_mb,
            used_mb=used,
            free_mb=free,
            reserved_mb=self.reserved_mb,
            available_for_models_mb=max(0, free - self.reserved_mb),
            device_name="Stub GPU",
            device_index=0,
        )


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider()


@pytest.fixture
def stub_vram() -> StubVRAMManager:
    return StubVRAMManager()


@pytest.fixture
def sample_specs() -> list[ModelSpec]:
    return [
        ModelSpec(name="model-a", provider="stub", vram_mb=2000, tags=["llm"]),
        ModelSpec(name="model-b", provider="stub", vram_mb=3000, tags=["stt"]),
        ModelSpec(name="model-c", provider="stub", vram_mb=1500, tags=["ocr"]),
        ModelSpec(name="model-huge", provider="stub", vram_mb=20000, tags=["llm"]),
    ]
