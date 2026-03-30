"""Abstract base class for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from maitre.models import ModelSpec


class BaseProvider(ABC):
    """Every provider must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable identifier (e.g. 'faster_whisper')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g. 'Faster Whisper')."""

    @abstractmethod
    async def load_model(self, spec: ModelSpec) -> Any:
        """Load a model into GPU/CPU memory and return a handle."""

    @abstractmethod
    async def unload_model(self, handle: Any) -> None:
        """Release a model from memory."""

    @abstractmethod
    def is_installed(self) -> bool:
        """Return True if the underlying library is importable."""

    @abstractmethod
    def install_instructions(self) -> str:
        """Return a string describing how to install this provider's dependencies."""

    def estimated_vram_mb(self, spec: ModelSpec) -> int:
        """Override for dynamic VRAM estimation. Default: use spec.vram_mb."""
        return spec.vram_mb

    async def health_check(self, handle: Any) -> bool:
        """Return True if a loaded model handle is still functional."""
        return handle is not None

    async def infer(self, handle: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Run inference. Provider-specific request/response format."""
        raise NotImplementedError(f"{self.display_name} does not support direct inference via maitre")
