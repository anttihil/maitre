"""VRAM-aware LRU cache for loaded models."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from maitre.models import LoadedModelInfo, ModelSpec, ModelStatus
from maitre.providers.base import BaseProvider
from maitre.vram import VRAMManager

log = logging.getLogger(__name__)


class InsufficientVRAMError(Exception):
    """Raised when a model cannot fit even after full eviction."""


class _LoadedEntry:
    """Internal bookkeeping for a loaded model."""

    __slots__ = ("spec", "handle", "info")

    def __init__(self, spec: ModelSpec, handle: Any) -> None:
        self.spec = spec
        self.handle = handle
        now = datetime.now(timezone.utc)
        self.info = LoadedModelInfo(spec=spec, loaded_at=now, last_accessed=now)


class ModelCache:
    """VRAM-aware LRU cache that loads/unloads models through providers."""

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        vram_manager: VRAMManager,
        max_loaded: int = 10,
    ) -> None:
        self._providers = providers
        self._vram = vram_manager
        self._max_loaded = max_loaded
        self._loaded: OrderedDict[str, _LoadedEntry] = OrderedDict()
        self._allocated_mb: int = 0
        self._lock = asyncio.Lock()
        # Track models that are currently loading
        self._loading: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def load(self, spec: ModelSpec) -> LoadedModelInfo:
        """Ensure *spec* is loaded, evicting LRU models if needed."""
        async with self._lock:
            # Already loaded? Bump to most-recently-used.
            if spec.name in self._loaded:
                entry = self._loaded[spec.name]
                self._loaded.move_to_end(spec.name)
                entry.info.last_accessed = datetime.now(timezone.utc)
                log.info("Model %s already loaded (LRU bumped)", spec.name)
                return entry.info

            provider = self._providers.get(spec.provider)
            if provider is None:
                raise ValueError(f"Unknown provider: {spec.provider!r}")
            if not provider.is_installed():
                raise RuntimeError(
                    f"Provider {spec.provider!r} is not installed. "
                    f"{provider.install_instructions()}"
                )

            needed = provider.estimated_vram_mb(spec)
            await self._ensure_budget(needed)

            log.info("Loading model %s (provider=%s, vram=%d MB) …", spec.name, spec.provider, needed)
            self._loading.add(spec.name)
            try:
                handle = await provider.load_model(spec)
            except Exception:
                self._loading.discard(spec.name)
                raise
            self._loading.discard(spec.name)

            entry = _LoadedEntry(spec, handle)
            self._loaded[spec.name] = entry
            self._allocated_mb += needed
            log.info("Model %s loaded. Allocated VRAM: %d MB", spec.name, self._allocated_mb)
            return entry.info

    async def unload(self, name: str) -> None:
        """Explicitly unload a model by name."""
        async with self._lock:
            await self._evict(name)

    async def get_handle(self, name: str) -> Any:
        """Return the raw handle for a loaded model, bumping LRU."""
        async with self._lock:
            entry = self._loaded.get(name)
            if entry is None:
                return None
            self._loaded.move_to_end(name)
            entry.info.last_accessed = datetime.now(timezone.utc)
            return entry.handle

    def status(self, name: str) -> LoadedModelInfo | None:
        """Return info for a loaded model, or None."""
        entry = self._loaded.get(name)
        if entry is not None:
            return entry.info
        if name in self._loading:
            return LoadedModelInfo(
                spec=ModelSpec(name=name, provider="", vram_mb=0),
                status=ModelStatus.LOADING,
            )
        return None

    @property
    def loaded_names(self) -> list[str]:
        return list(self._loaded.keys())

    @property
    def allocated_mb(self) -> int:
        return self._allocated_mb

    # ------------------------------------------------------------------
    # Eviction internals
    # ------------------------------------------------------------------

    async def _ensure_budget(self, needed_mb: int) -> None:
        """Evict LRU models until *needed_mb* fits within budget."""
        # Also respect max_loaded cap
        while len(self._loaded) >= self._max_loaded:
            if not self._loaded:
                break
            victim_name = next(iter(self._loaded))
            await self._evict(victim_name)

        # VRAM budget eviction
        if self._vram.has_gpu:
            while not self._vram.can_fit(needed_mb, self._allocated_mb):
                if not self._loaded:
                    status = self._vram.query_device()
                    raise InsufficientVRAMError(
                        f"Model requires {needed_mb} MB but only "
                        f"{status.available_for_models_mb} MB available "
                        f"(total={status.total_mb}, reserved={status.reserved_mb})"
                    )
                victim_name = next(iter(self._loaded))
                await self._evict(victim_name)

    async def _evict(self, name: str) -> None:
        """Unload a single model by name."""
        entry = self._loaded.pop(name, None)
        if entry is None:
            return

        provider = self._providers.get(entry.spec.provider)
        if provider:
            try:
                await provider.unload_model(entry.handle)
            except Exception:
                log.exception("Error unloading model %s", name)

        freed = entry.spec.vram_mb
        self._allocated_mb = max(0, self._allocated_mb - freed)
        log.info("Evicted %s (freed ~%d MB, allocated now %d MB)", name, freed, self._allocated_mb)

        # Best-effort memory cleanup
        del entry
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def unload_all(self) -> None:
        """Unload every loaded model."""
        async with self._lock:
            names = list(self._loaded.keys())
            for name in names:
                await self._evict(name)
