"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from maitre.cache import ModelCache
from maitre.config import load_config
from maitre.providers import discover_providers
from maitre.registry import Registry
from maitre.vram import VRAMManager

log = logging.getLogger(__name__)


class AppState:
    """Shared application state accessible from request handlers."""

    registry: Registry
    cache: ModelCache
    vram: VRAMManager
    config: dict[str, Any]


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Startup / shutdown lifecycle."""
    log.info("Maitre starting up …")
    yield
    log.info("Maitre shutting down – unloading all models …")
    await state.cache.unload_all()


def create_app(config_path: Path | None = None) -> FastAPI:
    """Build and return the configured FastAPI application."""
    cfg = load_config(config_path)
    state.config = cfg

    # VRAM
    vram_cfg = cfg.get("vram", {})
    state.vram = VRAMManager(
        device_index=vram_cfg.get("device_index", 0),
        reserved_mb=vram_cfg.get("reserved_mb", 512),
        backend=vram_cfg.get("backend", "auto"),
    )

    # Providers + registry
    providers = discover_providers()
    state.registry = Registry(providers, cfg)

    # Cache
    cache_cfg = cfg.get("cache", {})
    state.cache = ModelCache(
        providers=providers,
        vram_manager=state.vram,
        max_loaded=cache_cfg.get("max_loaded_models", 10),
    )

    app = FastAPI(
        title="Maitre",
        description="Model manager service for local ML models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register routers
    from maitre.api.models import router as models_router
    from maitre.api.providers import router as providers_router
    from maitre.api.system import router as system_router

    app.include_router(models_router, prefix="/api/v1/models", tags=["models"])
    app.include_router(providers_router, prefix="/api/v1/providers", tags=["providers"])
    app.include_router(system_router, prefix="/api/v1/system", tags=["system"])

    return app
