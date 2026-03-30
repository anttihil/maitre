"""API routes for system status."""

from __future__ import annotations

from fastapi import APIRouter

from maitre.models import VRAMStatus

router = APIRouter()


def _get_state():  # noqa: ANN202
    from maitre.server import state

    return state


@router.get("/health")
async def health():  # noqa: ANN201
    return {"status": "ok"}


@router.get("/vram", response_model=VRAMStatus)
async def vram_status():  # noqa: ANN201
    """Current VRAM status (device + bookkeeping)."""
    s = _get_state()
    device = s.vram.query_device()
    return device


@router.get("/vram/budget")
async def vram_budget():  # noqa: ANN201
    """VRAM budget summary including model allocations."""
    s = _get_state()
    device = s.vram.query_device()
    return {
        "device": device.model_dump(),
        "allocated_by_models_mb": s.cache.allocated_mb,
        "loaded_models": s.cache.loaded_names,
    }


@router.get("/config")
async def show_config():  # noqa: ANN201
    """Return the running configuration (sanitized)."""
    s = _get_state()
    return s.config
