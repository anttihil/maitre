"""API routes for model management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from maitre.cache import InsufficientVRAMError
from maitre.models import ModelListItem, ModelStatus

router = APIRouter()


def _get_state():  # noqa: ANN202
    from maitre.server import state

    return state


@router.get("/", response_model=list[ModelListItem])
async def list_models(provider: str | None = None, tag: str | None = None):  # noqa: ANN201
    """List all registered models with their loaded/unloaded status."""
    s = _get_state()
    specs = s.registry.list_models(provider=provider, tag=tag)
    items = []
    for spec in specs:
        info = s.cache.status(spec.name)
        status = info.status if info else ModelStatus.UNLOADED
        items.append(
            ModelListItem(
                name=spec.name,
                provider=spec.provider,
                vram_mb=spec.vram_mb,
                tags=spec.tags,
                status=status,
            )
        )
    return items


@router.get("/{name}/status")
async def model_status(name: str):  # noqa: ANN201
    """Detailed status of a specific model."""
    s = _get_state()
    spec = s.registry.get_model(name)
    if spec is None:
        raise HTTPException(404, f"Model {name!r} not found in registry")
    info = s.cache.status(name)
    return {
        "spec": spec.model_dump(),
        "loaded": info is not None and info.status == ModelStatus.LOADED,
        "info": info.model_dump() if info else None,
    }


@router.post("/{name}/load")
async def load_model(name: str):  # noqa: ANN201
    """Load a model (evicting LRU models if necessary)."""
    s = _get_state()
    spec = s.registry.get_model(name)
    if spec is None:
        raise HTTPException(404, f"Model {name!r} not found in registry")
    try:
        info = await s.cache.load(spec)
    except InsufficientVRAMError as exc:
        raise HTTPException(507, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    return info.model_dump()


@router.post("/{name}/unload")
async def unload_model(name: str):  # noqa: ANN201
    """Explicitly unload a model."""
    s = _get_state()
    if s.cache.status(name) is None:
        raise HTTPException(404, f"Model {name!r} is not loaded")
    await s.cache.unload(name)
    return {"status": "unloaded", "name": name}


@router.post("/{name}/infer")
async def infer(name: str, request: dict[str, Any]):  # noqa: ANN201
    """Forward an inference request to a loaded model."""
    s = _get_state()
    spec = s.registry.get_model(name)
    if spec is None:
        raise HTTPException(404, f"Model {name!r} not found in registry")

    # Auto-load if not already loaded
    info = s.cache.status(name)
    if info is None or info.status != ModelStatus.LOADED:
        try:
            await s.cache.load(spec)
        except (InsufficientVRAMError, RuntimeError) as exc:
            raise HTTPException(507, str(exc)) from exc

    handle = await s.cache.get_handle(name)
    if handle is None:
        raise HTTPException(500, "Model loaded but handle unavailable")

    provider = s.registry.get_provider(spec.provider)
    if provider is None:
        raise HTTPException(500, f"Provider {spec.provider!r} not found")

    try:
        result = await provider.infer(handle, request)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Inference error: {exc}") from exc

    return result
