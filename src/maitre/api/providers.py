"""API routes for provider management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from maitre.models import ProviderInfo

router = APIRouter()


def _get_state():  # noqa: ANN202
    from maitre.server import state

    return state


@router.get("/", response_model=list[ProviderInfo])
async def list_providers():  # noqa: ANN201
    """List all registered providers with install/enabled status."""
    s = _get_state()
    return s.registry.list_providers()


@router.get("/{name}", response_model=ProviderInfo)
async def get_provider(name: str):  # noqa: ANN201
    """Get details for a single provider."""
    s = _get_state()
    for info in s.registry.list_providers():
        if info.name == name:
            return info
    raise HTTPException(404, f"Provider {name!r} not found")
