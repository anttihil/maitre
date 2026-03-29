"""Pydantic data models shared across the application."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelSpec(BaseModel):
    """A registered model in the config/registry."""

    name: str
    provider: str
    vram_mb: int
    params: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class LoadedModelInfo(BaseModel):
    """Runtime information about a currently-loaded model."""

    spec: ModelSpec
    status: ModelStatus = ModelStatus.LOADED
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


class ModelListItem(BaseModel):
    """Response item when listing models."""

    name: str
    provider: str
    vram_mb: int
    tags: list[str]
    status: ModelStatus


class VRAMStatus(BaseModel):
    """Snapshot of GPU VRAM state."""

    total_mb: int
    used_mb: int
    free_mb: int
    reserved_mb: int
    available_for_models_mb: int
    device_name: str
    device_index: int


class ProviderInfo(BaseModel):
    """Information about a registered provider."""

    name: str
    display_name: str
    installed: bool
    enabled: bool
    install_instructions: str
