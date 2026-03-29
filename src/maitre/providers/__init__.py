"""Provider discovery and registration."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maitre.providers.base import BaseProvider

log = logging.getLogger(__name__)

_BUILTIN_PROVIDERS: dict[str, str] = {
    "faster_whisper": "maitre.providers.faster_whisper:FasterWhisperProvider",
    "glm_ocr": "maitre.providers.glm_ocr:GLMOCRProvider",
    "llama_cpp": "maitre.providers.llama_cpp:LlamaCppProvider",
}


def _import_class(dotted: str) -> type[BaseProvider]:
    module_path, cls_name = dotted.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def discover_providers() -> dict[str, BaseProvider]:
    """Return a mapping of provider name -> instantiated provider."""
    providers: dict[str, BaseProvider] = {}

    # Built-in providers
    for name, target in _BUILTIN_PROVIDERS.items():
        try:
            cls = _import_class(target)
            providers[name] = cls()
        except Exception:
            log.exception("Failed to load built-in provider %s", name)

    # Entry-point plugins
    try:
        eps = importlib.metadata.entry_points(group="maitre.providers")
    except TypeError:
        eps = importlib.metadata.entry_points().get("maitre.providers", [])  # type: ignore[assignment]

    for ep in eps:
        if ep.name in providers:
            continue  # built-in already loaded
        try:
            cls = ep.load()
            providers[ep.name] = cls()
        except Exception:
            log.exception("Failed to load plugin provider %s", ep.name)

    return providers
