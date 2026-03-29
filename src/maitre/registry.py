"""Model and provider registry -- wires config to the runtime objects."""

from __future__ import annotations

import logging
from typing import Any

from maitre.models import ModelSpec, ProviderInfo
from maitre.providers.base import BaseProvider

log = logging.getLogger(__name__)


class Registry:
    """Holds all known model specs and provider instances."""

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        config: dict[str, Any],
    ) -> None:
        self._providers = providers
        self._config = config
        self._models: dict[str, ModelSpec] = {}
        self._provider_enabled: dict[str, bool] = {}

        self._load_from_config(config)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_from_config(self, cfg: dict[str, Any]) -> None:
        # Provider enable/disable flags
        prov_cfg = cfg.get("providers", {})
        for name in self._providers:
            section = prov_cfg.get(name, {})
            self._provider_enabled[name] = section.get("enabled", True)

        # Model entries
        for entry in cfg.get("models", []):
            spec = ModelSpec(**entry)
            self._models[spec.name] = spec
            log.debug("Registered model: %s (provider=%s, vram=%d MB)", spec.name, spec.provider, spec.vram_mb)

        log.info(
            "Registry loaded: %d models, %d providers (%d installed)",
            len(self._models),
            len(self._providers),
            sum(1 for p in self._providers.values() if p.is_installed()),
        )

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------

    def get_model(self, name: str) -> ModelSpec | None:
        return self._models.get(name)

    def list_models(self, provider: str | None = None, tag: str | None = None) -> list[ModelSpec]:
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        if tag:
            models = [m for m in models if tag in m.tags]
        return models

    def add_model(self, spec: ModelSpec) -> None:
        self._models[spec.name] = spec
        log.info("Added model to registry: %s", spec.name)

    def remove_model(self, name: str) -> ModelSpec | None:
        return self._models.pop(name, None)

    # ------------------------------------------------------------------
    # Provider queries
    # ------------------------------------------------------------------

    def get_provider(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[ProviderInfo]:
        infos = []
        for name, prov in self._providers.items():
            infos.append(
                ProviderInfo(
                    name=name,
                    display_name=prov.display_name,
                    installed=prov.is_installed(),
                    enabled=self._provider_enabled.get(name, True),
                    install_instructions=prov.install_instructions(),
                )
            )
        return infos

    def is_provider_enabled(self, name: str) -> bool:
        return self._provider_enabled.get(name, False)
