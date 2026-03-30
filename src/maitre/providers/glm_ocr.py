"""GLM OCR provider -- document OCR using zai-org/GLM-OCR (glmocr package)."""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

from maitre.models import ModelSpec
from maitre.providers.base import BaseProvider

log = logging.getLogger(__name__)


class GLMOCRProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "glm_ocr"

    @property
    def display_name(self) -> str:
        return "GLM OCR"

    def is_installed(self) -> bool:
        try:
            import glmocr  # noqa: F401

            return True
        except ImportError:
            return False

    def install_instructions(self) -> str:
        return 'pip install "glmocr[selfhosted]"'

    async def load_model(self, spec: ModelSpec) -> Any:
        from glmocr import GlmOcr

        layout_device = spec.params.get("layout_device", "cuda")

        log.info("Loading GLM OCR (layout_device=%s)", layout_device)

        loop = asyncio.get_running_loop()

        def _load() -> GlmOcr:
            ocr = GlmOcr(layout_device=layout_device)
            # Eagerly initialise so VRAM is allocated now, not on first request
            ocr.__enter__()
            return ocr

        return await loop.run_in_executor(None, _load)

    async def unload_model(self, handle: Any) -> None:
        try:
            handle.__exit__(None, None, None)
        except Exception:
            log.exception("Error during GlmOcr cleanup")
        del handle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def infer(self, handle: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Run OCR on one or more images.

        Request: {"image_path": "/path/to/image.png"}
                 or {"image_paths": ["/path/a.png", "/path/b.png"]}
                 or {"output_dir": "./results"}
        """
        image_path = request.get("image_path")
        image_paths = request.get("image_paths")
        output_dir = request.get("output_dir")

        if not image_path and not image_paths:
            raise ValueError("'image_path' or 'image_paths' is required")

        target = image_paths if image_paths else image_path

        loop = asyncio.get_running_loop()

        def _run() -> dict[str, Any]:
            result = handle.parse(target)
            response: dict[str, Any] = {"json_result": result.json_result}
            if output_dir:
                result.save(output_dir=output_dir)
                response["saved_to"] = output_dir
            return response

        return await loop.run_in_executor(None, _run)
