"""GLM OCR provider -- vision/OCR using GLM-Edge-V2 or similar."""

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
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except ImportError:
            return False

    def install_instructions(self) -> str:
        return "pip install transformers torch pillow accelerate"

    async def load_model(self, spec: ModelSpec) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = spec.params.get("model_id", "THUDM/glm-edge-v2-9b")
        dtype = spec.params.get("dtype", "float16")
        torch_dtype = getattr(torch, dtype, torch.float16)

        log.info("Loading GLM OCR model %s (dtype=%s)", model_id, dtype)

        loop = asyncio.get_running_loop()

        def _load() -> dict[str, Any]:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            return {"model": model, "tokenizer": tokenizer, "model_id": model_id}

        return await loop.run_in_executor(None, _load)

    async def unload_model(self, handle: Any) -> None:
        if isinstance(handle, dict):
            handle.get("model", None)  # reference for deletion
            handle.clear()
        del handle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def infer(self, handle: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Run OCR on an image.

        Request: {"image_path": "/path/to/image.png", "prompt": "OCR this image"}
        """
        from PIL import Image

        image_path = request.get("image_path")
        if not image_path:
            raise ValueError("'image_path' is required")

        prompt = request.get("prompt", "Please perform OCR on this image and return the text content.")
        model = handle["model"]
        tokenizer = handle["tokenizer"]

        loop = asyncio.get_running_loop()

        def _run() -> str:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=True, return_tensors="pt", tokenize=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.get("max_tokens", 2048),
                do_sample=False,
            )
            # Decode only the generated part
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(generated, skip_special_tokens=True)

        text = await loop.run_in_executor(None, _run)
        return {"text": text}
