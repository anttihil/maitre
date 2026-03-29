"""llama-cpp-python provider -- local LLM inference via GGUF models."""

from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from typing import Any

from maitre.models import ModelSpec
from maitre.providers.base import BaseProvider

log = logging.getLogger(__name__)


def _resolve_model_path(spec: ModelSpec) -> str:
    """Return a local file path to the GGUF model, downloading via huggingface_hub if needed."""
    # Direct path
    if "model_path" in spec.params:
        return spec.params["model_path"]

    # Download from HuggingFace Hub
    repo_id = spec.params.get("repo_id")
    filename = spec.params.get("filename")
    if not repo_id or not filename:
        raise ValueError("llama_cpp models need either 'model_path' or 'repo_id' + 'filename' in params")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    log.info("Downloaded model to %s", path)
    return path


class LlamaCppProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "llama_cpp"

    @property
    def display_name(self) -> str:
        return "llama.cpp"

    def is_installed(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def install_instructions(self) -> str:
        return 'pip install "maitre[llama-cpp]"  # or: pip install llama-cpp-python'

    async def load_model(self, spec: ModelSpec) -> Any:
        from llama_cpp import Llama

        loop = asyncio.get_running_loop()
        model_path = await loop.run_in_executor(None, _resolve_model_path, spec)

        n_gpu_layers = spec.params.get("n_gpu_layers", -1)
        n_ctx = spec.params.get("context_length", 4096)
        verbose = spec.params.get("verbose", False)

        log.info(
            "Loading llama.cpp model %s (n_gpu_layers=%s, n_ctx=%d)",
            Path(model_path).name,
            n_gpu_layers,
            n_ctx,
        )

        model = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose,
            ),
        )
        return model

    async def unload_model(self, handle: Any) -> None:
        if hasattr(handle, "close"):
            handle.close()
        del handle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def infer(self, handle: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Run chat completion.

        Request: {"messages": [{"role": "user", "content": "Hello"}], ...}
        """
        messages = request.get("messages")
        if not messages:
            raise ValueError("'messages' is required")

        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": request.get("max_tokens", 512),
            "temperature": request.get("temperature", 0.7),
        }
        if "stop" in request:
            kwargs["stop"] = request["stop"]

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: handle.create_chat_completion(**kwargs),
        )
        return result
