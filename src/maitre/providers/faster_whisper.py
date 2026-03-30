"""Faster Whisper provider -- speech-to-text."""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

from maitre.models import ModelSpec
from maitre.providers.base import BaseProvider

log = logging.getLogger(__name__)


class FasterWhisperProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "faster_whisper"

    @property
    def display_name(self) -> str:
        return "Faster Whisper"

    def is_installed(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def install_instructions(self) -> str:
        return 'pip install "maitre[faster-whisper]"  # or: pip install faster-whisper'

    async def load_model(self, spec: ModelSpec) -> Any:
        from faster_whisper import WhisperModel

        model_size = spec.params.get("model_size", "large-v3")
        compute_type = spec.params.get("compute_type", "float16")
        device = spec.params.get("device", "cuda")

        log.info("Loading Whisper model %s (compute=%s, device=%s)", model_size, compute_type, device)

        loop = asyncio.get_running_loop()
        model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(model_size, device=device, compute_type=compute_type),
        )
        return model

    async def unload_model(self, handle: Any) -> None:
        del handle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def infer(self, handle: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Transcribe an audio file.

        Request: {"audio_path": "/path/to/audio.wav", "language": "en", ...}
        """
        audio_path = request.get("audio_path")
        if not audio_path:
            raise ValueError("'audio_path' is required")

        kwargs: dict[str, Any] = {}
        if "language" in request:
            kwargs["language"] = request["language"]
        if "beam_size" in request:
            kwargs["beam_size"] = request["beam_size"]

        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: handle.transcribe(audio_path, **kwargs),
        )
        # Materialise segments (they're a generator)
        segment_list = await loop.run_in_executor(
            None,
            lambda: [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ],
        )

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segment_list,
            "text": " ".join(s["text"].strip() for s in segment_list),
        }
