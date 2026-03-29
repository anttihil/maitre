"""VRAM querying and budget management."""

from __future__ import annotations

import logging
import shutil
import subprocess

from maitre.models import VRAMStatus

log = logging.getLogger(__name__)


class VRAMManager:
    """Query GPU VRAM and track budget for model loading decisions."""

    def __init__(
        self,
        device_index: int = 0,
        reserved_mb: int = 512,
        backend: str = "auto",
    ) -> None:
        self.device_index = device_index
        self.reserved_mb = reserved_mb
        self._backend = self._resolve_backend(backend)
        log.info("VRAM backend: %s (device %d, reserved %d MB)", self._backend, device_index, reserved_mb)

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self, requested: str) -> str:
        if requested == "pynvml" or (requested == "auto" and self._pynvml_available()):
            return "pynvml"
        if requested == "nvidia-smi" or (requested == "auto" and self._nvsmi_available()):
            return "nvidia-smi"
        return "stub"

    @staticmethod
    def _pynvml_available() -> bool:
        try:
            import pynvml  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _nvsmi_available() -> bool:
        return shutil.which("nvidia-smi") is not None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_device(self) -> VRAMStatus:
        """Return a snapshot of current GPU VRAM state."""
        if self._backend == "pynvml":
            return self._query_pynvml()
        if self._backend == "nvidia-smi":
            return self._query_nvsmi()
        return self._query_stub()

    def _query_pynvml(self) -> VRAMStatus:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            total = int(info.total / 1024 / 1024)
            used = int(info.used / 1024 / 1024)
            free = int(info.free / 1024 / 1024)
        finally:
            pynvml.nvmlShutdown()

        return VRAMStatus(
            total_mb=total,
            used_mb=used,
            free_mb=free,
            reserved_mb=self.reserved_mb,
            available_for_models_mb=max(0, free - self.reserved_mb),
            device_name=name,
            device_index=self.device_index,
        )

    def _query_nvsmi(self) -> VRAMStatus:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.device_index}",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        name = parts[0]
        total = int(parts[1])
        used = int(parts[2])
        free = int(parts[3])

        return VRAMStatus(
            total_mb=total,
            used_mb=used,
            free_mb=free,
            reserved_mb=self.reserved_mb,
            available_for_models_mb=max(0, free - self.reserved_mb),
            device_name=name,
            device_index=self.device_index,
        )

    def _query_stub(self) -> VRAMStatus:
        """Stub for systems without an NVIDIA GPU -- reports unlimited budget."""
        log.warning("No NVIDIA GPU detected; using stub VRAM backend (unlimited budget)")
        return VRAMStatus(
            total_mb=0,
            used_mb=0,
            free_mb=0,
            reserved_mb=self.reserved_mb,
            available_for_models_mb=0,
            device_name="none",
            device_index=self.device_index,
        )

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def can_fit(self, model_vram_mb: int, allocated_mb: int) -> bool:
        """Check whether *model_vram_mb* fits in the remaining budget.

        Uses both bookkeeping (*allocated_mb*) and actual device state
        to be conservative.
        """
        status = self.query_device()
        # Bookkeeping-based remaining budget
        bookkeeping_remaining = max(0, status.total_mb - self.reserved_mb - allocated_mb)
        # Actual device remaining budget
        actual_remaining = status.available_for_models_mb
        # Use the more conservative of the two
        effective = min(bookkeeping_remaining, actual_remaining)
        return model_vram_mb <= effective

    @property
    def has_gpu(self) -> bool:
        return self._backend != "stub"
