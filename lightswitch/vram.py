"""VRAM-aware model manager."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Optional

from .errors import (
    InsufficientVRAMError,
    ModelAlreadyRegisteredError,
    ModelNotRegisteredError,
    VRAMQueryError,
)
from .model import AIModel

MIB = 1024 * 1024


@dataclass(frozen=True)
class VRAMInfo:
    """GPU memory usage in bytes."""

    total_bytes: int
    used_bytes: int

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)


class VRAMManager:
    """Register models and offload inactive models when VRAM is tight."""

    def __init__(self, gpu_index: int = 0) -> None:
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        self.gpu_index = gpu_index
        self._models: Dict[str, AIModel] = {}
        self._lock = RLock()

    @property
    def models(self) -> Dict[str, AIModel]:
        with self._lock:
            return dict(self._models)

    def register(self, model: AIModel) -> None:
        with self._lock:
            if model.name in self._models:
                raise ModelAlreadyRegisteredError(
                    f"model {model.name!r} is already registered"
                )
            self._models[model.name] = model

    def unregister(self, name: str) -> AIModel:
        with self._lock:
            try:
                return self._models.pop(name)
            except KeyError as exc:
                raise ModelNotRegisteredError(f"model {name!r} is not registered") from exc

    def get(self, name: str) -> AIModel:
        with self._lock:
            try:
                return self._models[name]
            except KeyError as exc:
                raise ModelNotRegisteredError(f"model {name!r} is not registered") from exc

    def vram_info(self) -> VRAMInfo:
        return query_nvidia_smi(self.gpu_index)

    def ensure_available(self, required_bytes: int, keep_loaded: Optional[str] = None) -> None:
        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")

        with self._lock:
            available = self.vram_info().available_bytes
            if available >= required_bytes:
                return

            for model in self._offload_candidates(keep_loaded=keep_loaded):
                model.offload()
                model.mark_unloaded()
                available += model.estimated_vram_bytes
                if available >= required_bytes:
                    return

        raise InsufficientVRAMError(
            f"insufficient VRAM: need {required_bytes} bytes, "
            f"but only {available} bytes are available after offloading"
        )

    def infer(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            model = self.get(name)
            if not model.is_loaded:
                self.ensure_available(model.estimated_vram_bytes, keep_loaded=name)
                model.load()
                model.mark_loaded()

            with model.use():
                return model.infer(*args, **kwargs)

    def _offload_candidates(self, keep_loaded: Optional[str] = None) -> Iterable[AIModel]:
        candidates = [
            model
            for model in self._models.values()
            if model.name != keep_loaded and model.is_loaded and not model.is_in_use
        ]
        return sorted(candidates, key=lambda model: model.last_used_at)


def query_nvidia_smi(gpu_index: int = 0) -> VRAMInfo:
    """Return total and used VRAM for one NVIDIA GPU using nvidia-smi."""

    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
        f"--id={gpu_index}",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise VRAMQueryError("nvidia-smi was not found") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise VRAMQueryError(f"nvidia-smi failed: {message}") from exc

    output = completed.stdout.strip().splitlines()
    if not output:
        raise VRAMQueryError("nvidia-smi returned no GPU memory data")

    first_line = output[0]
    try:
        total_mib, used_mib = [int(part.strip()) for part in first_line.split(",", 1)]
    except ValueError as exc:
        raise VRAMQueryError(f"could not parse nvidia-smi output: {first_line!r}") from exc

    return VRAMInfo(total_bytes=total_mib * MIB, used_bytes=used_mib * MIB)

