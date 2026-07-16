"""Framework-neutral AI model lifecycle primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import monotonic
from typing import Any, Iterator


class ModelState(str, Enum):
    """Memory residency states tracked by LightSwitch."""

    EVICTED = "evicted"
    CPU_RESIDENT = "cpu_resident"
    GPU_RESIDENT = "gpu_resident"
    IN_USE = "in_use"


@dataclass
class AIModel(ABC):
    """Recovery recipe and framework-specific hooks for one managed model.

    Subclasses retain only the lightweight configuration needed to reconstruct
    the model after eviction. The manager owns state transitions and calls the
    hooks below when their source state is active.
    """

    name: str
    estimated_ram_bytes: int
    estimated_vram_bytes: int
    state: ModelState = field(default=ModelState.EVICTED, init=False)
    last_used_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name must not be empty")
        if self.estimated_ram_bytes < 0:
            raise ValueError("estimated_ram_bytes must be non-negative")
        if self.estimated_vram_bytes < 0:
            raise ValueError("estimated_vram_bytes must be non-negative")
        self._lock = RLock()

    @property
    def is_resident(self) -> bool:
        return self.state is not ModelState.EVICTED

    @property
    def is_cpu_resident(self) -> bool:
        return self.state is ModelState.CPU_RESIDENT

    @property
    def is_gpu_resident(self) -> bool:
        return self.state in {ModelState.GPU_RESIDENT, ModelState.IN_USE}

    @property
    def is_in_use(self) -> bool:
        return self.state is ModelState.IN_USE

    @abstractmethod
    def load_to_cpu(self) -> None:
        """Reconstruct an evicted model in system RAM."""

    @abstractmethod
    def move_to_gpu(self) -> None:
        """Move a CPU-resident model to the managed GPU."""

    @abstractmethod
    def move_to_cpu(self) -> None:
        """Move a GPU-resident model to system RAM and release its VRAM."""

    @abstractmethod
    def evict_from_gpu(self) -> None:
        """Release a GPU-resident model without first moving it to CPU."""

    @abstractmethod
    def evict_from_cpu(self) -> None:
        """Release heavyweight CPU resources while retaining recovery data."""

    @abstractmethod
    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using the GPU-resident model."""

    def mark_cpu_resident(self) -> None:
        with self._lock:
            self.state = ModelState.CPU_RESIDENT

    def mark_gpu_resident(self) -> None:
        with self._lock:
            self.state = ModelState.GPU_RESIDENT
            self.touch()

    def mark_evicted(self) -> None:
        with self._lock:
            self.state = ModelState.EVICTED

    def touch(self) -> None:
        self.last_used_at = monotonic()

    @contextmanager
    def use(self) -> Iterator[None]:
        """Mark a GPU-resident model active while inference is in progress."""

        with self._lock:
            if self.state is not ModelState.GPU_RESIDENT:
                raise RuntimeError(
                    f"model {self.name!r} must be GPU-resident before use"
                )
            self.state = ModelState.IN_USE
            self.touch()

        try:
            yield
        finally:
            with self._lock:
                self.state = ModelState.GPU_RESIDENT
                self.touch()
