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
    """Lifecycle states tracked by LightSwitch."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    IN_USE = "in_use"


@dataclass
class AIModel(ABC):
    """Base class for framework-specific model adapters.

    Subclasses implement the actual model operations. LightSwitch only tracks
    state and calls these hooks at the right time.
    """

    name: str
    estimated_vram_bytes: int
    state: ModelState = field(default=ModelState.UNLOADED, init=False)
    last_used_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name must not be empty")
        if self.estimated_vram_bytes < 0:
            raise ValueError("estimated_vram_bytes must be non-negative")
        self._lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self.state in {ModelState.LOADED, ModelState.IN_USE}

    @property
    def is_in_use(self) -> bool:
        return self.state == ModelState.IN_USE

    @abstractmethod
    def load(self) -> None:
        """Load model resources into memory."""

    @abstractmethod
    def offload(self) -> None:
        """Release model resources from memory."""

    @abstractmethod
    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using framework-specific model code."""

    def mark_loaded(self) -> None:
        with self._lock:
            self.state = ModelState.LOADED
            self.touch()

    def mark_unloaded(self) -> None:
        with self._lock:
            self.state = ModelState.UNLOADED

    def touch(self) -> None:
        self.last_used_at = monotonic()

    @contextmanager
    def use(self) -> Iterator[None]:
        """Mark the model active while inference is in progress."""

        with self._lock:
            if self.state is not ModelState.LOADED:
                raise RuntimeError(f"model {self.name!r} must be loaded before use")
            self.state = ModelState.IN_USE
            self.touch()

        try:
            yield
        finally:
            with self._lock:
                self.state = ModelState.LOADED
                self.touch()

