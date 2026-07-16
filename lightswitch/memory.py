"""Unified system RAM and GPU VRAM model manager."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
from threading import Event, RLock, Thread, current_thread
from time import monotonic
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import psutil  # type: ignore[import-untyped]

from .errors import (
    InsufficientRAMError,
    InsufficientVRAMError,
    ModelAlreadyRegisteredError,
    ModelNotRegisteredError,
    ModelTransitionError,
    RAMQueryError,
)
from .model import AIModel, ModelState
from .vram import VRAMInfo, query_nvidia_smi


@dataclass(frozen=True)
class RAMInfo:
    """System memory usage in bytes."""

    total_bytes: int
    used_bytes: int
    available_bytes: int


@dataclass(frozen=True)
class MemoryAction:
    """One model transition made to relieve memory pressure."""

    model_name: str
    from_state: ModelState
    to_state: ModelState
    reason: str


@dataclass(frozen=True)
class PressureEvent:
    """Result of one simultaneous RAM and VRAM pressure check."""

    checked_at: float
    ram_before: Optional[RAMInfo]
    vram_before: Optional[VRAMInfo]
    ram_after: Optional[RAMInfo]
    vram_after: Optional[VRAMInfo]
    actions: Tuple[MemoryAction, ...]
    ram_pressure: bool
    vram_pressure: bool
    unresolved_ram_pressure: bool
    unresolved_vram_pressure: bool
    error: Optional[str]


@dataclass(frozen=True)
class MonitorStatus:
    """Thread-safe snapshot of the background monitor's latest cycle."""

    running: bool
    last_checked_at: Optional[float] = None
    ram_info: Optional[RAMInfo] = None
    vram_info: Optional[VRAMInfo] = None
    actions: Tuple[MemoryAction, ...] = ()
    ram_pressure: bool = False
    vram_pressure: bool = False
    unresolved_ram_pressure: bool = False
    unresolved_vram_pressure: bool = False
    last_error: Optional[str] = None


PressureCallback = Callable[[PressureEvent], None]


def query_system_ram() -> RAMInfo:
    """Return total, used, and available system RAM using psutil."""

    try:
        memory = psutil.virtual_memory()
    except Exception as exc:
        raise RAMQueryError(f"could not query system RAM: {exc}") from exc
    return RAMInfo(
        total_bytes=int(memory.total),
        used_bytes=int(memory.used),
        available_bytes=int(memory.available),
    )


class MemoryManager:
    """Manage recoverable models against RAM and VRAM reserves."""

    def __init__(
        self,
        ram_reserve_bytes: int,
        vram_reserve_bytes: int = 0,
        gpu_index: int = 0,
        poll_interval_seconds: float = 1.0,
        on_pressure: Optional[PressureCallback] = None,
        conservative: bool = True,
    ) -> None:
        if ram_reserve_bytes < 0:
            raise ValueError("ram_reserve_bytes must be non-negative")
        if vram_reserve_bytes < 0:
            raise ValueError("vram_reserve_bytes must be non-negative")
        if gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

        self.ram_reserve_bytes = ram_reserve_bytes
        self.vram_reserve_bytes = vram_reserve_bytes
        self.gpu_index = gpu_index
        self.poll_interval_seconds = poll_interval_seconds
        self.on_pressure = on_pressure
        self.conservative = conservative
        self._models: Dict[str, AIModel] = {}
        self._lock = RLock()
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None
        self._last_status = MonitorStatus(running=False)

    @property
    def models(self) -> Dict[str, AIModel]:
        with self._lock:
            return dict(self._models)

    @property
    def status(self) -> MonitorStatus:
        with self._lock:
            return replace(self._last_status, running=self.is_running)

    @property
    def is_running(self) -> bool:
        thread = self._monitor_thread
        return thread is not None and thread.is_alive()

    def __enter__(self) -> "MemoryManager":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def start(self) -> None:
        """Start the pressure watcher if it is not already running."""

        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._monitor_thread = Thread(
                target=self._monitor_loop,
                name="lightswitch-memory-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Request watcher shutdown and wait for it to finish."""

        self._stop_event.set()
        thread = self._monitor_thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout)

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
                raise ModelNotRegisteredError(
                    f"model {name!r} is not registered"
                ) from exc

    def get(self, name: str) -> AIModel:
        with self._lock:
            try:
                return self._models[name]
            except KeyError as exc:
                raise ModelNotRegisteredError(
                    f"model {name!r} is not registered"
                ) from exc

    def ram_info(self) -> RAMInfo:
        return query_system_ram()

    def vram_info(self) -> VRAMInfo:
        return query_nvidia_smi(self.gpu_index)

    def ensure_ram_available(
        self, required_bytes: int, keep_resident: Optional[str] = None
    ) -> None:
        """Ensure a new allocation can coexist with the configured RAM reserve."""

        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        target_available = self.ram_reserve_bytes + required_bytes

        with self._lock:
            available = self.ram_info().available_bytes
            if available >= target_available:
                return

            for model in self._ram_candidates(keep_resident):
                if model.state is ModelState.GPU_RESIDENT:
                    self._move_to_cpu(model)
                    available = self.ram_info().available_bytes
                    if available >= target_available:
                        return
                self._evict(model)
                available = self.ram_info().available_bytes
                if available >= target_available:
                    return

        raise InsufficientRAMError(
            f"insufficient system RAM: need {required_bytes} bytes plus a "
            f"{self.ram_reserve_bytes}-byte reserve, but only {available} bytes "
            "are available after eviction"
        )

    def ensure_vram_available(
        self, required_bytes: int, keep_resident: Optional[str] = None
    ) -> None:
        """Ensure a new allocation can coexist with the configured VRAM reserve."""

        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        target_available = self.vram_reserve_bytes + required_bytes

        with self._lock:
            available = self.vram_info().available_bytes
            if available >= target_available:
                return

            for model in self._vram_candidates(keep_resident):
                self._move_to_cpu(model)
                if self.ram_info().available_bytes < self.ram_reserve_bytes:
                    self._evict(model)
                available = self.vram_info().available_bytes
                if available >= target_available:
                    return

        raise InsufficientVRAMError(
            f"insufficient VRAM: need {required_bytes} bytes plus a "
            f"{self.vram_reserve_bytes}-byte reserve, but only {available} bytes "
            "are available after offloading"
        )

    def infer(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Recover a model as needed and run GPU inference."""

        with ExitStack() as usage:
            with self._lock:
                model = self.get(name)
                if model.state is ModelState.EVICTED:
                    if self.conservative:
                        self.ensure_vram_available(
                            model.estimated_vram_bytes, keep_resident=name
                        )
                    self.ensure_ram_available(
                        model.estimated_ram_bytes, keep_resident=name
                    )
                    self._load_to_cpu(model)

                if model.state is ModelState.CPU_RESIDENT:
                    self.ensure_ram_available(0, keep_resident=name)
                    self.ensure_vram_available(
                        model.estimated_vram_bytes, keep_resident=name
                    )
                    self._move_to_gpu(model)

                usage.enter_context(model.use())
            return model.infer(*args, **kwargs)

    def check_pressure(self) -> PressureEvent:
        """Run one pressure cycle and notify the configured callback."""

        with self._lock:
            event = self._check_pressure_locked()
            self._last_status = MonitorStatus(
                running=self.is_running,
                last_checked_at=event.checked_at,
                ram_info=event.ram_after,
                vram_info=event.vram_after,
                actions=event.actions,
                ram_pressure=event.ram_pressure,
                vram_pressure=event.vram_pressure,
                unresolved_ram_pressure=event.unresolved_ram_pressure,
                unresolved_vram_pressure=event.unresolved_vram_pressure,
                last_error=event.error,
            )

        if (event.ram_pressure or event.vram_pressure or event.error) and self.on_pressure:
            try:
                self.on_pressure(event)
            except Exception as exc:
                with self._lock:
                    self._last_status = replace(
                        self._last_status,
                        last_error=f"pressure callback failed: {exc}",
                    )
        return event

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.check_pressure()
            except Exception as exc:
                with self._lock:
                    self._last_status = replace(
                        self._last_status,
                        last_checked_at=monotonic(),
                        last_error=f"memory monitor failed: {exc}",
                    )
            self._stop_event.wait(self.poll_interval_seconds)

    def _check_pressure_locked(self) -> PressureEvent:
        errors: list[str] = []
        actions: list[MemoryAction] = []
        ram_before = self._safe_ram_info(errors)
        vram_before = self._safe_vram_info(errors)
        ram_current = ram_before
        vram_current = vram_before
        ram_pressure = (
            ram_before is not None
            and ram_before.available_bytes < self.ram_reserve_bytes
        )
        vram_pressure = (
            vram_before is not None
            and vram_before.available_bytes < self.vram_reserve_bytes
        )

        if ram_pressure:
            for model in self._ram_candidates():
                try:
                    if model.state is ModelState.GPU_RESIDENT:
                        actions.append(self._move_to_cpu(model, reason="ram"))
                        ram_current = self._safe_ram_info(errors)
                        vram_current = self._safe_vram_info(errors)
                        if (
                            ram_current is not None
                            and ram_current.available_bytes >= self.ram_reserve_bytes
                        ):
                            break
                    actions.append(self._evict(model, reason="ram"))
                    ram_current = self._safe_ram_info(errors)
                    vram_current = self._safe_vram_info(errors)
                except ModelTransitionError as exc:
                    errors.append(str(exc))
                if (
                    ram_current is not None
                    and ram_current.available_bytes >= self.ram_reserve_bytes
                ):
                    break

        if (
            vram_current is not None
            and vram_current.available_bytes < self.vram_reserve_bytes
        ):
            for model in self._vram_candidates():
                try:
                    actions.append(self._move_to_cpu(model, reason="vram"))
                    ram_current = self._safe_ram_info(errors)
                    vram_current = self._safe_vram_info(errors)
                    if (
                        ram_current is not None
                        and ram_current.available_bytes < self.ram_reserve_bytes
                    ):
                        actions.append(self._evict(model, reason="ram"))
                        ram_current = self._safe_ram_info(errors)
                        vram_current = self._safe_vram_info(errors)
                except ModelTransitionError as exc:
                    errors.append(str(exc))
                if (
                    vram_current is not None
                    and vram_current.available_bytes >= self.vram_reserve_bytes
                ):
                    break

        ram_after = self._safe_ram_info(errors)
        vram_after = self._safe_vram_info(errors)
        unresolved_ram = (
            ram_after is not None
            and ram_after.available_bytes < self.ram_reserve_bytes
        )
        unresolved_vram = (
            vram_after is not None
            and vram_after.available_bytes < self.vram_reserve_bytes
        )
        return PressureEvent(
            checked_at=monotonic(),
            ram_before=ram_before,
            vram_before=vram_before,
            ram_after=ram_after,
            vram_after=vram_after,
            actions=tuple(actions),
            ram_pressure=ram_pressure,
            vram_pressure=vram_pressure,
            unresolved_ram_pressure=unresolved_ram,
            unresolved_vram_pressure=unresolved_vram,
            error="; ".join(errors) or None,
        )

    def _safe_ram_info(self, errors: list[str]) -> Optional[RAMInfo]:
        try:
            return self.ram_info()
        except Exception as exc:
            errors.append(str(exc))
            return None

    def _safe_vram_info(self, errors: list[str]) -> Optional[VRAMInfo]:
        try:
            return self.vram_info()
        except Exception as exc:
            errors.append(str(exc))
            return None

    def _ram_candidates(
        self, keep_resident: Optional[str] = None
    ) -> Iterable[AIModel]:
        candidates = [
            model
            for model in self._models.values()
            if model.name != keep_resident
            and model.state in {ModelState.CPU_RESIDENT, ModelState.GPU_RESIDENT}
        ]
        return sorted(candidates, key=lambda model: model.last_used_at)

    def _vram_candidates(
        self, keep_resident: Optional[str] = None
    ) -> Iterable[AIModel]:
        candidates = [
            model
            for model in self._models.values()
            if model.name != keep_resident
            and model.state is ModelState.GPU_RESIDENT
        ]
        return sorted(candidates, key=lambda model: model.last_used_at)

    def _load_to_cpu(self, model: AIModel, reason: str = "recovery") -> MemoryAction:
        source = model.state
        if source is not ModelState.EVICTED:
            raise RuntimeError(f"cannot load {model.name!r} to CPU from {source.value}")
        try:
            model.load_to_cpu()
        except Exception as exc:
            raise ModelTransitionError(
                f"model {model.name!r} failed to load to CPU: {exc}"
            ) from exc
        model.mark_cpu_resident()
        return MemoryAction(model.name, source, ModelState.CPU_RESIDENT, reason)

    def _move_to_gpu(self, model: AIModel, reason: str = "recovery") -> MemoryAction:
        source = model.state
        if source is not ModelState.CPU_RESIDENT:
            raise RuntimeError(f"cannot move {model.name!r} to GPU from {source.value}")
        try:
            model.move_to_gpu()
        except Exception as exc:
            raise ModelTransitionError(
                f"model {model.name!r} failed to move to GPU: {exc}"
            ) from exc
        model.mark_gpu_resident()
        return MemoryAction(model.name, source, ModelState.GPU_RESIDENT, reason)

    def _move_to_cpu(self, model: AIModel, reason: str = "capacity") -> MemoryAction:
        source = model.state
        if source is not ModelState.GPU_RESIDENT:
            raise RuntimeError(f"cannot move {model.name!r} to CPU from {source.value}")
        try:
            model.move_to_cpu()
        except Exception as exc:
            raise ModelTransitionError(
                f"model {model.name!r} failed to move to CPU: {exc}"
            ) from exc
        model.mark_cpu_resident()
        return MemoryAction(model.name, source, ModelState.CPU_RESIDENT, reason)

    def _evict(self, model: AIModel, reason: str = "capacity") -> MemoryAction:
        source = model.state
        if source is not ModelState.CPU_RESIDENT:
            raise RuntimeError(f"cannot evict {model.name!r} from {source.value}")
        try:
            model.evict_from_cpu()
        except Exception as exc:
            raise ModelTransitionError(
                f"model {model.name!r} failed to evict from CPU: {exc}"
            ) from exc
        model.mark_evicted()
        return MemoryAction(model.name, source, ModelState.EVICTED, reason)
