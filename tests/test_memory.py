from threading import Event, Thread

import pytest

from lightswitch import (
    AIModel,
    InsufficientRAMError,
    InsufficientVRAMError,
    MemoryManager,
    ModelAlreadyRegisteredError,
    ModelNotRegisteredError,
    ModelState,
    ModelTransitionError,
    RAMInfo,
    RAMQueryError,
    VRAMInfo,
)
from lightswitch.memory import query_system_ram


class StaticMemoryManager(MemoryManager):
    def __init__(self, ram_available=100, vram_available=100, **kwargs):
        super().__init__(**kwargs)
        self.ram_available = ram_available
        self.vram_available = vram_available
        self.ram_query_count = 0
        self.vram_query_count = 0

    def ram_info(self):
        self.ram_query_count += 1
        return RAMInfo(100, 100 - self.ram_available, self.ram_available)

    def vram_info(self):
        self.vram_query_count += 1
        return VRAMInfo(100, 100 - self.vram_available)


class FakeModel(AIModel):
    def __init__(
        self, name, ram=10, vram=5, result=None, always_evict_from_gpu=False
    ):
        super().__init__(
            name=name,
            estimated_ram_bytes=ram,
            estimated_vram_bytes=vram,
            always_evict_from_gpu=always_evict_from_gpu,
        )
        self.result = name if result is None else result
        self.manager = None
        self.calls = []
        self.fail_on = None

    def _called(self, hook):
        self.calls.append(hook)
        if self.fail_on == hook:
            raise RuntimeError("adapter failure")

    def load_to_cpu(self):
        self._called("load_to_cpu")
        self.manager.ram_available -= self.estimated_ram_bytes

    def move_to_gpu(self):
        self._called("move_to_gpu")
        self.manager.vram_available -= self.estimated_vram_bytes

    def move_to_cpu(self):
        self._called("move_to_cpu")
        self.manager.vram_available += self.estimated_vram_bytes

    def evict_from_gpu(self):
        self._called("evict_from_gpu")
        self.manager.ram_available += self.estimated_ram_bytes
        self.manager.vram_available += self.estimated_vram_bytes

    def evict_from_cpu(self):
        self._called("evict_from_cpu")
        self.manager.ram_available += self.estimated_ram_bytes

    def infer(self, *args, **kwargs):
        self._called("infer")
        return self.result


def make_manager(ram=100, vram=100, **kwargs):
    values = {
        "ram_reserve_bytes": 0,
        "vram_reserve_bytes": 0,
        "ram_available": ram,
        "vram_available": vram,
    }
    values.update(kwargs)
    return StaticMemoryManager(**values)


def register(manager, *models):
    for model in models:
        model.manager = manager
        manager.register(model)


def test_manager_validates_configuration():
    with pytest.raises(ValueError, match="ram_reserve_bytes"):
        MemoryManager(-1)
    with pytest.raises(ValueError, match="vram_reserve_bytes"):
        MemoryManager(0, -1)
    with pytest.raises(ValueError, match="gpu_index"):
        MemoryManager(0, gpu_index=-1)
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        MemoryManager(0, poll_interval_seconds=0)

    assert MemoryManager(0).conservative is True
    assert MemoryManager(0, conservative=False).conservative is False


def test_register_duplicate_and_lookup_errors():
    manager = make_manager()
    model = FakeModel("a")
    register(manager, model)

    assert manager.get("a") is model
    assert manager.models == {"a": model}
    with pytest.raises(ModelAlreadyRegisteredError):
        manager.register(model)
    with pytest.raises(ModelNotRegisteredError):
        manager.get("missing")
    assert manager.unregister("a") is model
    with pytest.raises(ModelNotRegisteredError):
        manager.unregister("a")


def test_infer_recovers_evicted_model_through_cpu_and_gpu():
    manager = make_manager(ram=20, vram=20, ram_reserve_bytes=3, vram_reserve_bytes=4)
    model = FakeModel("a", ram=5, vram=6, result="ok")
    register(manager, model)

    assert manager.infer("a", "prompt") == "ok"
    assert model.calls == ["load_to_cpu", "move_to_gpu", "infer"]
    assert model.state is ModelState.GPU_RESIDENT
    assert manager.ram_available == 15
    assert manager.vram_available == 14


def test_conservative_recovery_fully_evicts_before_loading_by_default():
    call_order = []

    class OrderedModel(FakeModel):
        def _called(self, hook):
            call_order.append(f"{self.name}:{hook}")
            super()._called(hook)

    manager = make_manager(vram=2)
    old = OrderedModel("old", vram=4)
    target = OrderedModel("target", vram=6)
    old.mark_gpu_resident()
    register(manager, old, target)

    manager.infer("target")

    assert call_order == [
        "old:move_to_cpu",
        "old:evict_from_cpu",
        "target:load_to_cpu",
        "target:move_to_gpu",
        "target:infer",
    ]
    assert old.state is ModelState.EVICTED


def test_conservative_recovery_directly_evicts_when_ram_is_scarce():
    call_order = []

    class OrderedModel(FakeModel):
        def _called(self, hook):
            call_order.append(f"{self.name}:{hook}")
            super()._called(hook)

    manager = make_manager(ram=5, vram=2)
    old = OrderedModel("old", ram=10, vram=4)
    target = OrderedModel("target", ram=5, vram=6)
    old.mark_gpu_resident()
    register(manager, old, target)

    manager.infer("target")

    assert call_order == [
        "old:evict_from_gpu",
        "target:load_to_cpu",
        "target:move_to_gpu",
        "target:infer",
    ]
    assert old.state is ModelState.EVICTED


def test_non_conservative_recovery_preserves_load_first_order():
    call_order = []

    class OrderedModel(FakeModel):
        def _called(self, hook):
            call_order.append(f"{self.name}:{hook}")
            super()._called(hook)

    manager = make_manager(vram=2, conservative=False)
    old = OrderedModel("old", vram=4)
    target = OrderedModel("target", vram=6)
    old.mark_gpu_resident()
    register(manager, old, target)

    manager.infer("target")

    assert call_order == [
        "target:load_to_cpu",
        "old:move_to_cpu",
        "target:move_to_gpu",
        "target:infer",
    ]


def test_conservative_recovery_does_not_load_if_offload_fails():
    manager = make_manager(vram=0)
    old = FakeModel("old", vram=5)
    target = FakeModel("target", vram=5)
    old.mark_gpu_resident()
    old.fail_on = "move_to_cpu"
    register(manager, old, target)

    with pytest.raises(ModelTransitionError, match="adapter failure"):
        manager.infer("target")

    assert old.state is ModelState.GPU_RESIDENT
    assert target.state is ModelState.EVICTED
    assert target.calls == []


def test_conservative_recovery_does_not_load_if_eviction_fails():
    manager = make_manager(vram=0)
    old = FakeModel("old", vram=5)
    target = FakeModel("target", vram=5)
    old.mark_gpu_resident()
    old.fail_on = "evict_from_cpu"
    register(manager, old, target)

    with pytest.raises(ModelTransitionError, match="adapter failure"):
        manager.infer("target")

    assert old.state is ModelState.CPU_RESIDENT
    assert old.calls == ["move_to_cpu", "evict_from_cpu"]
    assert target.state is ModelState.EVICTED
    assert target.calls == []


def test_conservative_recovery_does_not_load_if_direct_eviction_fails():
    manager = make_manager(ram=0, vram=0)
    old = FakeModel("old", ram=5, vram=5)
    target = FakeModel("target", ram=5, vram=5)
    old.mark_gpu_resident()
    old.fail_on = "evict_from_gpu"
    register(manager, old, target)

    with pytest.raises(ModelTransitionError, match="adapter failure"):
        manager.infer("target")

    assert old.state is ModelState.GPU_RESIDENT
    assert old.calls == ["evict_from_gpu"]
    assert target.state is ModelState.EVICTED
    assert target.calls == []


@pytest.mark.parametrize("resource", ["ram", "vram"])
def test_conservative_capacity_checks_directly_evict_from_gpu(resource):
    manager = make_manager(
        ram=0,
        vram=0,
        ram_reserve_bytes=1,
        vram_reserve_bytes=1,
    )
    model = FakeModel("old", ram=5, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    if resource == "ram":
        manager.ensure_ram_available(1)
    else:
        manager.ensure_vram_available(1)

    assert model.calls == ["evict_from_gpu"]
    assert model.state is ModelState.EVICTED


def test_non_conservative_capacity_check_stages_eviction_through_cpu():
    manager = make_manager(
        ram=0,
        ram_reserve_bytes=1,
        conservative=False,
    )
    model = FakeModel("old", ram=5, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    manager.ensure_ram_available(1)

    assert model.calls == ["move_to_cpu", "evict_from_cpu"]
    assert model.state is ModelState.EVICTED


def test_infer_reuses_gpu_resident_model():
    manager = make_manager()
    model = FakeModel("a")
    model.mark_gpu_resident()
    register(manager, model)

    manager.infer("a")
    assert model.calls == ["infer"]


def test_vram_capacity_offloads_least_recently_used_model():
    manager = make_manager(vram=2)
    old = FakeModel("old", vram=4)
    newer = FakeModel("newer", vram=4)
    target = FakeModel("target", vram=6)
    for model, last_used in ((old, 1.0), (newer, 2.0)):
        model.mark_gpu_resident()
        model.last_used_at = last_used
    target.mark_cpu_resident()
    register(manager, old, newer, target)

    manager.infer("target")

    assert old.state is ModelState.CPU_RESIDENT
    assert old.calls == ["move_to_cpu"]
    assert newer.state is ModelState.GPU_RESIDENT
    assert target.state is ModelState.GPU_RESIDENT


def test_model_flag_directly_evicts_for_vram_capacity():
    manager = make_manager(vram=0, conservative=False)
    model = FakeModel("old", vram=5, always_evict_from_gpu=True)
    model.mark_gpu_resident()
    register(manager, model)

    manager.ensure_vram_available(1)

    assert model.calls == ["evict_from_gpu"]
    assert model.state is ModelState.EVICTED


def test_model_flag_directly_evicts_during_vram_pressure_check():
    manager = make_manager(
        vram=0,
        vram_reserve_bytes=5,
        conservative=False,
    )
    model = FakeModel("old", vram=5, always_evict_from_gpu=True)
    model.mark_gpu_resident()
    register(manager, model)

    event = manager.check_pressure()

    assert model.calls == ["evict_from_gpu"]
    assert model.state is ModelState.EVICTED
    assert [action.from_state for action in event.actions] == [
        ModelState.GPU_RESIDENT
    ]
    assert [action.to_state for action in event.actions] == [ModelState.EVICTED]
    assert [action.reason for action in event.actions] == ["vram"]
    assert not event.unresolved_vram_pressure


def test_ram_pressure_fully_evicts_gpu_model_and_requeries():
    manager = make_manager(ram=2, vram=1, ram_reserve_bytes=6)
    model = FakeModel("old", ram=4, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    event = manager.check_pressure()

    assert model.calls == ["evict_from_gpu"]
    assert model.state is ModelState.EVICTED
    assert [action.from_state for action in event.actions] == [
        ModelState.GPU_RESIDENT
    ]
    assert [action.to_state for action in event.actions] == [ModelState.EVICTED]
    assert [action.reason for action in event.actions] == ["ram"]
    assert not event.unresolved_ram_pressure
    assert manager.ram_query_count >= 3
    assert manager.vram_query_count >= 3


def test_simultaneous_pressure_uses_full_eviction_to_resolve_both():
    manager = make_manager(
        ram=0,
        vram=0,
        ram_reserve_bytes=5,
        vram_reserve_bytes=5,
    )
    model = FakeModel("old", ram=5, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    event = manager.check_pressure()

    assert model.state is ModelState.EVICTED
    assert event.ram_pressure
    assert event.vram_pressure
    assert not event.unresolved_ram_pressure
    assert not event.unresolved_vram_pressure


def test_vram_offload_completes_eviction_if_it_creates_ram_pressure():
    class CPUExpansionModel(FakeModel):
        def move_to_cpu(self):
            super().move_to_cpu()
            self.manager.ram_available -= 6

    manager = make_manager(
        ram=7,
        vram=0,
        ram_reserve_bytes=2,
        vram_reserve_bytes=5,
    )
    model = CPUExpansionModel("old", ram=5, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    event = manager.check_pressure()

    assert model.state is ModelState.EVICTED
    assert [action.to_state for action in event.actions] == [
        ModelState.CPU_RESIDENT,
        ModelState.EVICTED,
    ]
    assert not event.unresolved_ram_pressure
    assert not event.unresolved_vram_pressure


def test_active_models_are_not_transitioned_under_pressure():
    manager = make_manager(
        ram=0,
        vram=0,
        ram_reserve_bytes=1,
        vram_reserve_bytes=1,
    )
    model = FakeModel("active")
    model.state = ModelState.IN_USE
    register(manager, model)

    event = manager.check_pressure()

    assert model.calls == []
    assert event.unresolved_ram_pressure
    assert event.unresolved_vram_pressure


def test_capacity_errors_when_no_idle_models_can_help():
    manager = make_manager(ram=3, vram=3, ram_reserve_bytes=2, vram_reserve_bytes=2)

    with pytest.raises(InsufficientRAMError):
        manager.ensure_ram_available(2)
    with pytest.raises(InsufficientVRAMError):
        manager.ensure_vram_available(2)


def test_failed_transition_preserves_model_state():
    manager = make_manager(vram=0)
    model = FakeModel("broken", vram=5)
    model.mark_gpu_resident()
    model.fail_on = "move_to_cpu"
    register(manager, model)

    with pytest.raises(ModelTransitionError, match="adapter failure"):
        manager.ensure_vram_available(1)
    assert model.state is ModelState.GPU_RESIDENT


def test_pressure_cycle_continues_after_one_model_transition_fails():
    manager = make_manager(vram=0, vram_reserve_bytes=5)
    broken = FakeModel("broken", vram=5)
    healthy = FakeModel("healthy", vram=5)
    for model in (broken, healthy):
        model.mark_gpu_resident()
    broken.fail_on = "move_to_cpu"
    register(manager, broken, healthy)

    event = manager.check_pressure()

    assert broken.state is ModelState.GPU_RESIDENT
    assert healthy.state is ModelState.CPU_RESIDENT
    assert "adapter failure" in event.error
    assert not event.unresolved_vram_pressure


def test_vram_pressure_directly_evicts_when_ram_query_fails():
    manager = make_manager(vram=0, vram_reserve_bytes=5)
    model = FakeModel("old", ram=5, vram=5)
    model.mark_gpu_resident()
    register(manager, model)

    def fail_ram_query():
        raise RAMQueryError("RAM unavailable")

    manager.ram_info = fail_ram_query

    event = manager.check_pressure()

    assert model.calls == ["evict_from_gpu"]
    assert model.state is ModelState.EVICTED
    assert [action.reason for action in event.actions] == ["vram"]
    assert "RAM unavailable" in event.error
    assert not event.unresolved_vram_pressure


def test_pressure_check_can_evict_idle_models_during_inference():
    inference_started = Event()
    finish_inference = Event()

    class BlockingModel(FakeModel):
        def infer(self, *args, **kwargs):
            inference_started.set()
            assert finish_inference.wait(1)
            return self.result

    manager = make_manager(ram=0, ram_reserve_bytes=5)
    active = BlockingModel("active")
    idle = FakeModel("idle", ram=5)
    active.mark_gpu_resident()
    idle.mark_cpu_resident()
    register(manager, active, idle)

    inference = Thread(target=manager.infer, args=("active",))
    inference.start()
    assert inference_started.wait(1)

    event = manager.check_pressure()

    assert active.state is ModelState.IN_USE
    assert idle.state is ModelState.EVICTED
    assert not event.unresolved_ram_pressure
    finish_inference.set()
    inference.join(1)
    assert not inference.is_alive()
    assert active.state is ModelState.GPU_RESIDENT


def test_pressure_callback_and_status_include_resolved_cycle():
    events = []
    manager = make_manager(ram=0, ram_reserve_bytes=5, on_pressure=events.append)
    model = FakeModel("old", ram=5)
    model.mark_cpu_resident()
    register(manager, model)

    event = manager.check_pressure()

    assert events == [event]
    assert manager.status.last_checked_at == event.checked_at
    assert manager.status.actions == event.actions
    assert not manager.status.unresolved_ram_pressure


def test_callback_failure_is_recorded_without_escaping():
    def fail_callback(event):
        raise RuntimeError("callback broke")

    manager = make_manager(ram=0, ram_reserve_bytes=1, on_pressure=fail_callback)
    manager.check_pressure()
    assert manager.status.last_error == "pressure callback failed: callback broke"


def test_watcher_start_stop_and_context_manager():
    called = Event()
    manager = make_manager(poll_interval_seconds=0.01, on_pressure=lambda event: called.set())
    manager.ram_reserve_bytes = 101

    manager.start()
    manager.start()
    assert called.wait(1)
    assert manager.is_running
    manager.stop()
    manager.stop()
    assert not manager.is_running

    with manager:
        assert manager.is_running
    assert not manager.is_running


def test_query_system_ram_uses_psutil(monkeypatch):
    class Memory:
        total = 100
        used = 40
        available = 60

    monkeypatch.setattr("lightswitch.memory.psutil.virtual_memory", lambda: Memory())
    assert query_system_ram() == RAMInfo(100, 40, 60)


def test_query_system_ram_wraps_errors(monkeypatch):
    def fail():
        raise OSError("unavailable")

    monkeypatch.setattr("lightswitch.memory.psutil.virtual_memory", fail)
    with pytest.raises(RAMQueryError, match="unavailable"):
        query_system_ram()
