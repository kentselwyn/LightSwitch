import subprocess

import pytest

from lightswitch import (
    AIModel,
    InsufficientVRAMError,
    ModelAlreadyRegisteredError,
    ModelNotRegisteredError,
    ModelState,
    VRAMInfo,
    VRAMManager,
    VRAMQueryError,
)
from lightswitch.vram import MIB, query_nvidia_smi


class FakeModel(AIModel):
    def __init__(self, name, estimated_vram_bytes, result=None):
        super().__init__(name=name, estimated_vram_bytes=estimated_vram_bytes)
        self.result = result if result is not None else name
        self.load_calls = 0
        self.offload_calls = 0
        self.infer_calls = 0

    def load(self):
        self.load_calls += 1

    def offload(self):
        self.offload_calls += 1

    def infer(self, *args, **kwargs):
        self.infer_calls += 1
        return self.result


class StaticVRAMManager(VRAMManager):
    def __init__(self, available_bytes):
        super().__init__()
        self.available_bytes = available_bytes

    def vram_info(self):
        return VRAMInfo(
            total_bytes=10 * 1024**3,
            used_bytes=(10 * 1024**3) - self.available_bytes,
        )


def test_register_duplicate_and_lookup_errors():
    manager = StaticVRAMManager(available_bytes=10)
    model = FakeModel("a", 1)

    manager.register(model)
    assert manager.get("a") is model

    with pytest.raises(ModelAlreadyRegisteredError):
        manager.register(model)

    with pytest.raises(ModelNotRegisteredError):
        manager.get("missing")

    assert manager.unregister("a") is model

    with pytest.raises(ModelNotRegisteredError):
        manager.unregister("a")


def test_infer_loads_unloaded_model_and_updates_usage():
    manager = StaticVRAMManager(available_bytes=10)
    model = FakeModel("a", 5, result="ok")
    manager.register(model)

    assert manager.infer("a", "prompt") == "ok"

    assert model.load_calls == 1
    assert model.infer_calls == 1
    assert model.state is ModelState.LOADED
    assert model.last_used_at > 0


def test_infer_reuses_loaded_model():
    manager = StaticVRAMManager(available_bytes=0)
    model = FakeModel("a", 5)
    model.mark_loaded()
    manager.register(model)

    manager.infer("a")

    assert model.load_calls == 0
    assert model.infer_calls == 1


def test_lru_unused_models_are_offloaded_until_enough_memory():
    manager = StaticVRAMManager(available_bytes=2)
    old = FakeModel("old", 4)
    newer = FakeModel("newer", 4)
    target = FakeModel("target", 6)

    old.mark_loaded()
    old.last_used_at = 1.0
    newer.mark_loaded()
    newer.last_used_at = 2.0

    manager.register(old)
    manager.register(newer)
    manager.register(target)

    manager.infer("target")

    assert old.offload_calls == 1
    assert old.state is ModelState.UNLOADED
    assert newer.offload_calls == 0
    assert newer.state is ModelState.LOADED
    assert target.load_calls == 1


def test_active_models_are_not_offloaded():
    manager = StaticVRAMManager(available_bytes=0)
    active = FakeModel("active", 10)
    target = FakeModel("target", 5)
    active.mark_loaded()
    active.state = ModelState.IN_USE

    manager.register(active)
    manager.register(target)

    with pytest.raises(InsufficientVRAMError):
        manager.infer("target")

    assert active.offload_calls == 0
    assert active.state is ModelState.IN_USE


def test_query_nvidia_smi_parses_memory(monkeypatch):
    def fake_run(command, check, capture_output, text):
        assert command[-1] == "--id=0"
        return subprocess.CompletedProcess(command, 0, stdout="8192, 1024\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    info = query_nvidia_smi(0)

    assert info.total_bytes == 8192 * MIB
    assert info.used_bytes == 1024 * MIB
    assert info.available_bytes == 7168 * MIB


def test_query_nvidia_smi_wraps_command_errors(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="no device")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(VRAMQueryError, match="no device"):
        query_nvidia_smi(0)


def test_query_nvidia_smi_rejects_bad_output(monkeypatch):
    def fake_run(command, check, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="not-memory\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(VRAMQueryError, match="could not parse"):
        query_nvidia_smi(0)

