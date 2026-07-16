import pytest

from lightswitch import AIModel, ModelState


class DummyModel(AIModel):
    def load_to_cpu(self):
        pass

    def move_to_gpu(self):
        pass

    def move_to_cpu(self):
        pass

    def evict_from_gpu(self):
        pass

    def evict_from_cpu(self):
        pass

    def infer(self, value=None):
        return value


def make_model(**overrides):
    values = {
        "name": "m",
        "estimated_ram_bytes": 2,
        "estimated_vram_bytes": 1,
    }
    values.update(overrides)
    return DummyModel(**values)


def test_model_validates_configuration():
    with pytest.raises(ValueError, match="name"):
        make_model(name="")
    with pytest.raises(ValueError, match="estimated_ram_bytes"):
        make_model(estimated_ram_bytes=-1)
    with pytest.raises(ValueError, match="estimated_vram_bytes"):
        make_model(estimated_vram_bytes=-1)


def test_model_residency_properties():
    model = make_model()

    assert model.state is ModelState.EVICTED
    assert not model.is_resident

    model.mark_cpu_resident()
    assert model.is_resident
    assert model.is_cpu_resident
    assert not model.is_gpu_resident

    model.mark_gpu_resident()
    assert model.is_gpu_resident
    assert model.last_used_at > 0

    model.mark_evicted()
    assert model.state is ModelState.EVICTED


def test_model_use_tracks_gpu_inference_and_restores_state_on_error():
    model = make_model()
    model.mark_gpu_resident()

    with pytest.raises(RuntimeError, match="inference failed"):
        with model.use():
            assert model.state is ModelState.IN_USE
            assert model.is_in_use
            raise RuntimeError("inference failed")

    assert model.state is ModelState.GPU_RESIDENT
    assert model.last_used_at > 0


def test_model_use_requires_gpu_residency():
    model = make_model()
    model.mark_cpu_resident()

    with pytest.raises(RuntimeError, match="GPU-resident"):
        with model.use():
            pass
