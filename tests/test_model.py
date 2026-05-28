import pytest

from lightswitch import AIModel, ModelState


class DummyModel(AIModel):
    def load(self):
        self.loaded = True

    def offload(self):
        self.loaded = False

    def infer(self, value):
        return value


def test_model_validates_name():
    with pytest.raises(ValueError, match="name"):
        DummyModel(name="", estimated_vram_bytes=1)


def test_model_validates_estimated_vram():
    with pytest.raises(ValueError, match="estimated_vram_bytes"):
        DummyModel(name="bad", estimated_vram_bytes=-1)


def test_model_lifecycle_helpers():
    model = DummyModel(name="m", estimated_vram_bytes=1)

    assert model.state is ModelState.UNLOADED
    assert not model.is_loaded

    model.mark_loaded()
    assert model.state is ModelState.LOADED
    assert model.is_loaded
    first_used = model.last_used_at

    with model.use():
        assert model.state is ModelState.IN_USE
        assert model.is_in_use

    assert model.state is ModelState.LOADED
    assert model.last_used_at >= first_used

    model.mark_unloaded()
    assert model.state is ModelState.UNLOADED


def test_model_use_requires_loaded_state():
    model = DummyModel(name="m", estimated_vram_bytes=1)

    with pytest.raises(RuntimeError, match="must be loaded"):
        with model.use():
            pass

