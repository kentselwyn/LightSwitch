# LightSwitch

LightSwitch provides a tiny, framework-agnostic layer for loading, unloading, and
running AI models with simple VRAM-aware scheduling.

The package does not import TensorFlow, PyTorch, or any other machine learning
framework. Model wrappers own their framework-specific behavior by subclassing
`AIModel`.

```python
from lightswitch import AIModel, VRAMManager


class MyModel(AIModel):
    def load(self):
        ...

    def offload(self):
        ...

    def infer(self, prompt: str):
        ...


manager = VRAMManager(gpu_index=0)
model = MyModel(name="chat", estimated_vram_bytes=4 * 1024**3)

manager.register(model)
result = manager.infer("chat", "hello")
```

