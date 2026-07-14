# LightSwitch

LightSwitch is a framework-neutral memory manager for recoverable AI models. It
monitors system RAM and NVIDIA VRAM, moves idle models from GPU to CPU, and fully
evicts least-recently-used models when system RAM is under pressure.

An evicted model remains registered as a lightweight recovery recipe. Calling
`MemoryManager.infer()` reconstructs it in CPU memory, moves it to the GPU, and
runs inference without application-side reinitialization.

## Model adapters

Subclass `AIModel` and retain the configuration required by `load_to_cpu()` after
`evict_from_cpu()` deletes the heavyweight runtime object. State changes are
owned by the manager; adapter hooks should not update `model.state` themselves.

```python
class MyModel(AIModel):
    def __init__(self, name, model_id, estimated_ram_bytes, estimated_vram_bytes):
        super().__init__(
            name=name,
            estimated_ram_bytes=estimated_ram_bytes,
            estimated_vram_bytes=estimated_vram_bytes,
        )
        self.model_id = model_id
        self.runtime = None

    def load_to_cpu(self):
        self.runtime = framework.load(self.model_id, device="cpu")

    def move_to_gpu(self):
        self.runtime.to("cuda")

    def move_to_cpu(self):
        self.runtime.to("cpu")
        framework.release_gpu_cache()

    def evict_from_cpu(self):
        self.runtime = None
        framework.release_cpu_cache()

    def infer(self, prompt):
        return self.runtime(prompt)
```

`estimated_ram_bytes` should conservatively cover the model's system RAM
footprint in either resident state. `estimated_vram_bytes` should cover its GPU
allocation.

## Memory management

Construction does not start a thread. Use the manager as a context manager, or
call `start()` and `stop()` explicitly.

```python
from lightswitch import MemoryManager

GiB = 1024**3

manager = MemoryManager(
    ram_reserve_bytes=8 * GiB,
    vram_reserve_bytes=2 * GiB,
    gpu_index=0,
    poll_interval_seconds=1.0,
)
manager.register(
    MyModel(
        name="chat",
        model_id="organization/model",
        estimated_ram_bytes=16 * GiB,
        estimated_vram_bytes=12 * GiB,
    )
)

with manager:
    result = manager.infer("chat", "hello")
```

Every watcher cycle samples both resources. RAM pressure fully evicts idle
models in LRU order. VRAM-only pressure leaves them CPU-resident for faster
recovery. Models currently running inference are never transitioned.

Use `manager.status` for the last pressure snapshot and `on_pressure` for active
notification:

```python
def report(event):
    if event.error:
        logger.error(event.error)
    elif event.unresolved_ram_pressure or event.unresolved_vram_pressure:
        logger.warning("memory pressure remains after eviction")


manager = MemoryManager(
    ram_reserve_bytes=8 * GiB,
    vram_reserve_bytes=2 * GiB,
    on_pressure=report,
)
```

The RAM backend uses `psutil`. GPU monitoring requires `nvidia-smi` and currently
supports one NVIDIA GPU per manager.

## Migrating from 0.1

The recoverable lifecycle intentionally replaces the former `load()` and
`offload()` hooks. Add `estimated_ram_bytes`, implement the four residency hooks,
and replace `VRAMManager` with `MemoryManager`. The initial model state is now
`ModelState.EVICTED`.
