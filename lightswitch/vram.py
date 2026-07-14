"""NVIDIA VRAM measurement primitives."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .errors import VRAMQueryError

MIB = 1024 * 1024


@dataclass(frozen=True)
class VRAMInfo:
    """GPU memory usage in bytes."""

    total_bytes: int
    used_bytes: int

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)


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
