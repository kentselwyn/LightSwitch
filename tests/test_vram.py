import subprocess

import pytest

from lightswitch import VRAMQueryError
from lightswitch.vram import MIB, query_nvidia_smi


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
