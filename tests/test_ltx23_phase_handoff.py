from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/check_ltx23_phase_handoff.py"
SPEC = importlib.util.spec_from_file_location("check_ltx23_phase_handoff", SCRIPT)
HANDOFF = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HANDOFF)


def write_prompt_cache(path: Path, producer_pid: int = 99_999_999) -> None:
    header = {
        "__metadata__": {
            "schema_name": "lightx2v.ltx23.prompt_embeddings",
            "schema_version": "1",
            "process_isolated": "true",
            "producer_pid": str(producer_pid),
        }
    }
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def write_gpu(root: Path, card: int, *, used: int, total: int = 32 * 1024**3) -> None:
    device = root / f"card{card}" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text(f"{used}\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text(f"{total}\n", encoding="utf-8")


def test_handoff_passes_only_after_pid_exit_and_both_cards_are_low(tmp_path):
    sysfs = tmp_path / "drm"
    write_gpu(sysfs, 0, used=64 * 1024**2)
    write_gpu(sysfs, 1, used=96 * 1024**2)
    cache = tmp_path / "prompt.safetensors"
    write_prompt_cache(cache)

    report = HANDOFF.wait_for_handoff(
        gpu_indices=(0, 1),
        prompt_cache=cache,
        limit_bytes=1024**3,
        timeout_seconds=0,
        interval_seconds=0.01,
        sysfs_root=sysfs,
    )

    assert report["status"] == "ok"
    assert report["producer_pid_alive"] is False
    assert [gpu["rocm_index"] for gpu in report["gpus"]] == [0, 1]


def test_handoff_fails_when_physical_vram_is_still_charged(tmp_path):
    sysfs = tmp_path / "drm"
    write_gpu(sysfs, 0, used=64 * 1024**2)
    write_gpu(sysfs, 1, used=30 * 1024**3)
    cache = tmp_path / "prompt.safetensors"
    write_prompt_cache(cache)

    report = HANDOFF.wait_for_handoff(
        gpu_indices=(0, 1),
        prompt_cache=cache,
        limit_bytes=1024**3,
        timeout_seconds=0,
        interval_seconds=0.01,
        sysfs_root=sysfs,
    )

    assert report["status"] == "failed"
    assert report["gpus"][1]["used_bytes"] == 30 * 1024**3
