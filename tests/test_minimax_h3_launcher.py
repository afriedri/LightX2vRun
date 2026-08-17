from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/run_minimax_h3_fl2av_4step_768p_day0.sh"


def run_dry(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    result_dir = tmp_path / "image-result"
    arguments = [
        "bash",
        str(ENTRY),
        "--image",
        "example.invalid/minimax-h3:day0",
        "--model-dir",
        str(tmp_path / "MiniMax-H3"),
        "--turbo-dir",
        str(tmp_path / "Minimax-h3-Turbo"),
        "--first-frame",
        str(tmp_path / "first.png"),
        "--last-frame",
        str(tmp_path / "last.png"),
        "--result-dir",
        str(result_dir),
        "--preflight-only",
        *extra,
    ]
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_REPO_DIR": str(REPO_ROOT),
            "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        },
    )
    return completed, result_dir


def test_h3_dry_run_uses_image_source_without_overlay(tmp_path):
    completed, result_dir = run_dry(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "H3 source mode: image" in completed.stdout
    assert "--env H3_SOURCE_MODE=image" in completed.stdout
    assert "target=/workspace/LightX2V\\,readonly" not in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "image"
    assert manifest["task"] == "fl2av"
    assert manifest["gpus"] == [0]
    assert manifest["parallel"]["tensor_parallel"] is False
    assert manifest["parallel"]["seq_parallel"] is False


def test_h3_dry_run_accepts_two_gpu_ulysses_without_tp(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--gpus", "0,1")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert r"HIP_VISIBLE_DEVICES=0\,1" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == [0, 1]
    assert manifest["gpu_count"] == 2
    assert manifest["parallel"] == {
        "seq_p_attn_type": "ulysses",
        "seq_p_size": 2,
        "seq_parallel": True,
        "tensor_parallel": False,
        "vae_decode_parallel": True,
    }


def test_h3_launcher_rejects_duplicate_or_more_than_two_gpus(tmp_path):
    duplicate, _ = run_dry(tmp_path, "--gpus", "0,0")
    assert duplicate.returncode == 2
    assert "must be unique" in duplicate.stderr

    too_many, _ = run_dry(tmp_path, "--gpus", "0,1,2")
    assert too_many.returncode == 2
    assert "one GPU id or two comma-separated ids" in too_many.stderr


def test_h3_launcher_rejects_host_source_overlay(tmp_path):
    completed, _ = run_dry(tmp_path, "--lightx2v-dir", str(tmp_path / "LightX2V"))
    assert completed.returncode == 2
    assert "unknown argument: --lightx2v-dir" in completed.stderr
