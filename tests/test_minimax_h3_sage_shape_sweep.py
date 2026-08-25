from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/collector/run_minimax_h3_sage_shape_sweep.sh"


def run_dry(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    result_dir = tmp_path / "sage-sweep"
    completed = subprocess.run(
        [
            "bash",
            str(ENTRY),
            "--image",
            "example.invalid/minimax-h3:sp8",
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
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_REPO_DIR": str(REPO_ROOT),
            "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
            "GPU_DEVICES": "",
            "H3_MODEL_DIR": "",
            "H3_TURBO_DIR": "",
            "H3_FIRST_FRAME": "",
            "H3_LAST_FRAME": "",
            "H3_RESULT_DIR": "",
            "H3_PROMPT_FILE": "",
            "H3_PROMPT_PRESET": "",
            "PROMPT": "",
        },
    )
    return completed, result_dir


def test_sweep_dry_run_launches_sp4_sp8_then_ring16_without_full_collector(tmp_path):
    completed, result_dir = run_dry(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SP4 Ulysses: GPUs 0,1,2,3" in completed.stdout
    assert "SP8 Ulysses: GPUs 0,1,2,3,4,5,6,7" in completed.stdout
    assert "Ring16: GPUs 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in completed.stdout
    assert completed.stdout.index("SP4 Ulysses: GPUs") < completed.stdout.index(
        "SP8 Ulysses: GPUs"
    ) < completed.stdout.index("Ring16: GPUs")
    assert completed.stdout.count("--env LIGHTX2V_H3_RUN_COLLECTOR=0") == 3
    assert "Dry run complete; Sage shape extraction skipped." in completed.stdout

    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sp_sizes"] == [4, 8, 16]
    assert manifest["topologies"] == {"4": "ulysses", "8": "ulysses", "16": "ring"}
    assert manifest["collector"]["full_runcollector_enabled"] is False
    assert manifest["collector"]["schema_version"] == 2
    sp4 = json.loads((result_dir / "sp4/run_manifest.json").read_text(encoding="utf-8"))
    sp8 = json.loads((result_dir / "sp8/run_manifest.json").read_text(encoding="utf-8"))
    ring16 = json.loads((result_dir / "sp16/run_manifest.json").read_text(encoding="utf-8"))
    assert sp4["gpu_count"] == 4
    assert sp8["gpu_count"] == 8
    assert ring16["gpu_count"] == 16
    assert ring16["parallel"]["seq_p_attn_type"] == "ring"
    assert sp4["runcollector"]["enabled"] is False
    assert sp8["runcollector"]["enabled"] is False
    assert ring16["runcollector"]["enabled"] is False


def test_sweep_requires_enough_gpu_ids(tmp_path):
    completed, _ = run_dry(tmp_path, "--gpus", "0,1,2,3")
    assert completed.returncode == 2
    assert "SP16 requires 16 GPU ids, got 4" in completed.stderr


def test_sweep_accepts_single_sp4_selection(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--counts", "4", "--gpus", "4,5,6,7")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SP4 Ulysses: GPUs 4,5,6,7" in completed.stdout
    assert "SP8 Ulysses: GPUs" not in completed.stdout
    assert "Ring16: GPUs" not in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sp_sizes"] == [4]


def test_sweep_rejects_ring16_without_ulysses_reference(tmp_path):
    completed, _ = run_dry(
        tmp_path,
        "--counts",
        "16",
        "--gpus",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    )
    assert completed.returncode == 2
    assert "Ring16 requires SP4 or SP8" in completed.stderr
