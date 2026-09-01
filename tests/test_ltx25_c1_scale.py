from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/run_10_ltx25_c1_704x1280_121f_24fps_scale.sh"
INNER = REPO_ROOT / "scripts/lib/run_ltx25_c1_in_container.sh"
BUILDER = REPO_ROOT / "scripts/lib/build_ltx25_c1_runtime_config.py"
SUMMARY = REPO_ROOT / "scripts/lib/summarize_timings.py"


def _build_config(tmp_path: Path, size: int) -> dict:
    output = tmp_path / f"sp{size}.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model-root",
            str(tmp_path / "models"),
            "--seq-parallel-size",
            str(size),
            "--output",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_ltx25_c1_scale_configs_use_pure_ulysses(tmp_path):
    for size in (2, 4, 8, 16):
        config = _build_config(tmp_path, size)
        parallel = config["parallel"]
        assert parallel["seq_p_size"] == size
        assert parallel["seq_p_attn_type"] == "ulysses"
        assert config["ltx25_distributed_video_vae"] is (size >= 4)
        assert config["ltx25_distributed_video_vae_overlap"] == 4
        assert "ring_p_size" not in parallel
        assert "ulysses_p_size" not in parallel


def test_ltx25_c1_scale_entry_builds_4_8_16_commands_and_manifest(tmp_path):
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "outputs"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SKIP_MODEL_CHECKS": "1",
        "MEMORY_MONITOR": "0",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "TIMING_RESULT_DIR": str(result_dir),
        "TIMING_RUN_TAG": "ltx25-scale-test",
        "OUTPUT_DIR": str(output_dir),
    }
    completed = subprocess.run(
        ["bash", str(ENTRY), "--image", "example.invalid/ltx25:scale"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.replace("\\,", ",")
    assert "GPU counts: 4,8,16" in stdout
    assert "LTX-2.5 topology: SP2/SP4/SP8/SP16=pure Ulysses" in stdout
    for size in (4, 8, 16):
        assert f"LTX25_WORLD_SIZE={size}" in stdout
        assert f"SAVE_RESULT_PATH=/workspace/LightX2V/save_results/ltx25-scale-test/{size}gpu.mp4" in stdout

    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "ltx25_c1"
    assert manifest["gpu_counts"] == [4, 8, 16]
    assert manifest["parallel_topologies"] == {
        "4": "ulysses",
        "8": "ulysses",
        "16": "ulysses",
    }


def test_ltx25_c1_scale_requires_explicit_image(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MEMORY_MONITOR": "0",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "TIMING_RESULT_DIR": str(tmp_path / "results"),
    }
    completed = subprocess.run(
        ["bash", str(ENTRY), "--counts", "4"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "requires --image" in completed.stderr


def test_ltx25_c1_summary_uses_eleven_step_label(tmp_path):
    lines = [
        "[Profile] Rank 0 - Level1_Log Run Text Encoder cost 0.010 seconds",
        "[Profile] Rank 0 - Level2_Log Run Encoders cost 0.011 seconds",
    ]
    lines.extend(
        f"[Profile] Rank 0 - Level1_Log Run Dit every step cost {second:.3f} seconds"
        for second in range(1, 12)
    )
    lines.extend(
        [
            "[Profile] Rank 0 - Level1_Log Run VAE Decoder cost 3.000 seconds",
            "[Profile] Rank 0 - Level1_Log RUN pipeline cost 70.000 seconds",
            "[Profile] Rank 0 - Level1_Log Total Cost cost 71.000 seconds",
        ]
    )
    (tmp_path / "4gpu.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--model",
            "ltx25_c1",
            "--log-dir",
            str(tmp_path),
            "--counts",
            "4",
            "--width",
            "1280",
            "--height",
            "704",
            "--frames",
            "121",
            "--fps",
            "24",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    table = (tmp_path / "timings.md").read_text(encoding="utf-8")
    assert "| DiT 11 steps |" in table
    assert "66.000s" in table
    assert "2/4/8/16 卡均使用纯 Ulysses" in table


def test_ltx25_c1_inner_fans_out_one_scale_union_cache_without_tuning():
    text = INNER.read_text(encoding="utf-8")
    assert "if (( world_size > 2 )); then" in text
    assert 'rank % 2 == 0' not in text
    assert 'cp -f -- "${tunableop_cache_rank0}"' in text
    assert 'cmp -s -- "${tunableop_cache_rank0}" "${tunableop_cache_rank1}"' in text
    assert "one verified SP2/4/8/16 union cache" in text
    assert 'tunableop-sp${world_size}' in text
    assert 'export PYTORCH_TUNABLEOP_FILENAME="${tunableop_cache_pattern}"' in text
    assert 'PYTORCH_TUNABLEOP_RECORD_UNTUNED=0' in text
    assert 'artifact_dir="${video_path%.mp4}_artifacts"' in text
    assert "RCCL P2P gather; rank0 BF16 blend" in text


def test_ltx25_c1_scale_scripts_are_valid_bash():
    for path in (ENTRY, INNER, REPO_ROOT / "scripts/lib/run_timing_sweep.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
