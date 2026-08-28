from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_ltx25_c1_704x1280_5s.sh"
IN_CONTAINER = REPO_ROOT / "scripts/lib/run_ltx25_c1_in_container.sh"
BUILDER = REPO_ROOT / "scripts/lib/build_ltx25_c1_runtime_config.py"


def test_ltx25_c1_config_promotes_only_evidenced_semantic_roles(tmp_path):
    model_root = tmp_path / "LTX-2.5"
    output = tmp_path / "c1.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--model-root",
            str(model_root),
            "--output",
            str(output),
        ],
        check=True,
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["ltx25_c1_gfx1201"] is True
    assert config["ltx25_video_to_audio_attn_type"] == "aiter_triton_dense_bf16_attn"
    assert config["ltx25_qk_rms_norm_type"] == "sgl-kernel"
    assert config["rms_norm_type"] == "torch_native"
    assert config["ltx25_video_self_attn_type"] == "torch_sdpa"
    assert config["ltx25_video_text_attn_type"] == "torch_sdpa"
    assert config["ltx25_audio_self_attn_type"] == "torch_sdpa"
    assert config["ltx25_audio_text_attn_type"] == "torch_sdpa"
    assert config["ltx25_audio_to_video_attn_type"] == "torch_sdpa"
    assert config["rope_type"] == "ltx25_triton_strict_rope"


def test_ltx25_c1_host_entry_builds_read_only_two_rank_docker_command(tmp_path):
    model_root = tmp_path / "models" / "LTX-2.5"
    model_root.mkdir(parents=True)
    source = tmp_path / "LightX2V"
    (source / "lightx2v").mkdir(parents=True)
    cache = tmp_path / "tunableop_results_full0.csv"
    cache.write_text("Validator,PT_VERSION,0\n", encoding="utf-8")
    conv3d_csv = tmp_path / "future_conv3d_v1_tuned.csv"
    result_dir = tmp_path / "results"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
    }

    completed = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--image",
            "example.invalid/ltx25:c1",
            "--tunableop-cache",
            str(cache),
            "--lightx2v-source",
            str(source),
            "--conv3d-csv",
            str(conv3d_csv),
            "--model-root",
            str(model_root),
            "--result-dir",
            str(result_dir),
            "--gpus",
            "2,3",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.replace("\\,", ",")
    assert "Mode:           ltx25_c1" in stdout
    assert "GPU devices:    2,3" in stdout
    assert "strict RoPE" in stdout
    assert "PYTORCH_TUNABLEOP_ENABLED=1" in stdout
    assert "PYTORCH_TUNABLEOP_TUNING=0" in stdout
    assert stdout.count(f"source={cache}") == 2
    assert "target=/workspace/ltx25-c1/tunableop_results_full0.csv,readonly" in stdout
    assert "target=/workspace/ltx25-c1/tunableop_results_full1.csv,readonly" in stdout
    assert "PYTORCH_TUNABLEOP_FILENAME=/workspace/ltx25-c1/tunableop_results_full%d.csv" in stdout
    assert "LIGHTX2V_LTX25_C1_IN_CONTAINER=1" in stdout
    assert "LIGHTX2V_CK_CONV3D_PROFILE_DIR=/workspace/LightX2V/save_results/ck_conv3d_routes" in stdout
    assert f"source={conv3d_csv},target=/workspace/ck-conv3d-config/conv3d_v1_tuned.csv,readonly" in stdout
    assert "run_ltx25_c1_in_container.sh" in stdout
    assert f"source={source},target=/workspace/LightX2V,readonly" in stdout


def test_ltx25_c1_host_entry_defaults_to_image_replay_cache(tmp_path):
    model_root = tmp_path / "models" / "LTX-2.5"
    model_root.mkdir(parents=True)
    result_dir = tmp_path / "results"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
    }
    env.pop("LTX25_TUNABLEOP_CACHE", None)

    completed = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--image",
            "example.invalid/ltx25:c1-integrated",
            "--model-root",
            str(model_root),
            "--result-dir",
            str(result_dir),
            "--gpus",
            "0,1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.replace("\\,", ",")
    cache_root = "/workspace/LightX2V/configs/ltx25/gfx1201"
    assert f"PYTORCH_TUNABLEOP_FILENAME={cache_root}/tunableop_results_full%d.csv" in stdout
    assert f"LTX25_TUNABLEOP_CACHE_RANK0={cache_root}/tunableop_results_full0.csv" in stdout
    assert f"LTX25_TUNABLEOP_CACHE_RANK1={cache_root}/tunableop_results_full1.csv" in stdout
    assert "target=/workspace/ltx25-c1/tunableop_results_full0.csv" not in stdout
    assert "image cache, read-only" in stdout


def test_ltx25_c1_inner_helper_rejects_direct_host_execution():
    completed = subprocess.run(
        ["bash", str(IN_CONTAINER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "inside the ROCm image" in completed.stderr


def test_ltx25_c1_prompt_source_supplies_required_runcollector_output():
    text = IN_CONTAINER.read_text(encoding="utf-8")

    assert 'prompt_collector="${result_dir}/ltx25_c1_prompt_collector.json"' in text
    assert '--collector-output "${prompt_collector}"' in text
    assert '"${prompt_collector}" "${video_path}"' in text


def test_ltx25_c1_image_source_requires_and_verifies_strict_rope():
    text = IN_CONTAINER.read_text(encoding="utf-8")

    assert "lightx2v/models/networks/ltx2/infer/strict_rope.py" in text
    assert 'expected = "ltx25_triton_strict_rope"' in text
    assert "LTX-2.5 C1 RoPE route verified" in text
    assert "runtime config selected" in text


def test_ltx25_c1_scripts_are_valid_bash():
    for path in (LAUNCHER, IN_CONTAINER, REPO_ROOT / "lightx2vModelrun"):
        subprocess.run(["bash", "-n", str(path)], check=True)
