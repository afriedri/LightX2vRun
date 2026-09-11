"""Run the actual Ref2V sweep and dedicated entry with a fake container launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE_ENTRY = ROOT / "scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh"
TURBO_ENTRY = ROOT / "scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh"
SUMMARY = ROOT / "scripts/lib/summarize_timings.py"
DEFAULT_TURBO_IMAGE = "compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4"
DEFAULT_MATERIAL_NAME = "rooftop_2ref_15s"
DEFAULT_REFERENCE_NAMES = ["Image Gen-1.png", "RYAN-2.png"]
NINE_REFERENCE_MATERIAL_NAME = "storeroom_courtyard_9ref_15s"
NINE_REFERENCE_NAMES = [
    "wang_hao_front.png",
    "wang_hao_full.png",
    "chen_fan_front.png",
    "chen_fan_full.png",
    "discipline_disciple.png",
    "su_xue_front.png",
    "su_xue_full.png",
    "character_layout.png",
    "zhao_xiaoman.png",
]


def run_sweep(
    tmp_path,
    counts="2,4,8,16",
    fail=False,
    entry=BASE_ENTRY,
    image="test:h3-scale",
    prompt_profile=None,
    prompt_profile_option="--prompt-profile",
):
    fake = tmp_path / "model-run"
    fake.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} - <<'PYTHON'\n" +
        "import json,os,pathlib\n" +
        "out=pathlib.Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True,exist_ok=True)\n" +
        "(out/'captured.json').write_text(json.dumps(dict(os.environ)))\n" +
        ("raise SystemExit(37)\n" if fail else "print('fake inference complete')\n") + "PYTHON\n")
    fake.chmod(0o755)
    env = {**os.environ, "MODEL_RUN": str(fake), "DRY_RUN": "1", "MEMORY_MONITOR": "0",
           "TIMING_RESULT_DIR": str(tmp_path / "sweep"), "H3_INFER_STEPS": "3", "PROMPT": "",
           "H3_PROMPT_FILE": "", "H3_REFERENCE_IMAGE": "", "H3_REFERENCE_IMAGES": "",
           "H3_PROMPT_PROFILE": "", "H3_REF2AV_MATERIAL_CONFIG": "",
           "H3_STEP_UPDATE": "comfy_res_multistep",
           "H3_TASK": "fl2av", "H3_SMOKE": "1", "H3_OUTPUT_DURATION": "5"}
    args = ["bash", str(entry), "--counts", counts]
    if image is not None:
        args.extend(("--image", image))
    if prompt_profile is not None:
        args.extend((prompt_profile_option, prompt_profile))
    result = subprocess.run(args, capture_output=True, text=True, env=env)
    return result, tmp_path / "sweep"


def test_scale_defaults_to_two_references_and_isolates_each_card_count(tmp_path):
    result, output = run_sweep(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["gpu_counts"] == [2, 4, 8, 16]
    assert manifest["native_frames"] == 362 and manifest["output_frames"] == 360
    assert manifest["infer_steps"] == 3 and manifest["lora"] is False
    assert manifest["prompt_profile"] == DEFAULT_MATERIAL_NAME
    assert manifest["material_name"] == DEFAULT_MATERIAL_NAME
    assert [Path(path).name for path in manifest["reference_images"]] == DEFAULT_REFERENCE_NAMES
    assert all(Path(path).is_file() for path in manifest["reference_images"])
    assert Path(manifest["prompt_file"]).is_file()
    refs = prompt = None
    for count in (2, 4, 8, 16):
        env = json.loads((output / f"{count}gpu/captured.json").read_text())
        assert env["H3_TASK"] == "ref2av" and env["H3_OUTPUT_DURATION"] == "15"
        assert env["H3_INFER_STEPS"] == "3" and env["H3_SMOKE"] == "0"
        assert env["IMAGE"] == "test:h3-scale"
        assert len(env["GPU_DEVICES"].split(",")) == count
        assert env["TARGET_HEIGHT"] == "768" and env["TARGET_WIDTH"] == "1344"
        assert env["TARGET_VIDEO_LENGTH"] == "362"
        assert env["LIGHTX2V_H3_RUN_COLLECTOR"] == "1"
        assert env["LIGHTX2V_VAE_CONV3D_BACKEND"] == "Aiter_CK_Conv3d_v1"
        assert env["LIGHTX2V_CK_CONV3D_PROFILE_DIR"] == "/workspace/LightX2V/save_results/ck_conv3d_routes"
        assert env["H3_EVIDENCE_DIR"] == "/workspace/LightX2V/save_results"
        assert env["SAVE_RESULT_PATH"].endswith("/minimax_h3_ref2av.mp4")
        assert [Path(path).name for path in env["H3_REFERENCE_IMAGES"].splitlines()] == DEFAULT_REFERENCE_NAMES
        if refs is not None:
            assert refs == env["H3_REFERENCE_IMAGES"] and prompt == env["PROMPT"]
        refs, prompt = env["H3_REFERENCE_IMAGES"], env["PROMPT"]
        assert "[Shot 6]" in prompt and "<Picture 1>" in prompt and "<Picture 2>" in prompt
        assert "<Picture 9>" not in prompt and "<Audio" not in prompt
        assert (output / f"{count}gpu.log").is_file()


def test_scale_propagates_inference_failure_and_stops_sweep(tmp_path):
    result, output = run_sweep(tmp_path, fail=True)
    assert result.returncode == 37, result.stdout + result.stderr
    assert (output / "2gpu/captured.json").is_file()
    assert not (output / "4gpu").exists()


def test_turbo_8step_scale_covers_cp2xu8_sp16_and_preserves_profile(tmp_path):
    result, output = run_sweep(tmp_path, entry=TURBO_ENTRY)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["gpu_counts"] == [2, 4, 8, 16]
    assert manifest["parallel_topologies"] == {
        "2": "ulysses",
        "4": "ulysses",
        "8": "ulysses",
        "16": "ulysses_ring_2d",
    }
    assert manifest["profile"] == "ref2av_turbo_8step"
    assert manifest["weights"] == "transformer_ref"
    assert manifest["lora"] is True
    assert manifest["lora_file"] == "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
    assert manifest["sampling"] == "training_euler"
    assert manifest["infer_steps"] == 9 and manifest["model_evaluations"] == 8
    for count in (2, 4, 8, 16):
        env = json.loads((output / f"{count}gpu/captured.json").read_text())
        assert env["H3_PROFILE"] == "ref2av_turbo_8step"
        assert env["H3_STEP_UPDATE"] == "training_euler"
        assert env["H3_INFER_STEPS"] == "9"
        assert env["H3_TURBO_DIR"] == "/disk/ssd/difumodels-Dis/Minimax-h3-Turbo"
        assert "H3_COMFY_DIR" not in env
        assert len(env["GPU_DEVICES"].split(",")) == count
        assert env["H3_PROMPT_PROFILE"] == DEFAULT_MATERIAL_NAME
        assert env["H3_REF2AV_MATERIAL_NAME"] == DEFAULT_MATERIAL_NAME
        assert [Path(path).name for path in env["H3_REFERENCE_IMAGES"].splitlines()] == DEFAULT_REFERENCE_NAMES
    assert "MiniMax-H3 topology: SP2/SP4/SP8=Ulysses, SP16=CP2xU8" in result.stdout


@pytest.mark.parametrize("entry", [BASE_ENTRY, TURBO_ENTRY])
@pytest.mark.parametrize("option", ["--prompt-profile", "--prompts-profile"])
def test_scale_selects_nine_reference_prompt_profile(tmp_path, entry, option):
    result, output = run_sweep(
        tmp_path,
        counts="2",
        entry=entry,
        prompt_profile=NINE_REFERENCE_MATERIAL_NAME,
        prompt_profile_option=option,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["prompt_profile"] == NINE_REFERENCE_MATERIAL_NAME
    assert manifest["material_name"] == NINE_REFERENCE_MATERIAL_NAME
    assert [Path(path).name for path in manifest["reference_images"]] == NINE_REFERENCE_NAMES
    env = json.loads((output / "2gpu/captured.json").read_text())
    assert env["H3_PROMPT_PROFILE"] == NINE_REFERENCE_MATERIAL_NAME
    assert [Path(path).name for path in env["H3_REFERENCE_IMAGES"].splitlines()] == NINE_REFERENCE_NAMES
    assert "[镜头 5]" in env["PROMPT"] and "<Picture 9>" in env["PROMPT"]
    assert "@音频" not in env["PROMPT"]


def test_scale_rejects_unknown_prompt_profile(tmp_path):
    result, output = run_sweep(tmp_path, counts="2", prompt_profile="missing")
    assert result.returncode == 2
    assert "unsupported --prompt-profile: missing" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("counts", ["2", "4", "8", "16"])
def test_scale_can_run_one_count(tmp_path, counts):
    result, output = run_sweep(tmp_path, counts=counts)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / f"{counts}gpu/captured.json").exists()


def test_turbo_scale_can_run_only_sp16(tmp_path):
    result, output = run_sweep(tmp_path, counts="16", entry=TURBO_ENTRY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "16gpu/captured.json").exists()
    assert not (output / "2gpu").exists()


def test_turbo_scale_defaults_to_v4_production_image(tmp_path):
    result, output = run_sweep(
        tmp_path,
        counts="16",
        entry=TURBO_ENTRY,
        image=None,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    env = json.loads((output / "16gpu/captured.json").read_text())
    assert env["IMAGE"] == DEFAULT_TURBO_IMAGE


def test_turbo_scale_summary_labels_eight_model_evaluations(tmp_path):
    (tmp_path / "16gpu.log").write_text(
        "\n".join(
            (
                "[Profile] Rank 0 - Run VAE Encoder cost 2.000 seconds",
                "[Profile] Rank 0 - Run Text Encoder cost 3.000 seconds",
                "DiT summary: status=complete, steps=8/8, total=40.000s, average=5.000s/it",
                "[Profile] Rank 0 - Run VAE Decoder cost 4.000 seconds",
                "[Profile] Rank 0 - RUN pipeline cost 50.000 seconds",
                "[Profile] Rank 0 - Total Cost cost 55.000 seconds",
            )
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "python3",
            str(SUMMARY),
            "--model",
            "minimax_h3",
            "--log-dir",
            str(tmp_path),
            "--counts",
            "16",
            "--dit-steps",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "| DiT 8 steps | — | — | — | 40.000s |" in (tmp_path / "timings.md").read_text()
    assert "DiT 8 steps,,,,40.000000" in (tmp_path / "timings.csv").read_text()
