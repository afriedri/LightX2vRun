"""Run the actual Ref2V sweep and dedicated entry with a fake container launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh"


def run_sweep(tmp_path, counts="4,8,16", fail=False):
    fake = tmp_path / "model-run"
    fake.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} - <<'PYTHON'\n" +
        "import json,os,pathlib\n" +
        "out=pathlib.Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True,exist_ok=True)\n" +
        "(out/'captured.json').write_text(json.dumps(dict(os.environ)))\n" +
        ("raise SystemExit(37)\n" if fail else "print('fake inference complete')\n") + "PYTHON\n")
    fake.chmod(0o755)
    env = {**os.environ, "MODEL_RUN": str(fake), "DRY_RUN": "1", "MEMORY_MONITOR": "0",
           "TIMING_RESULT_DIR": str(tmp_path / "sweep"), "H3_INFER_STEPS": "3", "PROMPT": "",
           "H3_PROMPT_FILE": "", "H3_REFERENCE_IMAGE": "", "H3_STEP_UPDATE": "comfy_res_multistep",
           "H3_TASK": "fl2av", "H3_SMOKE": "1", "H3_OUTPUT_DURATION": "5"}
    result = subprocess.run(["bash", str(ENTRY), "--counts", counts, "--image", "test:h3-scale"],
                            capture_output=True, text=True, env=env)
    return result, tmp_path / "sweep"


def test_scale_reuses_materials_and_isolates_each_card_count(tmp_path):
    result, output = run_sweep(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["gpu_counts"] == [4, 8, 16]
    assert manifest["native_frames"] == 362 and manifest["output_frames"] == 360
    assert manifest["infer_steps"] == 3 and manifest["lora"] is False
    refs = prompt = None
    for count in (4, 8, 16):
        env = json.loads((output / f"{count}gpu/captured.json").read_text())
        assert env["H3_TASK"] == "ref2av" and env["H3_OUTPUT_DURATION"] == "15"
        assert env["H3_INFER_STEPS"] == "3" and env["H3_SMOKE"] == "0"
        assert env["IMAGE"] == "test:h3-scale"
        assert len(env["GPU_DEVICES"].split(",")) == count
        assert env["TARGET_HEIGHT"] == "768" and env["TARGET_WIDTH"] == "1344"
        assert env["TARGET_VIDEO_LENGTH"] == "362"
        assert env["LIGHTX2V_H3_RUN_COLLECTOR"] == "1"
        assert env["H3_EVIDENCE_DIR"] == "/workspace/LightX2V/save_results"
        assert env["SAVE_RESULT_PATH"].endswith("/minimax_h3_ref2av.mp4")
        assert len(env["H3_REFERENCE_IMAGES"].splitlines()) == 2
        if refs is not None:
            assert refs == env["H3_REFERENCE_IMAGES"] and prompt == env["PROMPT"]
        refs, prompt = env["H3_REFERENCE_IMAGES"], env["PROMPT"]
        assert "[Shot 6]" in prompt and "<Audio" not in prompt
        assert (output / f"{count}gpu.log").is_file()


def test_scale_propagates_inference_failure_and_stops_sweep(tmp_path):
    result, output = run_sweep(tmp_path, fail=True)
    assert result.returncode == 37, result.stdout + result.stderr
    assert (output / "4gpu/captured.json").is_file()
    assert not (output / "8gpu").exists()


@pytest.mark.parametrize("counts", ["2", "4", "8", "16"])
def test_scale_can_run_one_count(tmp_path, counts):
    result, output = run_sweep(tmp_path, counts=counts)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / f"{counts}gpu/captured.json").exists()
