import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/run_ltx23_fp8_b0_704x1280_5s.sh"
CONTAINER_ENTRY = REPO_ROOT / "scripts/lib/run_ltx23_in_container.sh"
WEIGHT_PROBE_ENTRY = REPO_ROOT / "scripts/collector/run_ltx23_weight_probe.sh"


def run_dry(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "result"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "RUN_REPO_DIR": str(REPO_ROOT),
    }
    return subprocess.run(
        [
            "bash",
            str(ENTRY),
            "--image",
            "example.invalid/ltx23:b0",
            "--result-dir",
            str(result_dir),
            *extra,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_weight_probe_dry(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "probe-result"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "RUN_REPO_DIR": str(REPO_ROOT),
    }
    return subprocess.run(
        [
            "bash",
            str(WEIGHT_PROBE_ENTRY),
            "--image",
            "example.invalid/ltx23:b0",
            "--result-dir",
            str(result_dir),
            *extra,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_mounts_assets_and_enables_collector_by_default(tmp_path):
    completed = run_dry(tmp_path, "--gpus", "3,7")
    assert completed.returncode == 0, completed.stderr
    assert "Mode:           ltx23" in completed.stdout
    assert "GPU devices:    3,7" in completed.stdout
    assert "target=/models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors" in completed.stdout
    assert "target=/models/ltx23/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in completed.stdout
    assert "target=/models/ltx23/gemma" in completed.stdout
    manifest = json.loads((tmp_path / "result/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["collector"]["enabled"] is True
    assert manifest["gpus"] == [3, 7]
    assert manifest["residency"] == {
        "dynamic_activation_workspace_budget_gib": 16.0,
        "gpu_slots": {"bf16": 2, "fp8": 2},
        "mock_memory_report": "mock_memory.json",
        "pinned_host_slots": {"bf16": 2, "fp8": 2},
        "prompt_handoff": {
            "artifact": "ltx23_prompt_embeddings.safetensors",
            "consumer": "fresh_sp2_dit_processes",
            "producer": "single_source_process_accelerate_auto",
            "timeout_seconds": 60.0,
            "vram_release_gate": "phase_handoff.json",
            "vram_release_limit_mib": 1024.0,
        },
        "strategy": "process_isolated_gemma_then_disk_streamed_mixed_bf16_fp8",
    }
    assert manifest["workload"] == {
        "final_height": 704,
        "final_width": 1280,
        "fps": 24,
        "frames": 121,
        "stage1_height": 352,
        "stage1_steps": 8,
        "stage1_width": 640,
        "stage2_steps": 3,
    }


def test_no_collect_is_explicit(tmp_path):
    completed = run_dry(tmp_path, "--no-collect")
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "result/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["collector"]["enabled"] is False
    assert manifest["workload"]["frames"] == 121


def test_weight_probe_reuses_exact_ltx_container_path(tmp_path):
    completed = run_weight_probe_dry(tmp_path, "--gpus", "3,7", "--sha256")
    assert completed.returncode == 0, completed.stderr
    assert "Mode:           ltx23" in completed.stdout
    assert "GPU devices:    3,7" in completed.stdout
    assert "LTX action:     in-container weight probe only" in completed.stdout
    assert "LTX23_WEIGHT_PROBE_ONLY=1" in completed.stdout
    assert "LTX23_WEIGHT_PROBE_SHA256=1" in completed.stdout
    assert "target=/models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors" in completed.stdout
    assert "/workspace/LightX2VRun/scripts/lib/run_ltx23_in_container.sh" in completed.stdout
    manifest = json.loads(
        (tmp_path / "probe-result/run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["weight_probe_only"] is True
    assert manifest["weight_probe_sha256"] is True


def test_weight_probe_container_entry_writes_to_shared_result_mount():
    text = CONTAINER_ENTRY.read_text(encoding="utf-8")
    assert "collect_ltx23_weight_probe.py" in text
    assert '${evidence_dir}/ltx23_weight_probe.json' in text
    assert 'if [[ "${weight_probe_only}" == 1 ]]' in text


def test_launcher_rejects_non_two_gpu_contract(tmp_path):
    completed = run_dry(tmp_path, "--gpus", "0")
    assert completed.returncode == 2
    assert "exactly two" in completed.stderr


def test_container_entry_does_not_override_stage1_target_shape():
    text = CONTAINER_ENTRY.read_text(encoding="utf-8")
    assert "--target_shape" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "LIGHTX2V_WAN_RUNTUNE=1" in text
    assert "torchrun --standalone --nproc_per_node=2" in text
    assert "python3 -m lightx2v.utils.ltx2_prompt_cache" in text
    assert "check_ltx23_phase_handoff.py" in text
    assert "--gemma-device-map auto" in text
    assert 'HIP_VISIBLE_DEVICES="${gpu0}"' not in text
    assert 'export LIGHTX2V_LTX23_PROMPT_CACHE="${prompt_cache}"' in text
    assert text.index("lightx2v.utils.ltx2_prompt_cache") < text.index(
        "-m lightx2v.infer"
    )
    assert "wait_monitor" in ENTRY.read_text(encoding="utf-8")


def test_frozen_b0_drops_inherited_shape_and_schedule_overrides(tmp_path, monkeypatch):
    poisoned = {
        "TARGET_VIDEO_LENGTH": "105",
        "TARGET_HEIGHT": "720",
        "TARGET_WIDTH": "1280",
        "TARGET_FPS": "21",
        "INFER_STEPS": "99",
        "BASE_CONFIG_JSON": "/tmp/not-the-b0-config.json",
    }
    for name, value in poisoned.items():
        monkeypatch.setenv(name, value)

    completed = run_dry(tmp_path)

    assert completed.returncode == 0, completed.stderr
    for name, value in poisoned.items():
        assert f"{name}={value}" not in completed.stdout
