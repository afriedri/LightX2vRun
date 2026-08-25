from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY = REPO_ROOT / "scripts/run_minimax_h3_fl2av_4step_768p.sh"
SUPERCOLLECT_ENTRY = REPO_ROOT / "scripts/collector/run_minimax_h3_sp8_supercollect.sh"
MEGADISCOVERY_ENTRY = REPO_ROOT / "scripts/collector/run_minimax_h3_megadiscovery.sh"
DEFAULT_MODEL_DIR = "/disk/ssd1/difumodels/MiniMax-h3"
DEFAULT_TURBO_DIR = "/disk/ssd/difumodels-Dis/Minimax-h3-Turbo"
DEFAULT_FIRST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png"
DEFAULT_LAST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png"
DEFAULT_PROMPT_PRESET = "bird-stable"


def rendered_prompt(preset: str, duration: str) -> str:
    template = (REPO_ROOT / f"prompts/minimax_h3/{preset}.txt").read_text(encoding="utf-8").strip()
    return template.replace("{{DURATION_SECONDS}}", duration)


def run_dry(
    tmp_path: Path,
    *extra: str,
    include_model_paths: bool = True,
    include_frames: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    result_dir = tmp_path / "image-result"
    arguments = [
        "bash",
        str(ENTRY),
        "--image",
        "example.invalid/minimax-h3:production",
    ]
    if include_model_paths:
        arguments.extend(
            [
                "--model-dir",
                str(tmp_path / "MiniMax-H3"),
                "--turbo-dir",
                str(tmp_path / "Minimax-h3-Turbo"),
            ]
        )
    if include_frames:
        arguments.extend(
            [
                "--first-frame",
                str(tmp_path / "first.png"),
                "--last-frame",
                str(tmp_path / "last.png"),
            ]
        )
    arguments.extend(
        [
            "--result-dir",
            str(result_dir),
            "--preflight-only",
            *extra,
        ]
    )
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
            "H3_MODEL_DIR": "",
            "H3_TURBO_DIR": "",
            "GPU_DEVICES": "",
            "H3_PROMPT_FILE": "",
            "H3_PROMPT_PRESET": "",
            "PROMPT": "",
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
    assert manifest["gpus"] == [0, 1, 2, 3]
    assert manifest["parallel"]["tensor_parallel"] is False
    assert manifest["parallel"] == {
        "seq_p_attn_type": "ulysses",
        "seq_p_size": 4,
        "seq_parallel": True,
        "tensor_parallel": False,
        "vae_decode_parallel": True,
    }
    assert manifest["memory_policy"] == {
        "cpu_block_buffers": 2,
        "cpu_offload": True,
        "gpu_block_buffers": 2,
        "lazy_load": True,
        "offload_granularity": "block",
        "text_encoder_rank0_only": True,
        "text_encoder_unload_after_infer": True,
    }
    assert manifest["backends"] == {
        "dense_gemm_lora": {
            "measured_backend": "hipBLASLt",
            "route": "torch.mm -> default ROCm BLAS heuristic",
            "tunableop_cache": False,
            "tunableop_enabled": False,
        },
        "h1_patch_embed": {
            "aten_operator": "addmm",
            "calls_per_h1": 1,
            "frontend_api": "torch.nn.functional.linear",
            "matrix_mkn_per_rank": [8064, 1536, 1152],
            "measured_device_backend": "hipBLASLt",
            "name": "torch_linear",
            "sequence_sharded": False,
            "tunableop_enabled": False,
        },
        "h3_core_attention": {
            "name": "aiter_fav3_sage_bf16_attn",
            "numerics": "int8_qk_fp8_pv",
            "route": "sage_attn_v2_gfx1201",
        },
        "h3_core_qk_rms": "minimax_h3_aiter_triton_qk",
        "h3_core_rope": "minimax_h3_triton_strict_rope",
        "h3_refiner_attention": "aiter_triton_mha",
        "h4_video_attention": "aiter_triton_mha",
    }
    assert "--env PYTORCH_TUNABLEOP_ENABLED=0" in completed.stdout


def test_h3_launcher_defaults_to_published_host_frames(tmp_path):
    completed, result_dir = run_dry(tmp_path, include_frames=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "H3 frame source: host" in completed.stdout
    assert "--env H3_FRAME_SOURCE=host" in completed.stdout
    assert "--env IMAGE_PATH=/inputs/minimax_h3_first_frame" in completed.stdout
    assert "--env LAST_FRAME_PATH=/inputs/minimax_h3_last_frame" in completed.stdout
    assert f"source={DEFAULT_FIRST_FRAME}" in completed.stdout
    assert f"source={DEFAULT_LAST_FRAME}" in completed.stdout
    assert "target=/inputs/minimax_h3_first_frame" in completed.stdout
    assert "target=/inputs/minimax_h3_last_frame" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["first_frame"] == {"path": DEFAULT_FIRST_FRAME, "sha256": None, "source": "host"}
    assert manifest["last_frame"] == {"path": DEFAULT_LAST_FRAME, "sha256": None, "source": "host"}
    assert manifest["prompt"] == rendered_prompt(DEFAULT_PROMPT_PRESET, "5.17")
    assert manifest["prompt_source"] == f"preset:{DEFAULT_PROMPT_PRESET}"
    assert manifest["prompt_preset"] == DEFAULT_PROMPT_PRESET
    assert manifest["workload"] == {
        "fps": 24,
        "frames": 124,
        "height": 768,
        "model_evaluations": 4,
        "width": 1344,
    }


def test_h3_launcher_renders_smoke_duration_in_structured_prompt(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--smoke")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt"] == rendered_prompt(DEFAULT_PROMPT_PRESET, "5.17")
    assert manifest["workload"] == {
        "fps": 24,
        "frames": 124,
        "height": 480,
        "model_evaluations": 4,
        "width": 480,
    }
    assert manifest["backends"]["h1_patch_embed"]["matrix_mkn_per_rank"] == [1800, 1536, 1152]


def test_h3_launcher_selects_audio_prompt_preset(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--prompt-preset", "bird-audio")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt"] == rendered_prompt("bird-audio", "5.17")
    assert manifest["prompt_source"] == "preset:bird-audio"
    assert manifest["prompt_preset"] == "bird-audio"


def test_h3_launcher_accepts_custom_prompt_file(tmp_path):
    prompt_file = tmp_path / "custom-prompt.txt"
    prompt_file.write_text("custom ending at {{DURATION_SECONDS}} seconds", encoding="utf-8")
    completed, result_dir = run_dry(tmp_path, "--prompt-file", str(prompt_file))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt"] == "custom ending at 5.17 seconds"
    assert manifest["prompt_source"] == f"file:{prompt_file}"
    assert manifest["prompt_preset"] is None


def test_h3_launcher_rejects_unknown_prompt_preset(tmp_path):
    completed, _ = run_dry(tmp_path, "--prompt-preset", "not-a-preset")
    assert completed.returncode == 2
    assert "unknown MiniMax-H3 prompt preset" in completed.stderr


def test_h3_launcher_mounts_explicit_host_frames(tmp_path):
    completed, result_dir = run_dry(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "H3 frame source: host" in completed.stdout
    assert "--env H3_FRAME_SOURCE=host" in completed.stdout
    assert "target=/inputs/minimax_h3_first_frame" in completed.stdout
    assert "target=/inputs/minimax_h3_last_frame" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["first_frame"]["source"] == "host"
    assert manifest["last_frame"]["source"] == "host"


def test_h3_launcher_uses_published_default_model_locations(tmp_path):
    completed, result_dir = run_dry(tmp_path, include_model_paths=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"H3 model:       {DEFAULT_MODEL_DIR} -> /models/minimax_h3" in completed.stdout
    assert f"H3 turbo:       {DEFAULT_TURBO_DIR} -> /models/minimax_h3_turbo" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_dir"] == DEFAULT_MODEL_DIR
    assert manifest["turbo_dir"] == DEFAULT_TURBO_DIR


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


def test_h3_dry_run_accepts_four_gpu_ulysses_without_tp(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--gpus", "4,5,6,7")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert r"HIP_VISIBLE_DEVICES=4\,5\,6\,7" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == [4, 5, 6, 7]
    assert manifest["parallel"] == {
        "seq_p_attn_type": "ulysses",
        "seq_p_size": 4,
        "seq_parallel": True,
        "tensor_parallel": False,
        "vae_decode_parallel": True,
    }


def test_h3_dry_run_accepts_eight_gpu_ulysses_without_tp(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--gpus", "0,1,2,3,4,5,6,7")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert r"HIP_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert manifest["parallel"] == {
        "seq_p_attn_type": "ulysses",
        "seq_p_size": 8,
        "seq_parallel": True,
        "tensor_parallel": False,
        "vae_decode_parallel": True,
    }


def test_h3_dry_run_accepts_sixteen_gpu_cp2_ulysses8_without_tp(tmp_path):
    gpu_ids = ",".join(str(index) for index in range(16))
    completed, result_dir = run_dry(tmp_path, "--gpus", gpu_ids)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == list(range(16))
    assert manifest["parallel"] == {
        "context_p_size": 2,
        "seq_p_attn_type": "ulysses_ring",
        "seq_p_size": 16,
        "seq_parallel": True,
        "tensor_parallel": False,
        "ulysses_p_size": 8,
        "vae_decode_parallel": True,
    }
    assert manifest["backends"]["h3_core_attention"]["route"] == "sage_attn_v2_gfx1201_cp2_u8_2d"
    assert manifest["memory_policy"]["text_encoder_rank0_only"] is True


def test_h3_launcher_rejects_sp16_smoke_geometry(tmp_path):
    gpu_ids = ",".join(str(index) for index in range(16))
    completed, _ = run_dry(tmp_path, "--smoke", "--gpus", gpu_ids)

    assert completed.returncode == 2
    assert "full 124f 1344x768 workload" in completed.stderr


def test_h3_collect_is_opt_in_and_passed_to_each_rank(tmp_path):
    completed, result_dir = run_dry(tmp_path, "--collect", "--gpus", "0,1")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "H3 T1 collector:" in completed.stdout
    assert "--env LIGHTX2V_H3_RUN_COLLECTOR=1" in completed.stdout
    assert r"--env LIGHTX2V_H3_RUN_COLLECTOR_OUTPUT=/workspace/LightX2V/save_results/h3_runcollector.rank\{rank\}.json" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runcollector"] == {
        "cross_rank_summary": "h3_t1_summary.json",
        "enabled": True,
        "mode": "h3_t1_lifecycle",
        "numerical_path_changed": False,
        "output_pattern": "h3_runcollector.rank{rank}.json",
        "profiling_only": True,
        "schema_name": "lightx2v.minimax_h3.runcollector",
        "schema_version": 2,
    }


def test_h3_runcollector_is_disabled_by_default(tmp_path):
    completed, result_dir = run_dry(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runcollector"]["enabled"] is False
    assert "H3 T1 collector:" not in completed.stdout


def test_sp8_supercollect_is_observer_only_and_freezes_production_contract(tmp_path):
    result_dir = tmp_path / "supercollect"
    completed = subprocess.run(
        [
            "bash",
            str(SUPERCOLLECT_ENTRY),
            "--image",
            "example.invalid/minimax-h3:production",
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
            "--gpus",
            "8,9,10,11,12,13,14,15",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_REPO_DIR": str(REPO_ROOT),
            "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
            "H3_MODEL_DIR": "",
            "H3_TURBO_DIR": "",
            "GPU_DEVICES": "",
            "H3_PROMPT_FILE": "",
            "H3_PROMPT_PRESET": "",
            "PROMPT": "",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SP8 SuperCollect: enabled (observer-only; production contract unchanged)" in completed.stdout
    assert r"HIP_VISIBLE_DEVICES=8\,9\,10\,11\,12\,13\,14\,15" in completed.stdout
    assert "--env LIGHTX2V_H3_RUN_COLLECTOR=1" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == list(range(8, 16))
    assert manifest["parallel"]["seq_p_attn_type"] == "ulysses"
    assert manifest["memory_policy"]["cpu_block_buffers"] == 2
    assert manifest["memory_policy"]["gpu_block_buffers"] == 2
    assert manifest["runcollector"]["enabled"] is True
    assert manifest["supercollect"] == {
        "enabled": True,
        "host_snapshots": ["supercollect_host_before.json", "supercollect_host_after.json"],
        "memory_monitor_interval_seconds": 1.0,
        "observer_only": True,
        "production_contract_changed": False,
        "profile": "sp8_system_wide_v1",
        "runtime_evidence": "memory.jsonl",
    }


def test_sp8_supercollect_rejects_non_sp8_or_smoke(tmp_path):
    common = ["bash", str(SUPERCOLLECT_ENTRY), "--image", "image", "--result-dir", str(tmp_path / "result")]
    four = subprocess.run(common + ["--gpus", "0,1,2,3"], capture_output=True, text=True, check=False)
    smoke = subprocess.run(common + ["--smoke"], capture_output=True, text=True, check=False)
    assert four.returncode == 2
    assert "exactly eight" in four.stderr
    assert smoke.returncode == 2
    assert "full production workload" in smoke.stderr


def test_cp2_ulysses8_sp16_megadiscovery_is_observer_only(tmp_path):
    result_dir = tmp_path / "megadiscovery"
    gpu_ids = ",".join(str(index) for index in range(16))
    completed = subprocess.run(
        [
            "bash",
            str(MEGADISCOVERY_ENTRY),
            "--image",
            "example.invalid/minimax-h3:production",
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
            "--gpus",
            gpu_ids,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "RUN_REPO_DIR": str(REPO_ROOT),
            "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
            "H3_MODEL_DIR": "",
            "H3_TURBO_DIR": "",
            "GPU_DEVICES": "",
            "H3_PROMPT_FILE": "",
            "H3_PROMPT_PRESET": "",
            "PROMPT": "",
            "NCCL_DEBUG": "",
            "NCCL_DEBUG_SUBSYS": "",
            "NCCL_DEBUG_FILE": "",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MegaDiscovery: enabled for 16 GPUs (observer-only; production contract unchanged)" in completed.stdout
    assert "--env H3_MEGADISCOVERY=1" in completed.stdout
    assert "--env H3_MEGADISCOVERY_TIMEOUT_SECONDS=300" in completed.stdout
    assert "--env LIGHTX2V_H3_RUN_COLLECTOR=1" in completed.stdout
    assert "--env NCCL_DEBUG=INFO" in completed.stdout
    assert "NCCL_DEBUG_FILE=/workspace/LightX2V/save_results/rccl.%h.%p.log" in completed.stdout
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["gpus"] == list(range(16))
    assert manifest["parallel"]["seq_p_attn_type"] == "ulysses_ring"
    assert manifest["memory_policy"]["cpu_block_buffers"] == 2
    assert manifest["memory_policy"]["gpu_block_buffers"] == 2
    assert manifest["runcollector"]["enabled"] is True
    assert manifest["megadiscovery"] == {
        "collective_benchmark": "megadiscovery_collectives.json",
        "enabled": True,
        "host_snapshots": ["megadiscovery_host_before.json", "megadiscovery_host_after.json"],
        "memory_monitor_interval_seconds": 0.5,
        "observer_only": True,
        "probe_order": "after_production_inference",
        "probe_timeout_seconds": 300,
        "production_collective_contract": {
            "a2a_backend": "torch",
            "communication_quantization": None,
            "context_p_size": 2,
            "prepost_backend": "torch",
            "tensor_fusion": False,
            "topology": "cp2_ulysses8_sp16",
            "ulysses_p_size": 8,
        },
        "production_contract_changed": False,
        "profile": "cp2_ulysses8_sp16_full_stack_v1",
        "rccl_debug_pattern": "rccl.*.log",
        "rccl_environment": {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_FILE": "/workspace/LightX2V/save_results/rccl.%h.%p.log",
            "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,TUNING,P2P,SHM,NET",
        },
        "reports": ["megadiscovery.json", "megadiscovery.md"],
    }


def test_megadiscovery_rejects_non_sp8_ring16_or_smoke(tmp_path):
    common = ["bash", str(MEGADISCOVERY_ENTRY), "--image", "image", "--result-dir", str(tmp_path / "result")]
    four = subprocess.run(common + ["--gpus", "0,1,2,3"], capture_output=True, text=True, check=False)
    smoke = subprocess.run(common + ["--smoke"], capture_output=True, text=True, check=False)
    assert four.returncode == 2
    assert "exactly eight or sixteen" in four.stderr
    assert smoke.returncode == 2
    assert "full production workload" in smoke.stderr


def test_h3_launcher_rejects_duplicate_or_unsupported_gpu_counts(tmp_path):
    duplicate, _ = run_dry(tmp_path, "--gpus", "0,0")
    assert duplicate.returncode == 2
    assert "must be unique" in duplicate.stderr

    unsupported, _ = run_dry(tmp_path, "--gpus", "0,1,2")
    assert unsupported.returncode == 2
    assert "supports exactly 1, 2, 4, 8, or 16 GPUs" in unsupported.stderr

    too_many, _ = run_dry(tmp_path, "--gpus", ",".join(str(index) for index in range(17)))
    assert too_many.returncode == 2
    assert "one, two, four, eight, or sixteen comma-separated GPU ids" in too_many.stderr


def test_h3_launcher_rejects_host_source_overlay(tmp_path):
    completed, _ = run_dry(tmp_path, "--lightx2v-dir", str(tmp_path / "LightX2V"))
    assert completed.returncode == 2
    assert "unknown argument: --lightx2v-dir" in completed.stderr
