from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_wan22_ckconv3d_v1_720p_5s_lazy_16gpu_memdiag.sh"
CONFIG = REPO_ROOT / "configs/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201_lazy.json"
MODEL_RUN_WRAPPER = REPO_ROOT / "scripts/run_wan22_lazy_16gpu_modelrun.sh"
EXPERIMENT_A_LAUNCHER = REPO_ROOT / "scripts/run_wan22_lazy_16gpu_experiment_a.sh"
EXPERIMENT_A_SUMMARIZER = REPO_ROOT / "scripts/summarize_wan22_lazy_experiment_a.py"
COMPATIBILITY_ROOT = REPO_ROOT / "python_compat/wan22_lazy"


class Wan22Lazy16GpuMemoryDiagnosticTest(unittest.TestCase):
    def test_config_enables_only_supported_lazy_combination(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(config["lazy_load"])
        self.assertTrue(config["cpu_offload"])
        self.assertEqual(config["offload_granularity"], "block")
        self.assertEqual(config["offload_ratio"], 1.0)
        self.assertTrue(config["t5_cpu_offload"])
        self.assertFalse(config["t5_lazy_load"])
        self.assertEqual(config["parallel"]["seq_p_size"], 16)
        self.assertEqual(config["parallel"]["seq_p_attn_type"], "ring")

    def test_launcher_runs_only_16gpu_720p121f_with_monitoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_modelrun = root / "lightx2vModelrun"
            call_log = root / "calls.jsonl"
            fake_modelrun.write_text(
                "#!/usr/bin/env bash\n"
                "python3 -c 'import json, os, sys; print(json.dumps({**{key: os.environ.get(key) for key in (\"GPU_DEVICES\", \"BASE_CONFIG_JSON\", \"PROFILE_WORKLOAD\", \"TARGET_VIDEO_LENGTH\", \"TARGET_HEIGHT\", \"TARGET_WIDTH\", \"TARGET_FPS\", \"MEMORY_MONITOR\", \"OFFLOAD_RATIO\")}, \"argv\": sys.argv[1:]}, sort_keys=True))' \"$@\" >> \"$CALL_LOG\"\n",
                encoding="utf-8",
            )
            fake_modelrun.chmod(0o755)
            completed = subprocess.run(
                ["bash", str(LAUNCHER)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "MODEL_RUN": str(fake_modelrun),
                    "CALL_LOG": str(call_log),
                    "OUTPUT_DIR": str(root / "outputs"),
                    "SWEEP_COUNTS": "2,4,8,16",
                    "SWEEP_WORKLOADS": "480p",
                    "MEMORY_MONITOR": "0",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["GPU_DEVICES"], ",".join(str(index) for index in range(16)))
        self.assertEqual(
            calls[0]["BASE_CONFIG_JSON"],
            "/workspace/LightX2VRun/configs/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201_lazy.json",
        )
        self.assertEqual(calls[0]["PROFILE_WORKLOAD"], "720p121f")
        self.assertEqual(calls[0]["TARGET_VIDEO_LENGTH"], "121")
        self.assertEqual(calls[0]["TARGET_HEIGHT"], "720")
        self.assertEqual(calls[0]["TARGET_WIDTH"], "1280")
        self.assertEqual(calls[0]["TARGET_FPS"], "24")
        self.assertEqual(calls[0]["MEMORY_MONITOR"], "1")
        self.assertEqual(calls[0]["OFFLOAD_RATIO"], "1.0")
        self.assertEqual(
            calls[0]["argv"],
            [
                "/bin/bash",
                "/workspace/LightX2VRun/scripts/run_wan22_lazy_16gpu_inside.sh",
            ],
        )
        self.assertIn("Wan2.2 dedicated lazy-load memory diagnostic", completed.stdout)

    def test_attention_compatibility_adds_noop_disk_loader(self):
        completed = subprocess.run(
            [
                "python3",
                "-S",
                "-c",
                "import runpy, sys, types; "
                "template_module = types.ModuleType('lightx2v_platform.ops.attn.template'); "
                "template_module.AttnWeightTemplate = type('AttnWeightTemplate', (), {}); "
                "sys.modules['lightx2v_platform.ops.attn.template'] = template_module; "
                "runpy.run_path(sys.argv[1]); "
                "template = template_module.AttnWeightTemplate; "
                "assert callable(getattr(template, 'load_state_dict_from_disk', None)); "
                "assert template.load_state_dict_from_disk(object(), 0) is None",
                str(COMPATIBILITY_ROOT / "sitecustomize.py"),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_lazy_fp8_compatibility_casts_cuda_bias_to_infer_dtype(self):
        completed = subprocess.run(
            [
                "python3",
                "-S",
                "-c",
                "import runpy, sys, types; "
                "attention_module = types.ModuleType('lightx2v_platform.ops.attn.template'); "
                "attention_module.AttnWeightTemplate = type('AttnWeightTemplate', (), {}); "
                "mm_module = types.ModuleType('lightx2v.common.ops.mm.mm_weight'); "
                "mm_module.torch = types.SimpleNamespace(float32='fp32'); "
                "calls = []; "
                "mm_module.MMWeightQuantTemplate = type('MMWeightQuantTemplate', (), {'post_process': lambda self: calls.append('original')}); "
                "sys.modules['lightx2v_platform.ops.attn.template'] = attention_module; "
                "sys.modules['lightx2v.common.ops.mm.mm_weight'] = mm_module; "
                "runpy.run_path(sys.argv[1]); "
                "buffer = types.SimpleNamespace(dtype='fp32', to=lambda dtype: calls.append(('cast', dtype)) or types.SimpleNamespace(dtype=dtype)); "
                "weight = types.SimpleNamespace(lazy_load=True, bias_force_fp32=False, infer_dtype='bf16', bias_cuda_buffer=buffer); "
                "mm_module.MMWeightQuantTemplate.post_process(weight); "
                "assert calls == ['original', ('cast', 'bf16')], calls; "
                "assert weight.bias_cuda_buffer.dtype == 'bf16'",
                str(COMPATIBILITY_ROOT / "sitecustomize.py"),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_dedicated_modelrun_rejects_other_modes(self):
        completed = subprocess.run(
            ["bash", str(MODEL_RUN_WRAPPER), "--wan22-ckconv3d-v1-release"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("only accepts --profile-wan22-ckconv3d-v1", completed.stderr)

    def test_launcher_rejects_non_16gpu_selection(self):
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "GPU_DEVICES": "0,1", "DRY_RUN": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires exactly 16 GPU_DEVICES", completed.stderr)

    def test_experiment_a_requires_instrumented_image(self):
        completed = subprocess.run(
            ["bash", str(EXPERIMENT_A_LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env={key: value for key, value in os.environ.items() if key != "IMAGE"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("IMAGE must name an image", completed.stderr)

    def test_experiment_a_dry_run_enables_ranked_runtune_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                ["bash", str(EXPERIMENT_A_LAUNCHER)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "IMAGE": "example.invalid/lightx2v:experiment-a",
                    "DRY_RUN": "1",
                    "MODEL_RUN": "/bin/true",
                    "OUTPUT_DIR": str(root),
                    "SWEEP_TAG": "experiment_a_test",
                    "WAN22_CACHE_STATE": "warm",
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Experiment A instrumentation: enabled", completed.stdout)
        self.assertIn("Experiment A upload: enabled", completed.stdout)
        self.assertIn("Experiment A dry-run complete", completed.stdout)

    def test_experiment_a_preflight_checks_wan_override_and_uploads_bundle(self):
        launcher = EXPERIMENT_A_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("models/runners/wan/wan_runner.py", launcher)
        self.assertIn('runtuner.begin_scope("request_end_to_end")', launcher)
        self.assertIn("--experiment-a", launcher)
        self.assertIn("diagnostic bundle upload was still attempted", launcher)

    def test_experiment_a_summarizes_all_three_timing_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "runtune"
            reports.mkdir()
            for rank in range(2):
                payload = {
                    "schema_version": 2,
                    "backend": "wan22-lazy-full-offload-baseline-warm-cache",
                    "environment": {"rank": rank, "world_size": 2},
                    "steps": [
                        {"success": True, "wall_ms": 1000.0 + rank + step}
                        for step in range(4)
                    ],
                    "scopes": [
                        {"name": "transformer_lifecycle", "success": True, "wall_ms": 6000.0 + rank},
                        {"name": "request_end_to_end", "success": True, "wall_ms": 8000.0 + rank},
                    ],
                    "events": [{"category": "transformer.release", "elapsed_ms": 10.0 + rank}],
                    "device_memory_snapshots": [
                        {"label": "request_start", "allocated_bytes": 10 + rank, "reserved_bytes": 20 + rank}
                    ],
                    "summary": {
                        "cpu": {
                            "offload.host_fill": {"total_ms": 100.0 + rank},
                            "offload.host_fill_wait": {"total_ms": 50.0 + rank},
                            "transformer.construct": {"total_ms": 25.0 + rank},
                        },
                        "gpu": {
                            "offload.initial_copy": {"total_ms": 5.0 + rank},
                            "offload.prefetch_copy": {"total_ms": 20.0 + rank},
                            "offload.block_compute": {"total_ms": 900.0 + rank},
                        },
                    },
                }
                (reports / f"rank{rank:03d}.json").write_text(json.dumps(payload), encoding="utf-8")
            log = root / "run.log"
            log.write_text(
                "DiT summary: status=complete, steps=4/4, total=89.293s, average=22.323s/it, throughput=0.0448it/s\n",
                encoding="utf-8",
            )
            output = root / "summary.json"
            completed = subprocess.run(
                [
                    "python3",
                    str(EXPERIMENT_A_SUMMARIZER),
                    "--input-dir",
                    str(reports),
                    "--log",
                    str(log),
                    "--output",
                    str(output),
                    "--expected-ranks",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["legacy_four_step_denoising"]["critical_path_seconds"], 89.293)
        self.assertEqual(summary["scopes"]["transformer_lifecycle"]["max_ms"], 6001.0)
        self.assertEqual(summary["scopes"]["request_end_to_end"]["max_ms"], 8001.0)
        self.assertEqual(summary["runtuner_step_wall"]["critical_path_total_ms"], 4010.0)

    def test_experiment_a_missing_scope_error_identifies_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "runtune"
            reports.mkdir()
            payload = {
                "schema_version": 2,
                "backend": "wan22-lazy-full-offload-baseline-warm-cache",
                "environment": {"rank": 0, "world_size": 1},
                "steps": [
                    {"success": True, "wall_ms": 1000.0 + step}
                    for step in range(4)
                ],
                "scopes": [
                    {
                        "name": "transformer_lifecycle",
                        "success": True,
                        "wall_ms": 6000.0,
                    }
                ],
            }
            (reports / "rank000.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            log = root / "run.log"
            log.write_text(
                "DiT summary: status=complete, steps=4/4, total=89.293s\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    str(EXPERIMENT_A_SUMMARIZER),
                    "--input-dir",
                    str(reports),
                    "--log",
                    str(log),
                    "--output",
                    str(root / "summary.json"),
                    "--expected-ranks",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "rank 0 report must contain one successful 'request_end_to_end' scope",
            completed.stderr,
        )
        self.assertIn("available=['transformer_lifecycle:True']", completed.stderr)


if __name__ == "__main__":
    unittest.main()
