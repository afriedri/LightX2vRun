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


if __name__ == "__main__":
    unittest.main()
