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
                "python3 -c 'import json, os; print(json.dumps({key: os.environ.get(key) for key in (\"GPU_DEVICES\", \"BASE_CONFIG_JSON\", \"PROFILE_WORKLOAD\", \"TARGET_VIDEO_LENGTH\", \"TARGET_HEIGHT\", \"TARGET_WIDTH\", \"TARGET_FPS\", \"MEMORY_MONITOR\", \"OFFLOAD_RATIO\")}, sort_keys=True))' >> \"$CALL_LOG\"\n",
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
        self.assertIn("Wan2.2 dedicated lazy-load memory diagnostic", completed.stdout)

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
