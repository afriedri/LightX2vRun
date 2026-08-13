from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
WAN22_INSIDE = SCRIPTS / "lib/run_wan22_in_container.sh"
SUMMARY = SCRIPTS / "lib/summarize_timings.py"


TASKS = {
    "run_01_wan21_720p_81f_16fps_1gpu.sh": ("wan21", 720, 1280, 81, 16, [1]),
    "run_02_wan21_720p_81f_16fps_scale.sh": ("wan21", 720, 1280, 81, 16, [1, 2, 4, 8, 16]),
    "run_04_wan22_480p_81f_16fps_scale.sh": ("wan22", 480, 832, 81, 16, [2, 4, 8, 16]),
    "run_05_wan22_480p_121f_24fps_scale.sh": ("wan22", 480, 832, 121, 24, [2, 4, 8, 16]),
    "run_06_wan22_720p_81f_16fps_scale.sh": ("wan22", 720, 1280, 81, 16, [2, 4, 8, 16]),
    "run_07_wan22_720p_121f_24fps_scale.sh": ("wan22", 720, 1280, 121, 24, [2, 4, 8, 16]),
}


class ConsolidatedWanLauncherTest(unittest.TestCase):
    def test_six_numbered_entries_have_exact_workloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, expected in TASKS.items():
                with self.subTest(entry=name):
                    result_dir = root / name.removesuffix(".sh")
                    completed = subprocess.run(
                        ["bash", str(SCRIPTS / name), "--gpu-count", str(expected[-1][0])],
                        check=False,
                        capture_output=True,
                        text=True,
                        env={
                            **os.environ,
                            "DRY_RUN": "1",
                            "WAN_RESULT_DIR": str(result_dir),
                            "RUN_REPO_DIR": str(REPO_ROOT),
                        },
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        completed.stdout + completed.stderr,
                    )
                    manifest = json.loads(
                        (result_dir / "run_manifest.json").read_text(encoding="utf-8")
                    )
                    model, height, width, frames, fps, counts = expected
                    self.assertEqual(manifest["model"], model)
                    self.assertEqual(manifest["height"], height)
                    self.assertEqual(manifest["width"], width)
                    self.assertEqual(manifest["frames"], frames)
                    self.assertEqual(manifest["fps"], fps)
                    self.assertEqual(manifest["gpu_counts"], [counts[0]])
                    self.assertIn(
                        f"scripts/lib/run_{model}_in_container.sh",
                        completed.stdout,
                    )

    def wan22_config(self, count: int, schedule: str = "auto") -> tuple[dict, str]:
        devices = ",".join(str(index) for index in range(count))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                ["bash", str(WAN22_INSIDE)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HIP_VISIBLE_DEVICES": devices,
                    "CONFIG_ONLY": "1",
                    "AITER_JIT_DIR": str(root / "aiter"),
                    "RUNTIME_CONFIG_DIR": str(root / "runtime"),
                    "LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE": schedule,
                },
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        config = json.loads(completed.stdout[completed.stdout.index("{\n") :])
        return config, completed.stdout

    def test_wan22_auto_policy_and_overrides(self):
        for count, expected in ((2, "baseline"), (4, "baseline"), (8, "b1"), (16, "b1")):
            with self.subTest(count=count):
                config, output = self.wan22_config(count)
                self.assertTrue(config["lazy_load"])
                self.assertEqual(config["offload_granularity"], "block")
                self.assertEqual(config["offload_ratio"], 1.0)
                self.assertEqual(config["wan_block_offload_schedule_request"], "auto")
                self.assertEqual(config["wan_block_offload_schedule"], expected)
                self.assertIn(
                    f"Wan block-offload schedule: {expected} (request=auto)",
                    output,
                )

        forced_on, _ = self.wan22_config(2, "b1")
        forced_off, _ = self.wan22_config(16, "baseline")
        self.assertEqual(forced_on["wan_block_offload_schedule"], "b1")
        self.assertEqual(forced_off["wan_block_offload_schedule"], "baseline")

    def test_single_count_selection_runs_only_that_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            completed = subprocess.run(
                ["bash", str(SCRIPTS / "run_07_wan22_720p_121f_24fps_scale.sh"), "2"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "WAN_RESULT_DIR": str(result_dir),
                    "RUN_REPO_DIR": str(REPO_ROOT),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue((result_dir / "2gpu.log").is_file())
            self.assertFalse((result_dir / "4gpu.log").exists())

    def test_summary_keeps_previous_metric_table_contract(self):
        log = """
[Profile] Rank 0 - Run Image Encoder cost 1.000 seconds
[Profile] Rank 0 - Run VAE Encoder cost 2.000 seconds
[Profile] Rank 0 - Run Text Encoder cost 3.000 seconds
[Profile] Rank 0 - Run Encoders cost 6.100 seconds
DiT summary: status=complete, steps=4/4, total=40.000s, average=10.000s/it
[Profile] Rank 0 - Run VAE Decoder cost 4.000 seconds
[Profile] Rank 0 - RUN pipeline cost 50.000 seconds
[Profile] Rank 0 - Total Cost cost 55.000 seconds
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "1gpu.log").write_text(log, encoding="utf-8")
            completed = subprocess.run(
                [
                    "python3",
                    str(SUMMARY),
                    "--model",
                    "wan21",
                    "--log-dir",
                    str(root),
                    "--counts",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            table = (root / "timings.md").read_text(encoding="utf-8")
            csv_text = (root / "timings.csv").read_text(encoding="utf-8")
        self.assertIn("| DiT 4 steps | 40.000s |", table)
        self.assertIn("| Total Cost | 55.000s |", table)
        self.assertIn("Encoders 合计,6.100000", csv_text)


if __name__ == "__main__":
    unittest.main()
