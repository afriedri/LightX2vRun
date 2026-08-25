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
FIX1_IMAGE = (
    "compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:"
    "gfx1201-R9600-FP8-260813-CKConv3dV1-Hot28-B0B1-Fix1"
)
H3_IMAGE = "example.invalid/lightx2v:h3-production"
H3_768_FIRST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png"
H3_768_LAST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png"
H3_480_FIRST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/FL2V_START_480P.jpg"
H3_480_LAST_FRAME = "/disk/ssd/lightx2v-runtime/inputs/FL2V_END_480P.jpg"


TASKS = {
    "run_01_wan21_720p_81f_16fps_1gpu.sh": ("wan21", 720, 1280, 81, 16, [1]),
    "run_02_wan21_720p_81f_16fps_scale.sh": ("wan21", 720, 1280, 81, 16, [1, 2, 4, 8, 16]),
    "run_04_wan22_480p_81f_16fps_scale.sh": ("wan22", 480, 832, 81, 16, [2, 4, 8, 16]),
    "run_05_wan22_480p_121f_24fps_scale.sh": ("wan22", 480, 832, 121, 24, [2, 4, 8, 16]),
    "run_06_wan22_720p_81f_16fps_scale.sh": ("wan22", 720, 1280, 81, 16, [2, 4, 8, 16]),
    "run_07_wan22_720p_121f_24fps_scale.sh": ("wan22", 720, 1280, 121, 24, [2, 4, 8, 16]),
    "run_08_minimax_h3_768p_124f_24fps_scale.sh": ("minimax_h3", 768, 1344, 124, 24, [2, 4, 8, 16]),
    "run_09_minimax_h3_480p_124f_24fps_scale.sh": ("minimax_h3", 480, 832, 124, 24, [2, 4, 8, 16]),
}

THERMAL_GPU_GROUPS = {
    "2": "1,2",
    "4": "15,14,13,12",
    "8": "0,1,2,3,4,5,6,7",
    "16": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
}

DEFAULT_GPU_GROUPS = {
    "run_02_wan21_720p_81f_16fps_scale.sh": {"1": "0", **THERMAL_GPU_GROUPS},
    "run_04_wan22_480p_81f_16fps_scale.sh": THERMAL_GPU_GROUPS,
    "run_05_wan22_480p_121f_24fps_scale.sh": THERMAL_GPU_GROUPS,
    "run_06_wan22_720p_81f_16fps_scale.sh": THERMAL_GPU_GROUPS,
    "run_07_wan22_720p_121f_24fps_scale.sh": THERMAL_GPU_GROUPS,
}


class ConsolidatedTimingLauncherTest(unittest.TestCase):
    def test_numbered_entries_have_exact_workloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, expected in TASKS.items():
                with self.subTest(entry=name):
                    result_dir = root / name.removesuffix(".sh")
                    model, height, width, frames, fps, counts = expected
                    command = [
                        "bash",
                        str(SCRIPTS / name),
                        "--gpu-count",
                        str(counts[0]),
                    ]
                    if model == "minimax_h3":
                        command.extend(("--image", H3_IMAGE))
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        env={
                            **os.environ,
                            "DRY_RUN": "1",
                            "IMAGE": "example.invalid/stale:hot26",
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
                    self.assertEqual(manifest["model"], model)
                    self.assertEqual(manifest["height"], height)
                    self.assertEqual(manifest["width"], width)
                    self.assertEqual(manifest["frames"], frames)
                    self.assertEqual(manifest["fps"], fps)
                    self.assertEqual(manifest["gpu_counts"], [counts[0]])
                    expected_group = "0"
                    if name in DEFAULT_GPU_GROUPS:
                        expected_group = DEFAULT_GPU_GROUPS[name][str(counts[0])]
                    elif model == "minimax_h3":
                        expected_group = THERMAL_GPU_GROUPS[str(counts[0])]
                    self.assertEqual(
                        manifest["gpu_groups"],
                        {str(counts[0]): expected_group},
                    )
                    self.assertIn(
                        f"scripts/lib/run_{model}_in_container.sh",
                        completed.stdout,
                    )
                    expected_image = H3_IMAGE if model == "minimax_h3" else FIX1_IMAGE
                    self.assertIn(f"Image:          {expected_image}", completed.stdout)
                    self.assertIn(
                        "Memory monitor: enabled (sample=2.0s, terminal=30.0s + phase changes)",
                        completed.stdout,
                    )
                    if model == "minimax_h3":
                        self.assertEqual(manifest["duration_seconds"], 5.17)
                        self.assertEqual(manifest["parallel_topologies"], {"2": "ulysses"})
                        expected_frames = (
                            (H3_480_FIRST_FRAME, H3_480_LAST_FRAME)
                            if height == 480
                            else (H3_768_FIRST_FRAME, H3_768_LAST_FRAME)
                        )
                        self.assertEqual(manifest["first_frame"], expected_frames[0])
                        self.assertEqual(manifest["last_frame"], expected_frames[1])
                        self.assertEqual(manifest["prompt_preset"], "bird-stable")
                        self.assertIn(
                            "MiniMax-H3 topology: SP2/SP4/SP8=Ulysses, SP16=CP2xU8",
                            completed.stdout,
                        )
                        self.assertIn("--env TARGET_VIDEO_LENGTH=124", completed.stdout)
                        self.assertIn(
                            f"H3 first frame: {expected_frames[0]} -> /inputs/minimax_h3_first_frame",
                            completed.stdout,
                        )
                        self.assertIn(
                            f"H3 last frame:  {expected_frames[1]} -> /inputs/minimax_h3_last_frame",
                            completed.stdout,
                        )

    def test_wan_default_sweeps_preserve_thermal_gpu_rotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, expected_groups in DEFAULT_GPU_GROUPS.items():
                with self.subTest(entry=name):
                    result_dir = root / name.removesuffix(".sh")
                    completed = subprocess.run(
                        ["bash", str(SCRIPTS / name)],
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
                    self.assertEqual(manifest["gpu_groups"], expected_groups)
                    for count, group in expected_groups.items():
                        self.assertIn(
                            f"{manifest['model']}: {count} GPUs ({group})",
                            completed.stdout,
                        )

    def test_minimax_h3_timing_requires_an_explicit_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run_08_minimax_h3_768p_124f_24fps_scale.sh"),
                    "--gpu-count",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "IMAGE": "example.invalid/stale:must-be-ignored",
                    "TIMING_RESULT_DIR": str(Path(temporary) / "result"),
                    "RUN_REPO_DIR": str(REPO_ROOT),
                },
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("MiniMax-H3 timing requires --image", completed.stderr)

    def test_minimax_h3_default_sweep_covers_all_production_topologies(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run_08_minimax_h3_768p_124f_24fps_scale.sh"),
                    "--image",
                    H3_IMAGE,
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "TIMING_RESULT_DIR": str(result_dir),
                    "RUN_REPO_DIR": str(REPO_ROOT),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads(
                (result_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["gpu_counts"], [2, 4, 8, 16])
            self.assertEqual(
                manifest["gpu_groups"],
                {
                    "2": "1,2",
                    "4": "15,14,13,12",
                    "8": "0,1,2,3,4,5,6,7",
                    "16": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                },
            )
            self.assertEqual(
                manifest["parallel_topologies"],
                {"2": "ulysses", "4": "ulysses", "8": "ulysses", "16": "ulysses_ring_2d"},
            )
            for count in (2, 4, 8, 16):
                self.assertTrue((result_dir / f"{count}gpu.log").is_file())
            for count, group in manifest["gpu_groups"].items():
                self.assertIn(
                    f"minimax_h3: {count} GPUs ({group})",
                    completed.stdout,
                )

    def test_minimax_h3_480p_default_sweep_uses_jpg_keyframes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "result"
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "run_09_minimax_h3_480p_124f_24fps_scale.sh"),
                    "--image",
                    H3_IMAGE,
                ],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "TIMING_RESULT_DIR": str(result_dir),
                    "RUN_REPO_DIR": str(REPO_ROOT),
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads(
                (result_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual((manifest["width"], manifest["height"]), (832, 480))
            self.assertEqual(manifest["gpu_counts"], [2, 4, 8, 16])
            self.assertEqual(manifest["gpu_groups"], THERMAL_GPU_GROUPS)
            self.assertEqual(manifest["first_frame"], H3_480_FIRST_FRAME)
            self.assertEqual(manifest["last_frame"], H3_480_LAST_FRAME)
            self.assertEqual(manifest["prompt_preset"], "bird-stable")
            self.assertEqual(
                manifest["parallel_topologies"],
                {"2": "ulysses", "4": "ulysses", "8": "ulysses", "16": "ulysses_ring_2d"},
            )
            self.assertIn("Workload: 832x480, 124 frames @ 24 FPS (5.17s)", completed.stdout)
            self.assertIn(H3_480_FIRST_FRAME, completed.stdout)
            self.assertIn(H3_480_LAST_FRAME, completed.stdout)

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
        self.assertIn("Baseline 与 B1 使用完全相同的起止边界", table)
        self.assertIn("RUN 上升而 Total 下降并不表示漏计", table)
        self.assertIn("Encoders 合计,6.100000", csv_text)

    def test_minimax_h3_summary_uses_the_same_table_contract(self):
        log = """
[Profile] Rank 0 - Run VAE Encoder cost 2.000 seconds
[Profile] Rank 0 - Run Text Encoder cost 3.000 seconds
DiT summary: status=complete, steps=4/4, total=40.000s, average=10.000s/it
[Profile] Rank 0 - Run VAE Decoder cost 4.000 seconds
[Profile] Rank 0 - RUN pipeline cost 50.000 seconds
[Profile] Rank 0 - Total Cost cost 55.000 seconds
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "2gpu.log").write_text(log, encoding="utf-8")
            completed = subprocess.run(
                [
                    "python3",
                    str(SUMMARY),
                    "--model",
                    "minimax_h3",
                    "--log-dir",
                    str(root),
                    "--counts",
                    "2",
                    "--width",
                    "832",
                    "--height",
                    "480",
                    "--frames",
                    "124",
                    "--fps",
                    "24",
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            table = (root / "timings.md").read_text(encoding="utf-8")
            csv_text = (root / "timings.csv").read_text(encoding="utf-8")
        self.assertIn("| Image Encoder | N/A |", table)
        self.assertIn("| Encoders 合计 | 5.000s |", table)
        self.assertIn("| DiT 4 steps | 40.000s |", table)
        self.assertIn("832x480、124 帧、24 FPS", table)
        self.assertIn("2/4/8 卡使用 Ulysses，16 卡使用 CP2xU8", table)
        self.assertIn("Encoders 合计,5.000000", csv_text)


if __name__ == "__main__":
    unittest.main()
