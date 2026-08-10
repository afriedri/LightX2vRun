from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = REPO_ROOT / "lightx2vModelrun"
RELEASE = REPO_ROOT / "scripts/run_wan22_ckconv3d_v1_release.sh"
SWEEP = REPO_ROOT / "scripts/run_wan22_ckconv3d_v1_timing_sweep.sh"


class Wan22CKConv3dTimingSweepTest(unittest.TestCase):
    def run_release_config(self, gpu_count: int) -> tuple[dict[str, object], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = root / "base.json"
            base_config.write_text("{}\n", encoding="utf-8")
            completed = subprocess.run(
                ["bash", str(RELEASE)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "BASE_CONFIG_JSON": str(base_config),
                    "CONFIG_ONLY": "1",
                    "HIP_VISIBLE_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                    "LIGHTX2V_PATH": str(root / "LightX2V"),
                    "MODELS_ROOT": str(root / "models"),
                    "MODEL_PATH": str(root / "model"),
                    "RUNTIME_CONFIG_DIR": str(root / "runtime"),
                    "AITER_JIT_DIR": str(root / "aiter"),
                    "LIGHTX2V_CK_CONV3D_PROFILE_DIR": str(root / "must-not-exist"),
                    "PROFILE_TAG": "must-not-propagate",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertFalse((root / "must-not-exist").exists())
            return json.loads(completed.stdout[completed.stdout.index("{\n") :]), completed.stdout

    def test_release_matches_profile_workload_without_shape_collection(self):
        for gpu_count in (2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                config, output = self.run_release_config(gpu_count)
                self.assertTrue(config["cpu_offload"])
                self.assertEqual(config["offload_granularity"], "block")
                self.assertEqual(config["offload_ratio"], 1.0)
                self.assertTrue(config["t5_cpu_offload"])
                self.assertFalse(config["vae_cpu_offload"])
                self.assertFalse(config["use_tiling_vae"])
                self.assertEqual(config["parallel"]["seq_p_size"], gpu_count)
                self.assertEqual(config["parallel"]["vae_parallel"], True)
                self.assertEqual(
                    config["parallel"]["seq_p_attn_type"],
                    "ring" if gpu_count == 16 else "ulysses",
                )
                self.assertEqual(config["self_attn_1_type"], "aiter_fav3_sage_bf16_attn")
                self.assertIn("Shape collection: disabled", output)

    def test_release_rejects_single_gpu(self):
        completed = subprocess.run(
            ["bash", str(RELEASE)],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0", "CONFIG_ONLY": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("supports 2, 4, 8, or 16 GPUs", completed.stderr)

    def test_modelrun_selects_hot26_release_at_all_timing_scales(self):
        for gpu_count in (2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                env = os.environ.copy()
                env.pop("IMAGE", None)
                env.update(
                    {
                        "DRY_RUN": "1",
                        "GPU_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                        "RUN_REPO_DIR": str(REPO_ROOT),
                    }
                )
                completed = subprocess.run(
                    [str(MODEL_RUN), "--wan22-ckconv3d-v1-release"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn("CKConv3dV1-Hot26-AOTFinal", completed.stdout)
                self.assertIn("run_wan22_ckconv3d_v1_release.sh", completed.stdout)
                self.assertIn("Mode:           wan22_ckconv3d_v1_release", completed.stdout)

    def test_timing_sweep_runs_four_scales_without_profile_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_modelrun = root / "lightx2vModelrun"
            call_log = root / "calls.log"
            fake_modelrun.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s|%s|%s|%s|%s\\n' \"$GPU_DEVICES\" \"$IMAGE\" \"${LIGHTX2V_CK_CONV3D_PROFILE_DIR-unset}\" \"${PROFILE_TAG-unset}\" \"$*\" >> \"$CALL_LOG\"\n"
                "echo '[Profile] Single GPU - Level1_Log Run Image Encoder cost 1.000000 seconds'\n"
                "echo '[Profile] Single GPU - Level1_Log Run VAE Encoder cost 2.000000 seconds'\n"
                "echo '[Profile] Single GPU - Level1_Log Run Text Encoder cost 3.000000 seconds'\n"
                "echo '[Profile] Single GPU - Level2_Log Run Encoders cost 6.100000 seconds'\n"
                "echo '[Profile] Single GPU - Level1_Log Run VAE Decoder cost 7.000000 seconds'\n"
                "echo '[Profile] Single GPU - Level1_Log RUN pipeline cost 18.000000 seconds'\n"
                "echo '[Profile] Single GPU - Level1_Log Total Cost cost 20.000000 seconds'\n"
                "echo 'DiT summary: status=complete, steps=4/4, total=4.000s, average=1.000s/it, throughput=1.0000it/s'\n",
                encoding="utf-8",
            )
            fake_modelrun.chmod(0o755)
            stale_result_dir = root / "wan22_ckconv3d_targeted_replay_stale"
            stale_result_dir.mkdir()
            (stale_result_dir / "winners.csv").write_text("stale\n", encoding="utf-8")
            output_dir = root / "outputs"
            result_dir = output_dir / "test-wan22-timing"
            completed = subprocess.run(
                ["bash", str(SWEEP), "--image", "registry.example/hot26:test"],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "MODEL_RUN": str(fake_modelrun),
                    "CALL_LOG": str(call_log),
                    "OUTPUT_DIR": str(output_dir),
                    "TIMING_SWEEP_TAG": "test-wan22-timing",
                    "SWEEP_TAG": "wan22_ckconv3d_targeted_replay_stale",
                    "RESULT_DIR": str(stale_result_dir),
                    "LIGHTX2V_CK_CONV3D_PROFILE_DIR": "must-not-pass",
                    "PROFILE_TAG": "must-not-pass",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            calls = [line.split("|") for line in call_log.read_text(encoding="utf-8").splitlines()]
            with (result_dir / "timings.csv").open(newline="", encoding="utf-8") as source:
                rows = {row["metric"]: row for row in csv.DictReader(source)}
            stale_contents = (stale_result_dir / "winners.csv").read_text(encoding="utf-8")

        self.assertEqual(
            [call[0] for call in calls],
            [
                "1,2",
                "15,14,13,12",
                "0,1,2,3,4,5,6,7",
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            ],
        )
        self.assertTrue(all(call[1] == "registry.example/hot26:test" for call in calls))
        self.assertTrue(all(call[2:4] == ["unset", "unset"] for call in calls))
        self.assertTrue(all(call[4] == "--wan22-ckconv3d-v1-release" for call in calls))
        self.assertEqual(stale_contents, "stale\n")
        self.assertEqual(list(rows["Image Encoder"]), ["metric", "2", "4", "8", "16", "unit"])
        self.assertTrue(all(rows["Encoders 合计"][str(scale)] == "6.100000" for scale in (2, 4, 8, 16)))
        self.assertTrue(all(rows["DiT 4 steps"][str(scale)] == "4.000000" for scale in (2, 4, 8, 16)))
        self.assertIn("timing sweep complete", completed.stdout)


if __name__ == "__main__":
    unittest.main()
