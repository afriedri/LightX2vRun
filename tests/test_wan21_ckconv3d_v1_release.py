import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = REPO_ROOT / "lightx2vModelrun"
RELEASE_SCRIPT = REPO_ROOT / "scripts/run_wan21_ckconv3d_v1_release.sh"
SCALE_SWEEP_SCRIPT = REPO_ROOT / "scripts/run_wan21_ckconv3d_v1_scale_sweep.sh"


class Wan21CKConv3dV1ReleaseTest(unittest.TestCase):
    def run_release_config(self, gpu_count):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = root / "base.json"
            base_config.write_text("{}\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "BASE_CONFIG_JSON": str(base_config),
                    "CONFIG_ONLY": "1",
                    "HIP_VISIBLE_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                    "LIGHTX2V_PATH": str(root / "LightX2V"),
                    "MODELS_ROOT": str(root / "models"),
                    "MODEL_PATH": str(root / "model"),
                    "RUNTIME_CONFIG_DIR": str(root / "runtime"),
                    "AITER_JIT_DIR": str(root / "aiter"),
                }
            )
            result = subprocess.run(
                ["bash", str(RELEASE_SCRIPT)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        return json.loads(result.stdout[result.stdout.index("{\n") :]), result.stdout

    def test_release_config_is_resident_untiled_sage_ck_at_every_scale(self):
        for gpu_count in (1, 2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                config, output = self.run_release_config(gpu_count)
                self.assertFalse(config["use_tiling_vae"])
                self.assertFalse(config["vae_cpu_offload"])
                self.assertTrue(config["t5_cpu_offload"])
                self.assertTrue(config["clip_cpu_offload"])
                self.assertEqual(config["parallel"]["seq_p_size"], gpu_count)
                self.assertEqual(config["parallel"]["vae_parallel"], gpu_count > 1)
                self.assertEqual(config["self_attn_1_type"], "aiter_fav3_sage_bf16_attn")
                self.assertEqual(
                    config["parallel"]["seq_p_attn_type"],
                    "ring" if gpu_count == 16 else "ulysses",
                )
                self.assertIn(f"{gpu_count} GPU(s), VAE resident", output)
                self.assertIn("MIOpen: 0", output)

    def test_modelrun_selects_ck_aot_image_and_release_script(self):
        env = os.environ.copy()
        env.pop("IMAGE", None)
        env.update(
            {
                "DRY_RUN": "1",
                "GPU_DEVICES": "0",
                "RUN_REPO_DIR": str(REPO_ROOT),
            }
        )
        result = subprocess.run(
            [str(MODEL_RUN), "--wan21-ckconv3d-v1-release"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn("gfx1201-R9600-FP8-260806-CKConv3dV1-Hot9-AOT4", result.stdout)
        self.assertIn("run_wan21_ckconv3d_v1_release.sh", result.stdout)
        self.assertIn("Mode:           wan21_ckconv3d_v1_release", result.stdout)

    def test_modelrun_accepts_all_production_scales(self):
        for gpu_count in (1, 2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                env = os.environ.copy()
                env.update(
                    {
                        "DRY_RUN": "1",
                        "GPU_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                        "RUN_REPO_DIR": str(REPO_ROOT),
                    }
                )
                result = subprocess.run(
                    [str(MODEL_RUN), "--wan21-ckconv3d-v1-release"],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertIn("run_wan21_ckconv3d_v1_release.sh", result.stdout)
                self.assertIn("--network=host", result.stdout)
                self.assertNotIn("rocm-smi", result.stdout)

    def test_scale_sweep_runs_full_release_inference_on_disjoint_default_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_modelrun = root / "lightx2vModelrun"
            call_log = root / "calls.log"
            fake_modelrun.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s|%s|%s|%s|%s|%s\\n' \"$GPU_DEVICES\" \"$CONTAINER_NAME\" \"$SAVE_RESULT_PATH\" \"${LIGHTX2V_CK_CONV3D_PROFILE_DIR-unset}\" \"${PROFILE_TAG-unset}\" \"$*\" >> \"$CALL_LOG\"\n",
                encoding="utf-8",
            )
            fake_modelrun.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "MODEL_RUN": str(fake_modelrun),
                    "CALL_LOG": str(call_log),
                    "SWEEP_TAG": "test-release-sweep",
                    "LIGHTX2V_CK_CONV3D_PROFILE_DIR": "must-not-pass",
                    "PROFILE_TAG": "must-not-pass",
                }
            )
            result = subprocess.run(
                ["bash", str(SCALE_SWEEP_SCRIPT)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = [line.split("|") for line in call_log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(
            [call[0] for call in calls],
            [
                "0",
                "1,2",
                "3,4,5,6",
                "7,8,9,10,11,12,13,14",
                "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            ],
        )
        self.assertTrue(all(call[3:5] == ["unset", "unset"] for call in calls))
        self.assertTrue(all(call[5] == "--wan21-ckconv3d-v1-release" for call in calls))
        self.assertTrue(all("profile" not in call[1] for call in calls))
        self.assertTrue(
            all(
                call[2].endswith(f"_{count}gpu.mp4")
                for call, count in zip(calls, (1, 2, 4, 8, 16))
            )
        )
        self.assertIn("production scale sweep complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
