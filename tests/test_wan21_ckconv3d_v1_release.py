import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = REPO_ROOT / "lightx2vModelrun"
RELEASE_SCRIPT = REPO_ROOT / "scripts/run_wan21_ckconv3d_v1_release.sh"


class Wan21CKConv3dV1ReleaseTest(unittest.TestCase):
    def test_release_config_is_single_gpu_resident_untiled_ck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = root / "base.json"
            base_config.write_text("{}\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "BASE_CONFIG_JSON": str(base_config),
                    "CONFIG_ONLY": "1",
                    "HIP_VISIBLE_DEVICES": "0",
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

        config = json.loads(result.stdout[result.stdout.index("{\n") :])
        self.assertFalse(config["use_tiling_vae"])
        self.assertFalse(config["vae_cpu_offload"])
        self.assertTrue(config["t5_cpu_offload"])
        self.assertTrue(config["clip_cpu_offload"])
        self.assertEqual(config["parallel"]["seq_p_size"], 1)
        self.assertFalse(config["parallel"]["vae_parallel"])
        self.assertIn("single GPU, VAE resident", result.stdout)
        self.assertIn("MIOpen: 0", result.stdout)

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

        self.assertIn("gfx1201-R9600-FP8-260805-CKConv3dV1-AOT", result.stdout)
        self.assertIn("run_wan21_ckconv3d_v1_release.sh", result.stdout)
        self.assertIn("Mode:           wan21_ckconv3d_v1_release", result.stdout)

    def test_modelrun_rejects_multiple_gpus(self):
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "GPU_DEVICES": "0,1",
                "RUN_REPO_DIR": str(REPO_ROOT),
            }
        )
        result = subprocess.run(
            [str(MODEL_RUN), "--wan21-ckconv3d-v1-release"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires exactly one GPU_DEVICE", result.stderr)


if __name__ == "__main__":
    unittest.main()
