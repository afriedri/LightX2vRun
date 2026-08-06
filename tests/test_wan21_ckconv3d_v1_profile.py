from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUN = REPO_ROOT / "lightx2vModelrun"
SCRIPTS = REPO_ROOT / "scripts"


class Wan21CKConv3dV1ProfileTest(unittest.TestCase):
    def run_config(self, gpu_count: int):
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
                    "PROFILE_TAG": f"test_{gpu_count}gpu",
                    "SELF_ATTN_TYPE": "aiter_flydsl_bf16_flash_attn",
                }
            )
            script = SCRIPTS / f"run_wan21_ckconv3d_v1_profile_{gpu_count}gpu.sh"
            result = subprocess.run(
                ["bash", str(script)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            return json.loads(result.stdout[result.stdout.index("{\n") :]), result.stdout

    def test_all_scale_wrappers_use_resident_ck_vae(self):
        for gpu_count in (1, 2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                config, output = self.run_config(gpu_count)
                self.assertFalse(config["use_tiling_vae"])
                self.assertFalse(config["vae_cpu_offload"])
                self.assertEqual(config["parallel"]["seq_p_size"], gpu_count)
                self.assertEqual(config["parallel"]["vae_parallel"], gpu_count > 1)
                self.assertEqual(
                    config["self_attn_1_type"], "aiter_fav3_sage_bf16_attn"
                )
                self.assertEqual(
                    config["parallel"]["seq_p_attn_type"],
                    "ring" if gpu_count == 16 else "ulysses",
                )
                self.assertIn("MIOpen: 0", output)

    def test_modelrun_selects_wrapper_from_gpu_count(self):
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
                    [str(MODEL_RUN), "--profile-wan21-ckconv3d-v1"],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertIn(
                    f"run_wan21_ckconv3d_v1_profile_{gpu_count}gpu.sh",
                    result.stdout,
                )
                self.assertIn("--network=host", result.stdout)
                self.assertNotIn("rocm-smi", result.stdout)

    def test_modelrun_mounts_generated_tuned_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            tuned_csv = Path(temporary) / "conv3d_v1_tuned.csv"
            tuned_csv.write_text("placeholder\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "GPU_DEVICES": "0,1",
                    "RUN_REPO_DIR": str(REPO_ROOT),
                    "CK_CONV3D_TUNED_CSV": str(tuned_csv),
                }
            )
            result = subprocess.run(
                [str(MODEL_RUN), "--profile-wan21-ckconv3d-v1"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertIn("AITER_CONFIG_CONV3D_V1=/workspace/ck-conv3d-config/conv3d_v1_tuned.csv", result.stdout)
        self.assertIn(str(tuned_csv), result.stdout)


if __name__ == "__main__":
    unittest.main()
