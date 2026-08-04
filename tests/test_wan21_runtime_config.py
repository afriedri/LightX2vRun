import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts/run_wan21_i2v_distill_fp8_4step_gfx1201_1gpu_benchmark_4090.sh"


class Wan21RuntimeConfigTest(unittest.TestCase):
    def run_config(self, *, gpu_count=1, overrides=None):
        overrides = overrides or {}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_config = temp_path / "base.json"
            base_config.write_text("{}\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "BASE_CONFIG_JSON": str(base_config),
                    "CONFIG_ONLY": "1",
                    "EXPECTED_GPU_COUNT": str(gpu_count),
                    "HIP_VISIBLE_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                    "LIGHTX2V_PATH": str(temp_path / "LightX2V"),
                    "MODELS_ROOT": str(temp_path / "models"),
                    "MODEL_PATH": str(temp_path / "model"),
                    "RUNTIME_CONFIG_DIR": str(temp_path / "runtime"),
                    "AITER_JIT_DIR": str(temp_path / "aiter"),
                }
            )
            env.update(overrides)
            result = subprocess.run(
                ["bash", str(RUN_SCRIPT)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            config_start = result.stdout.index("{\n")
            return json.loads(result.stdout[config_start:]), result.stdout

    def test_legacy_vae_defaults(self):
        config, output = self.run_config()

        self.assertTrue(config["use_tiling_vae"])
        self.assertTrue(config["vae_cpu_offload"])
        self.assertTrue(config["t5_cpu_offload"])
        self.assertTrue(config["clip_cpu_offload"])
        self.assertFalse(config["parallel"]["vae_parallel"])
        self.assertIn("MIOpen: 0", output)
        self.assertIn("VAE channels_last_3d: 0", output)

    def test_optimized_vae_overrides(self):
        config, output = self.run_config(
            overrides={
                "LIGHTX2V_AMD_MIOPEN": "1",
                "USE_TILING_VAE": "0",
                "VAE_CPU_OFFLOAD": "0",
                "T5_CPU_OFFLOAD": "0",
                "CLIP_CPU_OFFLOAD": "0",
                "USE_CHANNELS_LAST_3D": "1",
            }
        )

        self.assertFalse(config["use_tiling_vae"])
        self.assertFalse(config["vae_cpu_offload"])
        self.assertFalse(config["t5_cpu_offload"])
        self.assertFalse(config["clip_cpu_offload"])
        self.assertIn("MIOpen: 1", output)
        self.assertIn("VAE channels_last_3d: 1", output)

    def test_two_gpu_explicitly_enables_vae_parallel(self):
        config, _ = self.run_config(gpu_count=2)

        self.assertEqual(config["parallel"]["seq_p_size"], 2)
        self.assertTrue(config["parallel"]["vae_parallel"])

    def test_rejects_non_binary_flag(self):
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            self.run_config(overrides={"USE_TILING_VAE": "true"})

        self.assertIn("USE_TILING_VAE must be 0 or 1", raised.exception.stderr)


if __name__ == "__main__":
    unittest.main()
