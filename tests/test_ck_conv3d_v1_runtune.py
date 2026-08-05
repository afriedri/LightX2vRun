from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/runtune_ck_conv3d_v1.py"
LAUNCHER_PATH = REPO_ROOT / "scripts/run_wan21_ck_conv3d_v1_runtune.sh"
SPEC = importlib.util.spec_from_file_location("runtune_ck_conv3d_v1", RUNNER_PATH)
RUNTUNE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTUNE)


def fake_result(arm: str, gpu_id: int, encode_seconds: float, decode_seconds: float):
    is_ck = arm == "ck_v1_resident"

    def phase(seconds: float):
        return {
            "measurement": {
                "wall_seconds": seconds,
                "gpu_seconds": seconds * 0.9,
                "peak_allocated_mib": 5000.0,
                "peak_reserved_mib": 5500.0,
            },
            "vs_pytorch_offload": {
                "shape_match": True,
                "cosine": 1.0,
                "max_abs": 0.0,
                "passed": True,
            },
            "route_counts": {"ck": 10} if is_ck else {"disabled": 10},
        }

    return {
        "arm": arm,
        "gpu_id": str(gpu_id),
        "configuration": {
            "backend": "Aiter_CK_Conv3d_v1" if is_ck else "pytorch",
            "miopen": False,
            "tiling": not is_ck,
            "vae_cpu_offload": not is_ck,
            "channels_last_3d": False,
            "dtype": "BF16",
            "automatic_frequency": True,
        },
        "workload": {
            "height": 720,
            "width": 1280,
            "frames": 81,
            "latent_channels": 16,
            "latent_frames": 21,
            "seed": 20260805,
            "rounds": 1,
        },
        "correctness_thresholds": {
            "atol": 0.05,
            "rtol": 0.05,
            "min_close_percent": 99.9,
            "min_cosine": 0.999,
        },
        "model_load": {"wall_seconds": 1.0},
        "rounds": [
            {
                "round": 0,
                "encode": phase(encode_seconds),
                "decode": phase(decode_seconds),
            }
        ],
        "gate": {"passed": True, "reasons": []},
    }


def write_result(root: Path, gpu_id: int, payload: dict) -> None:
    result_dir = root / f"gpu{gpu_id}"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


class CKConv3dV1RunTuneTest(unittest.TestCase):
    def test_launcher_uses_run_owned_runner_and_image_owned_lightx2v(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "GPU_IDS": "0,1",
                    "IMAGE": "ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260804-SageUtimateWithMM-T1",
                    "RESULT_DIR": str(Path(temporary) / "result"),
                    "RUN_REPO_DIR": str(REPO_ROOT),
                }
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER_PATH)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertIn("gfx1201-R9600-FP8-260805-CKConv3dV1-AOT", result.stdout)
        self.assertIn("image-bundled /workspace/LightX2V", result.stdout)
        self.assertIn(str(RUNNER_PATH), result.stdout)
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("LIGHTX2V_DIR", launcher)
        self.assertNotIn("rocm-smi", launcher)
        self.assertNotIn("TMPDIR=/task/tmp", launcher)
        self.assertIn("--env TMPDIR=/tmp", launcher)
        self.assertIn("source=${container_tmp},target=/tmp", launcher)
        self.assertIn("source=${RUNNER},target=${CONTAINER_RUNNER},readonly", launcher)

    def test_arm_contract_matches_t1_baseline_and_ck_resident_candidate(self):
        self.assertEqual(
            RUNTUNE.ARM_CONFIG,
            {
                "pytorch_offload": {
                    "backend": "pytorch",
                    "vae_cpu_offload": True,
                    "use_tiling": True,
                },
                "ck_v1_resident": {
                    "backend": "Aiter_CK_Conv3d_v1",
                    "vae_cpu_offload": False,
                    "use_tiling": False,
                },
            },
        )

    def test_aggregate_reports_encode_decode_and_strict_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_result(root, 0, fake_result("pytorch_offload", 0, 20.0, 30.0))
            write_result(root, 1, fake_result("ck_v1_resident", 1, 10.0, 15.0))

            summary = RUNTUNE.aggregate(root)

            self.assertEqual(summary["comparison"]["encode_wall_speedup"], 2.0)
            self.assertEqual(summary["comparison"]["decode_wall_speedup"], 2.0)
            self.assertEqual(summary["comparison"]["vae_wall_speedup"], 2.0)
            self.assertEqual(summary["arms"][1]["ck_calls"], 20)
            json.loads(
                (root / "summary.json").read_text(encoding="utf-8"),
                parse_constant=self.fail,
            )
            self.assertTrue((root / "summary.csv").is_file())

    def test_aggregate_rejects_a_non_vae_setting_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = fake_result("pytorch_offload", 0, 20.0, 30.0)
            candidate = fake_result("ck_v1_resident", 1, 10.0, 15.0)
            candidate["configuration"]["miopen"] = True
            write_result(root, 0, baseline)
            write_result(root, 1, candidate)

            with self.assertRaisesRegex(ValueError, "frozen non-VAE setting"):
                RUNTUNE.aggregate(root)

    def test_aggregate_rejects_a_changed_tiling_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = fake_result("pytorch_offload", 0, 20.0, 30.0)
            candidate = fake_result("ck_v1_resident", 1, 10.0, 15.0)
            candidate["configuration"]["tiling"] = True
            write_result(root, 0, baseline)
            write_result(root, 1, candidate)

            with self.assertRaisesRegex(ValueError, "changed its VAE contract"):
                RUNTUNE.aggregate(root)

    def test_aggregate_rejects_a_changed_vae_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = fake_result("pytorch_offload", 0, 20.0, 30.0)
            candidate = fake_result("ck_v1_resident", 1, 10.0, 15.0)
            candidate["configuration"]["vae_cpu_offload"] = True
            write_result(root, 0, baseline)
            write_result(root, 1, candidate)

            with self.assertRaisesRegex(ValueError, "changed its VAE contract"):
                RUNTUNE.aggregate(root)


if __name__ == "__main__":
    unittest.main()
