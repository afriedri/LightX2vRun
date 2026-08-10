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
SUMMARY = REPO_ROOT / "scripts/summarize_wan22_ckconv3d_scale.py"
SWEEP = REPO_ROOT / "scripts/run_wan22_ckconv3d_v1_scale_sweep.sh"
SCRIPTS = REPO_ROOT / "scripts"


def _route(*, calls: int, route: str = "ck") -> dict[str, object]:
    return {
        "route": route,
        "selection": "csv_lookup" if route == "ck" else "generic_fallback",
        "calls": calls,
        "input_shape": [1, 3, 4, 6, 8],
        "weight_shape": [5, 3, 3, 3, 3],
        "output_shape": [1, 5, 4, 6, 8],
        "input_stride": [576, 192, 48, 8, 1],
        "weight_stride": [81, 27, 9, 3, 1],
        "conv_stride": [1, 1, 1],
        "padding": [1, 1, 1],
        "dilation": [1, 1, 1],
        "groups": 1,
    }


def _write_profile(root: Path) -> Path:
    profile = root / "profile_2gpu"
    profile.mkdir()
    snapshot_names = []
    for rank in range(2):
        for phase, calls in (("encode", 2), ("decode", 3)):
            payload = {
                "kind": "lightx2v_ck_conv3d_profile",
                "phase": phase,
                "rank": rank,
                "world_size": 2,
                "gfx": "gfx1201",
                "wgp": 24,
                "backend": "Aiter_CK_Conv3d_v1",
                "routes": [
                    _route(
                        calls=calls,
                        route="pytorch" if rank == 1 and phase == "decode" else "ck",
                    )
                ],
            }
            snapshot_path = profile / f"rank{rank:03d}_{phase}_000.json"
            snapshot_path.write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            snapshot_names.append(snapshot_path.name)
    (profile / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "wan22_ck_conv3d_v1_profile_manifest",
                "world_size": 2,
                "snapshots": sorted(snapshot_names),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return profile


class Wan22CKConv3dScaleSweepTest(unittest.TestCase):
    def run_config(self, gpu_count: int) -> tuple[dict[str, object], str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = root / "base.json"
            base_config.write_text("{}\n", encoding="utf-8")
            completed = subprocess.run(
                ["bash", str(SCRIPTS / f"run_wan22_ckconv3d_v1_profile_{gpu_count}gpu.sh")],
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
                    "PROFILE_TAG": f"test_{gpu_count}gpu",
                },
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            return json.loads(completed.stdout[completed.stdout.index("{\n") :]), completed.stdout

    def test_all_scale_profiles_match_wan21_offload_policy(self):
        for gpu_count in (1, 2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                config, output = self.run_config(gpu_count)
                self.assertTrue(config["cpu_offload"])
                self.assertEqual(config["offload_granularity"], "block")
                self.assertEqual(config["offload_ratio"], 1.0)
                self.assertTrue(config["t5_cpu_offload"])
                self.assertFalse(config["vae_cpu_offload"])
                self.assertFalse(config["use_tiling_vae"])
                self.assertEqual(config["parallel"]["seq_p_size"], gpu_count)
                self.assertEqual(config["parallel"]["vae_parallel"], gpu_count > 1)
                self.assertEqual(
                    config["parallel"]["seq_p_attn_type"],
                    "ring" if gpu_count == 16 else "ulysses",
                )
                self.assertEqual(config["self_attn_1_type"], "aiter_fav3_sage_bf16_attn")
                self.assertIn("VAE resident, tiling=0", output)

    def test_modelrun_selects_wan22_profile_wrapper_from_gpu_count(self):
        for gpu_count in (1, 2, 4, 8, 16):
            with self.subTest(gpu_count=gpu_count):
                completed = subprocess.run(
                    [str(MODEL_RUN), "--profile-wan22-ckconv3d-v1"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "DRY_RUN": "1",
                        "GPU_DEVICES": ",".join(str(index) for index in range(gpu_count)),
                        "RUN_REPO_DIR": str(REPO_ROOT),
                    },
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn(
                    f"run_wan22_ckconv3d_v1_profile_{gpu_count}gpu.sh",
                    completed.stdout,
                )
                self.assertIn("CKConv3dV1-Hot26-AOTFinal", completed.stdout)

    def test_planner_uses_rank_critical_path_without_summing_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root)
            output = root / "plan"
            completed = subprocess.run(
                [
                    "python3",
                    "-B",
                    str(SUMMARY),
                    "--inputs",
                    str(profile),
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with (output / "full_tune_contracts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                contracts = list(csv.DictReader(handle))
            with (output / "routes.csv").open(newline="", encoding="utf-8") as handle:
                routes = list(csv.DictReader(handle))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0]["calls"], "5")
        self.assertEqual(contracts[0]["flops_per_call"], "155520")
        self.assertEqual(contracts[0]["projected_flops"], "777600")
        self.assertEqual(contracts[0]["input_shape"], "1x3x4x6x8")
        self.assertEqual(contracts[0]["output_shape"], "1x5x4x6x8")
        self.assertIn("runtime_fallback", contracts[0]["full_tune_reasons"])
        self.assertEqual(len(routes), 4)
        self.assertEqual(manifest["kind"], "wan22_ck_conv3d_v1_scale_sweep_summary")
        self.assertEqual(manifest["world_sizes"], [2])
        self.assertEqual(manifest["contract_count"], 1)
        self.assertEqual(manifest["ck_route_count"], 7)
        self.assertEqual(manifest["fallback_route_count"], 3)

    def test_planner_rejects_incomplete_rank_phase_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root)
            (profile / "rank001_decode_000.json").unlink()
            manifest_path = profile / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["snapshots"].remove("rank001_decode_000.json")
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "python3",
                    "-B",
                    str(SUMMARY),
                    "--inputs",
                    str(profile),
                    "--output-dir",
                    str(root / "plan"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("missing rank/phase snapshots", completed.stderr)

    def test_summary_rejects_undeclared_snapshot_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _write_profile(root)
            source = profile / "rank000_encode_000.json"
            (profile / "rank000_encode_001.json").write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    "python3",
                    "-B",
                    str(SUMMARY),
                    "--inputs",
                    str(profile),
                    "--output-dir",
                    str(root / "summary"),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("snapshot manifest mismatch", completed.stderr)

    def test_scale_sweep_dry_run_does_not_require_profile_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                ["bash", str(SWEEP)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "MODEL_RUN": "/bin/true",
                    "SWEEP_COUNTS": "1,2",
                    "SWEEP_TAG": "dry_run_test",
                    "OUTPUT_DIR": str(root / "outputs"),
                    "RESULT_DIR": str(root / "results"),
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("scale sweep dry-run complete", completed.stdout)

    def test_scale_sweep_passes_explicit_image_override(self):
        image = "registry.example/lightx2v:wan22-ck-aot-final"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = subprocess.run(
                ["bash", str(SWEEP), "--image", image],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "DRY_RUN": "1",
                    "SWEEP_COUNTS": "1",
                    "SWEEP_TAG": "image_override_test",
                    "OUTPUT_DIR": str(root / "outputs"),
                    "RESULT_DIR": str(root / "results"),
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(f"image override: {image}", completed.stdout)
        self.assertIn(f"Image:          {image}", completed.stdout)


if __name__ == "__main__":
    unittest.main()
