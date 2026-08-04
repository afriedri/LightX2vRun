import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/benchmark_wan21_vae_t0.py"
SPEC = importlib.util.spec_from_file_location("benchmark_wan21_vae_t0", MODULE_PATH)
T0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T0)


def fake_result(arm, gpu_id, encode_seconds, decode_seconds):
    def phase(seconds, cosine=1.0):
        return {
            "measurement": {
                "seconds": seconds,
                "peak_allocated_mib": 1000.0 + seconds,
                "peak_reserved_mib": 1100.0 + seconds,
            },
            "vs_legacy": {"cosine": cosine, "max_abs": 0.0},
        }

    return {
        "arm": arm,
        "gpu_id": str(gpu_id),
        "flags": {
            "miopen": arm != "legacy",
            "tiling": arm in {"legacy", "miopen"},
            "vae_cpu_offload": arm != "optimized",
            "channels_last_3d": False,
        },
        "model_load": {"seconds": 1.0},
        "rounds": [
            {
                "round": 0,
                "encode": phase(encode_seconds),
                "decode": phase(decode_seconds),
            }
        ],
    }


class Wan21VaeT0AggregationTest(unittest.TestCase):
    def test_aggregate_reports_incremental_speedups_and_strict_json(self):
        timings = {
            "legacy": (60.0, 90.0),
            "miopen": (30.0, 30.0),
            "miopen_untiled": (15.0, 10.0),
            "optimized": (10.0, 5.0),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for gpu_id, arm in enumerate(T0.ARM_ORDER):
                result_dir = root / f"gpu{gpu_id}"
                result_dir.mkdir()
                (result_dir / "result.json").write_text(
                    json.dumps(fake_result(arm, gpu_id, *timings[arm])),
                    encoding="utf-8",
                )

            summary = T0.aggregate(root)

            self.assertEqual(summary["comparisons"]["miopen_only_vs_legacy"]["decode_speedup"], 3.0)
            self.assertEqual(summary["comparisons"]["optimized_vs_legacy"]["vae_total_speedup"], 10.0)
            json.loads((root / "summary.json").read_text(encoding="utf-8"), parse_constant=self.fail)
            self.assertTrue((root / "summary.csv").is_file())

    def test_aggregate_requires_all_four_arms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "gpu0"
            result_dir.mkdir()
            (result_dir / "result.json").write_text(
                json.dumps(fake_result("legacy", 0, 1.0, 1.0)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected T0 arms"):
                T0.aggregate(root)


if __name__ == "__main__":
    unittest.main()
