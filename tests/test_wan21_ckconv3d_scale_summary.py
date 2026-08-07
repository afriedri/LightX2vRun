import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = REPO_ROOT / "scripts/summarize_wan21_ckconv3d_scale.py"


class Wan21CKConv3dScaleSummaryTest(unittest.TestCase):
    def test_distributed_critical_path_and_csv_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "2gpu.log"
            log.write_text(
                "\n".join(
                    [
                        "[Profile] Rank 0 - Level1_Log Run Image Encoder cost 1.000000 seconds",
                        "[Profile] Rank 1 - Level1_Log Run Image Encoder cost 1.100000 seconds",
                        "[Profile] Rank 0 - Level1_Log Run VAE Encoder cost 10.000000 seconds",
                        "[Profile] Rank 1 - Level1_Log Run VAE Encoder cost 9.000000 seconds",
                        "[Profile] Rank 0 - Level1_Log Run Text Encoder cost 0.500000 seconds",
                        "[Profile] Rank 1 - Level1_Log Run Text Encoder cost 0.600000 seconds",
                        "[Profile] Rank 0 - Level2_Log Run Encoders cost 12.000000 seconds",
                        "[Profile] Rank 1 - Level2_Log Run Encoders cost 12.500000 seconds",
                        "[Profile] Rank 0 - Level1_Log Run VAE Decoder cost 8.000000 seconds",
                        "[Profile] Rank 1 - Level1_Log Run VAE Decoder cost 8.500000 seconds",
                        "[Profile] Rank 0 - Level1_Log RUN pipeline cost 42.000000 seconds",
                        "[Profile] Rank 1 - Level1_Log RUN pipeline cost 43.000000 seconds",
                        "[Profile] Rank 0 - Level1_Log Total Cost cost 50.000000 seconds",
                        "[Profile] Rank 1 - Level1_Log Total Cost cost 51.000000 seconds",
                        "DiT summary: status=complete, steps=4/4, total=25.000s, average=6.250s/it, throughput=0.1600it/s",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(SUMMARY_SCRIPT),
                    "--log-dir",
                    str(root),
                    "--counts",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with (root / "timings.csv").open(newline="", encoding="utf-8") as source:
                rows = {row["metric"]: row for row in csv.DictReader(source)}

        self.assertEqual(list(rows["Image Encoder"]), ["metric", "1", "2", "4", "8", "16", "unit"])
        self.assertEqual(rows["Image Encoder"]["2"], "1.100000")
        self.assertEqual(rows["Encoders 合计"]["2"], "12.500000")
        self.assertEqual(rows["DiT 4 steps"]["2"], "25.000000")
        self.assertEqual(rows["DiT 每步平均"]["2"], "6.250000")
        self.assertEqual(rows["VAE Decoder"]["2"], "8.500000")
        self.assertEqual(rows["RUN pipeline"]["2"], "43.000000")
        self.assertEqual(rows["Total Cost"]["2"], "51.000000")
        self.assertIn("| metric | 1 | 2 | 4 | 8 | 16 |", result.stdout)


if __name__ == "__main__":
    unittest.main()
