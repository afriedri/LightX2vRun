from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_wan22_lazy_b1_scale.sh"
SUMMARIZER = REPO_ROOT / "scripts/summarize_wan22_lazy_b1_scale.py"


class Wan22LazyB1ScaleTest(unittest.TestCase):
    def test_launcher_dry_run_runs_2_4_8_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                ["bash", str(LAUNCHER)],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "IMAGE": "example.invalid/lightx2v:b1-fix2",
                    "WAN22_CACHE_STATE": "warm",
                    "OUTPUT_DIR": temporary,
                    "B1_SCALE_TAG": "b1_scale_test",
                    "MODEL_RUN": "/bin/true",
                    "DRY_RUN": "1",
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        for count in (2, 4, 8):
            self.assertIn(f"Starting B1 {count}-GPU arm", completed.stdout)
            self.assertIn(f"Experiment A GPU count: {count}", completed.stdout)
            self.assertIn(f"; {count} GPUs (", completed.stdout)
        self.assertIn("B1 scale dry-run complete", completed.stdout)

    @staticmethod
    def _write_scale_result(
        root: Path,
        physical_peak_bytes: int,
        physical_total_bytes: int = 32 * GIB,
    ) -> None:
        for count in (2, 4, 8):
            arm = root / f"{count}gpu"
            runtune = arm / "runtune"
            runtune.mkdir(parents=True)
            for rank in range(count):
                report = {
                    "schema_version": 2,
                    "backend": "wan22-lazy-full-offload-b1-warm-cache",
                    "environment": {"rank": rank, "world_size": count},
                    "steps": [
                        {"success": True, "wall_ms": 1000.0 + step}
                        for step in range(4)
                    ],
                    "scopes": [
                        {
                            "name": "transformer_lifecycle",
                            "success": True,
                            "wall_ms": 10_000.0 + rank,
                        },
                        {
                            "name": "request_end_to_end",
                            "success": True,
                            "wall_ms": 12_000.0 + rank,
                        },
                    ],
                    "summary": {
                        "cpu": {
                            "offload.host_fill": {
                                "calls": 156,
                                "details": [
                                    {"calls": 156, "detail": {"bytes": 10}}
                                ],
                            }
                        },
                        "gpu": {
                            "offload.initial_copy": {"calls": 0, "details": []},
                            "offload.prefetch_copy": {
                                "calls": 156,
                                "details": [
                                    {"calls": 156, "detail": {"bytes": 10}}
                                ],
                            },
                        },
                    },
                }
                (runtune / f"rank{rank:03d}.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )

            summary = {
                "backend_and_cache_state": "wan22-lazy-full-offload-b1-warm-cache",
                "legacy_four_step_denoising": {
                    "critical_path_seconds": 10.0
                },
                "runtuner_step_wall": {"critical_path_total_ms": 9_000.0},
                "scopes": {
                    "transformer_lifecycle": {"max_ms": 10_000.0 + count},
                    "request_end_to_end": {"max_ms": 12_000.0 + count},
                },
                "device_memory": {
                    str(rank): {
                        "max_allocator_allocated_bytes": 8 * GIB,
                        "max_allocator_reserved_bytes": 10 * GIB,
                    }
                    for rank in range(count)
                },
                "host_and_physical_memory": {
                    "physical_gpu_peak_used_bytes": {
                        f"card{rank}": physical_peak_bytes
                        for rank in range(count)
                    },
                    "physical_gpu_total_bytes": {
                        f"card{rank}": physical_total_bytes for rank in range(count)
                    },
                    "max_simultaneous_rank_sums_bytes": {
                        "Pss_bytes": count * GIB,
                        "Rss_bytes": count * 2 * GIB,
                    },
                    "max_observed_cgroup_current_bytes": count * GIB,
                },
            }
            (arm / "experiment_a_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (root / f"{count}gpu_720p121f_{count}gpu.mp4").write_bytes(
                b"same-output"
            )

    def _run_summarizer(self, root: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
        output = root / "b1_scale_summary.json"
        completed = subprocess.run(
            [
                "python3",
                str(SUMMARIZER),
                "--result-dir",
                str(root),
                "--output",
                str(output),
                "--counts",
                "2,4,8",
                "--gpu-budget-gib",
                "32",
                "--minimum-gpu-headroom-gib",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return completed, json.loads(output.read_text(encoding="utf-8"))

    def test_summarizer_accepts_complete_runs_with_memory_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_scale_result(root, physical_peak_bytes=30 * GIB)
            completed, summary = self._run_summarizer(root)

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue(summary["accepted"])
        self.assertTrue(summary["acceptance"]["all_gpu_memory_budgets"])
        self.assertEqual(
            summary["runs"]["2"]["memory"]["minimum_physical_headroom_bytes"],
            2 * GIB,
        )

    def test_summarizer_rejects_peak_inside_budget_but_below_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_scale_result(root, physical_peak_bytes=int(31.5 * GIB))
            completed, summary = self._run_summarizer(root)

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        self.assertFalse(summary["accepted"])
        self.assertFalse(summary["acceptance"]["all_gpu_memory_budgets"])
        self.assertFalse(summary["runs"]["2"]["memory"]["passed"])

    def test_summarizer_uses_measured_card_total_below_nominal_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_scale_result(
                root,
                physical_peak_bytes=int(30.9 * GIB),
                physical_total_bytes=int(31.5 * GIB),
            )
            completed, summary = self._run_summarizer(root)

        self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
        memory = summary["runs"]["2"]["memory"]
        self.assertFalse(memory["passed"])
        self.assertLess(memory["minimum_physical_headroom_bytes"], GIB)


if __name__ == "__main__":
    unittest.main()
