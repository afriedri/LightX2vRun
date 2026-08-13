from __future__ import annotations

import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/upload_wan22_memory_bundle.py"
SPEC = importlib.util.spec_from_file_location("upload_wan22_memory_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeS3Client:
    def __init__(self, size: int, digest: str) -> None:
        self.size = size
        self.digest = digest
        self.upload: tuple[object, ...] | None = None

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.upload = (filename, bucket, key, ExtraArgs)

    def head_object(self, Bucket, Key):
        return {"ContentLength": self.size, "Metadata": {"sha256": self.digest}}


class UploadWan22MemoryBundleTests(unittest.TestCase):
    def _result_dir(self, root: Path, workloads: tuple[str, ...] = ("720p121f_16gpu",)) -> Path:
        result = root / "wan22_memdiag_hot28_20260812_081324"
        result.mkdir()
        for workload in workloads:
            (result / f"{workload}.log").write_text("inference log\n", encoding="utf-8")
            memory = result / f"{workload}_memory.jsonl"
            memory.write_text('{"sample": "dit"}\n', encoding="utf-8")
            (result / f"{memory.name}.summary.json").write_text(
                '{"peak": 1}\n', encoding="utf-8"
            )
        (result / "unrelated.txt").write_text("ignore me\n", encoding="utf-8")
        return result

    def _experiment_arm(self, pair: Path, name: str, ranks: int = 2) -> Path:
        result = pair / name
        result.mkdir()
        workload = "720p121f_16gpu"
        (result / f"{workload}.log").write_text(
            "inference log\n", encoding="utf-8"
        )
        memory = result / f"{workload}_memory.jsonl"
        memory.write_text('{"sample": "dit"}\n', encoding="utf-8")
        (result / f"{memory.name}.summary.json").write_text(
            '{"peak": 1}\n', encoding="utf-8"
        )
        runtune = result / "runtune"
        runtune.mkdir()
        for rank in range(ranks):
            (runtune / f"rank{rank:03d}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": {"rank": rank, "world_size": ranks},
                    }
                ),
                encoding="utf-8",
            )
        (result / "experiment_a_summary.json").write_text(
            json.dumps({"arm": name}), encoding="utf-8"
        )
        (result / "experiment_a_summarizer.log").write_text(
            "summary complete\n", encoding="utf-8"
        )
        return result

    def test_build_archive_collects_complete_diagnostic_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root, ("480p121f_8gpu", "720p121f_16gpu"))
            archive_path, artifact_count = MODULE.build_archive(
                result, root / "bundle.tar.gz"
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                names = archive.getnames()

        self.assertEqual(artifact_count, 6)
        self.assertEqual(
            names,
            [
                f"{result.name}/480p121f_8gpu.log",
                f"{result.name}/480p121f_8gpu_memory.jsonl",
                f"{result.name}/480p121f_8gpu_memory.jsonl.summary.json",
                f"{result.name}/720p121f_16gpu.log",
                f"{result.name}/720p121f_16gpu_memory.jsonl",
                f"{result.name}/720p121f_16gpu_memory.jsonl.summary.json",
            ],
        )

    def test_archive_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            first, _ = MODULE.build_archive(result, root / "first.tar.gz")
            second, _ = MODULE.build_archive(result, root / "second.tar.gz")
            first_digest = MODULE.sha256_file(first)
            second_digest = MODULE.sha256_file(second)

        self.assertEqual(first_digest, second_digest)

    def test_experiment_a_archive_includes_rank_reports_and_summary_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            runtune = result / "runtune"
            runtune.mkdir()
            for rank in range(2):
                (runtune / f"rank{rank:03d}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "environment": {"rank": rank, "world_size": 2},
                        }
                    ),
                    encoding="utf-8",
                )
            (result / "experiment_a_summary.json").write_text(
                '{"kind": "wan22_lazy_experiment_a_summary"}\n',
                encoding="utf-8",
            )
            (result / "experiment_a_summarizer.log").write_text(
                "summary complete\n",
                encoding="utf-8",
            )

            archive_path, artifact_count = MODULE.build_archive(
                result,
                root / "experiment-a.tar.gz",
                experiment_a=True,
                expected_ranks=2,
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                names = archive.getnames()

        self.assertEqual(artifact_count, 7)
        self.assertIn(f"{result.name}/runtune/rank000.json", names)
        self.assertIn(f"{result.name}/runtune/rank001.json", names)
        self.assertIn(f"{result.name}/experiment_a_summary.json", names)
        self.assertIn(f"{result.name}/experiment_a_summarizer.log", names)

    def test_experiment_a_archive_rejects_incomplete_rank_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            runtune = result / "runtune"
            runtune.mkdir()
            (runtune / "rank000.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": {"rank": 0, "world_size": 2},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing/unexpected"):
                MODULE.build_archive(
                    result,
                    root / "experiment-a.tar.gz",
                    experiment_a=True,
                    expected_ranks=2,
                )

    def test_experiment_a_archive_allows_missing_final_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            runtune = result / "runtune"
            runtune.mkdir()
            (runtune / "rank000.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "environment": {"rank": 0, "world_size": 1},
                    }
                ),
                encoding="utf-8",
            )
            (result / "experiment_a_summarizer.log").write_text(
                "ValueError: missing request scope\n",
                encoding="utf-8",
            )

            archive_path, artifact_count = MODULE.build_archive(
                result,
                root / "failed-experiment-a.tar.gz",
                experiment_a=True,
                expected_ranks=1,
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                names = archive.getnames()

        self.assertEqual(artifact_count, 5)
        self.assertIn(f"{result.name}/experiment_a_summarizer.log", names)
        self.assertNotIn(f"{result.name}/experiment_a_summary.json", names)

    def test_b0_b1_archive_contains_both_complete_arms_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = root / "paired"
            pair.mkdir()
            self._experiment_arm(pair, "b0")
            self._experiment_arm(pair, "b1")
            (pair / "b0_b1_comparison.json").write_text(
                '{"accepted": true}\n', encoding="utf-8"
            )
            (pair / "b0_b1_comparison.log").write_text(
                "comparison complete\n", encoding="utf-8"
            )

            archive_path, artifact_count = MODULE.build_archive(
                pair,
                root / "paired.tar.gz",
                b0_b1=True,
                expected_ranks=2,
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                names = archive.getnames()

        self.assertEqual(artifact_count, 16)
        self.assertIn("paired/b0/runtune/rank000.json", names)
        self.assertIn("paired/b1/runtune/rank001.json", names)
        self.assertIn("paired/b0_b1_comparison.json", names)
        self.assertIn("paired/b0_b1_comparison.log", names)

    def test_missing_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            summary = result / "720p121f_16gpu_memory.jsonl.summary.json"
            summary.unlink()
            with self.assertRaisesRegex(ValueError, summary.name):
                MODULE.build_archive(result, root / "bundle.tar.gz")

    def test_upload_sets_and_verifies_sha256_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            archive_path, artifact_count = MODULE.build_archive(
                result, root / "bundle.tar.gz"
            )
            digest = MODULE.sha256_file(archive_path)
            client = FakeS3Client(archive_path.stat().st_size, digest)
            MODULE.upload_archive(
                client,
                archive_path,
                "bucket",
                "prefix/bundle.tar.gz",
                digest,
                artifact_count,
            )

        assert client.upload is not None
        self.assertEqual(client.upload[1:3], ("bucket", "prefix/bundle.tar.gz"))
        self.assertEqual(client.upload[3]["Metadata"]["sha256"], digest)
        self.assertEqual(client.upload[3]["Metadata"]["artifact-count"], "3")
        self.assertEqual(client.upload[3]["ContentType"], "application/gzip")

    def test_endpoint_without_scheme_defaults_to_http(self) -> None:
        self.assertEqual(
            MODULE.normalize_endpoint("compiler.hvat-ai.com:9000"),
            "http://compiler.hvat-ai.com:9000",
        )


if __name__ == "__main__":
    unittest.main()
