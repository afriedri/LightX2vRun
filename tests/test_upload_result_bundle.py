from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/upload_result_bundle.py"
SPEC = importlib.util.spec_from_file_location("upload_result_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeS3Client:
    def __init__(
        self,
        size: int,
        digest: str,
        artifact_count: int,
    ) -> None:
        self.size = size
        self.digest = digest
        self.artifact_count = artifact_count
        self.upload: tuple[object, ...] | None = None
        self.presign: tuple[object, ...] | None = None

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.upload = (filename, bucket, key, ExtraArgs)

    def head_object(self, Bucket, Key):
        return {
            "ContentLength": self.size,
            "Metadata": {
                "sha256": self.digest,
                "artifact-count": str(self.artifact_count),
            },
        }

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.presign = (operation, Params, ExpiresIn)
        return "http://minio.example/download"


class UploadResultBundleTests(unittest.TestCase):
    @staticmethod
    def _result_dir(root: Path) -> Path:
        result = root / "sample_result"
        result.mkdir()
        (result / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")
        (result / ".hidden").write_text("metadata\n", encoding="utf-8")
        nested = result / "nested"
        nested.mkdir()
        (nested / "memory.jsonl").write_text('{"rss": 1}\n', encoding="utf-8")
        (nested / "empty.txt").touch()
        (result / "empty-directory").mkdir()
        return result

    def test_build_archive_recursively_collects_the_complete_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            archive_path, artifact_count, source_bytes = MODULE.build_archive(
                result,
                root / "bundle.tar.gz",
            )
            expected_source_bytes = sum(
                path.stat().st_size for path in result.rglob("*") if path.is_file()
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                members = {member.name: member for member in archive.getmembers()}

        self.assertEqual(artifact_count, 4)
        self.assertEqual(
            set(members),
            {
                "sample_result",
                "sample_result/.hidden",
                "sample_result/empty-directory",
                "sample_result/nested",
                "sample_result/nested/empty.txt",
                "sample_result/nested/memory.jsonl",
                "sample_result/summary.json",
            },
        )
        self.assertTrue(members["sample_result/empty-directory"].isdir())
        self.assertEqual(members["sample_result/nested/empty.txt"].size, 0)
        self.assertEqual(source_bytes, expected_source_bytes)

    def test_archive_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            first, _, _ = MODULE.build_archive(result, root / "first.tar.gz")
            second, _, _ = MODULE.build_archive(result, root / "second.tar.gz")

            first_digest = MODULE.sha256_file(first)
            second_digest = MODULE.sha256_file(second)

        self.assertEqual(first_digest, second_digest)

    def test_executable_bit_is_normalized_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            executable = result / "reproduce.sh"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o751)

            archive_path, _, _ = MODULE.build_archive(result, root / "bundle.tar.gz")
            with tarfile.open(archive_path, "r:gz") as archive:
                member = archive.getmember("sample_result/reproduce.sh")

        self.assertEqual(member.mode, 0o755)

    def test_symbolic_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            os.symlink(root / "outside", result / "external")

            with self.assertRaisesRegex(ValueError, "symbolic links"):
                MODULE.build_archive(result, root / "bundle.tar.gz")

    def test_archive_inside_result_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)

            with self.assertRaisesRegex(ValueError, "outside the result directory"):
                MODULE.build_archive(result, result / "bundle.tar.gz")

    def test_existing_custom_archive_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            output = root / "bundle.tar.gz"
            output.write_bytes(b"keep")

            with self.assertRaisesRegex(ValueError, "--overwrite"):
                MODULE.build_archive(result, output)
            self.assertEqual(output.read_bytes(), b"keep")

    def test_upload_sets_and_verifies_integrity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            archive_path, artifact_count, _ = MODULE.build_archive(
                result,
                root / "bundle.tar.gz",
            )
            digest = MODULE.sha256_file(archive_path)
            client = FakeS3Client(
                archive_path.stat().st_size,
                digest,
                artifact_count,
            )
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
        self.assertEqual(
            client.upload[3]["Metadata"]["artifact-count"],
            str(artifact_count),
        )
        self.assertEqual(client.upload[3]["ContentType"], "application/gzip")

    def test_upload_rejects_remote_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            archive_path, artifact_count, _ = MODULE.build_archive(
                result,
                root / "bundle.tar.gz",
            )
            digest = MODULE.sha256_file(archive_path)
            client = FakeS3Client(
                archive_path.stat().st_size,
                "wrong",
                artifact_count,
            )

            with self.assertRaisesRegex(RuntimeError, "SHA256 metadata"):
                MODULE.upload_archive(
                    client,
                    archive_path,
                    "bucket",
                    "prefix/bundle.tar.gz",
                    digest,
                    artifact_count,
                )

    def test_download_url_is_scoped_to_uploaded_object(self) -> None:
        client = FakeS3Client(1, "digest", 1)
        url = MODULE.create_download_url(
            client,
            "bucket",
            "prefix/bundle.tar.gz",
            3600,
        )

        self.assertEqual(url, "http://minio.example/download")
        self.assertEqual(
            client.presign,
            (
                "get_object",
                {"Bucket": "bucket", "Key": "prefix/bundle.tar.gz"},
                3600,
            ),
        )

    def test_destination_normalization(self) -> None:
        self.assertEqual(
            MODULE.normalize_endpoint("compiler.hvat-ai.com:9000"),
            "http://compiler.hvat-ai.com:9000",
        )
        self.assertEqual(
            MODULE.object_key_for("run.tar.gz", "/minimax-h3/day0/", None),
            "minimax-h3/day0/run.tar.gz",
        )
        self.assertEqual(
            MODULE.object_key_for(
                "ignored.tar.gz",
                "ignored",
                "/exact/run.tar.gz/",
            ),
            "exact/run.tar.gz",
        )

    def test_package_only_cli_does_not_require_boto3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._result_dir(root)
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"S3_BUCKET": "", "S3_ENDPOINT_URL": ""},
            ):
                with contextlib.redirect_stdout(output):
                    status = MODULE.main([str(result), "--package-only"])

            archive_path = root / "sample_result.tar.gz"
            self.assertEqual(status, 0)
            self.assertTrue(archive_path.is_file())
            self.assertIn("Uploaded     : skipped", output.getvalue())


if __name__ == "__main__":
    unittest.main()
