#!/usr/bin/env python3
"""Package Wan2.2 memory diagnostics and upload them to S3/MinIO."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


DEFAULT_ENDPOINT = "http://compiler.hvat-ai.com:9000"
DEFAULT_BUCKET = "aitertunecollection"
DEFAULT_PREFIX = "wan22/memory-diagnostics"
EXPERIMENT_A_PREFIX = "wan22/experiment-a"
B0_B1_PREFIX = "wan22/experiment-b0-b1"
MEMORY_SUFFIX = "_memory.jsonl"
SUMMARY_SUFFIX = ".summary.json"


def normalize_endpoint(value: str) -> str:
    endpoint = value.strip()
    if not endpoint:
        raise ValueError("S3 endpoint must not be empty")
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid S3 endpoint: {value!r}")
    return endpoint.rstrip("/")


def diagnostic_paths(result_dir: Path) -> tuple[Path, list[Path]]:
    result_dir = result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise ValueError(f"result directory not found: {result_dir}")

    memory_files = sorted(result_dir.glob(f"*{MEMORY_SUFFIX}"))
    if not memory_files:
        raise ValueError(f"no *{MEMORY_SUFFIX} files found in {result_dir}")

    artifacts: list[Path] = []
    missing: list[str] = []
    empty: list[str] = []
    for memory_path in memory_files:
        workload = memory_path.name[: -len(MEMORY_SUFFIX)]
        group = (
            result_dir / f"{workload}.log",
            memory_path,
            result_dir / f"{memory_path.name}{SUMMARY_SUFFIX}",
        )
        for path in group:
            if not path.is_file():
                missing.append(path.name)
            elif path.stat().st_size == 0:
                empty.append(path.name)
            else:
                artifacts.append(path)
    if missing:
        raise ValueError(f"missing diagnostic artifacts: {', '.join(missing)}")
    if empty:
        raise ValueError(f"empty diagnostic artifacts: {', '.join(empty)}")
    return result_dir, artifacts


def experiment_a_paths(
    result_dir: Path, expected_ranks: int
) -> tuple[Path, list[Path]]:
    result_dir, artifacts = diagnostic_paths(result_dir)
    runtune_dir = result_dir / "runtune"
    if not runtune_dir.is_dir():
        raise ValueError(f"Experiment-A runtune directory not found: {runtune_dir}")

    reports: dict[int, Path] = {}
    for path in sorted(runtune_dir.glob("rank*.json")):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"empty Experiment-A rank report: {path}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            environment = report.get("environment", {})
            rank = int(environment.get("rank", -1))
            world_size = int(environment.get("world_size", -1))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Experiment-A rank report {path}: {exc}") from exc
        if report.get("schema_version") != 2:
            raise ValueError(f"{path} is not an Experiment-A schema-2 report")
        if world_size != expected_ranks:
            raise ValueError(
                f"{path} world_size={world_size}, expected {expected_ranks}"
            )
        if path.name != f"rank{rank:03d}.json":
            raise ValueError(f"rank report filename does not match rank {rank}: {path}")
        if rank in reports:
            raise ValueError(f"duplicate Experiment-A rank {rank}: {path}")
        reports[rank] = path

    expected = set(range(expected_ranks))
    if set(reports) != expected:
        raise ValueError(
            "missing/unexpected Experiment-A ranks: "
            f"expected {sorted(expected)}, got {sorted(reports)}"
        )
    artifacts.extend(reports[rank] for rank in sorted(reports))

    for optional_name in (
        "experiment_a_summary.json",
        "experiment_a_summarizer.log",
    ):
        path = result_dir / optional_name
        if path.exists():
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"empty Experiment-A artifact: {path}")
            artifacts.append(path)
    return result_dir, artifacts


def b0_b1_paths(
    result_dir: Path, expected_ranks: int
) -> tuple[Path, list[Path]]:
    result_dir = result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise ValueError(f"B0/B1 result directory not found: {result_dir}")

    artifacts: list[Path] = []
    for arm in ("b0", "b1"):
        _, arm_artifacts = experiment_a_paths(result_dir / arm, expected_ranks)
        artifacts.extend(arm_artifacts)

    for required_name in ("b0_b1_comparison.json", "b0_b1_comparison.log"):
        path = result_dir / required_name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty B0/B1 artifact: {path}")
        artifacts.append(path)
    return result_dir, artifacts


def _add_regular_file(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = source.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def build_archive(
    result_dir: Path,
    archive_path: Path,
    overwrite: bool = False,
    *,
    experiment_a: bool = False,
    b0_b1: bool = False,
    expected_ranks: int = 16,
) -> tuple[Path, int]:
    if experiment_a and b0_b1:
        raise ValueError("experiment_a and b0_b1 archive modes are mutually exclusive")
    if b0_b1:
        result_dir, artifacts = b0_b1_paths(result_dir, expected_ranks)
    elif experiment_a:
        result_dir, artifacts = experiment_a_paths(result_dir, expected_ranks)
    else:
        result_dir, artifacts = diagnostic_paths(result_dir)
    archive_path = archive_path.expanduser().resolve()
    if archive_path.exists() and not overwrite:
        raise ValueError(f"archive already exists: {archive_path}; pass --overwrite to replace it")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            dir=archive_path.parent,
            delete=False,
        ) as raw:
            temporary_path = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for source in artifacts:
                        _add_regular_file(
                            archive,
                            source,
                            f"{result_dir.name}/{source.relative_to(result_dir).as_posix()}",
                        )
        os.replace(temporary_path, archive_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return archive_path, len(artifacts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_s3_client(endpoint: str) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required; install it with: python3 -m pip install boto3") from exc

    client_options: dict[str, Any] = {
        "endpoint_url": endpoint,
        "region_name": os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1",
        "config": Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    }
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY")
    if minio_access_key or minio_secret_key:
        if not (minio_access_key and minio_secret_key):
            raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set together")
        client_options["aws_access_key_id"] = minio_access_key
        client_options["aws_secret_access_key"] = minio_secret_key
    return boto3.client("s3", **client_options)


def upload_archive(
    client: Any,
    archive_path: Path,
    bucket: str,
    object_key: str,
    digest: str,
    artifact_count: int,
) -> None:
    size = archive_path.stat().st_size
    metadata = {"sha256": digest, "artifact-count": str(artifact_count)}
    try:
        client.upload_file(
            str(archive_path),
            bucket,
            object_key,
            ExtraArgs={"ContentType": "application/gzip", "Metadata": metadata},
        )
        uploaded = client.head_object(Bucket=bucket, Key=object_key)
    except Exception as exc:
        raise RuntimeError(f"S3 upload failed: {type(exc).__name__}: {exc}") from exc
    if int(uploaded.get("ContentLength", -1)) != size:
        raise RuntimeError("uploaded object size does not match the local archive")
    uploaded_digest = str(uploaded.get("Metadata", {}).get("sha256", ""))
    if uploaded_digest != digest:
        raise RuntimeError("uploaded object SHA256 metadata does not match the local archive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="Wan2.2 memory diagnostic result directory")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("S3_ENDPOINT_URL", DEFAULT_ENDPOINT),
        help=f"S3-compatible endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET", DEFAULT_BUCKET),
        help=f"destination bucket (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("S3_PREFIX", DEFAULT_PREFIX),
        help=f"object key prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument("--object-key", help="exact destination object key")
    parser.add_argument("--archive", type=Path, help="local archive path")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --archive path")
    archive_mode = parser.add_mutually_exclusive_group()
    archive_mode.add_argument(
        "--experiment-a",
        action="store_true",
        help="also require and package the complete per-rank Experiment-A reports",
    )
    archive_mode.add_argument(
        "--b0-b1",
        action="store_true",
        help="package paired b0/b1 report directories and their comparison",
    )
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=16,
        help="expected Experiment-A rank count (default: 16)",
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="build and verify the local archive without uploading it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result_dir = args.result_dir.expanduser().resolve()
        custom_archive = args.archive is not None
        if args.b0_b1:
            archive_suffix = "_b0_b1.tar.gz"
        elif args.experiment_a:
            archive_suffix = "_experiment_a.tar.gz"
        else:
            archive_suffix = "_memory_diagnostics.tar.gz"
        archive_path = args.archive or result_dir.parent / f"{result_dir.name}{archive_suffix}"
        archive_path, artifact_count = build_archive(
            result_dir,
            archive_path,
            overwrite=args.overwrite or not custom_archive,
            experiment_a=args.experiment_a,
            b0_b1=args.b0_b1,
            expected_ranks=args.expected_ranks,
        )
        digest = sha256_file(archive_path)

        bucket = args.bucket.strip()
        if not bucket:
            raise ValueError("bucket must not be empty")
        prefix = args.prefix.strip().strip("/")
        if args.b0_b1 and args.prefix == DEFAULT_PREFIX:
            prefix = B0_B1_PREFIX
        elif args.experiment_a and args.prefix == DEFAULT_PREFIX:
            prefix = EXPERIMENT_A_PREFIX
        object_key = (
            args.object_key.strip().strip("/")
            if args.object_key is not None
            else "/".join(
                part for part in (prefix, result_dir.name, archive_path.name) if part
            )
        )
        if not object_key:
            raise ValueError("object key must not be empty")

        endpoint = normalize_endpoint(args.endpoint)
        if not args.package_only:
            client = create_s3_client(endpoint)
            upload_archive(
                client,
                archive_path,
                bucket,
                object_key,
                digest,
                artifact_count,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Artifacts: {artifact_count}")
    print(f"Archive  : {archive_path}")
    print(f"SHA256   : {digest}")
    if args.package_only:
        print("Uploaded : skipped (--package-only)")
    else:
        print(f"Uploaded : s3://{bucket}/{object_key}")
        print(f"Endpoint : {endpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
