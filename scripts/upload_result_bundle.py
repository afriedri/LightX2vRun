#!/usr/bin/env python3
"""Deterministically package any result directory and upload it to S3/MinIO."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import stat
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


DEFAULT_ENDPOINT = "http://compiler.hvat-ai.com:9000"
DEFAULT_BUCKET = "aitertunecollection"
DEFAULT_PREFIX = "lightx2v/results"
DEFAULT_COMPRESSION_LEVEL = 1
MAX_PRESIGN_SECONDS = 7 * 24 * 60 * 60


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


def normalize_bucket(value: str) -> str:
    bucket = value.strip()
    if not bucket:
        raise ValueError("bucket must not be empty")
    if "/" in bucket or "://" in bucket:
        raise ValueError(f"bucket must be a name, not a path or URI: {value!r}")
    return bucket


def normalize_object_key(value: str) -> str:
    key = value.strip().strip("/")
    if not key:
        raise ValueError("object key must not be empty")
    if any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError(f"object key contains an invalid path component: {value!r}")
    return key


def object_key_for(archive_name: str, prefix: str, exact_key: str | None) -> str:
    if exact_key is not None:
        return normalize_object_key(exact_key)
    clean_prefix = prefix.strip().strip("/")
    if clean_prefix:
        normalize_object_key(clean_prefix)
        return f"{clean_prefix}/{archive_name}"
    return normalize_object_key(archive_name)


def parse_presign_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("presign seconds must be an integer") from exc
    if not 1 <= seconds <= MAX_PRESIGN_SECONDS:
        raise argparse.ArgumentTypeError(
            f"presign seconds must be between 1 and {MAX_PRESIGN_SECONDS}"
        )
    return seconds


def _relative_name(root: Path, path: Path) -> str:
    if path == root:
        return root.name
    relative = path.relative_to(root).as_posix()
    return f"{root.name}/{relative}"


def collect_entries(result_dir: Path) -> tuple[Path, list[Path], list[Path]]:
    requested = result_dir.expanduser()
    if requested.is_symlink():
        raise ValueError(f"result directory must not be a symbolic link: {requested}")
    try:
        root = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"result directory not found: {requested}") from exc
    if not root.is_dir():
        raise ValueError(f"result path is not a directory: {root}")
    if not root.name:
        raise ValueError("filesystem root cannot be packaged as a result directory")

    directories = [root]
    files: list[Path] = []
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic links are not supported in result bundles: {path}")
        if stat.S_ISDIR(mode):
            directories.append(path)
        elif stat.S_ISREG(mode):
            files.append(path)
        else:
            raise ValueError(f"special files are not supported in result bundles: {path}")
    if not files:
        raise ValueError(f"result directory contains no regular files: {root}")
    return root, directories, files


def _base_tar_info(arcname: str, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.mtime = 0
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_directory(archive: tarfile.TarFile, arcname: str) -> None:
    info = _base_tar_info(f"{arcname.rstrip('/')}/", 0o755)
    info.type = tarfile.DIRTYPE
    archive.addfile(info)


def _add_regular_file(archive: tarfile.TarFile, source: Path, arcname: str) -> int:
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"result entry changed type while packaging: {source}")
    mode = 0o755 if before.st_mode & 0o111 else 0o644
    info = _base_tar_info(arcname, mode)
    info.size = before.st_size
    with source.open("rb") as handle:
        archive.addfile(info, handle)
    after = source.lstat()
    if not stat.S_ISREG(after.st_mode) or (
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"result file changed while packaging: {source}")
    return before.st_size


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_archive(
    result_dir: Path,
    archive_path: Path,
    overwrite: bool = False,
    *,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> tuple[Path, int, int]:
    if not 0 <= compression_level <= 9:
        raise ValueError("compression level must be between 0 and 9")
    root, directories, files = collect_entries(result_dir)
    output = archive_path.expanduser().resolve()
    if _is_within(output, root):
        raise ValueError(f"archive must be outside the result directory: {output}")
    if output.exists() and not overwrite:
        raise ValueError(f"archive already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    source_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as raw:
            temporary_path = Path(raw.name)
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compression_level,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for directory in directories:
                        _add_directory(archive, _relative_name(root, directory))
                    for source in files:
                        source_bytes += _add_regular_file(
                            archive,
                            source,
                            _relative_name(root, source),
                        )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output, len(files), source_bytes


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
        raise RuntimeError(
            "boto3 is required for upload; install it with: python3 -m pip install boto3"
        ) from exc

    options: dict[str, Any] = {
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
            raise ValueError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set together"
            )
        options["aws_access_key_id"] = minio_access_key
        options["aws_secret_access_key"] = minio_secret_key
    return boto3.client("s3", **options)


def upload_archive(
    client: Any,
    archive_path: Path,
    bucket: str,
    object_key: str,
    digest: str,
    artifact_count: int,
) -> None:
    size = archive_path.stat().st_size
    metadata = {
        "sha256": digest,
        "artifact-count": str(artifact_count),
    }
    try:
        client.upload_file(
            str(archive_path),
            bucket,
            object_key,
            ExtraArgs={
                "ContentType": "application/gzip",
                "Metadata": metadata,
            },
        )
        uploaded = client.head_object(Bucket=bucket, Key=object_key)
    except Exception as exc:
        raise RuntimeError(f"S3 upload failed: {type(exc).__name__}: {exc}") from exc

    if int(uploaded.get("ContentLength", -1)) != size:
        raise RuntimeError("uploaded object size does not match the local archive")
    remote_metadata = uploaded.get("Metadata", {})
    if str(remote_metadata.get("sha256", "")) != digest:
        raise RuntimeError("uploaded object SHA256 metadata does not match the archive")
    if str(remote_metadata.get("artifact-count", "")) != str(artifact_count):
        raise RuntimeError("uploaded object artifact-count metadata does not match")


def create_download_url(
    client: Any,
    bucket: str,
    object_key: str,
    expires_in: int,
) -> str:
    try:
        return str(
            client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=expires_in,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"S3 presigned URL generation failed: {type(exc).__name__}: {exc}"
        ) from exc


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{value} B"
        amount /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="result directory to package recursively")
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
        default=os.environ.get("RESULT_BUNDLE_PREFIX", DEFAULT_PREFIX),
        help=f"destination key prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument("--object-key", help="exact destination object key; overrides --prefix")
    parser.add_argument("--archive", type=Path, help="local archive path (default: <result-dir>.tar.gz)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing custom --archive")
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(10),
        default=DEFAULT_COMPRESSION_LEVEL,
        metavar="0..9",
        help=f"gzip compression level (default: {DEFAULT_COMPRESSION_LEVEL})",
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="create and verify the archive without uploading it",
    )
    parser.add_argument(
        "--presign-seconds",
        type=parse_presign_seconds,
        help="print a temporary download URL after upload (maximum: 604800)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.package_only and args.presign_seconds is not None:
        print("ERROR: --presign-seconds cannot be used with --package-only", file=sys.stderr)
        return 1

    try:
        requested = args.result_dir.expanduser()
        result_dir = requested.resolve(strict=True)
        custom_archive = args.archive is not None
        archive_path = args.archive or result_dir.parent / f"{result_dir.name}.tar.gz"
        archive_path, artifact_count, source_bytes = build_archive(
            requested,
            archive_path,
            overwrite=args.overwrite or not custom_archive,
            compression_level=args.compression_level,
        )
        digest = sha256_file(archive_path)

        bucket = None
        object_key = None
        endpoint = None
        download_url = None
        if not args.package_only:
            bucket = normalize_bucket(args.bucket)
            object_key = object_key_for(
                archive_path.name,
                args.prefix,
                args.object_key,
            )
            endpoint = normalize_endpoint(args.endpoint)
            client = create_s3_client(endpoint)
            if args.presign_seconds is not None:
                # URL signing does not require the object to exist. Generate it
                # before upload so a signing failure cannot leave a successful
                # upload reported as an undifferentiated command failure.
                download_url = create_download_url(
                    client,
                    bucket,
                    object_key,
                    args.presign_seconds,
                )
            upload_archive(
                client,
                archive_path,
                bucket,
                object_key,
                digest,
                artifact_count,
            )
    except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Files        : {artifact_count}")
    print(f"Source bytes : {source_bytes} ({format_bytes(source_bytes)})")
    print(f"Archive      : {archive_path}")
    print(f"Archive bytes: {archive_path.stat().st_size} ({format_bytes(archive_path.stat().st_size)})")
    print(f"SHA256       : {digest}")
    if args.package_only:
        print("Uploaded     : skipped (--package-only)")
    else:
        assert bucket is not None and object_key is not None and endpoint is not None
        print(f"Uploaded     : s3://{bucket}/{object_key}")
        print(f"Endpoint     : {endpoint}")
        if download_url is not None:
            print(f"Download URL : {download_url}")
            print(f"URL expires  : {args.presign_seconds} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
