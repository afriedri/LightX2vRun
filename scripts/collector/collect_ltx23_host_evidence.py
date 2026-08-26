#!/usr/bin/env python3
"""Capture read-only host evidence around an LTX-2.3 FP8 B0 run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lightx2v.ltx23_fp8.b0.host"
SCHEMA_VERSION = 1


def run(argv: list[str], timeout: float = 20.0) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def repository(path: Path) -> dict[str, Any]:
    revision = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    branch = run(["git", "-C", str(path), "branch", "--show-current"])
    status = run(["git", "-C", str(path), "status", "--short"])
    return {
        "path": str(path),
        "revision": revision.get("stdout") if revision.get("returncode") == 0 else None,
        "branch": branch.get("stdout") if branch.get("returncode") == 0 else None,
        "dirty": bool(status.get("stdout")) if status.get("returncode") == 0 else None,
    }


def image(reference: str) -> dict[str, Any]:
    result = run(["docker", "image", "inspect", "--format", "{{json .}}", reference])
    if result.get("returncode") != 0:
        return {"reference": reference, "probe": result}
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        return {"reference": reference, "error": f"invalid docker inspect JSON: {exc}"}
    return {
        "reference": reference,
        "id": payload.get("Id"),
        "repo_digests": payload.get("RepoDigests") or [],
        "created": payload.get("Created"),
        "size_bytes": payload.get("Size"),
    }


def path_evidence(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return item
    stat = path.stat()
    filesystem = os.statvfs(path)
    item.update(
        {
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": stat.st_size if path.is_file() else None,
            "device_major": os.major(stat.st_dev),
            "device_minor": os.minor(stat.st_dev),
            "filesystem_available_bytes": filesystem.f_bavail * filesystem.f_frsize,
        }
    )
    return item


def gpu_snapshot(indices: tuple[int, ...]) -> dict[str, Any]:
    result = run(["rocm-smi", "--showproductname", "--showuniqueid", "--showmeminfo", "vram", "--json"])
    return {"selected_rocm_indices": list(indices), "rocm_smi": result}


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU ids must be comma-separated integers") from exc
    if len(result) != 2 or len(set(result)) != 2:
        raise argparse.ArgumentTypeError("LTX-2.3 B0 requires exactly two unique GPU ids")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after", "failure"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpus, required=True)
    args = parser.parse_args()
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "meminfo": Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace"),
        },
        "run_repository": repository(args.run_root),
        "image": image(args.image),
        "gpu": gpu_snapshot(args.gpu_indices),
        "artifacts": {
            "checkpoint": path_evidence(args.checkpoint),
            "upscaler": path_evidence(args.upscaler),
            "gemma": path_evidence(args.gemma_dir),
        },
    }
    write_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
