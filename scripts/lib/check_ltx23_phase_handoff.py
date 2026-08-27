#!/usr/bin/env python3
"""Gate LTX DiT startup on Gemma-process exit and physical VRAM release."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lightx2v.ltx23.phase_handoff"
LTX25_SCHEMA_NAME = "lightx2v.ltx25.phase_handoff"
SCHEMA_VERSION = 1
PROMPT_CACHE_SCHEMAS = {
    "lightx2v.ltx23.prompt_embeddings": ("1", SCHEMA_NAME, "2.3"),
    "lightx2v.ltx2.prompt_embeddings": ("2", LTX25_SCHEMA_NAME, "2.5"),
}


def parse_indices(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers")
    result = tuple(int(item) for item in value.split(","))
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return result


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def sample_amd_gpus(sysfs_root: Path) -> list[dict[str, Any]]:
    indexed = []
    for path in sysfs_root.glob("card*"):
        match = re.fullmatch(r"card([0-9]+)", path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    result = []
    for _, card_path in sorted(indexed):
        device = card_path / "device"
        if _read_int(device / "vendor") != 0x1002:
            continue
        total = _read_int(device / "mem_info_vram_total")
        used = _read_int(device / "mem_info_vram_used")
        if total is None or used is None:
            continue
        result.append(
            {
                "rocm_index": len(result),
                "card": card_path.name,
                "used_bytes": used,
                "total_bytes": total,
            }
        )
    return result


def prompt_cache_metadata(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= 16 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header size {header_size}")
            header = json.loads(stream.read(header_size))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cannot read prompt-cache metadata from {path}: {exc}") from exc
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise RuntimeError("prompt cache has no safetensors metadata")
    schema_name = metadata.get("schema_name")
    if schema_name not in PROMPT_CACHE_SCHEMAS:
        raise RuntimeError("prompt cache has the wrong schema")
    expected_version, _, _ = PROMPT_CACHE_SCHEMAS[schema_name]
    if metadata.get("schema_version") != expected_version:
        raise RuntimeError("prompt cache has the wrong schema version")
    if metadata.get("process_isolated") != "true":
        raise RuntimeError("prompt cache does not declare process isolation")
    return metadata


def _pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def wait_for_handoff(
    *,
    gpu_indices: tuple[int, ...],
    prompt_cache: Path,
    limit_bytes: int,
    timeout_seconds: float,
    interval_seconds: float,
    sysfs_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any]:
    metadata = prompt_cache_metadata(prompt_cache)
    _, handoff_schema, model_generation = PROMPT_CACHE_SCHEMAS[
        metadata["schema_name"]
    ]
    try:
        producer_pid = int(metadata["producer_pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("prompt cache has no valid producer_pid") from exc
    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    final_gpus: list[dict[str, Any]] = []
    missing: list[int] = []
    producer_alive = True
    while True:
        attempts += 1
        by_index = {
            item["rocm_index"]: item for item in sample_amd_gpus(sysfs_root)
        }
        missing = [index for index in gpu_indices if index not in by_index]
        final_gpus = [by_index[index] for index in gpu_indices if index in by_index]
        producer_alive = _pid_alive(producer_pid)
        released = (
            not missing
            and not producer_alive
            and all(item["used_bytes"] <= limit_bytes for item in final_gpus)
        )
        now = time.monotonic()
        if released or now >= deadline:
            break
        time.sleep(min(interval_seconds, max(0.0, deadline - now)))
    return {
        "schema_name": handoff_schema,
        "schema_version": SCHEMA_VERSION,
        "model_generation": model_generation,
        "status": "ok" if released else "failed",
        "producer_pid": producer_pid,
        "producer_pid_alive": producer_alive,
        "gpu_indices": list(gpu_indices),
        "missing_gpu_indices": missing,
        "vram_release_limit_bytes": limit_bytes,
        "gpus": final_gpus,
        "attempts": attempts,
        "wait_seconds": time.monotonic() - started,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-indices", type=parse_indices, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-mib", type=float, default=1024.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument(
        "--sysfs-root", type=Path, default=Path("/sys/class/drm"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.limit_mib <= 0 or args.timeout < 0 or args.interval <= 0:
        parser.error("limit-mib and interval must be positive; timeout must be non-negative")
    try:
        report = wait_for_handoff(
            gpu_indices=args.gpu_indices,
            prompt_cache=args.prompt_cache,
            limit_bytes=round(args.limit_mib * 1024**2),
            timeout_seconds=args.timeout,
            interval_seconds=args.interval,
            sysfs_root=args.sysfs_root,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, report)
    for gpu in report["gpus"]:
        print(
            f"LTX-{report['model_generation']} handoff GPU "
            f"{gpu['rocm_index']}: {gpu['used_bytes'] / 1024**2:.1f} MiB used "
            f"(limit {report['vram_release_limit_bytes'] / 1024**2:.1f} MiB)"
        )
    if report["status"] != "ok":
        print(
            "ERROR: Gemma process/VRAM was not released before the DiT deadline",
            file=sys.stderr,
        )
        return 1
    print(
        f"LTX-{report['model_generation']} handoff PASS: producer PID "
        f"{report['producer_pid']} exited "
        "and physical VRAM is available for fresh SP2 ranks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
