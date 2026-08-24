#!/usr/bin/env python3
"""Capture read-only host evidence for a MiniMax-H3 SuperCollect run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitor_run_memory import (
    parse_gpu_indices,
    read_optional_int,
    read_optional_text,
    sample_diskstats,
    sample_gpus,
    sample_host_memory,
    sample_pressure,
    sample_vmstat,
)


SCHEMA_NAME = "lightx2v.minimax_h3.sp8_supercollect.host"
SCHEMA_VERSION = 1


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def parse_path_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", label) or not raw_path:
        raise argparse.ArgumentTypeError("path must use LABEL=/absolute/path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("evidence paths must be absolute")
    return label, path


def run_command(argv: list[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    return payload


def snapshot_repository(path: Path) -> dict[str, Any]:
    revision = run_command(["git", "-C", str(path), "rev-parse", "HEAD"])
    branch = run_command(["git", "-C", str(path), "branch", "--show-current"])
    status = run_command(["git", "-C", str(path), "status", "--short"])
    return {
        "path": str(path),
        "revision": revision.get("stdout") if revision.get("returncode") == 0 else None,
        "branch": branch.get("stdout") if branch.get("returncode") == 0 else None,
        "dirty": bool(status.get("stdout")) if status.get("returncode") == 0 else None,
        "errors": [
            item
            for item in (revision, branch, status)
            if item.get("returncode", 0) != 0 or item.get("error")
        ],
    }


def snapshot_image(image: str) -> dict[str, Any]:
    template = "{{json .}}"
    result = run_command(["docker", "image", "inspect", "--format", template, image])
    if result.get("returncode") != 0:
        return {"reference": image, "probe": result}
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        return {"reference": image, "probe": result, "error": f"invalid inspect JSON: {exc}"}
    return {
        "reference": image,
        "id": payload.get("Id"),
        "repo_digests": payload.get("RepoDigests") or [],
        "created": payload.get("Created"),
        "architecture": payload.get("Architecture"),
        "os": payload.get("Os"),
        "size_bytes": payload.get("Size"),
    }


def snapshot_paths(paths: list[tuple[str, Path]]) -> dict[str, Any]:
    result = {}
    for label, path in paths:
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            filesystem = os.statvfs(path)
            item.update(
                {
                    "device_major": os.major(stat.st_dev),
                    "device_minor": os.minor(stat.st_dev),
                    "filesystem_total_bytes": filesystem.f_blocks * filesystem.f_frsize,
                    "filesystem_available_bytes": filesystem.f_bavail * filesystem.f_frsize,
                }
            )
            mount = run_command(["findmnt", "--json", "--target", str(path)])
            if mount.get("returncode") == 0:
                try:
                    item["mount"] = json.loads(mount["stdout"])
                except json.JSONDecodeError:
                    item["mount_probe"] = mount
            else:
                item["mount_probe"] = mount
        result[label] = item
    return result


def snapshot_numa_nodes() -> dict[str, Any]:
    result = {}
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        result[node_path.name] = {
            "cpulist": read_text(node_path / "cpulist"),
            "distance": read_text(node_path / "distance"),
            "meminfo": read_text(node_path / "meminfo"),
            "numastat": read_text(node_path / "numastat"),
        }
    return result


def enrich_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    item = dict(gpu)
    pci_address = item.get("pci_address")
    if not pci_address:
        return item
    device = Path("/sys/bus/pci/devices") / str(pci_address)
    item["pci"] = {
        "numa_node": read_optional_int(device / "numa_node"),
        "local_cpulist": read_optional_text(device / "local_cpulist"),
        "current_link_speed": read_optional_text(device / "current_link_speed"),
        "current_link_width": read_optional_text(device / "current_link_width"),
        "max_link_speed": read_optional_text(device / "max_link_speed"),
        "max_link_width": read_optional_text(device / "max_link_width"),
    }
    return item


def snapshot_block_devices() -> dict[str, Any]:
    result = {}
    for path in sorted(Path("/sys/block").glob("*")):
        result[path.name] = {
            "device_model": read_optional_text(path / "device/model"),
            "device_numa_node": read_optional_int(path / "device/numa_node"),
            "rotational": read_optional_int(path / "queue/rotational"),
            "logical_block_size": read_optional_int(path / "queue/logical_block_size"),
            "physical_block_size": read_optional_int(path / "queue/physical_block_size"),
            "read_ahead_kb": read_optional_int(path / "queue/read_ahead_kb"),
            "scheduler": read_optional_text(path / "queue/scheduler"),
        }
    return result


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


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    selected = set(args.gpu_indices)
    all_gpus = sample_gpus()
    selected_gpus = [enrich_gpu(gpu) for gpu in all_gpus if gpu["rocm_index"] in selected]
    observed = {gpu["rocm_index"] for gpu in selected_gpus}
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "read_only": True,
        "workload_contract": {
            "model": "minimax_h3",
            "parallel": "ulysses_sp8",
            "width": 1344,
            "height": 768,
            "frames": 124,
            "fps": 24,
            "dit_dtype": "BF16",
            "dit_quant_scheme": "Default",
            "lazy_load": True,
            "cpu_block_buffers": 2,
            "gpu_block_buffers": 2,
        },
        "host": {
            "hostname": socket.gethostname(),
            "kernel": platform.release(),
            "platform": platform.platform(),
            "boot_id": read_text(Path("/proc/sys/kernel/random/boot_id")),
            "process_affinity": sorted(os.sched_getaffinity(0)),
            "memory": sample_host_memory(),
            "vmstat": sample_vmstat(),
            "pressure": sample_pressure(),
            "diskstats": sample_diskstats(),
            "numa_nodes": snapshot_numa_nodes(),
            "block_devices": snapshot_block_devices(),
        },
        "gpus": {
            "requested_rocm_indices": list(args.gpu_indices),
            "observed_rocm_indices": sorted(observed),
            "missing_rocm_indices": sorted(selected - observed),
            "devices": selected_gpus,
        },
        "paths": snapshot_paths(args.path),
        "container_image": snapshot_image(args.image),
        "run_repository": snapshot_repository(args.run_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--path", type=parse_path_spec, action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.gpu_indices) != 8:
        raise SystemExit("SP8 SuperCollect requires exactly eight GPU indices")
    write_atomic(args.output, build_snapshot(args))
    print(f"MiniMax-H3 SP8 SuperCollect host snapshot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
