#!/usr/bin/env python3
"""No-GPU E1 static memory gate derived from an LTX-2.5 census report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "lightx2v.ltx25.b0.memory_mock"
SCHEMA_VERSION = 1
EXPECTED_CENSUS_SCHEMA = "lightx2v.ltx25.weight_census"
DEFAULT_CARD_BYTES = 34_208_743_424
DEFAULT_DYNAMIC_BUDGET_GIB = 16.0
DEFAULT_RUNTIME_RESERVE_MIB = 512
DEFAULT_MIN_MARGIN_GIB = 8.0
GPU_SLOT_COUNT = 2
PINNED_SLOT_COUNT = 2

FROZEN_WORKLOAD = {
    "height": 704,
    "width": 1280,
    "frames": 121,
    "fps": 24,
    "stage1_model_evaluations": 8,
    "stage2_model_evaluations": 3,
    "sequence_parallel_world_size": 2,
    "transformer_blocks": 48,
    "tensors_per_block": 84,
}


class MemoryMockError(RuntimeError):
    pass


def _gib(value: int) -> float:
    return value / (1024**3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_census(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryMockError(f"cannot read census {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MemoryMockError("census JSON must be an object")
    return payload


def _require(condition: bool, errors: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    if not condition:
        errors.append({"code": code, "message": message, **context})


def build_report(
    census: Mapping[str, Any],
    *,
    card_bytes: int = DEFAULT_CARD_BYTES,
    dynamic_budget_gib: float = DEFAULT_DYNAMIC_BUDGET_GIB,
    runtime_reserve_mib: int = DEFAULT_RUNTIME_RESERVE_MIB,
    min_margin_gib: float = DEFAULT_MIN_MARGIN_GIB,
    census_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    _require(
        census.get("schema_name") == EXPECTED_CENSUS_SCHEMA,
        errors,
        "wrong_census_schema",
        "input is not an LTX-2.5 weight census",
        actual=census.get("schema_name"),
    )
    _require(
        census.get("status") == "ok" and census.get("state") == "CENSUS_OK",
        errors,
        "census_not_ok",
        "E1 requires a complete CENSUS_OK report",
        status=census.get("status"),
        state=census.get("state"),
    )
    _require(card_bytes > 0, errors, "invalid_card_bytes", "card_bytes must be positive")
    _require(dynamic_budget_gib >= 0, errors, "invalid_dynamic_budget", "dynamic budget cannot be negative")
    _require(runtime_reserve_mib >= 0, errors, "invalid_runtime_reserve", "runtime reserve cannot be negative")
    _require(min_margin_gib >= 0, errors, "invalid_min_margin", "minimum margin cannot be negative")

    files = census.get("files", {})
    _require(
        isinstance(files, Mapping) and set(files) == {
            "transformer",
            "text_encoder",
            "video_vae",
            "audio_vae",
            "spatial_upscaler",
        },
        errors,
        "asset_set_mismatch",
        "census must contain exactly the five frozen BF16 B0 components",
        actual=sorted(files) if isinstance(files, Mapping) else None,
    )
    transformer = files.get("transformer", {}) if isinstance(files, Mapping) else {}
    contracts = transformer.get("contracts", {}) if isinstance(transformer, Mapping) else {}
    block_contract = contracts.get("blocks", {}) if isinstance(contracts, Mapping) else {}
    block_rows = block_contract.get("blocks", []) if isinstance(block_contract, Mapping) else []
    block_indices = block_contract.get("indices", []) if isinstance(block_contract, Mapping) else []
    block_bytes_values = [row.get("bytes") for row in block_rows if isinstance(row, Mapping)]
    block_tensor_counts = [row.get("tensor_count") for row in block_rows if isinstance(row, Mapping)]

    _require(block_indices == list(range(48)), errors, "block_indices_mismatch", "Transformer blocks must be exactly 0..47", actual=block_indices)
    _require(block_tensor_counts == [84] * 48, errors, "block_tensor_count_mismatch", "every Transformer block must contain 84 checkpoint tensors", actual=block_tensor_counts)
    unique_block_bytes = sorted({value for value in block_bytes_values if isinstance(value, int)})
    _require(len(block_bytes_values) == 48 and len(unique_block_bytes) == 1, errors, "block_bytes_not_uniform", "all 48 Transformer blocks must have one identical byte size", actual=unique_block_bytes)
    block_bytes = unique_block_bytes[0] if len(unique_block_bytes) == 1 else 0
    non_block_device_bytes = contracts.get("l2_non_block_bytes", 0) if isinstance(contracts, Mapping) else 0
    connector_l1_only_bytes = contracts.get("connector_l1_only_bytes", 0) if isinstance(contracts, Mapping) else 0
    _require(isinstance(non_block_device_bytes, int) and non_block_device_bytes > 0, errors, "invalid_non_block_bytes", "census did not provide positive L2 non-block bytes", actual=non_block_device_bytes)
    _require(isinstance(connector_l1_only_bytes, int) and connector_l1_only_bytes > 0, errors, "invalid_connector_bytes", "census did not provide positive L1-only connector bytes", actual=connector_l1_only_bytes)

    dynamic_budget_bytes = int(round(dynamic_budget_gib * 1024**3))
    runtime_reserve_bytes = runtime_reserve_mib * 1024**2
    minimum_margin_bytes = int(math.ceil(min_margin_gib * 1024**3))
    gpu_slot_bytes = GPU_SLOT_COUNT * block_bytes
    pinned_slot_bytes = PINNED_SLOT_COUNT * block_bytes
    device_static_bytes = gpu_slot_bytes + int(non_block_device_bytes or 0)
    conservative_pinned_bytes = pinned_slot_bytes + int(non_block_device_bytes or 0)
    derived_peak_bytes = (
        device_static_bytes
        + conservative_pinned_bytes
        + dynamic_budget_bytes
        + runtime_reserve_bytes
    )
    derived_margin_bytes = card_bytes - derived_peak_bytes
    static_margin_bytes = card_bytes - device_static_bytes
    gates = {
        "census_ok": census.get("status") == "ok" and census.get("state") == "CENSUS_OK",
        "frozen_workload": True,
        "block_schema_48x84": block_indices == list(range(48)) and block_tensor_counts == [84] * 48,
        "static_fit": static_margin_bytes >= 0,
        "budgeted_fit": derived_margin_bytes >= 0,
        "design_margin_at_least_8_gib": derived_margin_bytes >= minimum_margin_bytes,
    }
    if not gates["budgeted_fit"]:
        errors.append({"code": "budgeted_peak_exceeds_card", "message": "derived conservative peak exceeds physical card capacity", "derived_peak_bytes": derived_peak_bytes, "card_bytes": card_bytes})
    if not gates["design_margin_at_least_8_gib"]:
        errors.append({"code": "design_margin_too_small", "message": "derived margin is below the frozen E1 design gate", "derived_margin_bytes": derived_margin_bytes, "minimum_margin_bytes": minimum_margin_bytes})

    status = "ok" if not errors and all(gates.values()) else "error"
    census_identity: dict[str, Any] = {}
    if census_path is not None:
        stat = census_path.stat()
        census_identity = {
            "path": str(census_path),
            "realpath": os.path.realpath(census_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(census_path),
        }
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "state": "E1_MEMORY_OK" if status == "ok" else "E1_MEMORY_FAILED",
        "census": census_identity,
        "workload": dict(FROZEN_WORKLOAD),
        "card_physical_bytes": card_bytes,
        "block_bytes": block_bytes,
        "gpu_slot_count": GPU_SLOT_COUNT,
        "gpu_slot_bytes": gpu_slot_bytes,
        "pinned_slot_count": PINNED_SLOT_COUNT,
        "pinned_slot_bytes": pinned_slot_bytes,
        "non_block_device_bytes": non_block_device_bytes,
        "connector_l1_only_bytes": connector_l1_only_bytes,
        "device_static_bytes": device_static_bytes,
        "conservative_pinned_bytes": conservative_pinned_bytes,
        "dynamic_budget_bytes": dynamic_budget_bytes,
        "runtime_reserve_bytes": runtime_reserve_bytes,
        "derived_peak_bytes": derived_peak_bytes,
        "derived_margin_bytes": derived_margin_bytes,
        "minimum_margin_bytes": minimum_margin_bytes,
        "gib": {
            "card_physical": _gib(card_bytes),
            "block": _gib(block_bytes),
            "device_static": _gib(device_static_bytes),
            "conservative_pinned": _gib(conservative_pinned_bytes),
            "dynamic_budget": _gib(dynamic_budget_bytes),
            "runtime_reserve": _gib(runtime_reserve_bytes),
            "derived_peak": _gib(derived_peak_bytes),
            "derived_margin": _gib(derived_margin_bytes),
        },
        "gates": gates,
        "assumptions": [
            "single BF16 scheme with exactly two GPU block slots per rank",
            "exactly two pinned host block slots per rank",
            "L2 non-block tensors are resident once per rank",
            "connectors are L1-only and have zero L2 residency",
            "conservative accounting charges pinned slots plus non-block staging against the per-card envelope",
            "16 GiB default dynamic activation/kernel-workspace budget",
            "704x1280, 121 frames, 24 FPS, Stage-1 8 eval plus Stage-2 3 eval are immutable",
            "this static upper-bound gate does not replace the E5 physical resident-set gate",
        ],
        "errors": errors,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--card-bytes", type=int, default=DEFAULT_CARD_BYTES)
    parser.add_argument("--dynamic-budget-gib", type=float, default=DEFAULT_DYNAMIC_BUDGET_GIB)
    parser.add_argument("--runtime-reserve-mib", type=int, default=DEFAULT_RUNTIME_RESERVE_MIB)
    parser.add_argument("--min-margin-gib", type=float, default=DEFAULT_MIN_MARGIN_GIB)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        census = load_census(args.census)
        report = build_report(
            census,
            card_bytes=args.card_bytes,
            dynamic_budget_gib=args.dynamic_budget_gib,
            runtime_reserve_mib=args.runtime_reserve_mib,
            min_margin_gib=args.min_margin_gib,
            census_path=args.census,
        )
    except MemoryMockError as exc:
        print(f"E1_MEMORY_FAILED: {exc}", file=sys.stderr)
        return 2
    write_json(args.output, report)
    print(
        f"{report['state']} output={args.output} "
        f"peak_gib={report['gib']['derived_peak']:.3f} "
        f"margin_gib={report['gib']['derived_margin']:.3f}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
