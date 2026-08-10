#!/usr/bin/env python3
"""Summarize a Wan2.2 CK Conv3D scale sweep and emit exact contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


WORLD_SIZES = (1, 2, 4, 8, 16)
PLAN_FIELDS = (
    "contract_id",
    "calls",
    "priority_score",
    "flops_per_call",
    "projected_flops",
    "n",
    "k",
    "c",
    "d",
    "h",
    "w",
    "z",
    "y",
    "x",
    "stride_d",
    "stride_h",
    "stride_w",
    "dilation_d",
    "dilation_h",
    "dilation_w",
    "left_pad_d",
    "left_pad_h",
    "left_pad_w",
    "right_pad_d",
    "right_pad_h",
    "right_pad_w",
    "gfx",
    "wgp",
    "dtype",
    "input_shape",
    "weight_shape",
    "output_shape",
    "conv_stride",
    "padding",
    "dilation",
    "groups",
    "full_tune_reasons",
)
ROUTE_FIELDS = (
    "contract_id",
    "world_size",
    "rank",
    "phase",
    "route",
    "selection",
    "calls",
    "flops_per_call",
    "projected_flops",
    "input_shape",
    "weight_shape",
    "output_shape",
    "conv_stride",
    "padding",
    "dilation",
)


def shape(value: Any, size: int) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if len(result) != size:
        raise ValueError(f"expected {size} dimensions, got {value!r}")
    return result


def shape_text(value: Iterable[int]) -> str:
    return "x".join(str(int(item)) for item in value)


def output_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[int, ...]:
    spatial = tuple(
        (input_shape[axis + 2] + 2 * padding[axis] - dilation[axis] * (weight_shape[axis + 2] - 1) - 1)
        // stride[axis]
        + 1
        for axis in range(3)
    )
    return (input_shape[0], weight_shape[0], *spatial)


def flops_per_call(input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output: tuple[int, ...]) -> int:
    return 2 * output[0] * output[1] * output[2] * output[3] * output[4] * weight_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4]


def contract_key(route: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(route["gfx"]),
        int(route["wgp"]),
        "bfloat16",
        shape(route["input_shape"], 5),
        shape(route["weight_shape"], 5),
        shape(route["output_shape"], 5),
        shape(route["conv_stride"], 3),
        shape(route["padding"], 3),
        shape(route["dilation"], 3),
        int(route["groups"]),
    )


def contract_id(key: tuple[Any, ...]) -> str:
    payload = json.dumps(
        [list(item) if isinstance(item, tuple) else item for item in key],
        separators=(",", ":"),
    )
    return f"wan22_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def load_profile(path: Path) -> tuple[int, list[dict[str, Any]]]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing profile manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "wan22_ck_conv3d_v1_profile_manifest":
        raise ValueError(f"unexpected profile manifest: {manifest_path}")
    world_size = int(manifest["world_size"])
    if world_size not in WORLD_SIZES:
        raise ValueError(f"unsupported world size in {manifest_path}: {world_size}")
    expected = {
        (rank, phase)
        for rank in range(world_size)
        for phase in ("encode", "decode")
    }
    declared_names = manifest.get("snapshots")
    if not isinstance(declared_names, list) or not all(
        isinstance(name, str) and name for name in declared_names
    ):
        raise ValueError(f"invalid snapshot list in {manifest_path}")
    if len(set(declared_names)) != len(declared_names):
        raise ValueError(f"duplicate snapshot names in {manifest_path}")
    actual_paths = {snapshot_path.name: snapshot_path for snapshot_path in path.glob("rank*_*.json")}
    declared_set = set(declared_names)
    if declared_set != set(actual_paths):
        missing_names = sorted(declared_set - set(actual_paths))
        unexpected_names = sorted(set(actual_paths) - declared_set)
        raise ValueError(
            f"{path}: snapshot manifest mismatch; missing={missing_names}, "
            f"unexpected={unexpected_names}"
        )

    snapshots: dict[tuple[int, str], Path] = {}
    for snapshot_name in declared_names:
        snapshot_path = actual_paths[snapshot_name]
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if payload.get("kind") != "lightx2v_ck_conv3d_profile":
            raise ValueError(f"unexpected snapshot kind: {snapshot_path}")
        rank = int(payload["rank"])
        phase = str(payload["phase"])
        if int(payload.get("world_size", -1)) != world_size:
            raise ValueError(f"world size mismatch in {snapshot_path}")
        if rank not in range(world_size) or phase not in ("encode", "decode"):
            raise ValueError(f"invalid rank/phase in {snapshot_path}: {(rank, phase)}")
        expected_name = f"rank{rank:03d}_{phase}_000.json"
        if snapshot_path.name != expected_name:
            raise ValueError(
                f"unexpected snapshot sequence: expected {expected_name}, "
                f"got {snapshot_path.name}"
            )
        if payload.get("backend") != "Aiter_CK_Conv3d_v1":
            raise ValueError(f"unexpected Conv3D backend in {snapshot_path}")
        key = (rank, phase)
        if key in snapshots:
            raise ValueError(f"{path}: duplicate rank/phase snapshot: {key}")
        snapshots[key] = snapshot_path
    missing = expected - set(snapshots)
    if missing:
        raise ValueError(f"{path}: missing rank/phase snapshots: {sorted(missing)}")

    routes: list[dict[str, Any]] = []
    for (rank, phase), snapshot_path in sorted(snapshots.items()):
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for route in payload.get("routes", []):
            if int(route.get("calls", 0)) <= 0:
                continue
            normalized = dict(route)
            normalized.update(
                world_size=world_size,
                rank=rank,
                phase=phase,
                gfx=payload.get("gfx", "gfx1201"),
                wgp=int(payload.get("wgp", 24) or 24),
            )
            routes.append(normalized)
    return world_size, routes


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(profile_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    profiles: dict[int, Path] = {}
    routes: list[dict[str, Any]] = []
    for directory in profile_dirs:
        world_size, profile_routes = load_profile(directory)
        if world_size in profiles:
            raise ValueError(
                f"duplicate world size {world_size}: {profiles[world_size]} and {directory}"
            )
        profiles[world_size] = directory
        routes.extend(profile_routes)
    if not routes:
        raise ValueError("profile inputs contain no Conv3D routes")

    by_contract: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        key = contract_key(route)
        input_shape, weight_shape, observed_output, stride, padding, dilation = key[3:9]
        if key[9] != 1:
            continue
        if input_shape[1] != weight_shape[1]:
            continue
        if observed_output != output_shape(input_shape, weight_shape, stride, padding, dilation):
            raise ValueError(f"invalid runtime Conv3D output contract: {route}")
        by_contract[key].append(route)
    if not by_contract:
        raise ValueError("no groups=1 CK-compatible Conv3D routes were observed")

    plan_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for key, contract_routes in by_contract.items():
        gfx, wgp, dtype, input_shape, weight_shape, output, stride, padding, dilation, groups = key
        identifier = contract_id(key)
        per_call = flops_per_call(input_shape, weight_shape, output)
        rank_calls: dict[tuple[int, int], int] = defaultdict(int)
        phases: set[str] = set()
        route_names: set[str] = set()
        for route in contract_routes:
            calls = int(route["calls"])
            rank_calls[(int(route["world_size"]), int(route["rank"]))] += calls
            phases.add(str(route["phase"]))
            route_names.add(str(route["route"]))
            route_rows.append(
                {
                    "contract_id": identifier,
                    "world_size": route["world_size"],
                    "rank": route["rank"],
                    "phase": route["phase"],
                    "route": route["route"],
                    "selection": route.get("selection") or "",
                    "calls": calls,
                    "flops_per_call": per_call,
                    "projected_flops": calls * per_call,
                    "input_shape": shape_text(input_shape),
                    "weight_shape": shape_text(weight_shape),
                    "output_shape": shape_text(output),
                    "conv_stride": shape_text(stride),
                    "padding": shape_text(padding),
                    "dilation": shape_text(dilation),
                }
            )
        calls = max(rank_calls.values())
        projected = calls * per_call
        reasons = ["runtime_observed"]
        if any(route != "ck" for route in route_names):
            reasons.append("runtime_fallback")
        plan_rows.append(
            {
                "contract_id": identifier,
                "calls": calls,
                "priority_score": f"{projected:.12g}",
                "flops_per_call": per_call,
                "projected_flops": projected,
                "n": input_shape[0],
                "k": weight_shape[0],
                "c": input_shape[1],
                "d": input_shape[2],
                "h": input_shape[3],
                "w": input_shape[4],
                "z": weight_shape[2],
                "y": weight_shape[3],
                "x": weight_shape[4],
                "stride_d": stride[0],
                "stride_h": stride[1],
                "stride_w": stride[2],
                "dilation_d": dilation[0],
                "dilation_h": dilation[1],
                "dilation_w": dilation[2],
                "left_pad_d": padding[0],
                "left_pad_h": padding[1],
                "left_pad_w": padding[2],
                "right_pad_d": padding[0],
                "right_pad_h": padding[1],
                "right_pad_w": padding[2],
                "gfx": gfx,
                "wgp": wgp,
                "dtype": dtype,
                "input_shape": shape_text(input_shape),
                "weight_shape": shape_text(weight_shape),
                "output_shape": shape_text(output),
                "conv_stride": shape_text(stride),
                "padding": shape_text(padding),
                "dilation": shape_text(dilation),
                "groups": groups,
                "full_tune_reasons": ",".join(reasons),
            }
        )

    plan_rows.sort(key=lambda row: (-int(row["projected_flops"]), row["contract_id"]))
    route_rows.sort(
        key=lambda row: (
            -int(row["projected_flops"]),
            row["contract_id"],
            row["world_size"],
            row["rank"],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "full_tune_contracts.csv", PLAN_FIELDS, plan_rows)
    write_csv(output_dir / "routes.csv", ROUTE_FIELDS, route_rows)
    manifest = {
        "kind": "wan22_ck_conv3d_v1_scale_sweep_summary",
        "profile_dirs": [str(path.resolve()) for path in profile_dirs],
        "world_sizes": sorted(profiles),
        "observed_route_count": len(routes),
        "contract_count": len(plan_rows),
        "ck_route_count": sum(int(route["calls"]) for route in routes if route.get("route") == "ck"),
        "fallback_route_count": sum(int(route["calls"]) for route in routes if route.get("route") != "ck"),
        "outputs": {
            "contracts": "full_tune_contracts.csv",
            "routes": "routes.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build_summary(args.inputs, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Wan2.2 CK Conv3D scale sweep: {manifest['contract_count']} contracts")
    print(f"Contracts: {args.output_dir / 'full_tune_contracts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
