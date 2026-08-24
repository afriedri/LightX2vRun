#!/usr/bin/env python3
"""Collect MiniMax-H3 Sage shapes from normalized runtime dispatch logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_minimax_h3_run import ANSI_RE, parse_runtime_events


SCHEMA_NAME = "lightx2v.minimax_h3.sage_shape_sweep"
SCHEMA_VERSION = 1
EXPECTED_BACKEND = "aiter_fav3_sage_bf16_attn"
EXPECTED_ROUTE = "sage_attn_v2_gfx1201"
EXPECTED_COMPONENTS = ("dit.core.attention", "attention.sage")
EXPECTED_KERNEL = "sage_attention2"
EXPECTED_LAYOUT = "BSHD"
EXPECTED_NUMERICS = "int8_qk_fp8_pv"
EXPECTED_GLOBAL_HEADS = 56
EXPECTED_HEAD_DIM = 128
SUPPORTED_SP_SIZES = (1, 2, 4, 8)


def parse_run_spec(value: str) -> tuple[int, Path]:
    try:
        raw_sp, raw_path = value.split("=", 1)
        sp_size = int(raw_sp)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("run must use SP_SIZE=LOG_PATH") from error
    if sp_size not in SUPPORTED_SP_SIZES:
        raise argparse.ArgumentTypeError(
            f"SP_SIZE must be one of {SUPPORTED_SP_SIZES}, got {sp_size}"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("LOG_PATH may not be empty")
    return sp_size, Path(raw_path)


def parse_rank(value: object) -> tuple[int, int]:
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(f"Sage dispatch rank must use rank/world format, got {value!r}")
    raw_rank, raw_world = value.split("/", 1)
    try:
        return int(raw_rank), int(raw_world)
    except ValueError as error:
        raise ValueError(f"Sage dispatch rank must be numeric, got {value!r}") from error


def _shape(value: object) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Sage dispatch shape must be four-dimensional BSHD, got {value!r}")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value):
        raise ValueError(f"Sage dispatch shape must contain positive integers, got {value!r}")
    return list(value)


def collect_log(log_path: Path, expected_world_size: int) -> dict:
    if not log_path.is_file():
        raise ValueError(f"Sage run log does not exist: {log_path}")
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    events = [
        event
        for event in parse_runtime_events(text)
        if event.get("kind") == "dispatch"
        and event.get("model") == "minimax_h3"
        and event.get("component") in EXPECTED_COMPONENTS
        and event.get("backend") == EXPECTED_BACKEND
    ]
    if not events:
        raise ValueError(
            f"no MiniMax-H3 Sage dispatch found in {log_path}; "
            "the run must reach real H3 Core attention"
        )

    by_rank: dict[int, dict] = {}
    for event in events:
        rank, world_size = parse_rank(event.get("rank"))
        if world_size != expected_world_size:
            raise ValueError(
                f"{log_path} reports world_size={world_size} for rank {rank}, "
                f"expected {expected_world_size}"
            )
        if rank < 0 or rank >= expected_world_size:
            raise ValueError(f"{log_path} reports out-of-range Sage rank {rank}/{world_size}")
        if rank in by_rank:
            raise ValueError(
                f"{log_path} contains more than one Sage dispatch for rank {rank}; "
                "expected exactly one"
            )
        if event.get("route") != EXPECTED_ROUTE:
            raise ValueError(
                f"rank {rank} used Sage route {event.get('route')!r}, expected {EXPECTED_ROUTE!r}"
            )
        for field, expected in (
            ("kernel", EXPECTED_KERNEL),
            ("layout", EXPECTED_LAYOUT),
            ("numerics", EXPECTED_NUMERICS),
        ):
            if event.get(field) != expected:
                raise ValueError(
                    f"rank {rank} Sage dispatch {field}={event.get(field)!r}, expected {expected!r}"
                )
        if event.get("causal") is not False:
            raise ValueError(
                f"rank {rank} Sage dispatch must be noncausal, got causal={event.get('causal')!r}"
            )
        shape = _shape(event.get("shape"))
        batch, sequence, local_heads, head_dim = shape
        if batch != 1 or head_dim != EXPECTED_HEAD_DIM:
            raise ValueError(
                f"rank {rank} Sage shape must be [1,S,H,{EXPECTED_HEAD_DIM}], got {shape}"
            )
        if local_heads * expected_world_size != EXPECTED_GLOBAL_HEADS:
            raise ValueError(
                f"rank {rank} local_heads={local_heads} does not reconstruct "
                f"MiniMax-H3 global_heads={EXPECTED_GLOBAL_HEADS} at SP{expected_world_size}"
            )
        by_rank[rank] = {
            "rank": rank,
            "shape_bshd": shape,
            "sequence_length": sequence,
            "local_heads": local_heads,
        }

    missing = sorted(set(range(expected_world_size)) - set(by_rank))
    if missing:
        raise ValueError(f"{log_path} is missing Sage dispatches for ranks {missing}")
    shapes = {tuple(item["shape_bshd"]) for item in by_rank.values()}
    if len(shapes) != 1:
        raise ValueError(f"{log_path} has cross-rank Sage shape disagreement: {sorted(shapes)}")
    components = {str(event["component"]) for event in events}
    if len(components) != 1:
        raise ValueError(f"{log_path} has cross-rank Sage component disagreement: {sorted(components)}")
    shape = list(next(iter(shapes)))
    return {
        "sp_size": expected_world_size,
        "world_size": expected_world_size,
        "source_log": str(log_path),
        "backend": EXPECTED_BACKEND,
        "route": EXPECTED_ROUTE,
        "component": next(iter(components)),
        "kernel": EXPECTED_KERNEL,
        "layout": EXPECTED_LAYOUT,
        "numerics": EXPECTED_NUMERICS,
        "shape_bshd": shape,
        "sequence_length": shape[1],
        "local_heads": shape[2],
        "global_heads": EXPECTED_GLOBAL_HEADS,
        "head_dim": shape[3],
        "rank_count": len(by_rank),
        "per_rank": [by_rank[rank] for rank in sorted(by_rank)],
    }


def build_summary(run_specs: list[tuple[int, Path]]) -> dict:
    if not run_specs:
        raise ValueError("at least one --run SP_SIZE=LOG_PATH is required")
    sp_sizes = [sp_size for sp_size, _ in run_specs]
    if len(sp_sizes) != len(set(sp_sizes)):
        raise ValueError(f"duplicate SP sizes are not allowed: {sp_sizes}")
    runs = [collect_log(path, sp_size) for sp_size, path in run_specs]
    sequence_lengths = {run["sequence_length"] for run in runs}
    if len(sequence_lengths) != 1:
        raise ValueError(
            f"Sage sweep logs do not share one global sequence length: {sorted(sequence_lengths)}"
        )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source": "normalized runtime dispatch; schema-2 runcollector disabled",
        "numerical_path_changed": False,
        "sp_sizes": sp_sizes,
        "sequence_length": next(iter(sequence_lengths)),
        "runs": runs,
    }


def write_atomic(path: Path, payload: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="SP_SIZE=LOG_PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = build_summary(args.run)
    except ValueError as error:
        parser.error(str(error))
    write_atomic(args.output, payload)
    print(f"MiniMax-H3 Sage shapes: {args.output}")
    for run in payload["runs"]:
        print(f"SP{run['sp_size']}: {run['shape_bshd']} ({run['rank_count']} ranks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
