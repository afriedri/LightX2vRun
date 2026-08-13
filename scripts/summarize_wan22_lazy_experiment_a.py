#!/usr/bin/env python3
"""Summarize per-rank Wan2.2 lazy Experiment-A runtune reports."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DIT_SUMMARY_RE = re.compile(
    r"DiT summary: status=(?P<status>\w+), steps=(?P<steps>\d+)/(?P<total>\d+), "
    r"total=(?P<seconds>\d+(?:\.\d+)?)s"
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(by_rank: dict[int, float], unit: str = "ms") -> dict[str, Any]:
    values = list(by_rank.values())
    max_rank = max(by_rank, key=by_rank.get) if by_rank else None
    return {
        f"p50_{unit}": percentile(values, 0.50),
        f"p95_{unit}": percentile(values, 0.95),
        f"max_{unit}": max(values, default=0.0),
        "max_rank": max_rank,
        f"by_rank_{unit}": {
            str(rank): value for rank, value in sorted(by_rank.items())
        },
    }


def scope(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [entry for entry in report.get("scopes", []) if entry.get("name") == name]
    if len(matches) != 1 or not matches[0].get("success"):
        rank = report.get("environment", {}).get("rank", "unknown")
        available = [
            f"{entry.get('name', '<unnamed>')}:{entry.get('success', False)}"
            for entry in report.get("scopes", [])
        ]
        raise ValueError(
            f"rank {rank} report must contain one successful {name!r} scope; "
            f"found {len(matches)}, available={available}"
        )
    return matches[0]


def category_total(report: dict[str, Any], kind: str, name: str) -> float:
    return float(report.get("summary", {}).get(kind, {}).get(name, {}).get("total_ms", 0.0))


def category_bytes(report: dict[str, Any], kind: str, name: str) -> float:
    category = report.get("summary", {}).get(kind, {}).get(name, {})
    return float(
        sum(
            int(entry.get("calls", 0)) * int(entry.get("detail", {}).get("bytes", 0))
            for entry in category.get("details", [])
        )
    )


def category_calls(report: dict[str, Any], kind: str, name: str) -> float:
    return float(
        report.get("summary", {}).get(kind, {}).get(name, {}).get("calls", 0)
    )


def load_reports(input_dir: Path, expected_ranks: int) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("rank*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 2:
            raise ValueError(f"{path} is not an Experiment-A schema-2 report")
        environment = report.get("environment", {})
        rank = int(environment.get("rank", -1))
        world_size = int(environment.get("world_size", -1))
        if world_size != expected_ranks:
            raise ValueError(f"{path} world_size={world_size}, expected {expected_ranks}")
        if rank in reports:
            raise ValueError(f"duplicate rank {rank}: {path}")
        steps = report.get("steps", [])
        if len(steps) != 4 or any(not step.get("success") for step in steps):
            raise ValueError(f"{path} must contain four successful steps")
        reports[rank] = report
    expected = set(range(expected_ranks))
    if set(reports) != expected:
        raise ValueError(f"missing/unexpected ranks: expected {sorted(expected)}, got {sorted(reports)}")
    return reports


def parse_legacy_dit(log_path: Path) -> dict[str, Any]:
    matches = list(DIT_SUMMARY_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")))
    if not matches:
        raise ValueError(f"legacy DiT summary is missing from {log_path}")
    match = matches[-1]
    return {
        "status": match.group("status"),
        "steps": int(match.group("steps")),
        "total_steps": int(match.group("total")),
        "critical_path_seconds": float(match.group("seconds")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-summary", type=Path)
    parser.add_argument("--expected-ranks", type=int, default=16)
    args = parser.parse_args()

    reports = load_reports(args.input_dir, args.expected_ranks)
    backends = {str(report.get("backend", "")) for report in reports.values()}
    if len(backends) != 1:
        raise ValueError(f"rank reports disagree on backend/cache label: {sorted(backends)}")
    step_critical_ms = [
        max(float(report["steps"][step_index]["wall_ms"]) for report in reports.values())
        for step_index in range(4)
    ]
    summary: dict[str, Any] = {
        "schema_version": 2,
        "kind": "wan22_lazy_experiment_a_summary",
        "ranks": args.expected_ranks,
        "backend_and_cache_state": next(iter(backends)),
        "legacy_four_step_denoising": parse_legacy_dit(args.log),
        "runtuner_step_wall": {
            "critical_path_step_ms": step_critical_ms,
            "critical_path_total_ms": sum(step_critical_ms),
            "note": "Diagnostic inner-step wall; the legacy profiler metric above remains the comparison baseline.",
        },
        "scopes": {},
        "components": {"cpu": {}, "gpu": {}, "events": {}},
        "component_bytes": {"cpu": {}, "gpu": {}},
        "component_calls": {"cpu": {}, "gpu": {}},
        "device_memory": {},
        "component_timing_note": "Component times can overlap and must not be summed to derive latency.",
    }

    for name in ("transformer_lifecycle", "request_end_to_end"):
        summary["scopes"][name] = distribution(
            {rank: float(scope(report, name)["wall_ms"]) for rank, report in reports.items()}
        )

    for name in (
        "offload.host_fill",
        "offload.host_fill_wait",
        "offload.init_wait",
        "offload.load_wait",
        "offload.compute_wait",
        "transformer.construct",
    ):
        summary["components"]["cpu"][name] = distribution(
            {rank: category_total(report, "cpu", name) for rank, report in reports.items()}
        )
    for name in ("offload.initial_copy", "offload.prefetch_copy", "offload.block_compute"):
        summary["components"]["gpu"][name] = distribution(
            {rank: category_total(report, "gpu", name) for rank, report in reports.items()}
        )
    for name in ("offload.host_fill",):
        summary["component_bytes"]["cpu"][name] = distribution(
            {
                rank: category_bytes(report, "cpu", name)
                for rank, report in reports.items()
            },
            "bytes",
        )
        summary["component_calls"]["cpu"][name] = distribution(
            {
                rank: category_calls(report, "cpu", name)
                for rank, report in reports.items()
            },
            "calls",
        )
    for name in ("offload.initial_copy", "offload.prefetch_copy"):
        summary["component_bytes"]["gpu"][name] = distribution(
            {
                rank: category_bytes(report, "gpu", name)
                for rank, report in reports.items()
            },
            "bytes",
        )
        summary["component_calls"]["gpu"][name] = distribution(
            {
                rank: category_calls(report, "gpu", name)
                for rank, report in reports.items()
            },
            "calls",
        )
    summary["components"]["events"]["transformer.release"] = distribution(
        {
            rank: sum(
                float(event["elapsed_ms"])
                for event in report.get("events", [])
                if event.get("category") == "transformer.release"
            )
            for rank, report in reports.items()
        }
    )

    for rank, report in reports.items():
        snapshots = report.get("device_memory_snapshots", [])
        summary["device_memory"][str(rank)] = {
            "max_allocated_bytes": max((int(item["allocated_bytes"]) for item in snapshots), default=0),
            "max_reserved_bytes": max((int(item["reserved_bytes"]) for item in snapshots), default=0),
            "snapshots": snapshots,
        }

    if args.memory_summary is not None:
        summary["host_and_physical_memory"] = json.loads(
            args.memory_summary.read_text(encoding="utf-8")
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    legacy = summary["legacy_four_step_denoising"]
    lifecycle = summary["scopes"]["transformer_lifecycle"]
    request = summary["scopes"]["request_end_to_end"]
    print(
        "Experiment A: "
        f"legacy DiT={legacy['critical_path_seconds']:.3f}s, "
        f"transformer lifecycle max={lifecycle['max_ms'] / 1000:.3f}s, "
        f"request E2E max={request['max_ms'] / 1000:.3f}s"
    )
    print(f"Summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
