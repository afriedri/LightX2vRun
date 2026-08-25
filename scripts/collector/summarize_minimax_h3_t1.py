#!/usr/bin/env python3
"""Merge per-rank MiniMax-H3 schema-2 collectors into one AI-readable T1 report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


COLLECTOR_SCHEMA = "lightx2v.minimax_h3.runcollector"
COLLECTOR_VERSION = 2
SUMMARY_SCHEMA = "lightx2v.minimax_h3.t1_summary"
SUMMARY_VERSION = 1


def load_reports(result_dir: Path, expected_ranks: int) -> list[tuple[Path, dict]]:
    paths = sorted(result_dir.glob("h3_runcollector.rank*.json"))
    if len(paths) != expected_ranks:
        raise ValueError(f"expected {expected_ranks} H3 collector files, found {len(paths)} in {result_dir}")

    reports = []
    ranks = set()
    world_sizes = set()
    workload_contract = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_name") != COLLECTOR_SCHEMA or payload.get("schema_version") != COLLECTOR_VERSION:
            raise ValueError(
                f"{path.name} is not {COLLECTOR_SCHEMA} v{COLLECTOR_VERSION}: "
                f"got {payload.get('schema_name')!r} v{payload.get('schema_version')!r}"
            )
        if not isinstance(payload.get("h3_t1"), dict):
            raise ValueError(f"{path.name} is missing h3_t1")
        environment = payload.get("environment") or {}
        rank = int(environment["rank"])
        world_size = int(environment["world_size"])
        if rank in ranks:
            raise ValueError(f"duplicate collector rank {rank}")
        ranks.add(rank)
        world_sizes.add(world_size)
        workload = payload.get("workload") or {}
        canonical_workload = json.dumps(workload, sort_keys=True, separators=(",", ":"))
        if workload_contract is None:
            workload_contract = canonical_workload
        elif canonical_workload != workload_contract:
            raise ValueError(f"{path.name} workload does not match the other ranks")
        reports.append((path, payload))

    if ranks != set(range(expected_ranks)):
        raise ValueError(f"collector ranks must be 0..{expected_ranks - 1}, got {sorted(ranks)}")
    if world_sizes != {expected_ranks}:
        raise ValueError(f"collector world_size must be {expected_ranks}, got {sorted(world_sizes)}")
    return reports


def build_summary(reports: list[tuple[Path, dict]]) -> dict:
    by_rank = {int(payload["environment"]["rank"]): payload for _, payload in reports}
    step_indices = sorted(
        {
            int(step["step_index"])
            for payload in by_rank.values()
            for step in payload["h3_t1"]["summary"].get("steps", [])
        }
    )
    steps = []
    for step_index in step_indices:
        per_rank = {}
        for rank, payload in sorted(by_rank.items()):
            matches = [
                step
                for step in payload["h3_t1"]["summary"].get("steps", [])
                if int(step["step_index"]) == step_index
            ]
            if len(matches) != 1:
                raise ValueError(f"rank {rank} has {len(matches)} records for H3 step {step_index}")
            per_rank[str(rank)] = matches[0]
        for rank, step in per_rank.items():
            if "h3_step_gpu_ms" not in (step.get("metrics_ms") or {}):
                raise ValueError(f"rank {rank} H3 step {step_index} is missing h3_step_gpu_ms")
        metric_names = sorted(
            {
                name
                for step in per_rank.values()
                for name in (step.get("metrics_ms") or {})
            }
        )
        per_metric_rank_max = {
            name: max(float(step.get("metrics_ms", {}).get(name, 0.0)) for step in per_rank.values())
            for name in metric_names
        }
        critical_rank = max(
            sorted(per_rank, key=int),
            key=lambda rank: float(per_rank[rank]["metrics_ms"]["h3_step_gpu_ms"]),
        )
        critical = {
            name: float(per_rank[critical_rank].get("metrics_ms", {}).get(name, 0.0))
            for name in metric_names
        }
        steps.append(
            {
                "step_index": step_index,
                "critical_rank": int(critical_rank),
                "critical_path_ms": critical,
                "per_metric_rank_max_ms": per_metric_rank_max,
                "per_rank": per_rank,
            }
        )

    metric_names = sorted({name for step in steps for name in step["critical_path_ms"]})
    critical_totals = {
        name: sum(float(step["critical_path_ms"].get(name, 0.0)) for step in steps)
        for name in metric_names
    }
    opportunity_names = (
        "elementwise_and_unattributed_gpu_ms",
        "dynamic_lora_gpu_ms",
        "sp_layout_and_communication_gpu_ms",
        "exposed_h2d_gpu_ms",
        "refiner_gpu_ms",
        "pre_non_refiner_gpu_ms",
        "post_gpu_ms",
        "core_non_block_gpu_ms",
        "scheduler_gpu_ms",
    )
    opportunities = sorted(
        (
            {
                "metric": name,
                "critical_path_total_ms": critical_totals.get(name, 0.0),
                "share_of_h3_step_gpu": (
                    critical_totals.get(name, 0.0) / critical_totals["h3_step_gpu_ms"]
                    if critical_totals.get("h3_step_gpu_ms", 0.0) > 0.0
                    else None
                ),
            }
            for name in opportunity_names
        ),
        key=lambda item: item["critical_path_total_ms"],
        reverse=True,
    )
    wall_diagnostic_names = (
        "host_prefetch_wait_wall_ms",
        "swap_wait_wall_ms",
    )
    rank_max_wall_diagnostic_totals = {
        name: sum(
            float(step["per_metric_rank_max_ms"].get(name, 0.0))
            for step in steps
        )
        for name in wall_diagnostic_names
    }
    first_payload = by_rank[min(by_rank)]
    return {
        "schema_name": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_VERSION,
        "status": "complete",
        "profiling_only": True,
        "numerical_path_changed": False,
        "world_size": len(by_rank),
        "source_files": [path.name for path, _ in reports],
        "workload": first_payload.get("workload") or {},
        "methodology": first_payload["h3_t1"].get("methodology") or {},
        "critical_rank_selection": "For each H3 step, select the rank with maximum h3_step_gpu_ms and take every additive metric from that same rank.",
        "steps": steps,
        "critical_path_totals_ms": critical_totals,
        "opportunity_ranking": opportunities,
        "rank_max_wall_diagnostic_totals_ms": rank_max_wall_diagnostic_totals,
        "per_rank_totals_ms": {
            str(rank): payload["h3_t1"]["summary"].get("totals_ms", {})
            for rank, payload in sorted(by_rank.items())
        },
        "interpretation_guardrails": [
            "Collector mode inserts accelerator events, Python hooks, and a final materialization synchronization; do not use it as the production latency baseline.",
            "Dense attention is nested inside Ulysses inclusive time and is not additive with SP layout/communication.",
            "Total H2D includes overlapped copies; only exposed_h2d_gpu_ms is a critical-path optimization candidate.",
            "core_non_block_gpu_ms includes exposed transfer/barrier work and must not be added to exposed_h2d_gpu_ms.",
            "swap_wait_wall_ms is a host-barrier diagnostic and must not be added to accelerator-event metrics.",
            "host_prefetch_wait_wall_ms is time blocked on safetensors/page-cache-to-pinned-CPU fill; use rank_max_wall_diagnostic_totals_ms, not GPU critical-path shares.",
            "elementwise_and_unattributed is a residual bucket and requires a follow-up single-variable experiment before backend work.",
            "per_metric_rank_max_ms is an imbalance diagnostic only; do not add values that came from different ranks.",
        ],
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
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")
    output = args.output or args.result_dir / "h3_t1_summary.json"
    reports = load_reports(args.result_dir, args.expected_ranks)
    write_atomic(output, build_summary(reports))
    print(f"MiniMax-H3 T1 summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
