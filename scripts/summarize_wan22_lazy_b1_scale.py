#!/usr/bin/env python3
"""Validate Wan2.2 lazy B1 runs across multiple GPU counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compare_wan22_lazy_b0_b1 import (
    BACKEND_RE,
    load_reports,
    sha256_file,
    transfer_contract,
)


GIB = 1024**3


def parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc
    if not counts or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("counts must be non-empty and unique")
    if any(count not in (2, 4, 8, 16) for count in counts):
        raise argparse.ArgumentTypeError("counts may contain only 2, 4, 8, and 16")
    return counts


def scope_max(summary: dict[str, Any], name: str) -> float:
    return float(summary["scopes"][name]["max_ms"])


def memory_accounting(
    summary: dict[str, Any],
    count: int,
    budget_bytes: int,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    physical = summary.get("host_and_physical_memory", {})
    physical_peaks = {
        str(card): int(value)
        for card, value in physical.get("physical_gpu_peak_used_bytes", {}).items()
    }
    device_memory = summary.get("device_memory", {})
    expected_ranks = {str(rank) for rank in range(count)}
    allocator_by_rank: dict[str, dict[str, int]] = {}
    for rank, item in sorted(device_memory.items(), key=lambda pair: int(pair[0])):
        allocator_by_rank[str(rank)] = {
            "max_allocated_bytes": int(
                item.get("max_allocator_allocated_bytes", item.get("max_allocated_bytes", 0))
            ),
            "max_reserved_bytes": int(
                item.get("max_allocator_reserved_bytes", item.get("max_reserved_bytes", 0))
            ),
        }

    physical_peak = max(physical_peaks.values(), default=0)
    physical_min = min(physical_peaks.values(), default=0)
    allocator_allocated_peak = max(
        (item["max_allocated_bytes"] for item in allocator_by_rank.values()),
        default=0,
    )
    allocator_reserved_peak = max(
        (item["max_reserved_bytes"] for item in allocator_by_rank.values()),
        default=0,
    )
    usable_limit = budget_bytes - minimum_headroom_bytes
    monitoring_complete = (
        len(physical_peaks) == count and set(allocator_by_rank) == expected_ranks
    )
    within_budget = (
        monitoring_complete
        and physical_peak <= usable_limit
        and allocator_reserved_peak <= usable_limit
    )
    return {
        "passed": within_budget,
        "monitoring_complete": monitoring_complete,
        "expected_gpu_count": count,
        "physical_gpu_count": len(physical_peaks),
        "instrumented_rank_count": len(allocator_by_rank),
        "budget_bytes_per_gpu": budget_bytes,
        "minimum_required_headroom_bytes": minimum_headroom_bytes,
        "usable_limit_bytes_per_gpu": usable_limit,
        "physical_peak_used_bytes": physical_peak,
        "physical_min_of_card_peaks_bytes": physical_min,
        "physical_peak_skew_bytes": physical_peak - physical_min,
        "physical_headroom_bytes": budget_bytes - physical_peak,
        "physical_peak_fraction_of_budget": (
            physical_peak / budget_bytes if budget_bytes else 0.0
        ),
        "allocator_max_allocated_bytes": allocator_allocated_peak,
        "allocator_max_reserved_bytes": allocator_reserved_peak,
        "physical_peaks_by_card_bytes": physical_peaks,
        "allocator_peaks_by_rank_bytes": allocator_by_rank,
        "host_peak": {
            "simultaneous_rank_pss_bytes": int(
                physical.get("max_simultaneous_rank_sums_bytes", {}).get("Pss_bytes", 0)
            ),
            "simultaneous_rank_rss_bytes": int(
                physical.get("max_simultaneous_rank_sums_bytes", {}).get("Rss_bytes", 0)
            ),
            "sampled_cgroup_current_bytes": int(
                physical.get("max_observed_cgroup_current_bytes", 0)
            ),
        },
    }


def video_accounting(path: Path) -> dict[str, Any]:
    present = path.is_file() and path.stat().st_size > 0
    return {
        "present_and_nonempty": present,
        "path": str(path),
        "size_bytes": path.stat().st_size if present else 0,
        "sha256": sha256_file(path) if present else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=parse_counts, default=(2, 4, 8))
    parser.add_argument("--gpu-budget-gib", type=float, default=32.0)
    parser.add_argument("--minimum-gpu-headroom-gib", type=float, default=1.0)
    args = parser.parse_args()

    if args.gpu_budget_gib <= 0:
        parser.error("--gpu-budget-gib must be positive")
    if not 0 <= args.minimum_gpu_headroom_gib < args.gpu_budget_gib:
        parser.error("--minimum-gpu-headroom-gib must be non-negative and below the budget")

    try:
        result_dir = args.result_dir.expanduser().resolve()
        budget_bytes = int(args.gpu_budget_gib * GIB)
        minimum_headroom_bytes = int(args.minimum_gpu_headroom_gib * GIB)
        runs: dict[str, Any] = {}
        cache_states: set[str] = set()
        all_contracts = True
        all_memory = True
        all_outputs = True

        for count in args.counts:
            arm_dir = result_dir / f"{count}gpu"
            reports = load_reports(arm_dir, "b1", count)
            contract = transfer_contract(reports, "b1")
            summary = json.loads(
                (arm_dir / "experiment_a_summary.json").read_text(encoding="utf-8")
            )
            memory = memory_accounting(
                summary, count, budget_bytes, minimum_headroom_bytes
            )
            video = video_accounting(
                result_dir / f"{count}gpu_720p121f_{count}gpu.mp4"
            )
            for report in reports.values():
                match = BACKEND_RE.fullmatch(str(report.get("backend", "")))
                if match is None or match.group("mode") != "b1":
                    raise ValueError(f"{count}-GPU report has an invalid B1 backend label")
                cache_states.add(match.group("cache"))

            all_contracts = all_contracts and bool(contract["passed"])
            all_memory = all_memory and bool(memory["passed"])
            all_outputs = all_outputs and bool(video["present_and_nonempty"])
            runs[str(count)] = {
                "ranks": count,
                "backend_and_cache_state": summary["backend_and_cache_state"],
                "timings_ms": {
                    "legacy_four_step_denoising": float(
                        summary["legacy_four_step_denoising"]["critical_path_seconds"]
                    )
                    * 1000.0,
                    "runtuner_four_step_critical": float(
                        summary["runtuner_step_wall"]["critical_path_total_ms"]
                    ),
                    "transformer_lifecycle_max": scope_max(
                        summary, "transformer_lifecycle"
                    ),
                    "request_end_to_end_max": scope_max(summary, "request_end_to_end"),
                },
                "transfer_contract": contract,
                "memory": memory,
                "output": video,
            }

        if len(cache_states) != 1:
            raise ValueError(f"B1 scale cache labels differ: {sorted(cache_states)}")
        video_digests = {
            run["output"]["sha256"]
            for run in runs.values()
            if run["output"]["sha256"] is not None
        }
        exact_cross_scale_match = all_outputs and len(video_digests) == 1
        required_counts_complete = args.counts == (2, 4, 8)
        accepted = (
            required_counts_complete and all_contracts and all_memory and all_outputs
        )
        result = {
            "schema_version": 1,
            "kind": "wan22_lazy_b1_scale_summary",
            "gpu_counts": list(args.counts),
            "cache_state": next(iter(cache_states)),
            "accepted": accepted,
            "decision": (
                "ready_for_lightx2v_integration"
                if accepted
                else "inspect_failed_scale_acceptance_checks"
            ),
            "acceptance": {
                "required_gpu_counts_2_4_8_complete": required_counts_complete,
                "all_b1_transfer_contracts": all_contracts,
                "all_gpu_memory_budgets": all_memory,
                "all_outputs_present_and_nonempty": all_outputs,
                "exact_cross_scale_output_file_match": exact_cross_scale_match,
                "exact_output_match_is_informational": True,
            },
            "gpu_memory_policy": {
                "budget_gib_per_gpu": args.gpu_budget_gib,
                "minimum_headroom_gib_per_gpu": args.minimum_gpu_headroom_gib,
                "note": "Acceptance uses both physical-card peak usage and allocator reserved peak. Physical monitoring must cover every selected GPU.",
            },
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for count in args.counts:
        run = runs[str(count)]
        memory = run["memory"]
        print(
            f"B1 {count} GPU: DiT={run['timings_ms']['legacy_four_step_denoising'] / 1000:.3f}s, "
            f"request={run['timings_ms']['request_end_to_end_max'] / 1000:.3f}s, "
            f"physical VRAM peak={memory['physical_peak_used_bytes'] / GIB:.3f} GiB, "
            f"headroom={memory['physical_headroom_bytes'] / GIB:.3f} GiB, "
            f"memory_passed={memory['passed']}"
        )
    print(f"Exact cross-scale MP4 match (informational): {exact_cross_scale_match}")
    print(f"Accepted: {accepted}")
    print(f"Summary: {args.output}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
