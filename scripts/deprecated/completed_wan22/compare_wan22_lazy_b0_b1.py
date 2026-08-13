#!/usr/bin/env python3
"""Validate and compare paired Wan2.2 lazy B0/B1 16-rank runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


BACKEND_RE = re.compile(
    r"^wan22-lazy-full-offload-(?P<mode>b0|b1)-(?P<cache>cold|warm)-cache$"
)
TRANSFER_CATEGORIES = (
    ("cpu", "offload.host_fill"),
    ("gpu", "offload.initial_copy"),
    ("gpu", "offload.prefetch_copy"),
)
EXPECTED_CALLS = {
    "offload.host_fill": 156,
    "offload.initial_copy": 0,
    "offload.prefetch_copy": 156,
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def distribution(by_rank: dict[int, float], unit: str) -> dict[str, Any]:
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


def load_reports(result_dir: Path, mode: str, expected_ranks: int) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    for path in sorted((result_dir / "runtune").glob("rank*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        environment = report.get("environment", {})
        rank = int(environment.get("rank", -1))
        if report.get("schema_version") != 2:
            raise ValueError(f"{path} is not a schema-2 report")
        if int(environment.get("world_size", -1)) != expected_ranks:
            raise ValueError(f"{path} does not report world_size={expected_ranks}")
        if rank in reports:
            raise ValueError(f"duplicate {mode} rank {rank}")
        if len(report.get("steps", [])) != 4 or any(
            not step.get("success") for step in report.get("steps", [])
        ):
            raise ValueError(f"{path} does not contain four successful steps")
        scope_names = {
            scope.get("name")
            for scope in report.get("scopes", [])
            if scope.get("success")
        }
        if not {"transformer_lifecycle", "request_end_to_end"} <= scope_names:
            raise ValueError(f"{path} is missing a successful timing scope")
        match = BACKEND_RE.fullmatch(str(report.get("backend", "")))
        if match is None or match.group("mode") != mode:
            raise ValueError(
                f"{path} backend does not identify the {mode} arm: {report.get('backend')!r}"
            )
        reports[rank] = report

    expected = set(range(expected_ranks))
    if set(reports) != expected:
        raise ValueError(
            f"{mode} ranks differ: expected {sorted(expected)}, got {sorted(reports)}"
        )
    return reports


def category(report: dict[str, Any], kind: str, name: str) -> dict[str, Any]:
    return report.get("summary", {}).get(kind, {}).get(name, {})


def transfer_contract(
    reports: dict[int, dict[str, Any]], mode: str
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    passed = True
    block_sizes: set[int] = set()
    for kind, name in TRANSFER_CATEGORIES:
        calls_by_rank: dict[int, float] = {}
        bytes_by_rank: dict[int, float] = {}
        for rank, report in reports.items():
            item = category(report, kind, name)
            calls_by_rank[rank] = float(item.get("calls", 0))
            byte_count = 0
            for detail in item.get("details", []):
                calls = int(detail.get("calls", 0))
                size = int(detail.get("detail", {}).get("bytes", 0))
                if calls and size:
                    block_sizes.add(size)
                byte_count += calls * size
            bytes_by_rank[rank] = float(byte_count)
        expected_calls = EXPECTED_CALLS[name]
        category_passed = set(calls_by_rank.values()) == {float(expected_calls)}
        passed = passed and category_passed
        categories[name] = {
            "expected_calls_per_rank": expected_calls,
            "passed": category_passed,
            "calls": distribution(calls_by_rank, "calls"),
            "bytes": distribution(bytes_by_rank, "bytes"),
        }

    if len(block_sizes) != 1:
        passed = False
    block_bytes = next(iter(block_sizes), 0)
    expected_stream_bytes = 156 * block_bytes
    for name in ("offload.host_fill", "offload.prefetch_copy"):
        actual = set(categories[name]["bytes"]["by_rank_bytes"].values())
        byte_passed = actual == {float(expected_stream_bytes)}
        categories[name]["expected_bytes_per_rank"] = expected_stream_bytes
        categories[name]["passed"] = categories[name]["passed"] and byte_passed
        passed = passed and byte_passed

    initial_bytes = set(
        categories["offload.initial_copy"]["bytes"]["by_rank_bytes"].values()
    )
    initial_passed = initial_bytes == {0.0}
    categories["offload.initial_copy"]["expected_bytes_per_rank"] = 0
    categories["offload.initial_copy"]["passed"] = (
        categories["offload.initial_copy"]["passed"] and initial_passed
    )
    passed = passed and initial_passed
    return {
        "mode": mode,
        "passed": passed,
        "block_bytes": block_bytes,
        "categories": categories,
    }


def already_done_summary(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    true_by_rank: dict[int, float] = {}
    false_by_rank: dict[int, float] = {}
    for rank, report in reports.items():
        true_calls = false_calls = 0
        item = category(report, "cpu", "offload.host_fill_wait")
        for detail in item.get("details", []):
            calls = int(detail.get("calls", 0))
            if detail.get("detail", {}).get("already_done"):
                true_calls += calls
            else:
                false_calls += calls
        true_by_rank[rank] = float(true_calls)
        false_by_rank[rank] = float(false_calls)
    return {
        "already_done_true": distribution(true_by_rank, "calls"),
        "already_done_false": distribution(false_by_rank, "calls"),
    }


def compute_concurrency(reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for concurrent in (False, True):
        means: dict[int, float] = {}
        calls_by_rank: dict[int, float] = {}
        for rank, report in reports.items():
            calls = 0
            total_ms = 0.0
            for detail in category(
                report, "gpu", "offload.block_compute"
            ).get("details", []):
                if bool(detail.get("detail", {}).get("h2d_enqueued")) == concurrent:
                    calls += int(detail.get("calls", 0))
                    total_ms += float(detail.get("total_ms", 0.0))
            calls_by_rank[rank] = float(calls)
            means[rank] = total_ms / calls if calls else 0.0
        label = "with_concurrent_h2d" if concurrent else "without_concurrent_h2d"
        result[label] = {
            "calls": distribution(calls_by_rank, "calls"),
            "mean_block_ms": distribution(means, "ms"),
        }
    return result


def scope_max(summary: dict[str, Any], name: str) -> float:
    return float(summary["scopes"][name]["max_ms"])


def timing_comparison(b0: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "legacy_four_step_denoising_ms": (
            float(b0["legacy_four_step_denoising"]["critical_path_seconds"]) * 1000,
            float(b1["legacy_four_step_denoising"]["critical_path_seconds"]) * 1000,
        ),
        "runtuner_four_step_critical_ms": (
            float(b0["runtuner_step_wall"]["critical_path_total_ms"]),
            float(b1["runtuner_step_wall"]["critical_path_total_ms"]),
        ),
        "transformer_lifecycle_max_ms": (
            scope_max(b0, "transformer_lifecycle"),
            scope_max(b1, "transformer_lifecycle"),
        ),
        "request_end_to_end_max_ms": (
            scope_max(b0, "request_end_to_end"),
            scope_max(b1, "request_end_to_end"),
        ),
        "host_fill_wait_max_rank_total_ms": (
            float(b0["components"]["cpu"]["offload.host_fill_wait"]["max_ms"]),
            float(b1["components"]["cpu"]["offload.host_fill_wait"]["max_ms"]),
        ),
    }
    result = {}
    for name, (b0_ms, b1_ms) in metrics.items():
        result[name] = {
            "b0_ms": b0_ms,
            "b1_ms": b1_ms,
            "b1_minus_b0_ms": b1_ms - b0_ms,
            "b1_speedup_percent": ((b0_ms - b1_ms) / b0_ms * 100.0)
            if b0_ms
            else 0.0,
        }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_comparison(b0_video: Path, b1_video: Path) -> dict[str, Any]:
    for path in (b0_video, b1_video):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"output video is missing or empty: {path}")
    b0_digest = sha256_file(b0_video)
    b1_digest = sha256_file(b1_video)
    return {
        "exact_file_match": b0_digest == b1_digest,
        "b0": {
            "path": str(b0_video),
            "size_bytes": b0_video.stat().st_size,
            "sha256": b0_digest,
        },
        "b1": {
            "path": str(b1_video),
            "size_bytes": b1_video.stat().st_size,
            "sha256": b1_digest,
        },
        "note": "An exact MP4 match is a strong deterministic-output check; a mismatch requires frame/tensor-level investigation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b0-video", type=Path, required=True)
    parser.add_argument("--b1-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=16)
    args = parser.parse_args()

    try:
        b0_reports = load_reports(args.b0_dir, "b0", args.expected_ranks)
        b1_reports = load_reports(args.b1_dir, "b1", args.expected_ranks)
        b0_summary = json.loads(
            (args.b0_dir / "experiment_a_summary.json").read_text(encoding="utf-8")
        )
        b1_summary = json.loads(
            (args.b1_dir / "experiment_a_summary.json").read_text(encoding="utf-8")
        )
        cache_states = {
            BACKEND_RE.fullmatch(report["backend"]).group("cache")
            for report in (*b0_reports.values(), *b1_reports.values())
        }
        if len(cache_states) != 1:
            raise ValueError(f"B0/B1 cache labels differ: {sorted(cache_states)}")

        b0_contract = transfer_contract(b0_reports, "b0")
        b1_contract = transfer_contract(b1_reports, "b1")
        outputs = output_comparison(args.b0_video, args.b1_video)
        timings = timing_comparison(b0_summary, b1_summary)
        performance_improved = (
            timings["transformer_lifecycle_max_ms"]["b1_minus_b0_ms"] < 0
            and timings["request_end_to_end_max_ms"]["b1_minus_b0_ms"] < 0
        )
        accepted = (
            b0_contract["passed"]
            and b1_contract["passed"]
            and outputs["exact_file_match"]
            and performance_improved
        )
        comparison = {
            "schema_version": 1,
            "kind": "wan22_lazy_b0_b1_comparison",
            "ranks": args.expected_ranks,
            "cache_state": next(iter(cache_states)),
            "accepted": accepted,
            "decision": "proceed_from_b1" if accepted else "inspect_failed_acceptance_checks",
            "acceptance": {
                "b0_transfer_contract": b0_contract["passed"],
                "b1_transfer_contract": b1_contract["passed"],
                "exact_output_file_match": outputs["exact_file_match"],
                "lifecycle_and_request_improved": performance_improved,
            },
            "timings": timings,
            "transfer_contracts": {"b0": b0_contract, "b1": b1_contract},
            "host_fill_completion": {
                "b0": already_done_summary(b0_reports),
                "b1": already_done_summary(b1_reports),
            },
            "block_compute_concurrency": {
                "b0": compute_concurrency(b0_reports),
                "b1": compute_concurrency(b1_reports),
            },
            "outputs": outputs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    lifecycle = timings["transformer_lifecycle_max_ms"]
    request = timings["request_end_to_end_max_ms"]
    wait = timings["host_fill_wait_max_rank_total_ms"]
    print(
        "B0/B1: "
        f"lifecycle {lifecycle['b0_ms'] / 1000:.3f}s -> {lifecycle['b1_ms'] / 1000:.3f}s "
        f"({lifecycle['b1_speedup_percent']:+.2f}%), "
        f"request {request['b0_ms'] / 1000:.3f}s -> {request['b1_ms'] / 1000:.3f}s "
        f"({request['b1_speedup_percent']:+.2f}%), "
        f"fill wait {wait['b0_ms'] / 1000:.3f}s -> {wait['b1_ms'] / 1000:.3f}s"
    )
    print(f"Exact output match: {outputs['exact_file_match']}")
    print(f"Accepted: {accepted}")
    print(f"Comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
