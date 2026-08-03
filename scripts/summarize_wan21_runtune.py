#!/usr/bin/env python3
"""Summarize paired Wan2.1 FlyDSL/Sage runtime-tuning reports."""

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    ("dit_step_wall", "step_wall", None),
    ("self_phase", "gpu", "wan.self_phase"),
    ("self_attention", "gpu", None),
    ("cross_phase", "gpu", "wan.cross_phase"),
    ("cross_attention", "gpu", "attention.triton_bf16"),
    ("ffn_phase", "gpu", "wan.ffn_phase"),
    ("fp8_gemm", "gpu", "fp8.gemm"),
    ("fp8_quant", "gpu", "fp8.quant"),
    ("offload_initial_copy", "gpu", "offload.initial_copy"),
    ("offload_prefetch_copy", "gpu", "offload.prefetch_copy"),
    ("offload_init_wait", "cpu", "offload.init_wait"),
    ("offload_load_wait", "cpu", "offload.load_wait"),
    ("offload_compute_wait", "cpu", "offload.compute_wait"),
    ("offload_phase_wait", "cpu", "offload.phase_wait"),
    ("offload_device_wait", "cpu", "offload.device_wait"),
)

SELF_ATTENTION_CATEGORY = {
    "flydsl": "attention.flydsl_bf16",
    "sage": "attention.sage_v2",
}
TARGET_SELF_SHAPE = {"sequence": 74592, "heads": 40, "head_dim": 128}


def load_report(path):
    with Path(path).open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if report.get("schema_version") != 1:
        raise ValueError(f"unsupported runtune schema in {path}")
    if not report.get("steps"):
        raise ValueError(f"runtune report has no steps: {path}")
    successful = [step for step in report["steps"] if step.get("success")]
    if len(successful) != 4 or len(successful) != len(report["steps"]):
        raise ValueError(
            f"runtune report must contain exactly four successful steps: {path}"
        )
    return report


def select_steps(report, steady):
    steps = [step for step in report["steps"] if step.get("success")]
    if steady and len(steps) > 1:
        return steps[1:]
    return steps


def category_total(steps, sample_kind, category):
    if sample_kind == "step_wall":
        return sum(float(step["wall_ms"]) for step in steps)
    return sum(
        float(step.get(sample_kind, {}).get(category, {}).get("total_ms", 0.0))
        for step in steps
    )


def metric_total(report, backend, sample_kind, category, steady):
    steps = select_steps(report, steady)
    if category is None and sample_kind == "gpu":
        category = SELF_ATTENTION_CATEGORY[backend]
    return category_total(steps, sample_kind, category), len(steps)


def self_attention_shapes(report, backend):
    category = report.get("summary", {}).get("gpu", {}).get(
        SELF_ATTENTION_CATEGORY[backend], {}
    )
    return category.get("details", [])


def validate_target_shape(report, backend):
    details = self_attention_shapes(report, backend)
    for entry in details:
        q_shape = entry.get("detail", {}).get("q", {})
        if all(q_shape.get(key) == value for key, value in TARGET_SELF_SHAPE.items()):
            return entry
    observed = [entry.get("detail", {}).get("q", {}) for entry in details]
    raise ValueError(
        f"{backend} report does not contain the target self-attention shape "
        f"{TARGET_SELF_SHAPE}; observed={observed}"
    )


def collect_scope(flydsl, sage, steady):
    rows = []
    for name, sample_kind, category in METRICS:
        flydsl_ms, flydsl_steps = metric_total(
            flydsl, "flydsl", sample_kind, category, steady
        )
        sage_ms, sage_steps = metric_total(sage, "sage", sample_kind, category, steady)
        step_count = min(flydsl_steps, sage_steps)
        rows.append(
            {
                "metric": name,
                "steps": step_count,
                "flydsl_total_ms": flydsl_ms,
                "sage_total_ms": sage_ms,
                "flydsl_mean_per_step_ms": flydsl_ms / flydsl_steps if flydsl_steps else 0.0,
                "sage_mean_per_step_ms": sage_ms / sage_steps if sage_steps else 0.0,
                "sage_speedup": flydsl_ms / sage_ms if sage_ms else None,
            }
        )
    return rows


def write_csv(path, scopes):
    fieldnames = [
        "scope",
        "metric",
        "steps",
        "flydsl_total_ms",
        "sage_total_ms",
        "flydsl_mean_per_step_ms",
        "sage_mean_per_step_ms",
        "sage_speedup",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for scope, rows in scopes.items():
            for row in rows:
                writer.writerow({"scope": scope, **row})


def detail_rows(report, backend):
    rows = []
    for sample_kind in ("gpu", "cpu"):
        categories = report.get("summary", {}).get(sample_kind, {})
        for category, category_stats in sorted(categories.items()):
            for entry in category_stats.get("details", []):
                rows.append(
                    {
                        "backend": backend,
                        "sample_kind": sample_kind,
                        "category": category,
                        "detail_json": json.dumps(
                            entry["detail"], sort_keys=True, separators=(",", ":")
                        ),
                        "calls": entry["calls"],
                        "total_ms": entry["total_ms"],
                        "mean_ms": entry["mean_ms"],
                        "max_ms": entry["max_ms"],
                    }
                )
    return rows


def write_details_csv(path, rows):
    fieldnames = [
        "backend",
        "sample_kind",
        "category",
        "detail_json",
        "calls",
        "total_ms",
        "mean_ms",
        "max_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scope(name, rows):
    print(f"\n{name}")
    print(f"{'metric':24} {'FlyDSL ms/step':>16} {'Sage ms/step':>14} {'speedup':>10}")
    for row in rows:
        speedup = row["sage_speedup"]
        speedup_text = f"{speedup:.4f}x" if speedup is not None else "n/a"
        print(
            f"{row['metric']:24} "
            f"{row['flydsl_mean_per_step_ms']:16.3f} "
            f"{row['sage_mean_per_step_ms']:14.3f} "
            f"{speedup_text:>10}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flydsl", required=True)
    parser.add_argument("--sage", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    flydsl = load_report(args.flydsl)
    sage = load_report(args.sage)
    target_shapes = {
        "flydsl": validate_target_shape(flydsl, "flydsl"),
        "sage": validate_target_shape(sage, "sage"),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes = {
        "all_steps": collect_scope(flydsl, sage, steady=False),
        "steady_steps_excluding_first": collect_scope(flydsl, sage, steady=True),
    }
    summary = {
        "schema_version": 1,
        "flydsl_environment": flydsl.get("environment", {}),
        "sage_environment": sage.get("environment", {}),
        "target_self_attention": {
            "expected_q": TARGET_SELF_SHAPE,
            "flydsl": target_shapes["flydsl"],
            "sage": target_shapes["sage"],
        },
        "breakdown": {
            "flydsl": flydsl.get("summary", {}),
            "sage": sage.get("summary", {}),
        },
        "scopes": scopes,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=True)
        summary_file.write("\n")
    write_csv(output_dir / "summary.csv", scopes)
    write_details_csv(
        output_dir / "details.csv",
        detail_rows(flydsl, "flydsl") + detail_rows(sage, "sage"),
    )

    print_scope("All steps", scopes["all_steps"])
    print_scope("Steady steps (first step excluded)", scopes["steady_steps_excluding_first"])
    print(
        "\nTarget self attention: "
        f"S={TARGET_SELF_SHAPE['sequence']}, H={TARGET_SELF_SHAPE['heads']}, "
        f"D={TARGET_SELF_SHAPE['head_dim']}"
    )
    print(f"\nJSON: {output_dir / 'summary.json'}")
    print(f"CSV : {output_dir / 'summary.csv'}")
    print(f"Shape details: {output_dir / 'details.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
