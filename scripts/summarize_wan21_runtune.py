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

SAGE_BREAKDOWN_METRICS = (
    ("sage_prepare", "attention.sage_v2.prepare"),
    ("sage_prepare_layout", "attention.sage_v2.prepare.layout"),
    ("sage_k_mean", "attention.sage_v2.prepare.k_mean"),
    ("sage_v_amax", "attention.sage_v2.prepare.v_amax"),
    ("sage_qkv_quant", "attention.sage_v2.prepare.qkv_quant"),
    ("sage_prepare_padding", "attention.sage_v2.prepare.padding"),
    ("sage_core", "attention.sage_v2.core"),
    ("sage_output", "attention.sage_v2.output"),
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
        flydsl_ms, flydsl_steps = (None, 0)
        sage_ms, sage_steps = (None, 0)
        if flydsl is not None:
            flydsl_ms, flydsl_steps = metric_total(
                flydsl, "flydsl", sample_kind, category, steady
            )
        if sage is not None:
            sage_ms, sage_steps = metric_total(
                sage, "sage", sample_kind, category, steady
            )
        step_count = (
            min(flydsl_steps, sage_steps)
            if flydsl is not None and sage is not None
            else max(flydsl_steps, sage_steps)
        )
        rows.append(
            {
                "metric": name,
                "steps": step_count,
                "flydsl_total_ms": flydsl_ms,
                "sage_total_ms": sage_ms,
                "flydsl_mean_per_step_ms": (
                    flydsl_ms / flydsl_steps if flydsl_steps else None
                ),
                "sage_mean_per_step_ms": (
                    sage_ms / sage_steps if sage_steps else None
                ),
                "sage_speedup": (
                    flydsl_ms / sage_ms
                    if flydsl_ms is not None and sage_ms
                    else None
                ),
            }
        )
    if sage is not None:
        for name, category in SAGE_BREAKDOWN_METRICS:
            sage_ms, sage_steps = metric_total(
                sage, "sage", "gpu", category, steady
            )
            rows.append(
                {
                    "metric": name,
                    "steps": sage_steps,
                    "flydsl_total_ms": None,
                    "sage_total_ms": sage_ms,
                    "flydsl_mean_per_step_ms": None,
                    "sage_mean_per_step_ms": (
                        sage_ms / sage_steps if sage_steps else None
                    ),
                    "sage_speedup": None,
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
        flydsl_ms = row["flydsl_mean_per_step_ms"]
        sage_ms = row["sage_mean_per_step_ms"]
        flydsl_text = f"{flydsl_ms:.3f}" if flydsl_ms is not None else "n/a"
        sage_text = f"{sage_ms:.3f}" if sage_ms is not None else "n/a"
        print(
            f"{row['metric']:24} "
            f"{flydsl_text:>16} "
            f"{sage_text:>14} "
            f"{speedup_text:>10}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flydsl")
    parser.add_argument("--sage")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.flydsl and not args.sage:
        parser.error("at least one of --flydsl or --sage is required")

    flydsl = load_report(args.flydsl) if args.flydsl else None
    sage = load_report(args.sage) if args.sage else None
    target_shapes = {}
    if flydsl is not None:
        target_shapes["flydsl"] = validate_target_shape(flydsl, "flydsl")
    if sage is not None:
        target_shapes["sage"] = validate_target_shape(sage, "sage")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes = {
        "all_steps": collect_scope(flydsl, sage, steady=False),
        "steady_steps_excluding_first": collect_scope(flydsl, sage, steady=True),
    }
    summary = {
        "schema_version": 1,
        "flydsl_environment": flydsl.get("environment", {}) if flydsl else None,
        "sage_environment": sage.get("environment", {}) if sage else None,
        "target_self_attention": {
            "expected_q": TARGET_SELF_SHAPE,
            **target_shapes,
        },
        "breakdown": {
            **({"flydsl": flydsl.get("summary", {})} if flydsl else {}),
            **({"sage": sage.get("summary", {})} if sage else {}),
        },
        "scopes": scopes,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=True)
        summary_file.write("\n")
    write_csv(output_dir / "summary.csv", scopes)
    write_details_csv(
        output_dir / "details.csv",
        (detail_rows(flydsl, "flydsl") if flydsl else [])
        + (detail_rows(sage, "sage") if sage else []),
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
