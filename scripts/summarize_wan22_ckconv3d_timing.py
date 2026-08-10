#!/usr/bin/env python3
"""Summarize Wan2.2 timings with the Wan2.1 CK scale-sweep methodology."""

import argparse
import csv
from pathlib import Path

from summarize_wan21_ckconv3d_scale import METRICS, parse_log


SCALES = (2, 4, 8, 16)


def parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc
    if not counts or len(set(counts)) != len(counts) or any(count not in SCALES for count in counts):
        raise argparse.ArgumentTypeError("counts must be unique values from 2,4,8,16")
    return counts


def format_seconds(value):
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, results) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("metric", *(str(scale) for scale in SCALES), "unit"))
        for metric in METRICS:
            writer.writerow(
                (metric, *(format_seconds(results[scale].get(metric)) for scale in SCALES), "seconds")
            )


def markdown_table(results) -> str:
    lines = [
        "| metric | 2 | 4 | 8 | 16 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        cells = []
        for scale in SCALES:
            value = results[scale].get(metric)
            cells.append("—" if value is None else f"{value:.3f}s")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Wan2.2 CK Conv3D 2/4/8/16-GPU stage timings."
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--counts", type=parse_counts, default=SCALES)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    output_csv = args.output_csv or args.log_dir / "timings.csv"
    output_md = args.output_md or args.log_dir / "timings.md"
    results = {scale: {} for scale in SCALES}
    errors = []
    for scale in args.counts:
        log_path = args.log_dir / f"{scale}gpu.log"
        if not log_path.is_file():
            errors.append(f"missing {scale}-GPU log: {log_path}")
            continue
        results[scale] = parse_log(log_path)
        missing = [metric for metric in METRICS if results[scale].get(metric) is None]
        if missing:
            errors.append(f"{scale}-GPU log is missing: {', '.join(missing)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, results)
    table = markdown_table(results)
    output_md.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"CSV: {output_csv}")
    print(f"Table: {output_md}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
