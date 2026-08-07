#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


SCALES = (1, 2, 4, 8, 16)
METRICS = (
    "Image Encoder",
    "VAE Encoder",
    "Text Encoder",
    "Encoders 合计",
    "DiT 4 steps",
    "DiT 每步平均",
    "VAE Decoder",
    "RUN pipeline",
    "Total Cost",
)
PROFILE_RE = re.compile(
    r"\[Profile\]\s+(?:Rank (?P<rank>\d+)|(?P<single>Single GPU))\s+-\s+"
    r"(?:Level\d+_Log\s+)?(?P<name>.+?)\s+cost\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)\s+seconds"
)
DIT_SUMMARY_RE = re.compile(
    r"DiT summary:\s+status=(?P<status>\w+),\s+"
    r"steps=(?P<completed>\d+)/(?P<total>\d+),\s+"
    r"total=(?P<seconds>\d+(?:\.\d+)?)s,\s+"
    r"average=(?P<average>\d+(?:\.\d+)?)s/it"
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc
    if not counts or len(set(counts)) != len(counts) or any(count not in SCALES for count in counts):
        raise argparse.ArgumentTypeError("counts must be unique values from 1,2,4,8,16")
    return counts


def _critical_path(profiles, name: str):
    by_rank = profiles.get(name)
    if not by_rank:
        return None
    return max(sum(samples) for samples in by_rank.values())


def _encoder_total(profiles, values):
    measured = _critical_path(profiles, "Run Encoders")
    if measured is not None:
        return measured

    component_names = ("Run Image Encoder", "Run VAE Encoder", "Run Text Encoder")
    ranks = set().union(*(profiles.get(name, {}).keys() for name in component_names))
    rank_totals = []
    for rank in ranks:
        if all(rank in profiles.get(name, {}) for name in component_names):
            rank_totals.append(
                sum(sum(profiles[name][rank]) for name in component_names)
            )
    if rank_totals:
        return max(rank_totals)

    components = [values[name] for name in ("Image Encoder", "VAE Encoder", "Text Encoder")]
    if all(value is not None for value in components):
        return sum(components)
    return None


def _dit_times(text: str, profiles):
    summaries = list(DIT_SUMMARY_RE.finditer(text))
    for match in reversed(summaries):
        if match.group("status") == "complete" and match.group("completed") == match.group("total"):
            return float(match.group("seconds")), float(match.group("average"))

    by_rank = profiles.get("Run Dit every step")
    if not by_rank:
        return None, None
    synchronized_steps = min(len(samples) for samples in by_rank.values())
    if synchronized_steps == 0:
        return None, None
    total = sum(
        max(samples[step] for samples in by_rank.values())
        for step in range(synchronized_steps)
    )
    return total, total / synchronized_steps


def parse_log(path: Path) -> dict[str, float | None]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    profiles = defaultdict(lambda: defaultdict(list))
    for match in PROFILE_RE.finditer(text):
        rank = int(match.group("rank")) if match.group("rank") is not None else 0
        name = " ".join(match.group("name").split())
        profiles[name][rank].append(float(match.group("seconds")))

    values = {
        "Image Encoder": _critical_path(profiles, "Run Image Encoder"),
        "VAE Encoder": _critical_path(profiles, "Run VAE Encoder"),
        "Text Encoder": _critical_path(profiles, "Run Text Encoder"),
        "Encoders 合计": None,
        "DiT 4 steps": None,
        "DiT 每步平均": None,
        "VAE Decoder": _critical_path(profiles, "Run VAE Decoder"),
        "RUN pipeline": _critical_path(profiles, "RUN pipeline"),
        "Total Cost": _critical_path(profiles, "Total Cost"),
    }
    values["Encoders 合计"] = _encoder_total(profiles, values)
    values["DiT 4 steps"], values["DiT 每步平均"] = _dit_times(text, profiles)
    return values


def format_seconds(value):
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, results):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("metric", *(str(scale) for scale in SCALES), "unit"))
        for metric in METRICS:
            writer.writerow(
                (metric, *(format_seconds(results[scale].get(metric)) for scale in SCALES), "seconds")
            )


def markdown_table(results) -> str:
    lines = [
        "| metric | 1 | 2 | 4 | 8 | 16 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
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
        description="Summarize Wan2.1 CK Conv3D 1/2/4/8/16-GPU stage timings."
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
