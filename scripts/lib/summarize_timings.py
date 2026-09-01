#!/usr/bin/env python3
"""Print and persist rank-critical timing tables from LightX2V logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


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
SCALES = {
    "wan21": (1, 2, 4, 8, 16),
    "wan22": (2, 4, 8, 16),
    "minimax_h3": (2, 4, 8, 16),
    "ltx25_c1": (2, 4, 8, 16),
}
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
    if not counts or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("counts must be non-empty and unique")
    return counts


def critical_path(profiles, name: str) -> float | None:
    by_rank = profiles.get(name)
    if not by_rank:
        return None
    candidates = [
        sum(samples) for rank, samples in by_rank.items() if rank is not None
    ]
    # Profiler instances created before distributed initialization may label
    # every worker as Single GPU. Those samples execute concurrently.
    candidates.extend(by_rank.get(None, ()))
    return max(candidates) if candidates else None


def encoder_total(profiles, values) -> float | None:
    measured = critical_path(profiles, "Run Encoders")
    if measured is not None:
        return measured

    component_names = (
        "Run Image Encoder",
        "Run VAE Encoder",
        "Run Text Encoder",
    )
    ranks = set().union(*(profiles.get(name, {}).keys() for name in component_names))
    ranks.discard(None)
    rank_totals = []
    for rank in ranks:
        if all(rank in profiles.get(name, {}) for name in component_names):
            rank_totals.append(
                sum(sum(profiles[name][rank]) for name in component_names)
            )
    if rank_totals:
        return max(rank_totals)

    components = [
        values[name] for name in ("Image Encoder", "VAE Encoder", "Text Encoder")
    ]
    if all(value is not None for value in components):
        return sum(components)
    return None


def dit_times(text: str, profiles) -> tuple[float | None, float | None]:
    for match in reversed(list(DIT_SUMMARY_RE.finditer(text))):
        if (
            match.group("status") == "complete"
            and match.group("completed") == match.group("total")
        ):
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


def parse_log(path: Path, *, model: str) -> dict[str, float | None]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    profiles = defaultdict(lambda: defaultdict(list))
    for match in PROFILE_RE.finditer(text):
        rank = int(match.group("rank")) if match.group("rank") is not None else None
        name = " ".join(match.group("name").split())
        profiles[name][rank].append(float(match.group("seconds")))

    values = {
        "Image Encoder": critical_path(profiles, "Run Image Encoder"),
        "VAE Encoder": critical_path(profiles, "Run VAE Encoder"),
        "Text Encoder": critical_path(profiles, "Run Text Encoder"),
        "Encoders 合计": None,
        "DiT 4 steps": None,
        "DiT 每步平均": None,
        "VAE Decoder": critical_path(profiles, "Run VAE Decoder"),
        "RUN pipeline": critical_path(profiles, "RUN pipeline"),
        "Total Cost": critical_path(profiles, "Total Cost"),
    }
    values["Encoders 合计"] = encoder_total(profiles, values)
    if (
        model in ("wan22", "minimax_h3")
        and values["Encoders 合计"] is None
        and values["VAE Encoder"] is not None
        and values["Text Encoder"] is not None
    ):
        values["Encoders 合计"] = values["VAE Encoder"] + values["Text Encoder"]
    values["DiT 4 steps"], values["DiT 每步平均"] = dit_times(text, profiles)
    return values


def format_value(metric: str, value: float | None, *, model: str, csv_mode=False) -> str:
    if value is None:
        if (
            model in ("wan22", "minimax_h3") and metric == "Image Encoder"
        ) or (
            model == "ltx25_c1" and metric in ("Image Encoder", "VAE Encoder")
        ):
            return "N/A"
        return "" if csv_mode else "—"
    return f"{value:.6f}" if csv_mode else f"{value:.3f}s"


def metric_label(metric: str, *, model: str) -> str:
    if model == "ltx25_c1" and metric == "DiT 4 steps":
        return "DiT 11 steps"
    return metric


def write_csv(path: Path, scales, results, *, model: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("metric", *(str(scale) for scale in scales), "unit"))
        for metric in METRICS:
            writer.writerow(
                (
                    metric_label(metric, model=model),
                    *(
                        format_value(
                            metric,
                            results[scale].get(metric),
                            model=model,
                            csv_mode=True,
                        )
                        for scale in scales
                    ),
                    "seconds",
                )
            )


def markdown_table(
    scales,
    results,
    *,
    model: str,
    width: int = 1344,
    height: int = 768,
    frames: int = 124,
    fps: int = 24,
) -> str:
    lines = [
        "| metric | " + " | ".join(str(scale) for scale in scales) + " |",
        "| --- | " + " | ".join("---:" for _ in scales) + " |",
    ]
    for metric in METRICS:
        cells = [
            format_value(metric, results[scale].get(metric), model=model)
            for scale in scales
        ]
        lines.append(
            f"| {metric_label(metric, model=model)} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    if model == "minimax_h3":
        lines.append(
            f"> **MiniMax-H3 口径：**{width}x{height}、{frames} 帧、{fps} FPS；"
            "2/4/8 卡使用 Ulysses，16 卡使用 CP2xU8。跨 rank 指标取关键路径上的最慢 rank。"
        )
    elif model == "ltx25_c1":
        lines.append(
            f"> **LTX-2.5 C1 口径：**{width}x{height}、{frames} 帧、{fps} FPS、"
            "8+3 步；2/4/8/16 卡均使用纯 Ulysses。跨 rank 指标取关键路径上的最慢 rank。"
        )
    else:
        lines.append(
            "> **Total Cost 口径：**LightX2V 内部从 `init_runner(config)` 调用前，"
            "到 `runner.run_pipeline(input_info)` 返回后。Baseline 与 B1 使用完全"
            "相同的起止边界；B1 可能把 eager 初始化工作延迟到 `RUN pipeline` 内，"
            "因此 RUN 上升而 Total 下降并不表示漏计。"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(SCALES), required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--counts", type=parse_counts, required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    scales = SCALES[args.model]
    if any(count not in scales for count in args.counts):
        parser.error(f"{args.model} counts must be selected from {','.join(map(str, scales))}")

    results = {scale: {} for scale in scales}
    errors = []
    if args.model == "ltx25_c1":
        disabled_metrics = {"Image Encoder", "VAE Encoder"}
    elif args.model in ("wan22", "minimax_h3"):
        disabled_metrics = {"Image Encoder"}
    else:
        disabled_metrics = set()
    for scale in args.counts:
        log_path = args.log_dir / f"{scale}gpu.log"
        if not log_path.is_file():
            errors.append(f"missing {scale}-GPU log: {log_path}")
            continue
        results[scale] = parse_log(log_path, model=args.model)
        missing = [
            metric
            for metric in METRICS
            if metric not in disabled_metrics and results[scale].get(metric) is None
        ]
        if missing:
            errors.append(f"{scale}-GPU log is missing: {', '.join(missing)}")

    output_csv = args.log_dir / "timings.csv"
    output_md = args.log_dir / "timings.md"
    write_csv(output_csv, scales, results, model=args.model)
    table = markdown_table(
        scales,
        results,
        model=args.model,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
    )
    output_md.write_text(table, encoding="utf-8")
    print(table, end="")
    if args.model == "wan22":
        print("Image Encoder: N/A (Wan2.2 use_image_encoder=false)")
    elif args.model == "minimax_h3":
        print("Image Encoder: N/A (MiniMax-H3 keyframes are encoded by the VAE path)")
    elif args.model == "ltx25_c1":
        print("Image Encoder: N/A (LTX-2.5 T2AV has no image input)")
        print("VAE Encoder: N/A (LTX-2.5 T2AV starts from latent noise)")
    if args.model in ("minimax_h3", "ltx25_c1"):
        print(
            "Total Cost: LightX2V internal init_runner + run_pipeline; "
            "identical boundary across parallel scales"
        )
    else:
        print(
            "Total Cost: LightX2V internal init_runner + run_pipeline; "
            "identical boundary for Baseline and B1"
        )
    print(f"CSV: {output_csv}")
    print(f"Table: {output_md}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
