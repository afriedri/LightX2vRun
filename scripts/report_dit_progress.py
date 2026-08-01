#!/usr/bin/env python3
import argparse
import re
import sys


STEP_RE = re.compile(r"==> step_index: (\d+) / (\d+)")
DURATION_RE = re.compile(
    r"\[Profile\]\s+(?:Rank (?P<rank>\d+)|Single GPU)\s+-\s+"
    r"Level1_Log Run Dit every step cost (?P<seconds>\d+(?:\.\d+)?) seconds"
)


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(max(int(round(seconds)), 0), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_rate(seconds_per_iteration: float) -> str:
    if seconds_per_iteration >= 1.0:
        return f"{seconds_per_iteration:.2f}s/it"
    if seconds_per_iteration > 0:
        return f"{1.0 / seconds_per_iteration:.2f}it/s"
    return "?it/s"


def print_progress(completed: int, total: int, elapsed: float) -> None:
    width = 24
    ratio = completed / total if total else 0.0
    filled = min(int(ratio * width), width)
    bar = "#" * filled + "-" * (width - filled)
    average = elapsed / completed if completed else 0.0
    remaining = average * (total - completed) if completed else 0.0
    rate = format_rate(average)
    eta = format_duration(remaining) if completed else "?"
    print(
        f"DiT denoising: {ratio * 100:3.0f}%|{bar}| {completed}/{total} "
        f"[{format_duration(elapsed)}<{eta}, {rate}]",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forward LightX2V logs and report distributed DiT progress."
    )
    parser.add_argument("--ranks", type=int, required=True)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("--ranks must be at least 1")

    durations_by_rank = {rank: [] for rank in range(args.ranks)}
    completed = 0
    total = None
    critical_path_durations = []

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()

        step_match = STEP_RE.search(line)
        if step_match and total is None:
            total = int(step_match.group(2))
            print_progress(0, total, 0.0)

        duration_match = DURATION_RE.search(line)
        if not duration_match:
            continue

        rank_text = duration_match.group("rank")
        rank = int(rank_text) if rank_text is not None else 0
        if rank not in durations_by_rank:
            continue
        durations_by_rank[rank].append(float(duration_match.group("seconds")))

        while all(len(values) > completed for values in durations_by_rank.values()):
            step_seconds = max(
                durations_by_rank[rank_index][completed]
                for rank_index in range(args.ranks)
            )
            critical_path_durations.append(step_seconds)
            completed += 1
            if total is None:
                total = completed
            print_progress(completed, total, sum(critical_path_durations))

    if critical_path_durations:
        elapsed = sum(critical_path_durations)
        average = elapsed / completed
        status = "complete" if total == completed else "incomplete"
        print(
            f"DiT summary: status={status}, steps={completed}/{total}, "
            f"total={elapsed:.3f}s, average={average:.3f}s/it, "
            f"throughput={1.0 / average:.4f}it/s",
            flush=True,
        )
    else:
        print(
            "DiT summary: no synchronized per-step profiler samples were found",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
