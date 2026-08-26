#!/usr/bin/env python3
"""Validate and summarize one two-GPU LTX-2.3 FP8 B0 result directory."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUNTIME_RE = re.compile(r"\[Runtime\]\s+(?P<fields>.+)$")
PROFILE_RE = re.compile(
    r"\[Profile\]\s+(?:Rank (?P<rank>\d+)|(?P<single>Single GPU))\s+-\s+"
    r"(?:Level\d+_Log\s+)?(?P<name>.+?)\s+cost\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)\s+seconds"
)
B0_FPS = 24
B0_FRAMES = 121
EXPECTED_LOCAL_VIDEO_TOKENS = {"stage1": 1760, "stage2": 7040}
PROMPT_CACHE_TENSORS = {
    "v_context_p",
    "a_context_p",
    "v_context_n",
    "a_context_n",
}


def value(raw: str) -> Any:
    if raw == "none":
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def tokens(text: str) -> list[str]:
    result: list[str] = []
    start = depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "[{(":
            depth += 1
        elif character in "]})":
            depth = max(0, depth - 1)
        elif character.isspace() and depth == 0:
            if start < index:
                result.append(text[start:index])
            start = index + 1
    if start < len(text):
        result.append(text[start:])
    return result


def runtime_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        match = RUNTIME_RE.search(line)
        if match is None:
            continue
        fields = {}
        for token in tokens(match.group("fields")):
            if "=" in token:
                key, raw = token.split("=", 1)
                fields[key] = value(raw)
        if {"kind", "model", "stage", "component", "status"}.issubset(fields):
            events.append(fields)
    return events


def load_json(path: Path, required: bool = True) -> Any:
    if not path.is_file():
        if required:
            raise SystemExit(f"required evidence is missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_ffprobe(payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise SystemExit("LTX-2.3 B0 output must contain video and audio")
    if (int(video.get("width", 0)), int(video.get("height", 0))) != (1280, 704):
        raise SystemExit("LTX-2.3 B0 output must be 1280x704")
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    numerator, denominator = (int(item) for item in rate.split("/", 1))
    actual_fps = numerator / denominator if denominator else 0.0
    if not math.isclose(actual_fps, B0_FPS, rel_tol=0.0, abs_tol=0.01):
        raise SystemExit(f"output FPS is {actual_fps}, expected {B0_FPS}")
    observed = video.get("nb_read_frames") or video.get("nb_frames")
    if observed not in (None, "N/A") and int(observed) != B0_FRAMES:
        raise SystemExit(f"output has {observed} frames, expected {B0_FRAMES}")
    return {
        "width": 1280,
        "height": 704,
        "fps": actual_fps,
        "frames": int(observed) if observed not in (None, "N/A") else None,
        "has_audio": True,
        "format": payload.get("format", {}),
    }


def validate_preflights(result_dir: Path) -> dict[str, Any]:
    reports = {
        mode: load_json(result_dir / f"preflight_{mode}.json")
        for mode in ("host", "runtime", "distributed")
    }
    for mode, report in reports.items():
        if report.get("mode") != mode or report.get("status") != "ok":
            raise SystemExit(f"{mode} preflight did not complete successfully")
        if report.get("fps") != B0_FPS:
            raise SystemExit(f"{mode} preflight does not describe the frozen {B0_FPS} FPS workload")

    distributed = reports["distributed"]
    if distributed.get("world_size") != 2:
        raise SystemExit("distributed preflight world_size must be 2")
    if distributed.get("exact_shapes") is not True:
        raise SystemExit("distributed preflight must use exact B0 GEMM shapes")
    if distributed.get("local_video_tokens") != EXPECTED_LOCAL_VIDEO_TOKENS:
        raise SystemExit("distributed preflight local video-token shapes do not match B0")
    ranks = distributed.get("ranks", [])
    if {rank.get("rank") for rank in ranks} != {0, 1}:
        raise SystemExit("distributed preflight must cover ranks 0 and 1")
    expected_gemms = {
        (1760, 4096, 4096, False),
        (7040, 4096, 4096, False),
        (128, 2048, 2048, True),
    }
    for rank in ranks:
        if rank.get("rccl_all_reduce") != 3.0:
            raise SystemExit(f"rank {rank.get('rank')} RCCL all-reduce probe failed")
        observed = {
            (gemm.get("m"), gemm.get("n"), gemm.get("k"), gemm.get("bias"))
            for gemm in rank.get("gemms", [])
        }
        if observed != expected_gemms:
            raise SystemExit(f"rank {rank.get('rank')} exact FP8 GEMM shapes do not match B0")
    return reports


def validate_routes(events: list[dict[str, Any]]) -> dict[str, Any]:
    ltx = [event for event in events if event.get("model") == "ltx2"]
    stage_names = {event.get("stage") for event in ltx if event.get("kind") == "stage"}
    required_stages = {"L1", "L2-S1", "L2-Upsample", "L2-S2", "L3"}
    missing = required_stages - stage_names
    if missing:
        raise SystemExit(f"normalized LTX stages are missing: {sorted(missing)}")
    scaled_mm = [
        event for event in ltx
        if event.get("kind") == "dispatch"
        and event.get("component") == "dit.gemm.fp8"
        and event.get("backend") == "torch._scaled_mm"
        and event.get("status") == "active"
    ]
    ranks = {str(event.get("rank", "")).split("/", 1)[0] for event in scaled_mm}
    if ranks != {"0", "1"}:
        raise SystemExit(f"torch._scaled_mm dispatch evidence must cover ranks 0 and 1, got {sorted(ranks)}")
    prompt_routes = [
        event
        for event in ltx
        if event.get("kind") == "route"
        and event.get("component") == "prompt.encoder"
        and event.get("backend") == "safetensors.process_isolated"
    ]
    if not prompt_routes:
        raise SystemExit(
            "LTX-2.3 B0 must consume a process-isolated prompt cache"
        )
    return {
        "stage_names": sorted(stage_names),
        "scaled_mm_dispatch_ranks": sorted(ranks),
        "prompt_backend": "safetensors.process_isolated",
    }


def prompt_cache_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "ltx23_prompt_embeddings.safetensors"
    if not path.is_file():
        raise SystemExit(f"required prompt handoff is missing: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SystemExit("prompt handoff has a truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= 16 * 1024 * 1024:
            raise SystemExit(
                f"prompt handoff has an invalid header size: {header_bytes}"
            )
        try:
            header = json.loads(stream.read(header_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"prompt handoff header is invalid: {exc}") from exc
    metadata = header.get("__metadata__", {})
    if metadata.get("schema_name") != "lightx2v.ltx23.prompt_embeddings":
        raise SystemExit("prompt handoff schema is not LTX-2.3 embeddings v1")
    if metadata.get("schema_version") != "1":
        raise SystemExit("prompt handoff schema version must be 1")
    if metadata.get("process_isolated") != "true":
        raise SystemExit("prompt handoff does not prove process isolation")
    tensor_names = set(header) - {"__metadata__"}
    if tensor_names != PROMPT_CACHE_TENSORS:
        raise SystemExit(
            "prompt handoff tensor set mismatch: "
            f"expected={sorted(PROMPT_CACHE_TENSORS)}, got={sorted(tensor_names)}"
        )
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "schema_version": 1,
        "process_isolated": True,
        "producer_pid": int(metadata["producer_pid"]),
        "source_rank": int(metadata["source_rank"]),
        "tensor_shapes": {
            name: header[name].get("shape") for name in sorted(tensor_names)
        },
    }


def source_collector_summary(
    result_dir: Path, required: bool
) -> dict[str, Any] | None:
    path = result_dir / "ltx23_collector.gemma.json"
    report = load_json(path, required=required)
    if report is None:
        return None
    environment = report.get("environment", {})
    if int(environment.get("world_size", 0)) != 1:
        raise SystemExit("Gemma source collector must have world_size=1")
    labels = {
        item.get("label") for item in report.get("device_memory_snapshots", [])
    }
    required_labels = {
        "ltx23_l1_process_start",
        "ltx23_l1_gemma_device_resident",
        "ltx23_l1_prompt_cache_written",
    }
    if not required_labels.issubset(labels):
        raise SystemExit(
            "Gemma source collector is missing lifecycle labels: "
            f"{sorted(required_labels - labels)}"
        )
    scope = next(
        (
            item
            for item in report.get("scopes", [])
            if item.get("name") == "ltx23_l1_process"
        ),
        None,
    )
    if scope is None or not scope.get("success"):
        raise SystemExit("Gemma source collector did not close successfully")
    return {
        "path": str(path),
        "pid": environment.get("pid"),
        "world_size": 1,
        "wall_ms": scope.get("wall_ms"),
        "labels": sorted(labels - {None}),
    }


def phase_handoff_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "phase_handoff.json"
    report = load_json(path)
    if report.get("schema_name") != "lightx2v.ltx23.phase_handoff":
        raise SystemExit("phase handoff report has the wrong schema")
    if report.get("status") != "ok" or report.get("producer_pid_alive"):
        raise SystemExit("Gemma process did not exit cleanly before SP2 startup")
    gpus = report.get("gpus", [])
    if {int(item.get("rocm_index", -1)) for item in gpus} != set(
        report.get("gpu_indices", [])
    ):
        raise SystemExit("phase handoff report does not cover both selected GPUs")
    limit = int(report.get("vram_release_limit_bytes", 0))
    if limit <= 0 or any(int(item.get("used_bytes", limit + 1)) > limit for item in gpus):
        raise SystemExit("physical VRAM did not fall below the handoff limit")
    return report


def collector_summary(result_dir: Path, required: bool) -> dict[str, Any] | None:
    paths = sorted(result_dir.glob("ltx23_collector.rank*.json"))
    if not paths:
        if required:
            raise SystemExit("collector was requested but no per-rank reports were written")
        return None
    if len(paths) != 2:
        raise SystemExit(f"expected two collector reports, got {len(paths)}")
    reports = [load_json(path) for path in paths]
    ranks = {int(report.get("environment", {}).get("rank", -1)) for report in reports}
    if ranks != {0, 1}:
        raise SystemExit(f"collector ranks must be 0 and 1, got {sorted(ranks)}")
    per_rank = []
    for path, report in zip(paths, reports):
        if report.get("schema_version") != 2:
            raise SystemExit(f"collector schema must be 2: {path}")
        if int(report.get("environment", {}).get("world_size", 0)) != 2:
            raise SystemExit(f"collector world_size must be 2: {path}")
        steps = report.get("steps", [])
        stage1 = [item for item in steps if item.get("segment_index") == 0]
        stage2 = [item for item in steps if item.get("segment_index") == 1]
        if len(stage1) != 8 or len(stage2) != 3 or not all(item.get("success") for item in steps):
            raise SystemExit(f"collector must contain successful 8+3 denoise steps: {path}")
        memories = report.get("device_memory_snapshots", [])
        labels = {item.get("label") for item in memories}
        if "ltx23_l1_prompt_cache_loaded" not in labels:
            raise SystemExit(f"DiT rank did not load the prompt handoff: {path}")
        forbidden = {
            "ltx23_l1_gemma_host_model_ready",
            "ltx23_l1_gemma_device_resident",
        }
        if labels & forbidden:
            raise SystemExit(f"DiT rank rebuilt Gemma instead of using the handoff: {path}")
        per_rank.append(
            {
                "rank": report["environment"]["rank"],
                "path": str(path),
                "stage1_wall_ms": sum(float(item["wall_ms"]) for item in stage1),
                "stage2_wall_ms": sum(float(item["wall_ms"]) for item in stage2),
                "fp8_gemm_ms": float(report.get("summary", {}).get("gpu", {}).get("fp8.gemm", {}).get("total_ms", 0.0)),
                "peak_allocated_bytes": max((int(item.get("max_allocated_bytes", 0)) for item in memories), default=0),
                "peak_reserved_bytes": max((int(item.get("max_reserved_bytes", 0)) for item in memories), default=0),
            }
        )
    return {"schema_version": 2, "ranks": per_rank, "critical_rank": max(per_rank, key=lambda item: item["stage1_wall_ms"] + item["stage2_wall_ms"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expect-collector", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    log_path = args.result_dir / "run.log"
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    events = runtime_events(text)
    payload = {
        "schema_name": "lightx2v.ltx23_fp8.b0.summary",
        "schema_version": 1,
        "workload": {
            "final_resolution": [704, 1280],
            "stage1_resolution": [352, 640],
            "fps": B0_FPS,
            "frames": B0_FRAMES,
            "steps": [8, 3],
            "world_size": 2,
        },
        "preflight": validate_preflights(args.result_dir),
        "routes": validate_routes(events),
        "runtime_events": events,
        "profiles": [
            {"rank": int(match.group("rank") or 0), "name": match.group("name"), "seconds": float(match.group("seconds"))}
            for match in PROFILE_RE.finditer(text)
        ],
        "collector": collector_summary(args.result_dir, args.expect_collector),
        "source_collector": source_collector_summary(
            args.result_dir, args.expect_collector
        ),
        "prompt_handoff": prompt_cache_summary(args.result_dir),
        "phase_handoff": phase_handoff_summary(args.result_dir),
        "ffprobe": validate_ffprobe(load_json(args.result_dir / "ffprobe.json")),
        "memory_summary": load_json(args.result_dir / "memory.jsonl.summary.json", required=False),
    }
    output = args.output_json or args.result_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
