#!/usr/bin/env python3
"""Summarize MiniMax-H3 gfx1201 logs and probe the output MP4 on the GPU run host."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


LEGACY_STAGE_RE = re.compile(r"\[H3Stage\]\s+(?P<stage>H[0-6](?:-\w+)?)\s*(?P<detail>.*)")
RUNTIME_EVENT_RE = re.compile(r"\[Runtime\]\s+(?P<fields>.+)$")
PROFILE_RE = re.compile(
    r"\[Profile\]\s+(?:Rank (?P<rank>\d+)|(?P<single>Single GPU))\s+-\s+"
    r"(?:Level\d+_Log\s+)?(?P<name>.+?)\s+cost\s+"
    r"(?P<seconds>\d+(?:\.\d+)?)\s+seconds"
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _runtime_value(value: str):
    if value == "none":
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _runtime_tokens(text: str) -> list[str]:
    """Split key/value fields while preserving compact JSON and quoted text."""

    tokens = []
    start = 0
    depth = 0
    quote = None
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
                tokens.append(text[start:index])
            start = index + 1
    if start < len(text):
        tokens.append(text[start:])
    return tokens


def parse_runtime_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        match = RUNTIME_EVENT_RE.search(line)
        if match is None:
            continue
        fields = {}
        for token in _runtime_tokens(match.group("fields")):
            if "=" not in token:
                continue
            name, value = token.split("=", 1)
            fields[name] = _runtime_value(value)
        if {"kind", "model", "stage", "component", "status"}.issubset(fields):
            events.append(fields)
    return events


def parse_stages(text: str, runtime_events: list[dict] | None = None) -> list[dict]:
    events = parse_runtime_events(text) if runtime_events is None else runtime_events
    normalized = [event for event in events if event.get("kind") == "stage" and event.get("model") == "minimax_h3"]
    if normalized:
        return normalized
    return [
        {
            "kind": "stage",
            "model": "minimax_h3",
            "stage": match.group("stage"),
            "component": None,
            "status": None,
            "detail": match.group("detail"),
            "format": "legacy",
        }
        for match in LEGACY_STAGE_RE.finditer(text)
    ]


def summarize_probe(payload: dict) -> dict:
    kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
    return {
        "ok": True,
        "has_video": "video" in kinds,
        "has_audio": "audio" in kinds,
        "format": payload.get("format", {}),
        "streams": payload.get("streams", []),
    }


def ffprobe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"ok": False, "stderr": completed.stderr.strip()}
    return summarize_probe(json.loads(completed.stdout))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--probe-json", type=Path, help="ffprobe JSON captured inside the runtime container")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    text = ANSI_RE.sub("", args.log.read_text(encoding="utf-8", errors="replace"))
    runtime_events = parse_runtime_events(text)
    stages = parse_stages(text, runtime_events)
    profiles = [
        {
            "name": match.group("name"),
            "seconds": float(match.group("seconds")),
        }
        for match in PROFILE_RE.finditer(text)
    ]
    if args.probe_json and args.probe_json.is_file():
        probe = summarize_probe(json.loads(args.probe_json.read_text(encoding="utf-8")))
    else:
        probe = ffprobe(args.video) if args.video and args.video.is_file() else None
    payload = {
        "log": str(args.log),
        "runtime_events": runtime_events,
        "stages": stages,
        "profiles": profiles,
        "ffprobe": probe,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if probe is not None and not (probe.get("ok") and probe.get("has_video") and probe.get("has_audio")):
        raise SystemExit("H3 output MP4 must contain video and audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
