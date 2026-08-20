from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/summarize_minimax_h3_run.py"
SPEC = importlib.util.spec_from_file_location("summarize_minimax_h3_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_runtime_events_preserve_route_dispatch_and_stage_fields():
    text = "\n".join(
        (
            "[Runtime] kind=route model=minimax_h3 stage=H3-Core component=dit.core.rope "
            "status=configured backend=minimax_h3_triton_strict_rope rank=0/2",
            "[Runtime] kind=stage model=minimax_h3 stage=H3-Core component=dit.denoise "
            "status=progress rank=0/2 step=1 total_steps=4",
            "[Runtime] kind=dispatch model=minimax_h3 stage=H3-Core component=dit.core.rope "
            "status=active backend=minimax_h3_triton_strict_rope rank=1/2 "
            'config={"BLOCK_M":128,"PRELOAD_V":true} enable_fp_fusion=false '
            'message="first real tensor contract" shape=[22333,56,128]',
        )
    )

    events = SUMMARY.parse_runtime_events(text)
    assert [event["kind"] for event in events] == ["route", "stage", "dispatch"]
    assert events[1]["step"] == 1
    assert events[1]["total_steps"] == 4
    assert events[2]["enable_fp_fusion"] is False
    assert events[2]["config"] == {"BLOCK_M": 128, "PRELOAD_V": True}
    assert events[2]["message"] == "first real tensor contract"
    assert events[2]["shape"] == [22333, 56, 128]
    assert SUMMARY.parse_stages(text, events) == [events[1]]


def test_stage_parser_keeps_historical_logs_usable():
    stages = SUMMARY.parse_stages("[H3Stage] H4 video VAE decode\n")
    assert stages == [
        {
            "kind": "stage",
            "model": "minimax_h3",
            "stage": "H4",
            "component": None,
            "status": None,
            "detail": "video VAE decode",
            "format": "legacy",
        }
    ]
