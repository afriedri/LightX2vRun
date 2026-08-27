from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/mock_ltx25_b0_memory.py"
SPEC = importlib.util.spec_from_file_location("mock_ltx25_b0_memory", SCRIPT)
MOCK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOCK)


def fake_census(*, block_bytes: int = 773_546_368) -> dict:
    blocks = [
        {"block": block, "tensor_count": 84, "bytes": block_bytes}
        for block in range(48)
    ]
    return {
        "schema_name": "lightx2v.ltx25.weight_census",
        "status": "ok",
        "state": "CENSUS_OK",
        "files": {
            "transformer": {
                "contracts": {
                    "blocks": {
                        "indices": list(range(48)),
                        "blocks": blocks,
                    },
                    "l2_non_block_bytes": 854_954_496,
                    "connector_l1_only_bytes": 4_032_332_800,
                }
            },
            "text_encoder": {},
            "video_vae": {},
            "audio_vae": {},
            "spatial_upscaler": {},
        },
    }


def test_memory_mock_reproduces_frozen_static_accounting():
    report = MOCK.build_report(fake_census())

    assert report["block_bytes"] == 773_546_368
    assert report["gpu_slot_bytes"] == 2 * 773_546_368
    assert report["pinned_slot_bytes"] == 2 * 773_546_368
    assert report["device_static_bytes"] == 2_402_047_232
    assert report["conservative_pinned_bytes"] == 2_402_047_232
    assert report["derived_peak_bytes"] == 22_520_834_560
    assert report["derived_margin_bytes"] == 11_687_908_864
    assert report["gates"]["design_margin_at_least_8_gib"] is True
    assert report["workload"]["stage1_model_evaluations"] == 8
    assert report["workload"]["stage2_model_evaluations"] == 3
    assert report["status"] == "ok"


def test_memory_mock_rejects_wrong_48x84_schema():
    census = fake_census()
    census["files"]["transformer"]["contracts"]["blocks"]["blocks"][3][
        "tensor_count"
    ] = 83

    report = MOCK.build_report(census)

    assert report["status"] == "error"
    assert report["gates"]["block_schema_48x84"] is False
    assert "block_tensor_count_mismatch" in {
        error["code"] for error in report["errors"]
    }


def test_memory_mock_rejects_margin_below_gate():
    report = MOCK.build_report(
        fake_census(),
        card_bytes=24 * 1024**3,
    )

    assert report["status"] == "error"
    assert report["gates"]["budgeted_fit"] is True
    assert report["gates"]["design_margin_at_least_8_gib"] is False
