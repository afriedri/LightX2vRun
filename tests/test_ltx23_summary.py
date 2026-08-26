import importlib.util
import json
import struct
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/summarize_ltx23_b0.py"
SPEC = importlib.util.spec_from_file_location("summarize_ltx23_b0", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def runtime_log() -> str:
    lines = [
        "[Runtime] kind=route model=ltx2 stage=L1 component=prompt.encoder "
        "status=configured backend=safetensors.process_isolated rank=0/2"
    ]
    for stage, component in (
        ("L1", "prompt.encoder"),
        ("L2-S1", "dit.denoise"),
        ("L2-Upsample", "latent.spatial_upsampler"),
        ("L2-S2", "dit.denoise"),
        ("L3", "vae.decode"),
    ):
        lines.append(f"[Runtime] kind=stage model=ltx2 stage={stage} component={component} status=start rank=0/2")
    for rank in (0, 1):
        lines.append(
            f"[Runtime] kind=dispatch model=ltx2 stage=L2 component=dit.gemm.fp8 "
            f"status=active backend=torch._scaled_mm rank={rank}/2 m=1760 n=4096 k=4096"
        )
    return "\n".join(lines)


def collector(rank: int) -> dict:
    steps = [
        {"segment_index": 0, "step_index": index, "success": True, "wall_ms": 10.0}
        for index in range(8)
    ] + [
        {"segment_index": 1, "step_index": index, "success": True, "wall_ms": 20.0}
        for index in range(3)
    ]
    return {
        "schema_version": 2,
        "environment": {"rank": rank, "world_size": 2},
        "steps": steps,
        "summary": {"gpu": {"fp8.gemm": {"total_ms": 12.5 + rank}}},
        "device_memory_snapshots": [
            {
                "label": "ltx23_l1_prompt_cache_loaded",
                "max_allocated_bytes": 100 + rank,
                "max_reserved_bytes": 200 + rank,
            }
        ],
    }


def preflight(mode: str) -> dict:
    report = {"mode": mode, "status": "ok", "fps": 24}
    if mode == "distributed":
        report.update(
            {
                "world_size": 2,
                "exact_shapes": True,
                "local_video_tokens": {"stage1": 1760, "stage2": 7040},
                "ranks": [
                    {
                        "rank": rank,
                        "rccl_all_reduce": 3.0,
                        "gemms": [
                            {"m": 1760, "n": 4096, "k": 4096, "bias": False},
                            {"m": 7040, "n": 4096, "k": 4096, "bias": False},
                            {"m": 128, "n": 2048, "k": 2048, "bias": True},
                        ],
                    }
                    for rank in (0, 1)
                ],
            }
        )
    return report


def test_runtime_parser_preserves_json_details():
    events = SUMMARY.runtime_events(
        "[Runtime] kind=state model=ltx2 stage=B0 component=workload status=validated "
        "rank=0/2 resolution=[704,1280] pools={\"bf16\":2,\"fp8\":2}"
    )
    assert events[0]["resolution"] == [704, 1280]
    assert events[0]["pools"] == {"bf16": 2, "fp8": 2}


def test_collector_requires_exact_8_plus_3_and_two_ranks(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"ltx23_collector.rank{rank}.json").write_text(
            json.dumps(collector(rank)), encoding="utf-8"
        )
    result = SUMMARY.collector_summary(tmp_path, True)
    assert [item["rank"] for item in result["ranks"]] == [0, 1]
    assert result["ranks"][0]["stage1_wall_ms"] == 80.0
    assert result["ranks"][0]["stage2_wall_ms"] == 60.0


def test_route_validation_requires_scaled_mm_on_both_ranks():
    events = SUMMARY.runtime_events(runtime_log())
    assert SUMMARY.validate_routes(events)["scaled_mm_dispatch_ranks"] == ["0", "1"]
    with pytest.raises(SystemExit):
        SUMMARY.validate_routes(events[:-1])


def test_prompt_handoff_header_proves_process_boundary(tmp_path):
    header = {
        "__metadata__": {
            "schema_name": "lightx2v.ltx23.prompt_embeddings",
            "schema_version": "1",
            "process_isolated": "true",
            "producer_pid": "123",
            "source_rank": "0",
        },
        **{
            name: {"dtype": "BF16", "shape": [1, 8, 16], "data_offsets": [0, 0]}
            for name in SUMMARY.PROMPT_CACHE_TENSORS
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    (tmp_path / "ltx23_prompt_embeddings.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded
    )
    result = SUMMARY.prompt_cache_summary(tmp_path)
    assert result["process_isolated"] is True
    assert result["producer_pid"] == 123


def test_phase_handoff_requires_pid_exit_and_low_physical_vram(tmp_path):
    report = {
        "schema_name": "lightx2v.ltx23.phase_handoff",
        "status": "ok",
        "producer_pid_alive": False,
        "gpu_indices": [0, 1],
        "vram_release_limit_bytes": 1024**3,
        "gpus": [
            {"rocm_index": 0, "used_bytes": 64 * 1024**2},
            {"rocm_index": 1, "used_bytes": 64 * 1024**2},
        ],
    }
    (tmp_path / "phase_handoff.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    assert SUMMARY.phase_handoff_summary(tmp_path)["status"] == "ok"
    report["gpus"][1]["used_bytes"] = 2 * 1024**3
    (tmp_path / "phase_handoff.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        SUMMARY.phase_handoff_summary(tmp_path)


def test_ffprobe_contract_accepts_704x1280_av():
    payload = {
        "streams": [
            {"codec_type": "video", "width": 1280, "height": 704, "avg_frame_rate": "24/1", "nb_read_frames": "121"},
            {"codec_type": "audio", "sample_rate": "48000"},
        ],
        "format": {"duration": "5.041667"},
    }
    assert SUMMARY.validate_ffprobe(payload)["frames"] == 121


def test_preflight_contract_requires_two_rank_exact_shape_probe(tmp_path):
    for mode in ("host", "runtime", "distributed"):
        (tmp_path / f"preflight_{mode}.json").write_text(
            json.dumps(preflight(mode)), encoding="utf-8"
        )
    reports = SUMMARY.validate_preflights(tmp_path)
    assert reports["distributed"]["world_size"] == 2
    broken = preflight("distributed")
    broken["exact_shapes"] = False
    (tmp_path / "preflight_distributed.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        SUMMARY.validate_preflights(tmp_path)
