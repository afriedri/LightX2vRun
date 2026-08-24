from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/summarize_minimax_h3_t1.py"
SPEC = importlib.util.spec_from_file_location("summarize_minimax_h3_t1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_report(result_dir: Path, rank: int, world_size: int, step_values: list[dict], *, version: int = 2):
    payload = {
        "schema_name": "lightx2v.minimax_h3.runcollector",
        "schema_version": version,
        "environment": {"rank": rank, "world_size": world_size},
        "workload": {"model": "minimax_h3", "frames": 124, "seq_p_size": world_size},
        "h3_t1": {
            "methodology": {"block_compute": "events"},
            "summary": {
                "steps": [
                    {"step_index": index, "metrics_ms": values}
                    for index, values in enumerate(step_values)
                ],
                "totals_ms": {
                    name: sum(float(step.get(name, 0.0)) for step in step_values)
                    for name in {key for step in step_values for key in step}
                },
            },
        },
    }
    path = result_dir / f"h3_runcollector.rank{rank:03d}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cross_rank_summary_uses_per_step_critical_path(tmp_path):
    _write_report(
        tmp_path,
        0,
        2,
        [
            {"h3_step_gpu_ms": 10, "dynamic_lora_gpu_ms": 2, "exposed_h2d_gpu_ms": 0.5, "host_prefetch_wait_wall_ms": 4},
            {"h3_step_gpu_ms": 12, "dynamic_lora_gpu_ms": 3, "exposed_h2d_gpu_ms": 0.0, "host_prefetch_wait_wall_ms": 2},
        ],
    )
    _write_report(
        tmp_path,
        1,
        2,
        [
            {"h3_step_gpu_ms": 11, "dynamic_lora_gpu_ms": 1, "exposed_h2d_gpu_ms": 1.0, "host_prefetch_wait_wall_ms": 7},
            {"h3_step_gpu_ms": 9, "dynamic_lora_gpu_ms": 4, "exposed_h2d_gpu_ms": 0.25, "host_prefetch_wait_wall_ms": 5},
        ],
    )

    reports = MODULE.load_reports(tmp_path, 2)
    summary = MODULE.build_summary(reports)

    assert summary["schema_name"] == "lightx2v.minimax_h3.t1_summary"
    assert summary["numerical_path_changed"] is False
    assert summary["steps"][0]["critical_rank"] == 1
    assert summary["steps"][1]["critical_rank"] == 0
    assert summary["steps"][0]["critical_path_ms"]["h3_step_gpu_ms"] == 11
    assert summary["steps"][1]["critical_path_ms"]["h3_step_gpu_ms"] == 12
    assert summary["critical_path_totals_ms"]["h3_step_gpu_ms"] == 23
    assert summary["critical_path_totals_ms"]["dynamic_lora_gpu_ms"] == 4
    assert summary["critical_path_totals_ms"]["exposed_h2d_gpu_ms"] == pytest.approx(1.0)
    assert summary["steps"][0]["per_metric_rank_max_ms"]["dynamic_lora_gpu_ms"] == 2
    assert summary["steps"][1]["per_metric_rank_max_ms"]["dynamic_lora_gpu_ms"] == 4
    assert summary["rank_max_wall_diagnostic_totals_ms"]["host_prefetch_wait_wall_ms"] == 12
    assert summary["rank_max_wall_diagnostic_totals_ms"]["swap_wait_wall_ms"] == 0


def test_cross_rank_summary_rejects_old_schema(tmp_path):
    _write_report(tmp_path, 0, 1, [{"h3_step_gpu_ms": 10}], version=1)
    with pytest.raises(ValueError, match="v2"):
        MODULE.load_reports(tmp_path, 1)
