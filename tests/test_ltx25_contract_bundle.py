from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/collector/validate_ltx25_contract_bundle.py"
SPEC = importlib.util.spec_from_file_location("validate_ltx25_contract_bundle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _contract(family: str, *, semantic=None) -> dict:
    parameters = {"M": 16, "N": 32, "K": 64} if family == "gemm" else {"B": 1, "Sq": 16, "Sk": 16, "Hq": 8, "Hkv": 8, "D": 64}
    return {
        "contract_id": ("a" if family == "gemm" else "b") * 64,
        "family": family,
        "semantic": semantic,
        "parameters": parameters,
        "calls": 1,
    }


def _document(run_id: str, role: str, rank: int, world_size: int, stages: list[str]) -> dict:
    return {
        "schema_name": "lightx2v.ltx2.runcollector",
        "schema_version": 1,
        "model_generation": "2.5",
        "collector_mode": "discovery",
        "profiling_only": True,
        "numerical_path_changed": False,
        "identity": {
            "run_id": run_id,
            "process_role": role,
            "rank": rank,
            "world_size": world_size,
            "exit_status": "success",
        },
        "policy": {
            "reference_route_only": True,
            "kernel_probe_executed": False,
            "accelerator_synchronize": False,
        },
        "contracts": {
            "operator_contracts": [
                _contract("gemm"),
                _contract("attention", semantic="video_self"),
            ]
        },
        "validation": {
            "reference_route_preserved": True,
            "accelerator_synchronize_inserted": False,
            "observed_stages": stages,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_ltx25_contract_bundle_accepts_complete_reference_discovery(tmp_path):
    run_id = "fixture-ltx25-contract"
    _write(tmp_path / "ltx25_collector.l1.json", _document(run_id, "l1-prompt-source", 0, 1, ["L1"]))
    stages = ["L2-S1", "L2-Upsample", "L2-S2", "L3"]
    for rank in range(2):
        _write(tmp_path / f"ltx25_collector.rank{rank:03d}.json", _document(run_id, "denoise-rank", rank, 2, stages))

    summary = MODULE.validate_bundle(tmp_path, 2)
    assert summary["status"] == "CONTRACT_BUNDLE_COMPLETE"
    assert summary["b0_accepted"] is False
    assert summary["c1_pass"] is False
    assert set(summary["ranks"]) == {"0", "1"}


def test_validate_ltx25_contract_bundle_rejects_numerical_path_change(tmp_path):
    run_id = "fixture-ltx25-contract"
    l1 = _document(run_id, "l1-prompt-source", 0, 1, ["L1"])
    l1["numerical_path_changed"] = True
    _write(tmp_path / "ltx25_collector.l1.json", l1)
    for rank in range(2):
        _write(
            tmp_path / f"ltx25_collector.rank{rank:03d}.json",
            _document(run_id, "denoise-rank", rank, 2, ["L2-S1", "L2-Upsample", "L2-S2", "L3"]),
        )

    with pytest.raises(ValueError, match="numerical_path_changed"):
        MODULE.validate_bundle(tmp_path, 2)


def test_validate_ltx25_contract_bundle_rejects_cross_run_rank(tmp_path):
    run_id = "fixture-ltx25-contract"
    _write(tmp_path / "ltx25_collector.l1.json", _document(run_id, "l1-prompt-source", 0, 1, ["L1"]))
    stages = ["L2-S1", "L2-Upsample", "L2-S2", "L3"]
    _write(tmp_path / "ltx25_collector.rank000.json", _document(run_id, "denoise-rank", 0, 2, stages))
    _write(tmp_path / "ltx25_collector.rank001.json", _document("different-run", "denoise-rank", 1, 2, stages))

    with pytest.raises(ValueError, match="run_id differs"):
        MODULE.validate_bundle(tmp_path, 2)
