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
            "video_vae_route": "conv_diagnostic",
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


def _tensor(shape: list[int]) -> dict:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return {
        "shape": shape,
        "stride": list(reversed(stride)),
        "dtype": "bfloat16",
        "layout": "contiguous",
        "device": "cuda:0",
        "requires_grad": False,
    }


def _ultra_document(
    run_id: str, role: str, rank: int, world_size: int, stages: list[str]
) -> dict:
    payload = _document(run_id, role, rank, world_size, stages)
    payload["policy"]["contract_detail"] = "ultra"
    payload["provenance"] = {"lightx2v_source_revision": "1" * 40}
    payload["validation"]["collector_ultra_enabled"] = True
    rows = payload["contracts"]["operator_contracts"]
    collective_count = 0
    if role == "l1-prompt-source":
        rows.append(
            {
                "contract_id": "c" * 64,
                "family": "attention",
                "operator": "ltx25.l1.functional.sdpa",
                "method": "scaled_dot_product_attention",
                "inputs": {
                    "q": _tensor([1, 32, 128, 128]),
                    "k": _tensor([1, 32, 128, 128]),
                    "v": _tensor([1, 32, 128, 128]),
                },
                "parameters": {
                    "B": 1,
                    "Sq": 128,
                    "Sk": 128,
                    "Hq": 32,
                    "Hkv": 32,
                    "D": 128,
                    "qkv_layout": "BHSD",
                },
                "calls": 1,
            }
        )
    else:
        rows.extend(
            [
                {
                    "contract_id": "d" * 64,
                    "family": "rope",
                    "operator": "ltx25.l2.transformer.rope",
                    "method": "apply_single",
                    "inputs": {
                        "tensor": _tensor([1, 880, 32, 128]),
                        "freqs.0": _tensor([1, 32, 880, 128]),
                        "freqs.1": _tensor([1, 32, 880, 128]),
                    },
                    "parameters": {},
                    "calls": 1,
                },
                {
                    "contract_id": "e" * 64,
                    "family": "normalization",
                    "backend_class": "torch.nn.modules.normalization.GroupNorm",
                    "parameters": {"num_groups": 32, "num_channels": 128},
                    "calls": 1,
                },
                {
                    "contract_id": "f" * 64,
                    "family": "collective",
                    "operator": "ltx25.l2.ulysses.all_to_all",
                    "method": "all_to_all",
                    "semantic": "qkv.payload",
                    "inputs": {"input": _tensor([3, 880, 32, 128])},
                    "parameters": {
                        "output": _tensor([3, 880, 32, 128]),
                        "world_size": world_size,
                    },
                    "calls": 1,
                },
            ]
        )
        collective_count = 1
    payload["communication"] = {
        "observed": bool(collective_count),
        "contract_count": collective_count,
        "payload_metadata_only": True,
    }
    return payload


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
    assert summary["video_vae_route"] == "conv_diagnostic"
    assert summary["official_diffvae_contracts_collected"] is False
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


def test_validate_ltx25_collector_ultra_requires_all_exact_capabilities(tmp_path):
    run_id = "fixture-ltx25-ultra"
    _write(
        tmp_path / "ltx25_collector.l1.json",
        _ultra_document(run_id, "l1-prompt-source", 0, 1, ["L1"]),
    )
    stages = ["L2-S1", "L2-Upsample", "L2-S2", "L3"]
    for rank in range(2):
        _write(
            tmp_path / f"ltx25_collector.rank{rank:03d}.json",
            _ultra_document(run_id, "denoise-rank", rank, 2, stages),
        )

    summary = MODULE.validate_bundle(tmp_path, 2, require_ultra=True)

    assert summary["status"] == "CONTRACT_ULTRA_COMPLETE"
    assert summary["contract_detail"] == "ultra"
    assert summary["lightx2v_source_revision"] == "1" * 40
    assert summary["gates"]["c1_isolated_probe_input_complete"] is True
    assert summary["gates"]["selector_promotion_allowed"] is False
    assert summary["capabilities"]["projected_l1_qkv"]["status"] == "collected"
    assert summary["capabilities"]["rope_frequency_tensors"]["status"] == "collected"
    assert summary["capabilities"]["groupnorm_parameters"]["status"] == "collected"
    assert summary["capabilities"]["ulysses_collective_payloads"]["status"] == "collected"
    assert summary["capabilities"]["official_diffvae"]["status"] == "not_executed"


def test_validate_ltx25_collector_ultra_rejects_missing_projected_qkv(tmp_path):
    run_id = "fixture-ltx25-ultra-missing-qkv"
    l1 = _ultra_document(run_id, "l1-prompt-source", 0, 1, ["L1"])
    l1["contracts"]["operator_contracts"] = l1["contracts"]["operator_contracts"][:2]
    _write(tmp_path / "ltx25_collector.l1.json", l1)
    stages = ["L2-S1", "L2-Upsample", "L2-S2", "L3"]
    for rank in range(2):
        _write(
            tmp_path / f"ltx25_collector.rank{rank:03d}.json",
            _ultra_document(run_id, "denoise-rank", rank, 2, stages),
        )

    with pytest.raises(ValueError, match="projected L1 Q/K/V"):
        MODULE.validate_bundle(tmp_path, 2, require_ultra=True)
