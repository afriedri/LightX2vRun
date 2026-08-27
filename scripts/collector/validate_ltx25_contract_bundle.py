#!/usr/bin/env python3
"""Validate one LTX-2.5 B0 discovery contract bundle.

This validator proves that the reference run emitted usable C1 input data.  It
does not label the reference run as B0 accepted and does not claim any C1
kernel passed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "lightx2v.ltx2.runcollector"
SCHEMA_VERSION = 1
SUMMARY_SCHEMA = "lightx2v.ltx25.contract_bundle"
SUMMARY_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read collector {path}: {error}") from error
    _require(isinstance(payload, dict), f"collector {path} must be a JSON object")
    return payload


def _validate_common(path: Path, payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_name") == SCHEMA_NAME, f"{path}: wrong schema_name")
    _require(payload.get("schema_version") == SCHEMA_VERSION, f"{path}: wrong schema_version")
    _require(payload.get("model_generation") == "2.5", f"{path}: wrong model_generation")
    _require(payload.get("collector_mode") == "discovery", f"{path}: collector_mode must be discovery")
    _require(payload.get("profiling_only") is True, f"{path}: profiling_only must be true")
    _require(payload.get("numerical_path_changed") is False, f"{path}: numerical_path_changed must be false")
    identity = payload.get("identity")
    _require(isinstance(identity, Mapping), f"{path}: missing identity")
    _require(identity.get("exit_status") == "success", f"{path}: collector did not finalize successfully")
    policy = payload.get("policy")
    _require(isinstance(policy, Mapping), f"{path}: missing policy")
    _require(policy.get("reference_route_only") is True, f"{path}: reference_route_only must be true")
    _require(policy.get("kernel_probe_executed") is False, f"{path}: kernel_probe_executed must be false")
    _require(policy.get("accelerator_synchronize") is False, f"{path}: collector inserted accelerator synchronization")
    validation = payload.get("validation")
    _require(isinstance(validation, Mapping), f"{path}: missing validation")
    _require(validation.get("reference_route_preserved") is True, f"{path}: reference route was not preserved")
    _require(validation.get("accelerator_synchronize_inserted") is False, f"{path}: synchronization validation failed")


def _contracts(path: Path, payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    contracts_root = payload.get("contracts")
    _require(isinstance(contracts_root, Mapping), f"{path}: missing contracts object")
    rows = contracts_root.get("operator_contracts")
    _require(isinstance(rows, list) and rows, f"{path}: operator_contracts must be non-empty")
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"{path}: operator contract {index} is not an object")
        contract_id = row.get("contract_id")
        _require(isinstance(contract_id, str) and len(contract_id) == 64, f"{path}: operator contract {index} has invalid contract_id")
        _require(int(row.get("calls", 0)) > 0, f"{path}: operator contract {index} has no calls")
    return rows


def _observed_stages(payload: Mapping[str, Any]) -> set[str]:
    validation = payload.get("validation") or {}
    stages = validation.get("observed_stages") or []
    return {str(stage) for stage in stages}


def _has_gemm_contract(rows: list[Mapping[str, Any]]) -> bool:
    return any(
        row.get("family") == "gemm"
        and all(name in (row.get("parameters") or {}) for name in ("M", "N", "K"))
        for row in rows
    )


def _has_attention_contract(rows: list[Mapping[str, Any]]) -> bool:
    return any(
        row.get("family") == "attention"
        and all(name in (row.get("parameters") or {}) for name in ("B", "Sq", "Hq", "D"))
        for row in rows
    )


def _is_tensor_contract(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("shape"), list)
        and isinstance(value.get("stride"), list)
        and isinstance(value.get("dtype"), str)
        and isinstance(value.get("layout"), str)
    )


def _projected_qkv_contracts(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("family") == "attention"
        and str(row.get("operator", "")).startswith("ltx25.l1.functional.")
        and all(
            _is_tensor_contract((row.get("inputs") or {}).get(name))
            for name in ("q", "k", "v")
        )
        and (row.get("parameters") or {}).get("qkv_layout") in {"BHSD", "BSHD"}
    ]


def _rope_frequency_contracts(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("family") == "rope"
        and _is_tensor_contract((row.get("inputs") or {}).get("freqs.0"))
        and _is_tensor_contract((row.get("inputs") or {}).get("freqs.1"))
    ]


def _groupnorm_contracts(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("family") == "normalization"
        and "GroupNorm" in str(row.get("backend_class", ""))
        and isinstance((row.get("parameters") or {}).get("num_groups"), int)
        and int((row.get("parameters") or {}).get("num_groups", 0)) > 0
    ]


def _ulysses_collective_contracts(
    rows: list[Mapping[str, Any]], expected_ranks: int
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("family") == "collective"
        and row.get("method") == "all_to_all"
        and str(row.get("operator", "")).startswith("ltx25.l2.ulysses.")
        and _is_tensor_contract((row.get("inputs") or {}).get("input"))
        and _is_tensor_contract((row.get("parameters") or {}).get("output"))
        and (row.get("parameters") or {}).get("world_size") == expected_ranks
    ]


def _require_ultra_report(
    path: Path,
    payload: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> None:
    validation = payload.get("validation") or {}
    _require(
        validation.get("collector_ultra_enabled") is True,
        f"{path}: CollectorUltra was not enabled",
    )
    _require(
        (payload.get("policy") or {}).get("contract_detail") == "ultra",
        f"{path}: contract_detail must be ultra",
    )
    collective_count = sum(row.get("family") == "collective" for row in rows)
    communication = payload.get("communication") or {}
    _require(
        int(communication.get("contract_count", 0)) == collective_count,
        f"{path}: communication summary disagrees with collective contracts",
    )
    _require(
        communication.get("payload_metadata_only") is True,
        f"{path}: communication payload policy must be metadata-only",
    )


def validate_bundle(
    output_dir: Path,
    expected_ranks: int,
    require_upsample: bool = True,
    require_ultra: bool = False,
) -> dict[str, Any]:
    _require(expected_ranks > 0, "expected_ranks must be positive")
    l1_path = output_dir / "ltx25_collector.l1.json"
    _require(l1_path.is_file(), f"missing L1 collector: {l1_path}")
    l1 = _load(l1_path)
    _validate_common(l1_path, l1)
    l1_identity = l1["identity"]
    _require(l1_identity.get("process_role") == "l1-prompt-source", "L1 collector has wrong process_role")
    _require(_observed_stages(l1) == {"L1"}, f"{l1_path}: expected only observed stage L1")
    l1_rows = _contracts(l1_path, l1)
    _require(_has_gemm_contract(l1_rows), f"{l1_path}: no exact GEMM contract")
    _require(_has_attention_contract(l1_rows), f"{l1_path}: no exact attention contract")
    projected_qkv = _projected_qkv_contracts(l1_rows)
    if require_ultra:
        _require_ultra_report(l1_path, l1, l1_rows)
        _require(
            bool(projected_qkv),
            f"{l1_path}: projected L1 Q/K/V contracts were not collected",
        )

    rank_paths = sorted(output_dir.glob("ltx25_collector.rank*.json"))
    _require(len(rank_paths) == expected_ranks, f"expected {expected_ranks} rank collectors, found {len(rank_paths)}")
    ranks: dict[int, dict[str, Any]] = {}
    required_stages = {"L2-S1", "L3"}
    if require_upsample:
        required_stages.update({"L2-Upsample", "L2-S2"})
    run_id = str(l1_identity.get("run_id", ""))
    _require(bool(run_id), "L1 collector is missing run_id")
    video_vae_routes: set[str] = set()
    rope_frequency: list[Mapping[str, Any]] = []
    groupnorm: list[Mapping[str, Any]] = []
    ulysses_collectives: list[Mapping[str, Any]] = []

    for path in rank_paths:
        payload = _load(path)
        _validate_common(path, payload)
        identity = payload["identity"]
        rank = int(identity.get("rank", -1))
        _require(rank not in ranks, f"duplicate collector rank {rank}")
        _require(identity.get("process_role") == "denoise-rank", f"{path}: wrong process_role")
        _require(int(identity.get("world_size", -1)) == expected_ranks, f"{path}: wrong world_size")
        _require(str(identity.get("run_id", "")) == run_id, f"{path}: run_id differs from L1")
        observed = _observed_stages(payload)
        _require(required_stages <= observed, f"{path}: missing observed stages {sorted(required_stages - observed)}")
        rows = _contracts(path, payload)
        _require(_has_gemm_contract(rows), f"{path}: no exact GEMM contract")
        _require(_has_attention_contract(rows), f"{path}: no exact attention contract")
        rank_rope = _rope_frequency_contracts(rows)
        rank_groupnorm = _groupnorm_contracts(rows)
        rank_collectives = _ulysses_collective_contracts(rows, expected_ranks)
        if require_ultra:
            _require_ultra_report(path, payload, rows)
            _require(
                bool(rank_rope),
                f"{path}: RoPE cos/sin frequency contracts were not collected",
            )
            _require(
                bool(rank_collectives),
                f"{path}: Ulysses all-to-all payload contracts were not collected",
            )
        rope_frequency.extend(rank_rope)
        groupnorm.extend(rank_groupnorm)
        ulysses_collectives.extend(rank_collectives)
        video_vae_route = str(
            (payload.get("policy") or {}).get("video_vae_route", "unspecified")
        )
        video_vae_routes.add(video_vae_route)
        ranks[rank] = {
            "path": str(path),
            "operator_contracts": len(rows),
            "observed_stages": sorted(observed),
            "families": sorted({str(row.get("family")) for row in rows}),
            "attention_semantics": sorted({str(row.get("semantic")) for row in rows if row.get("family") == "attention" and row.get("semantic")}),
            "video_vae_route": video_vae_route,
            "rope_frequency_contracts": len(rank_rope),
            "groupnorm_contracts": len(rank_groupnorm),
            "ulysses_collective_contracts": len(rank_collectives),
        }

    _require(set(ranks) == set(range(expected_ranks)), f"collector ranks must be 0..{expected_ranks - 1}, got {sorted(ranks)}")
    _require(
        len(video_vae_routes) == 1,
        f"denoise ranks disagree on video VAE route: {sorted(video_vae_routes)}",
    )
    video_vae_route = next(iter(video_vae_routes))
    if require_ultra:
        _require(
            bool(groupnorm),
            "CollectorUltra found no GroupNorm contract with num_groups",
        )
    status = "CONTRACT_ULTRA_COMPLETE" if require_ultra else "CONTRACT_BUNDLE_COMPLETE"
    meaning = (
        "Ultra operator contracts collected; strict input for isolated C1 MegaOlympics probes"
        if require_ultra
        else "B0 reference contracts collected; input for C1 probes only"
    )
    official_diffvae_collected = video_vae_route == "official_diffvae"
    return {
        "schema_name": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_VERSION,
        "status": status,
        "meaning": meaning,
        "contract_detail": "ultra" if require_ultra else "standard",
        "b0_accepted": False,
        "c1_pass": False,
        "video_vae_route": video_vae_route,
        "official_diffvae_contracts_collected": official_diffvae_collected,
        "numerical_path_changed": False,
        "run_id": run_id,
        "expected_ranks": expected_ranks,
        "l1": {
            "path": str(l1_path),
            "operator_contracts": len(l1_rows),
            "families": sorted({str(row.get("family")) for row in l1_rows}),
            "projected_qkv_contracts": len(projected_qkv),
        },
        "ranks": {str(rank): ranks[rank] for rank in sorted(ranks)},
        "capabilities": {
            "projected_l1_qkv": {
                "status": "collected" if projected_qkv else "not_collected",
                "contracts": len(projected_qkv),
            },
            "rope_frequency_tensors": {
                "status": "collected" if rope_frequency else "not_collected",
                "contracts": len(rope_frequency),
            },
            "groupnorm_parameters": {
                "status": "collected" if groupnorm else "not_collected",
                "contracts": len(groupnorm),
            },
            "ulysses_collective_payloads": {
                "status": "collected" if ulysses_collectives else "not_collected",
                "contracts": len(ulysses_collectives),
                "payload_metadata_only": True,
            },
            "official_diffvae": {
                "status": "collected" if official_diffvae_collected else "not_executed",
                "contracts": 0,
                "promotion_eligible": official_diffvae_collected,
                "reason": None
                if official_diffvae_collected
                else "current executable route is conv_diagnostic",
            },
        },
        "gates": {
            "c1_isolated_probe_input_complete": require_ultra,
            "selector_promotion_allowed": official_diffvae_collected,
            "promotion_blockers": []
            if official_diffvae_collected
            else ["official_diffvae_not_executed", "video_quality_baseline_invalid"],
        },
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=2)
    parser.add_argument("--no-require-upsample", action="store_true")
    parser.add_argument("--require-ultra", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    summary = validate_bundle(
        args.output_dir,
        args.expected_ranks,
        require_upsample=not args.no_require_upsample,
        require_ultra=args.require_ultra,
    )
    summary_path = args.summary or args.output_dir / "ltx25_contract_bundle.json"
    _atomic_write(summary_path, summary)
    print(f"LTX25_{summary['status']} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
