from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/collector/collect_ltx25_weight_census.py"
SPEC = importlib.util.spec_from_file_location("collect_ltx25_weight_census", SCRIPT)
CENSUS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CENSUS)


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
    *,
    metadata: dict | None = None,
    include_payload: bool = True,
) -> None:
    header: dict = {"__metadata__": {}}
    if metadata:
        header["__metadata__"] = {
            key: json.dumps(value, separators=(",", ":"))
            if not isinstance(value, str)
            else value
            for key, value in metadata.items()
        }
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = CENSUS._DTYPE_BYTES[dtype]
        for value in shape:
            size *= value
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = bytes(offset) if include_payload else b""
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_header_inventory_and_streaming_sha_are_payload_safe(tmp_path):
    checkpoint = tmp_path / "asset.safetensors"
    write_safetensors(
        checkpoint,
        {
            "weight": ("BF16", [4, 4]),
            "scale": ("F32", [1]),
        },
        metadata={"model_version": "2.5.0", "config": {"kind": "fixture"}},
    )
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    report = CENSUS.inspect_asset(
        checkpoint,
        component="fixture",
        relative_path=checkpoint.name,
        expected_size=checkpoint.stat().st_size,
        expected_sha256=expected_sha,
        hash_file=True,
    )

    assert report["status"] == "ok"
    assert report["summary"]["dtype_tensor_counts"] == {"BF16": 1, "F32": 1}
    assert report["safetensors"]["hole_count"] == 0
    assert report["safetensors"]["overlap_count"] == 0
    assert report["probe_contract"]["payload_read"] is False
    assert report["probe_contract"]["sha256_streaming_payload_read"] is True


def test_wrong_sha_is_a_hard_error(tmp_path):
    checkpoint = tmp_path / "asset.safetensors"
    write_safetensors(checkpoint, {"weight": ("BF16", [2, 2])})

    report = CENSUS.inspect_asset(
        checkpoint,
        component="fixture",
        relative_path=checkpoint.name,
        expected_sha256="0" * 64,
        hash_file=True,
    )

    assert report["status"] == "error"
    assert "sha256_mismatch" in {issue["code"] for issue in report["issues"]}


def test_truncated_header_is_rejected(tmp_path):
    checkpoint = tmp_path / "truncated.safetensors"
    checkpoint.write_bytes(struct.pack("<Q", 100) + b"{}")

    with pytest.raises(CENSUS.CensusError, match="truncated Safetensors header"):
        CENSUS.read_header(checkpoint)


def test_wrong_transformer_block_fixture_fails_contract(tmp_path):
    checkpoint = tmp_path / "transformer.safetensors"
    tensors = {
        f"model.diffusion_model.transformer_blocks.{block}.linear.weight": (
            "BF16",
            [4, 4],
        )
        for block in range(47)
    }
    write_safetensors(
        checkpoint,
        tensors,
        metadata={
            "model_version": "2.5.0",
            "gemma_source_checkpoint": {
                "ltx_version": "2.5.0",
                "gemma_version": "gemma4-12b-ltx-v1",
            },
            "config": {"transformer": {"num_layers": 48}},
        },
    )
    report = CENSUS.inspect_asset(
        checkpoint,
        component="transformer",
        relative_path=checkpoint.name,
    )

    CENSUS.validate_transformer(report)
    report["status"] = (
        "error"
        if any(issue["severity"] == "error" for issue in report["issues"])
        else "ok"
    )

    assert report["status"] == "error"
    assert "contract_block_indices" in {
        issue["code"] for issue in report["issues"]
    }


def test_metadata_contract_accepts_architecture_inside_json_list():
    checks = []
    issues = []

    CENSUS._check_metadata_any(
        checks,
        issues,
        {"architectures": ["Gemma4UnifiedForConditionalGeneration"]},
        name="gemma.architecture",
        keys=("architecture", "architectures"),
        expected="Gemma4UnifiedForConditionalGeneration",
    )

    assert checks[0]["ok"] is True
    assert issues == []
