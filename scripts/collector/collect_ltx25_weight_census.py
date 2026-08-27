#!/usr/bin/env python3
"""Header-only Day-0 census for the five LTX-2.5 BF16 B0 assets.

The only operation that reads tensor payload bytes is the optional, fixed-size
streaming SHA256 pass.  Safetensors inspection reads the 8-byte length prefix
and JSON header only; Torch and HIP are deliberately not imported.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_NAME = "lightx2v.ltx25.weight_census"
SCHEMA_VERSION = 1
DEFAULT_MODEL_ROOT = Path("/disk/ssd/difumodels-Dis/LTX-2.5")
DEFAULT_OUTPUT_DIR = Path("ltx25-day0")
MAX_HEADER_BYTES = 512 * 1024 * 1024
HASH_CHUNK_BYTES = 16 * 1024 * 1024

EXPECTED_ASSETS: dict[str, dict[str, Any]] = {
    "transformer": {
        "relative_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "size_bytes": 42_018_190_584,
        "sha256": "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
    },
    "text_encoder": {
        "relative_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "size_bytes": 26_263_858_182,
        "sha256": "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
    },
    "video_vae": {
        "relative_path": "vae/ltx-2.5-video-vae-bf16.safetensors",
        "size_bytes": 1_472_223_346,
        "sha256": "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
    },
    "audio_vae": {
        "relative_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "size_bytes": 364_866_540,
        "sha256": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    },
    "spatial_upscaler": {
        "relative_path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "size_bytes": 995_778_752,
        "sha256": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
    },
}

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
_BLOCK_RE = re.compile(
    r"^(?P<prefix>.*(?:^|\.)transformer_blocks)\."
    r"(?P<block>[0-9]+)\.(?P<suffix>.+)$"
)
_EXPECTED_LINEAR_SHAPES = {
    "4096x4096": 384,
    "2048x2048": 576,
    "2048x4096": 144,
    "4096x2048": 48,
    "16384x4096": 48,
    "4096x16384": 48,
    "8192x2048": 48,
    "2048x8192": 48,
    "32x4096": 144,
    "32x2048": 144,
}
_PACKED_GEMMA_ASSETS = {
    "tokenizer_json",
    "hf_asset__tokenizer_config.json",
    "hf_asset__processor_config.json",
    "hf_asset__chat_template.jinja",
    "hf_asset__generation_config.json",
}


class CensusError(RuntimeError):
    """A file cannot be inspected safely enough to produce a census."""


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    issues.append(
        {"severity": severity, "code": code, "message": message, **context}
    )


def _parse_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_metadata(header: Mapping[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    raw = header.get("__metadata__", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _issue(
            issues,
            "error",
            "invalid_metadata",
            "Safetensors __metadata__ must be an object",
        )
        return {}
    return {str(key): _parse_metadata_value(value) for key, value in raw.items()}


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            encoded_size = stream.read(8)
            if len(encoded_size) != 8:
                raise CensusError(f"truncated Safetensors length prefix: {path}")
            header_size = struct.unpack("<Q", encoded_size)[0]
            if not 2 <= header_size <= MAX_HEADER_BYTES:
                raise CensusError(
                    f"invalid Safetensors header size {header_size}: {path}"
                )
            encoded = stream.read(header_size)
    except OSError as exc:
        raise CensusError(f"cannot read checkpoint {path}: {exc}") from exc
    if len(encoded) != header_size:
        raise CensusError(
            f"truncated Safetensors header: expected {header_size}, got {len(encoded)}"
        )
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CensusError(f"invalid Safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CensusError("Safetensors header must be a JSON object")
    return header_size, header


def sha256_file(path: Path) -> tuple[str, float, float]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    elapsed = time.perf_counter() - started
    mib_s = 0.0 if elapsed == 0 else path.stat().st_size / (1024**2) / elapsed
    return digest.hexdigest(), elapsed, mib_s


def _tensor_nbytes(dtype: str, shape: Sequence[int]) -> int | None:
    itemsize = _DTYPE_BYTES.get(dtype)
    return None if itemsize is None else itemsize * math.prod(shape)


def _shape_key(shape: Sequence[int]) -> str:
    return "x".join(str(value) for value in shape) if shape else "scalar"


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def _validate_tensors(
    header: Mapping[str, Any],
    *,
    payload_size: int,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, Mapping):
            _issue(
                issues,
                "error",
                "invalid_tensor_entry",
                "tensor header entry is not an object",
                tensor=name,
            )
            continue
        dtype = str(raw.get("dtype", ""))
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if not isinstance(shape, list) or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in shape
        ):
            _issue(
                issues,
                "error",
                "invalid_tensor_shape",
                "tensor shape must contain non-negative integers",
                tensor=name,
                actual=shape,
            )
            continue
        expected_nbytes = _tensor_nbytes(dtype, shape)
        if expected_nbytes is None:
            _issue(
                issues,
                "error",
                "unknown_dtype",
                "unknown Safetensors dtype byte width",
                tensor=name,
                dtype=dtype,
            )
        valid_offsets = (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offsets
            )
        )
        if not valid_offsets:
            _issue(
                issues,
                "error",
                "invalid_data_offsets",
                "data_offsets must contain exactly two integers",
                tensor=name,
                actual=offsets,
            )
            continue
        start, end = offsets
        stored_nbytes = end - start
        if not 0 <= start <= end <= payload_size:
            _issue(
                issues,
                "error",
                "data_offsets_out_of_bounds",
                "tensor data_offsets are outside the file payload",
                tensor=name,
                data_offsets=[start, end],
                payload_size=payload_size,
            )
        else:
            spans.append((start, end, str(name)))
        if expected_nbytes is not None and stored_nbytes != expected_nbytes:
            _issue(
                issues,
                "error",
                "tensor_byte_size_mismatch",
                "dtype/shape byte count disagrees with data_offsets",
                tensor=name,
                expected_nbytes=expected_nbytes,
                stored_nbytes=stored_nbytes,
            )
        inventory.append(
            {
                "name": str(name),
                "dtype": dtype,
                "shape": list(shape),
                "rank": len(shape),
                "data_offsets": [start, end],
                "stored_nbytes": stored_nbytes,
            }
        )

    gaps: list[dict[str, int]] = []
    overlap_count = 0
    cursor = 0
    previous_name: str | None = None
    for start, end, name in sorted(spans, key=lambda item: (item[0], item[1], item[2])):
        if start > cursor:
            gaps.append({"start": cursor, "end": start, "bytes": start - cursor})
        elif start < cursor and end > start:
            overlap_count += 1
            if overlap_count <= 20:
                _issue(
                    issues,
                    "error",
                    "overlapping_tensor_payload",
                    "tensor payload overlaps the preceding covered range",
                    tensor=name,
                    previous_tensor=previous_name,
                    start=start,
                    covered_end=cursor,
                )
        if end > cursor:
            cursor = end
            previous_name = name
    if cursor < payload_size:
        gaps.append({"start": cursor, "end": payload_size, "bytes": payload_size - cursor})
    if gaps:
        _issue(
            issues,
            "error",
            "payload_holes",
            "tensor spans do not exactly cover the Safetensors payload",
            hole_count=len(gaps),
            hole_bytes=sum(item["bytes"] for item in gaps),
        )
    return inventory, {
        "max_data_offset_end": cursor,
        "overlap_count": overlap_count,
        "hole_count": len(gaps),
        "hole_bytes": sum(item["bytes"] for item in gaps),
        "holes": gaps[:20],
    }


def inspect_asset(
    path: Path,
    *,
    component: str,
    relative_path: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    hash_file: bool = False,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "component": component,
        "input_path": str(path),
        "relative_path": relative_path,
        "probe_contract": {
            "header_only": True,
            "payload_read": False,
            "sha256_streaming_payload_read": bool(hash_file),
            "sha256_chunk_bytes": HASH_CHUNK_BYTES if hash_file else None,
            "torch_imported": False,
            "hip_initialized": False,
        },
        "issues": issues,
    }
    if not path.is_file():
        _issue(issues, "error", "missing_asset", "required B0 asset is missing")
        result["status"] = "error"
        return result

    stat = path.stat()
    result["identity"] = {
        "realpath": os.path.realpath(path),
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "allocated_bytes": stat.st_blocks * 512,
    }
    if expected_size is not None and stat.st_size != expected_size:
        _issue(
            issues,
            "error",
            "size_mismatch",
            "asset size does not match the frozen B0 identity",
            expected=expected_size,
            actual=stat.st_size,
        )
    try:
        header_size, header = read_header(path)
    except CensusError as exc:
        _issue(issues, "error", "invalid_safetensors", str(exc))
        result["status"] = "error"
        return result

    payload_start = 8 + header_size
    payload_size = max(0, stat.st_size - payload_start)
    inventory, boundaries = _validate_tensors(
        header, payload_size=payload_size, issues=issues
    )
    metadata = _parse_metadata(header, issues)
    dtype_tensor_counts = _counter(item["dtype"] for item in inventory)
    dtype_bytes: dict[str, int] = collections.defaultdict(int)
    for item in inventory:
        dtype_bytes[item["dtype"]] += item["stored_nbytes"]
    result.update(
        {
            "safetensors": {
                "header_bytes": header_size,
                "payload_start": payload_start,
                "payload_bytes": payload_size,
                "minimum_complete_file_bytes": payload_start
                + boundaries["max_data_offset_end"],
                **boundaries,
            },
            "metadata": metadata,
            "metadata_keys": sorted(metadata),
            "summary": {
                "tensor_count": len(inventory),
                "dtype_tensor_counts": dtype_tensor_counts,
                "dtype_bytes": dict(sorted(dtype_bytes.items())),
                "rank_histogram": _counter(item["rank"] for item in inventory),
                "shape_histogram": _counter(
                    _shape_key(item["shape"]) for item in inventory
                ),
            },
            "largest_tensors": sorted(
                inventory, key=lambda item: item["stored_nbytes"], reverse=True
            )[:20],
            "tensor_inventory": inventory,
        }
    )
    if hash_file:
        try:
            digest, elapsed, mib_s = sha256_file(path)
        except OSError as exc:
            _issue(issues, "error", "sha256_read_failed", str(exc))
        else:
            result["sha256"] = {
                "value": digest,
                "wall_seconds": elapsed,
                "throughput_mib_s": mib_s,
            }
            if expected_sha256 is not None and digest != expected_sha256:
                _issue(
                    issues,
                    "error",
                    "sha256_mismatch",
                    "asset SHA256 does not match the frozen B0 identity",
                    expected=expected_sha256,
                    actual=digest,
                )
    else:
        result["sha256"] = {
            "value": None,
            "wall_seconds": None,
            "throughput_mib_s": None,
        }
        if expected_sha256 is not None:
            _issue(
                issues,
                "error",
                "sha256_not_computed",
                "CENSUS_OK requires --sha256 for every B0 asset",
            )
    result["status"] = (
        "error" if any(item["severity"] == "error" for item in issues) else "ok"
    )
    return result


def _check_equal(
    checks: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    ok = actual == expected
    checks.append({"name": name, "expected": expected, "actual": actual, "ok": ok})
    if not ok:
        _issue(
            issues,
            "error",
            f"contract_{name.replace('.', '_')}",
            f"asset contract mismatch for {name}",
            expected=expected,
            actual=actual,
        )


def _metadata_config(report: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = report.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    config = metadata.get("config", {})
    return config if isinstance(config, Mapping) else {}


def _block_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    blocks: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    prefixes: collections.Counter[str] = collections.Counter()
    non_block_bytes = 0
    for tensor in report.get("tensor_inventory", []):
        match = _BLOCK_RE.match(tensor["name"])
        if match is None:
            non_block_bytes += tensor["stored_nbytes"]
            continue
        block = int(match.group("block"))
        prefixes[match.group("prefix")] += 1
        blocks[block].append(
            {**tensor, "suffix": match.group("suffix")}
        )
    rows = []
    schemas: dict[int, tuple[Any, ...]] = {}
    for block, tensors in sorted(blocks.items()):
        schemas[block] = tuple(
            sorted(
                (
                    item["suffix"],
                    item["dtype"],
                    tuple(item["shape"]),
                    item["stored_nbytes"],
                )
                for item in tensors
            )
        )
        rows.append(
            {
                "block": block,
                "tensor_count": len(tensors),
                "bytes": sum(item["stored_nbytes"] for item in tensors),
                "dtype_tensor_counts": _counter(item["dtype"] for item in tensors),
                "dtype_bytes": dict(
                    sorted(
                        {
                            dtype: sum(
                                item["stored_nbytes"]
                                for item in tensors
                                if item["dtype"] == dtype
                            )
                            for dtype in {item["dtype"] for item in tensors}
                        }.items()
                    )
                ),
            }
        )
    reference = schemas[min(schemas)] if schemas else ()
    mismatched = [block for block, schema in schemas.items() if schema != reference]
    return {
        "indices": sorted(blocks),
        "prefix_tensor_counts": dict(sorted(prefixes.items())),
        "blocks": rows,
        "schema_identical": not mismatched,
        "schema_mismatch_blocks": mismatched,
        "block_bytes_total": sum(row["bytes"] for row in rows),
        "non_block_bytes": non_block_bytes,
    }


def validate_transformer(report: dict[str, Any]) -> None:
    issues = report["issues"]
    checks: list[dict[str, Any]] = []
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    config = _metadata_config(report)
    transformer = config.get("transformer", {}) if isinstance(config, Mapping) else {}
    source = metadata.get("gemma_source_checkpoint", {}) if isinstance(metadata, Mapping) else {}
    blocks = _block_contract(report)
    inventory = report.get("tensor_inventory", [])

    _check_equal(checks, issues, name="model_version", expected="2.5.0", actual=metadata.get("model_version"))
    _check_equal(checks, issues, name="gemma_source.ltx_version", expected="2.5.0", actual=source.get("ltx_version") if isinstance(source, Mapping) else None)
    _check_equal(checks, issues, name="gemma_source.gemma_version", expected="gemma4-12b-ltx-v1", actual=source.get("gemma_version") if isinstance(source, Mapping) else None)
    _check_equal(checks, issues, name="tensor_count", expected=4_349, actual=summary.get("tensor_count"))
    _check_equal(checks, issues, name="dtype_tensor_counts", expected={"BF16": 4_059, "F32": 290}, actual=summary.get("dtype_tensor_counts"))
    _check_equal(checks, issues, name="dtype_bytes", expected={"BF16": 41_998_589_440, "F32": 18_923_520}, actual=summary.get("dtype_bytes"))
    _check_equal(checks, issues, name="payload_bytes", expected=42_017_512_960, actual=report.get("safetensors", {}).get("payload_bytes"))
    _check_equal(checks, issues, name="block_indices", expected=list(range(48)), actual=blocks["indices"])
    _check_equal(checks, issues, name="block_tensor_counts", expected=[84] * 48, actual=[row["tensor_count"] for row in blocks["blocks"]])
    _check_equal(checks, issues, name="block_bytes", expected=[773_546_368] * 48, actual=[row["bytes"] for row in blocks["blocks"]])
    _check_equal(checks, issues, name="block_schema_identical", expected=True, actual=blocks["schema_identical"])
    _check_equal(checks, issues, name="block_bytes_total", expected=37_130_225_664, actual=blocks["block_bytes_total"])
    _check_equal(checks, issues, name="non_block_bytes", expected=4_887_287_296, actual=blocks["non_block_bytes"])

    video_connector_bytes = sum(
        item["stored_nbytes"]
        for item in inventory
        if ".video_embeddings_connector." in f".{item['name']}."
    )
    audio_connector_bytes = sum(
        item["stored_nbytes"]
        for item in inventory
        if ".audio_embeddings_connector." in f".{item['name']}."
    )
    connector_bytes = video_connector_bytes + audio_connector_bytes
    l2_non_block_bytes = blocks["non_block_bytes"] - connector_bytes
    _check_equal(checks, issues, name="video_connector_bytes", expected=3_225_092_608, actual=video_connector_bytes)
    _check_equal(checks, issues, name="audio_connector_bytes", expected=807_240_192, actual=audio_connector_bytes)
    _check_equal(checks, issues, name="connector_bytes", expected=4_032_332_800, actual=connector_bytes)
    _check_equal(checks, issues, name="l2_non_block_bytes", expected=854_954_496, actual=l2_non_block_bytes)

    expected_architecture = {
        "num_layers": 48,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "cross_attention_dim": 4096,
        "audio_cross_attention_dim": 2048,
        "ff_bias": False,
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_proj_input_norm": False,
        "caption_projection_second_linear": False,
        "frequencies_precision": "float64",
        "rope_type": "split",
        "use_keyframes_abs_pos_embedding": True,
        "use_embeddings_connector": True,
        "apply_gated_attention": True,
        "cross_attention_adaln": True,
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "connector_num_layers": 8,
        "audio_connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 64,
    }
    for key, expected in expected_architecture.items():
        _check_equal(checks, issues, name=f"config.transformer.{key}", expected=expected, actual=transformer.get(key) if isinstance(transformer, Mapping) else None)

    audio_connector_num_layers = (
        transformer.get(
            "audio_connector_num_layers", transformer.get("connector_num_layers")
        )
        if isinstance(transformer, Mapping)
        else None
    )
    _check_equal(
        checks,
        issues,
        name="config.transformer.effective_audio_connector_num_layers",
        expected=8,
        actual=audio_connector_num_layers,
    )

    names = {
        item["name"]
        for item in inventory
        if _BLOCK_RE.match(item["name"])
    }
    video_biases = sorted(
        name for name in names if re.search(r"\.ff\.net\.(?:0\.proj|2)\.bias$", name)
    )
    audio_biases = sorted(
        name for name in names if re.search(r"\.audio_ff\.net\.(?:0\.proj|2)\.bias$", name)
    )
    _check_equal(checks, issues, name="video_ff_bias_count", expected=0, actual=len(video_biases))
    _check_equal(checks, issues, name="audio_ff_bias_count", expected=96, actual=len(audio_biases))

    linear_shapes = collections.Counter(
        _shape_key(item["shape"])
        for item in inventory
        if _BLOCK_RE.match(item["name"])
        and item["dtype"] == "BF16"
        and item["rank"] == 2
        and item["name"].endswith(".weight")
    )
    _check_equal(checks, issues, name="block_bf16_linear_count", expected=1_632, actual=sum(linear_shapes.values()))
    _check_equal(checks, issues, name="block_bf16_linear_shapes", expected=_EXPECTED_LINEAR_SHAPES, actual=dict(sorted(linear_shapes.items())))
    forbidden_dtypes = sorted({item["dtype"] for item in inventory if _BLOCK_RE.match(item["name"])} - {"BF16", "F32"})
    _check_equal(checks, issues, name="block_forbidden_dtypes", expected=[], actual=forbidden_dtypes)
    report["contracts"] = {
        "checks": checks,
        "blocks": blocks,
        "video_connector_bytes": video_connector_bytes,
        "audio_connector_bytes": audio_connector_bytes,
        "connector_l1_only_bytes": connector_bytes,
        "l2_non_block_bytes": l2_non_block_bytes,
        "block_bf16_linear_shape_counts": dict(sorted(linear_shapes.items())),
        "video_ff_bias_present": bool(video_biases),
        "audio_ff_bias_present": bool(audio_biases),
        "effective_audio_connector_num_layers": audio_connector_num_layers,
        "gemma_version": source.get("gemma_version") if isinstance(source, Mapping) else None,
    }


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if str(child_key) == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _check_metadata_any(
    checks: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    *,
    name: str,
    keys: Sequence[str],
    expected: Any,
) -> None:
    direct_values = [
        item for key in keys for item in _recursive_values(metadata, key)
    ]
    values: list[Any] = []

    def add_value(value: Any) -> None:
        values.append(value)
        if isinstance(value, list):
            for child in value:
                add_value(child)

    for value in direct_values:
        add_value(value)
    ok = expected in values
    checks.append({"name": name, "expected": expected, "observed": values, "ok": ok})
    if not ok:
        _issue(issues, "error", f"contract_{name}", f"metadata contract mismatch for {name}", expected=expected, observed=values)


def validate_text_encoder(report: dict[str, Any]) -> None:
    issues = report["issues"]
    checks: list[dict[str, Any]] = []
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    inventory = report.get("tensor_inventory", [])
    names = {item["name"] for item in inventory}
    present_assets = sorted(
        asset for asset in _PACKED_GEMMA_ASSETS if asset in names or any(name.endswith(f".{asset}") for name in names)
    )
    _check_equal(checks, issues, name="gemma.tensor_count", expected=686, actual=summary.get("tensor_count"))
    _check_equal(checks, issues, name="gemma.dtype_tensor_counts", expected={"BF16": 681, "U8": 5}, actual=summary.get("dtype_tensor_counts"))
    _check_equal(checks, issues, name="gemma.packed_assets", expected=sorted(_PACKED_GEMMA_ASSETS), actual=present_assets)
    gemma_config = metadata.get("gemma_config", {}) if isinstance(metadata, Mapping) else {}
    _check_metadata_any(checks, issues, gemma_config, name="gemma.architecture", keys=("architecture", "architectures"), expected="Gemma4UnifiedForConditionalGeneration")
    _check_metadata_any(checks, issues, gemma_config, name="gemma.model_type", keys=("model_type",), expected="gemma4_unified")
    for name, keys, expected in (
        ("gemma.version", ("gemma_version",), "gemma4-12b-ltx-v1"),
        ("gemma.layers", ("num_hidden_layers", "num_layers"), 48),
        ("gemma.hidden_size", ("hidden_size",), 3840),
        ("gemma.intermediate_size", ("intermediate_size",), 15360),
        ("gemma.q_heads", ("num_attention_heads",), 16),
        ("gemma.kv_heads", ("num_key_value_heads",), 8),
        ("gemma.head_dim", ("head_dim",), 256),
        ("gemma.sliding_window", ("sliding_window",), 1024),
    ):
        _check_metadata_any(checks, issues, gemma_config, name=name, keys=keys, expected=expected)
    layer_types = [str(value).lower() for value in _recursive_values(gemma_config, "layer_types") for value in (value if isinstance(value, list) else [value])]
    sliding = sum("sliding" in value for value in layer_types)
    full = sum("full" in value or "global" in value for value in layer_types)
    _check_equal(checks, issues, name="gemma.sliding_layers", expected=40, actual=sliding)
    _check_equal(checks, issues, name="gemma.full_layers", expected=8, actual=full)
    versions = _recursive_values(gemma_config, "gemma_version")
    report["contracts"] = {
        "checks": checks,
        "packed_assets": present_assets,
        "gemma_version": versions[0] if versions else None,
        "sliding_attention_layers": sliding,
        "full_attention_layers": full,
    }


def _metadata_text(report: Mapping[str, Any]) -> str:
    return json.dumps(report.get("metadata", {}), sort_keys=True, ensure_ascii=False)


def _validate_component_metadata(
    report: dict[str, Any], component: str, checks_spec: Sequence[tuple[str, Sequence[str], Any]], required_text: Sequence[str]
) -> None:
    issues = report["issues"]
    checks: list[dict[str, Any]] = []
    metadata = report.get("metadata", {})
    for name, keys, expected in checks_spec:
        _check_metadata_any(checks, issues, metadata, name=name, keys=keys, expected=expected)
    haystack = (
        _metadata_text(report)
        + "\n"
        + "\n".join(
            item.get("name", "") for item in report.get("tensor_inventory", [])
        )
    ).lower()
    for marker in required_text:
        ok = marker.lower() in haystack
        checks.append({"name": f"{component}.metadata_marker.{marker}", "expected": True, "actual": ok, "ok": ok})
        if not ok:
            _issue(issues, "error", f"contract_{component}_{marker.lower()}", f"{component} metadata is missing marker {marker!r}")
    report["contracts"] = {"checks": checks}


def validate_video_vae(report: dict[str, Any]) -> None:
    _validate_component_metadata(
        report,
        "video_vae",
        (
            ("video_vae.default_num_inference_steps", ("default_num_inference_steps",), 1),
            ("video_vae.stage_channels", ("stage_channels", "channels", "dims"), [2048, 1024, 512, 512, 256]),
            ("video_vae.stage_depths", ("stage_depths", "depths"), [4, 6, 4, 2, 8]),
        ),
        ("CausalDiffusionVAE", "NADiffusionDecoder"),
    )


def validate_audio_vae(report: dict[str, Any]) -> None:
    _validate_component_metadata(
        report,
        "audio_vae",
        (
            ("audio_vae.mel_bins", ("mel_bins", "n_mels", "num_mels"), 64),
            ("audio_vae.sample_rate", ("sample_rate", "sampling_rate", "audio_sampling_rate"), 16000),
            ("audio_vae.output_sample_rate", ("output_sampling_rate",), 48000),
            ("audio_vae.stereo", ("stereo",), True),
        ),
        ("vocoder", "bwe"),
    )


def validate_spatial_upscaler(report: dict[str, Any]) -> None:
    _validate_component_metadata(
        report,
        "spatial_upscaler",
        (
            ("upscaler.spatial_scale", ("spatial_scale",), 2.0),
            ("upscaler.spatial", ("spatial_upsample",), True),
            ("upscaler.temporal", ("temporal_upsample",), False),
            ("upscaler.in_channels", ("in_channels",), 128),
            ("upscaler.mid_channels", ("mid_channels",), 1024),
        ),
        ("LatentUpsampler",),
    )


_VALIDATORS = {
    "transformer": validate_transformer,
    "text_encoder": validate_text_encoder,
    "video_vae": validate_video_vae,
    "audio_vae": validate_audio_vae,
    "spatial_upscaler": validate_spatial_upscaler,
}


def collect_census(
    model_root: Path,
    *,
    hash_files: bool,
    assets: Mapping[str, Mapping[str, Any]] = EXPECTED_ASSETS,
    validate_components: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = model_root.resolve(strict=False)
    reports: dict[str, dict[str, Any]] = {}
    for component, expected in assets.items():
        relative_path = str(expected["relative_path"])
        report = inspect_asset(
            model_root / relative_path,
            component=component,
            relative_path=relative_path,
            expected_size=expected.get("size_bytes"),
            expected_sha256=expected.get("sha256"),
            hash_file=hash_files,
        )
        if validate_components and "tensor_inventory" in report and component in _VALIDATORS:
            _VALIDATORS[component](report)
            report["status"] = "error" if any(item["severity"] == "error" for item in report["issues"]) else "ok"
        reports[component] = report

    cross_checks: list[dict[str, Any]] = []
    cross_issues: list[dict[str, Any]] = []
    transformer_version = reports.get("transformer", {}).get("contracts", {}).get("gemma_version")
    text_encoder_version = reports.get("text_encoder", {}).get("contracts", {}).get("gemma_version")
    _check_equal(
        cross_checks,
        cross_issues,
        name="transformer_frozen_gemma_version",
        expected="gemma4-12b-ltx-v1",
        actual=transformer_version,
    )
    _check_equal(
        cross_checks,
        cross_issues,
        name="text_encoder_frozen_gemma_version",
        expected="gemma4-12b-ltx-v1",
        actual=text_encoder_version,
    )
    _check_equal(
        cross_checks,
        cross_issues,
        name="cross_file_gemma_version",
        expected=transformer_version,
        actual=text_encoder_version,
    )

    status = "ok"
    if cross_issues or any(report.get("status") != "ok" for report in reports.values()):
        status = "error"
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_root": str(model_root),
        "model_root_realpath": str(root),
        "hash_files": hash_files,
        "probe_contract": {
            "gpu_count": 0,
            "torch_imported": False,
            "hip_initialized": False,
            "header_reads_tensor_payload": False,
            "sha256_streaming_chunk_bytes": HASH_CHUNK_BYTES if hash_files else None,
        },
        "expected_asset_count": len(assets),
        "files": reports,
        "cross_file_contract": {"checks": cross_checks, "issues": cross_issues},
        "summary": {
            "ok_components": sorted(name for name, report in reports.items() if report.get("status") == "ok"),
            "error_components": sorted(name for name, report in reports.items() if report.get("status") != "ok"),
            "logical_bytes": sum(report.get("identity", {}).get("size_bytes", 0) for report in reports.values()),
            "wall_seconds": time.perf_counter() - started,
        },
        "status": status,
        "state": "CENSUS_OK" if status == "ok" else "CENSUS_FAILED",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# LTX-2.5 Day-0 Weight Census",
        "",
        f"- State: `{report['state']}`",
        f"- Model root: `{report['model_root_realpath']}`",
        f"- Full SHA256: `{str(report['hash_files']).lower()}`",
        f"- Logical bytes: `{report['summary']['logical_bytes']}`",
        f"- Wall seconds: `{report['summary']['wall_seconds']:.3f}`",
        "",
        "| Component | Status | Size (B) | Tensors | SHA256 |",
        "|---|---|---:|---:|---|",
    ]
    for component, item in report["files"].items():
        digest = item.get("sha256", {}).get("value") or "not-computed"
        lines.append(
            f"| `{component}` | `{item.get('status', 'error')}` | "
            f"{item.get('identity', {}).get('size_bytes', 0)} | "
            f"{item.get('summary', {}).get('tensor_count', 0)} | `{digest}` |"
        )
    lines.extend(["", "## Issues", ""])
    issue_count = 0
    for component, item in report["files"].items():
        for issue in item.get("issues", []):
            issue_count += 1
            lines.append(
                f"- `{component}` `{issue['severity']}` `{issue['code']}`: {issue['message']}"
            )
    for issue in report["cross_file_contract"]["issues"]:
        issue_count += 1
        lines.append(
            f"- `cross-file` `{issue['severity']}` `{issue['code']}`: {issue['message']}"
        )
    if not issue_count:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="stream all five files and verify their frozen SHA256 identities",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_census(args.model_root, hash_files=args.sha256)
    json_path = args.output_dir / "ltx25_weight_census.json"
    markdown_path = args.output_dir / "ltx25_weight_census.md"
    write_json(json_path, report)
    write_markdown(markdown_path, report)
    print(
        f"{report['state']} json={json_path} markdown={markdown_path} "
        f"wall_seconds={report['summary']['wall_seconds']:.3f}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
