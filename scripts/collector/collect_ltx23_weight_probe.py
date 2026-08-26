#!/usr/bin/env python3
"""Collect a complete, payload-free LTX-2.3 checkpoint compatibility report.

The collector reads only the Safetensors JSON header.  It is safe to run on the
29.5 GB monolith because tensor payloads are never materialized and PyTorch is
not imported.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_NAME = "lightx2v.ltx23.weight_probe"
SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT = Path(
    "/disk/ssd/difumodels-Dis/LTX-2.3-fp8/"
    "ltx-2.3-22b-distilled-fp8.safetensors"
)
DEFAULT_OUTPUT = Path("ltx23_weight_probe.json")
OFFICIAL_SIZE_BYTES = 29_531_884_062
OFFICIAL_SHA256 = "d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2"
LIGHTX2V_MODEL_PREFIX = "model.diffusion_model"
LTX2_ATTENTION_PREFIXES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)
MAX_HEADER_BYTES = 512 * 1024 * 1024

_BLOCK_KEY_RE = re.compile(
    r"^(?:(?P<prefix>.+)\.)?transformer_blocks\."
    r"(?P<block>[0-9]+)\.(?P<suffix>.+)$"
)
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


class ProbeError(RuntimeError):
    """The checkpoint cannot be inspected safely."""


def _parse_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Read exactly the length prefix and JSON header, never tensor payloads."""

    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ProbeError(f"truncated Safetensors length prefix: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= MAX_HEADER_BYTES:
                raise ProbeError(
                    f"invalid Safetensors header size {header_size}: {path}"
                )
            encoded = stream.read(header_size)
    except OSError as exc:
        raise ProbeError(f"cannot read checkpoint {path}: {exc}") from exc
    if len(encoded) != header_size:
        raise ProbeError(
            f"truncated Safetensors header: expected {header_size}, got {len(encoded)}"
        )
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"invalid Safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ProbeError("Safetensors header must be a JSON object")
    return header_size, header


def sha256_file(path: Path) -> tuple[str, float]:
    started = time.perf_counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), time.perf_counter() - started


def _counter(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def _shape_key(shape: Iterable[int]) -> str:
    return "x".join(str(dim) for dim in shape) or "scalar"


def _tensor_nbytes(dtype: str, shape: list[int]) -> int | None:
    itemsize = _DTYPE_BYTES.get(dtype)
    return None if itemsize is None else itemsize * math.prod(shape)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            **context,
        }
    )


def _validate_tensor_entries(
    header: Mapping[str, Any],
    *,
    path: Path,
    header_size: int,
    issues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    payload_size = max(0, path.stat().st_size - 8 - header_size)
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
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in shape
        ):
            _issue(
                issues,
                "error",
                "invalid_tensor_shape",
                "tensor shape must be a list of non-negative integers",
                tensor=name,
                actual=shape,
            )
            continue
        descriptor: dict[str, Any] = {"dtype": dtype, "shape": list(shape)}
        expected_nbytes = _tensor_nbytes(dtype, shape)
        descriptor["expected_nbytes"] = expected_nbytes
        if dtype not in _DTYPE_BYTES:
            _issue(
                issues,
                "warning",
                "unknown_dtype",
                "collector does not know this dtype's byte width",
                tensor=name,
                dtype=dtype,
            )
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        ):
            _issue(
                issues,
                "error",
                "invalid_data_offsets",
                "data_offsets must contain two integers",
                tensor=name,
                actual=offsets,
            )
        else:
            start, end = offsets
            descriptor["data_offsets"] = [start, end]
            descriptor["stored_nbytes"] = end - start
            if not 0 <= start <= end <= payload_size:
                _issue(
                    issues,
                    "error",
                    "data_offsets_out_of_bounds",
                    "tensor data_offsets are outside the checkpoint payload",
                    tensor=name,
                    data_offsets=[start, end],
                    payload_size=payload_size,
                )
            else:
                spans.append((start, end, name))
            if expected_nbytes is not None and end - start != expected_nbytes:
                _issue(
                    issues,
                    "error",
                    "tensor_byte_size_mismatch",
                    "shape/dtype byte count disagrees with data_offsets",
                    tensor=name,
                    expected_nbytes=expected_nbytes,
                    stored_nbytes=end - start,
                )
        tensors[name] = descriptor

    previous_end = 0
    previous_name = None
    overlap_count = 0
    for start, end, name in sorted(spans):
        if start < previous_end:
            overlap_count += 1
            if overlap_count <= 20:
                _issue(
                    issues,
                    "error",
                    "overlapping_tensor_payload",
                    "tensor payload overlaps the preceding tensor",
                    tensor=name,
                    previous_tensor=previous_name,
                    start=start,
                    previous_end=previous_end,
                )
        if end > previous_end:
            previous_end = end
            previous_name = name
    if overlap_count > 20:
        _issue(
            issues,
            "error",
            "overlapping_tensor_payload_truncated",
            "additional overlapping tensor reports were suppressed",
            suppressed=overlap_count - 20,
        )
    return tensors


def _metadata(header: Mapping[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    raw = header.get("__metadata__", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _issue(
            issues,
            "error",
            "invalid_metadata",
            "__metadata__ is not an object",
            actual_type=type(raw).__name__,
        )
        return {"raw_value": raw}
    return {str(key): _parse_metadata_value(value) for key, value in raw.items()}


def _block_entries(
    tensors: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[str, int],
    list[str],
]:
    blocks: dict[int, dict[str, dict[str, Any]]] = {}
    prefixes: collections.Counter[str] = collections.Counter()
    unmatched_transformer_keys: list[str] = []
    for name, descriptor in tensors.items():
        match = _BLOCK_KEY_RE.match(name)
        if match is None:
            if "transformer_blocks." in name:
                unmatched_transformer_keys.append(name)
            continue
        block = int(match.group("block"))
        prefix = match.group("prefix") or ""
        suffix = match.group("suffix")
        prefixes[prefix] += 1
        value = dict(descriptor)
        value["full_name"] = name
        value["namespace_prefix"] = prefix
        previous = blocks.setdefault(block, {}).setdefault(suffix, value)
        if previous["full_name"] != name:
            # Keep every duplicate namespace in the complete tensor inventory;
            # use the first one for normalized block comparisons.
            previous.setdefault("alternate_full_names", []).append(name)
    return blocks, dict(sorted(prefixes.items())), sorted(unmatched_transformer_keys)


def _num_layers(metadata: Mapping[str, Any], blocks: Mapping[int, Any]) -> tuple[int, str]:
    config = metadata.get("config")
    transformer = config.get("transformer") if isinstance(config, Mapping) else None
    value = transformer.get("num_layers") if isinstance(transformer, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value, "metadata.config.transformer.num_layers"
    if blocks:
        return max(blocks) + 1, "inferred_from_highest_transformer_block"
    return 0, "unavailable"


def _block_report(
    blocks: Mapping[int, Mapping[str, Mapping[str, Any]]], num_layers: int
) -> dict[str, Any]:
    all_suffixes = sorted({suffix for block in blocks.values() for suffix in block})
    block_rows: list[dict[str, Any]] = []
    signature_groups: dict[str, dict[str, Any]] = {}
    for block_index in range(num_layers):
        entries = blocks.get(block_index, {})
        signature_payload = [
            [suffix, value.get("dtype"), value.get("shape")]
            for suffix, value in sorted(entries.items())
        ]
        signature = hashlib.sha256(
            json.dumps(signature_payload, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        group = signature_groups.setdefault(
            signature,
            {
                "signature": signature,
                "blocks": [],
                "tensor_count_per_block": len(entries),
                "dtype_counts_per_block": _counter(
                    value.get("dtype") for value in entries.values()
                ),
            },
        )
        group["blocks"].append(block_index)
        block_rows.append(
            {
                "block": block_index,
                "present": block_index in blocks,
                "tensor_count": len(entries),
                "dtype_counts": _counter(value.get("dtype") for value in entries.values()),
                "linear_weight_dtype_counts": _counter(
                    value.get("dtype")
                    for suffix, value in entries.items()
                    if suffix.endswith(".weight") and len(value.get("shape", [])) == 2
                ),
                "namespace_prefixes": sorted(
                    {str(value.get("namespace_prefix", "")) for value in entries.values()}
                ),
                "missing_from_union_count": len(set(all_suffixes) - entries.keys()),
                "signature": signature,
            }
        )

    suffix_catalog = []
    expected_blocks = set(range(num_layers))
    for suffix in all_suffixes:
        present = sorted(index for index, entries in blocks.items() if suffix in entries)
        dtype_groups: dict[str, list[int]] = collections.defaultdict(list)
        shape_groups: dict[str, list[int]] = collections.defaultdict(list)
        names = []
        for block_index in present:
            value = blocks[block_index][suffix]
            dtype_groups[str(value.get("dtype"))].append(block_index)
            shape_groups[_shape_key(value.get("shape", []))].append(block_index)
            names.append(str(value.get("full_name")))
            names.extend(value.get("alternate_full_names", []))
        suffix_catalog.append(
            {
                "suffix": suffix,
                "present_blocks": present,
                "missing_blocks": sorted(expected_blocks - set(present)),
                "dtype_groups": dict(sorted(dtype_groups.items())),
                "shape_groups": dict(sorted(shape_groups.items())),
                "full_names": sorted(names),
            }
        )
    return {
        "num_layers": num_layers,
        "discovered_blocks": sorted(blocks),
        "union_suffix_count": len(all_suffixes),
        "blocks": block_rows,
        "signature_groups": sorted(
            signature_groups.values(), key=lambda row: row["blocks"]
        ),
        "suffix_catalog": suffix_catalog,
    }


def _suffix_candidates(
    expected_name: str, tensors: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    marker = "transformer_blocks."
    tail = expected_name[expected_name.index(marker) :] if marker in expected_name else expected_name
    candidates = []
    for name, descriptor in tensors.items():
        if name == tail or name.endswith(f".{tail}"):
            candidates.append({"name": name, **dict(descriptor)})
    return sorted(candidates, key=lambda row: row["name"])


def _expected_key_status(
    expected_name: str,
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    fuzzy: bool = False,
) -> dict[str, Any]:
    exact = tensors.get(expected_name)
    candidates = _suffix_candidates(expected_name, tensors)
    if exact is not None:
        status = "exact"
    elif candidates:
        status = "alternate_prefix"
    else:
        status = "missing"
    result: dict[str, Any] = {
        "expected_name": expected_name,
        "status": status,
        "exact": None if exact is None else dict(exact),
        "suffix_candidates": candidates,
    }
    if fuzzy and status == "missing":
        result["close_name_candidates"] = difflib.get_close_matches(
            expected_name, list(tensors), n=12, cutoff=0.45
        )
    return result


def _gate_contract(
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    num_layers: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for block in range(num_layers):
        for attention in LTX2_ATTENTION_PREFIXES:
            for parameter in ("weight", "bias"):
                expected = (
                    f"{LIGHTX2V_MODEL_PREFIX}.transformer_blocks.{block}."
                    f"{attention}.to_gate_logits.{parameter}"
                )
                row = _expected_key_status(expected, tensors)
                row.update(
                    {
                        "block": block,
                        "attention": attention,
                        "parameter": parameter,
                    }
                )
                rows.append(row)

    statuses = _counter(row["status"] for row in rows)
    missing = [row["expected_name"] for row in rows if row["status"] == "missing"]
    alternate = [
        row["expected_name"] for row in rows if row["status"] == "alternate_prefix"
    ]
    if missing:
        _issue(
            issues,
            "error",
            "missing_gate_logits",
            "checkpoint lacks one or more gated-attention parameters under any namespace",
            count=len(missing),
            first=missing[0],
        )
    if alternate:
        _issue(
            issues,
            "error",
            "gate_logits_namespace_mismatch",
            "gate parameters exist, but not under the exact LightX2V namespace",
            count=len(alternate),
            first=alternate[0],
        )

    actual_gate_names = sorted(name for name in tensors if "gate" in name.lower())
    quantized_gate_names = [
        name
        for name in actual_gate_names
        if str(tensors[name].get("dtype", "")).startswith("F8_")
        and ".to_gate_logits." in name
    ]
    actual_gate_rows = [
        {"name": name, **dict(tensors[name])} for name in actual_gate_names
    ]
    return {
        "lightx2v_expected_prefix": LIGHTX2V_MODEL_PREFIX,
        "attention_prefixes": list(LTX2_ATTENTION_PREFIXES),
        "expected_parameter_count": len(rows),
        "status_counts": statuses,
        "missing_expected_names": missing,
        "alternate_prefix_expected_names": alternate,
        "expected_matrix": rows,
        "actual_gate_like_tensors": actual_gate_rows,
        "actual_gate_dtype_counts": _counter(
            tensors[name].get("dtype") for name in actual_gate_names
        ),
        "quantized_gate_logits_are_valid_fp8_linears": True,
        "quantized_to_gate_logits": quantized_gate_names,
    }


def _fp8_contract(
    tensors: Mapping[str, Mapping[str, Any]],
    *,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    fp8_names = sorted(
        name
        for name, descriptor in tensors.items()
        if str(descriptor.get("dtype", "")).startswith("F8_")
    )
    fp8_weights = [name for name in fp8_names if name.endswith(".weight")]
    invalid_scale_pairs = []
    fp8_by_block: collections.Counter[int] = collections.Counter()
    fp8_outside_blocks = []
    for weight_name in fp8_weights:
        match = _BLOCK_KEY_RE.match(weight_name)
        if match is None:
            fp8_outside_blocks.append(weight_name)
        else:
            fp8_by_block[int(match.group("block"))] += 1
        layer = weight_name.removesuffix(".weight")
        for kind in ("input_scale", "weight_scale"):
            scale_name = f"{layer}.{kind}"
            scale = tensors.get(scale_name)
            valid = (
                scale is not None
                and scale.get("dtype") == "F32"
                and math.prod(scale.get("shape", [])) == 1
            )
            if not valid:
                invalid_scale_pairs.append(
                    {
                        "weight": weight_name,
                        "expected_scale": scale_name,
                        "actual": None if scale is None else dict(scale),
                    }
                )
    if invalid_scale_pairs:
        _issue(
            issues,
            "error",
            "invalid_fp8_scale_contract",
            "FP8 weights require F32 scalar input_scale and weight_scale siblings",
            count=len(invalid_scale_pairs),
            first=invalid_scale_pairs[0],
        )
    if fp8_outside_blocks:
        _issue(
            issues,
            "warning",
            "fp8_weight_outside_transformer_blocks",
            "FP8 weights were found outside transformer_blocks",
            count=len(fp8_outside_blocks),
            first=fp8_outside_blocks[0],
        )
    return {
        "tensor_count": len(fp8_names),
        "weight_count": len(fp8_weights),
        "dtype_counts": _counter(tensors[name].get("dtype") for name in fp8_names),
        "weight_count_by_block": {
            str(block): count for block, count in sorted(fp8_by_block.items())
        },
        "weights_outside_transformer_blocks": fp8_outside_blocks,
        "invalid_or_missing_scale_count": len(invalid_scale_pairs),
        "invalid_or_missing_scales": invalid_scale_pairs,
        "fp8_weight_names": fp8_weights,
    }


def collect_report(
    checkpoint: Path,
    *,
    calculate_sha256: bool = False,
    expected_keys: Iterable[str] = (),
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ProbeError(f"checkpoint is not a file: {checkpoint}")
    stat = checkpoint.stat()
    issues: list[dict[str, Any]] = []
    header_size, header = read_header(checkpoint)
    metadata = _metadata(header, issues)
    tensors = _validate_tensor_entries(
        header,
        path=checkpoint,
        header_size=header_size,
        issues=issues,
    )
    blocks, block_prefixes, unmatched = _block_entries(tensors)
    num_layers, num_layers_source = _num_layers(metadata, blocks)

    if stat.st_size != OFFICIAL_SIZE_BYTES:
        _issue(
            issues,
            "warning",
            "official_size_mismatch",
            "checkpoint size differs from the pinned official LTX-2.3 FP8 monolith",
            actual=stat.st_size,
            expected=OFFICIAL_SIZE_BYTES,
        )
    if num_layers != 48:
        _issue(
            issues,
            "warning",
            "unexpected_layer_count",
            "LTX-2.3 22B is expected to contain 48 transformer blocks",
            actual=num_layers,
            expected=48,
            source=num_layers_source,
        )
    if unmatched:
        _issue(
            issues,
            "warning",
            "unparsed_transformer_block_keys",
            "some keys contain transformer_blocks but do not match the expected grammar",
            count=len(unmatched),
            first=unmatched[0],
        )

    digest = None
    hash_seconds = None
    if calculate_sha256:
        digest, hash_seconds = sha256_file(checkpoint)
        if digest != OFFICIAL_SHA256:
            _issue(
                issues,
                "error",
                "official_sha256_mismatch",
                "checkpoint SHA256 differs from the pinned official artifact",
                actual=digest,
                expected=OFFICIAL_SHA256,
            )

    expected_key_rows = [
        _expected_key_status(name, tensors, fuzzy=True) for name in expected_keys
    ]
    for row in expected_key_rows:
        if row["status"] != "exact":
            _issue(
                issues,
                "error",
                "requested_runtime_key_not_exact",
                "an explicitly requested runtime key is not present verbatim",
                expected_name=row["expected_name"],
                status=row["status"],
            )

    dtype_expected_bytes: collections.Counter[str] = collections.Counter()
    dtype_stored_bytes: collections.Counter[str] = collections.Counter()
    for descriptor in tensors.values():
        dtype = str(descriptor.get("dtype"))
        if descriptor.get("expected_nbytes") is not None:
            dtype_expected_bytes[dtype] += int(descriptor["expected_nbytes"])
        if descriptor.get("stored_nbytes") is not None:
            dtype_stored_bytes[dtype] += int(descriptor["stored_nbytes"])

    top_level_prefixes = _counter(name.split(".", 1)[0] for name in tensors)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "probe_contract": {
            "payload_read": False,
            "bytes_read_without_sha256": 8 + header_size,
            "torch_imported": False,
            "complete_tensor_inventory_included": True,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header_size_bytes": header_size,
            "payload_size_bytes": max(0, stat.st_size - 8 - header_size),
            "official_identity": {
                "expected_size_bytes": OFFICIAL_SIZE_BYTES,
                "size_matches": stat.st_size == OFFICIAL_SIZE_BYTES,
                "expected_sha256": OFFICIAL_SHA256,
                "sha256": digest,
                "sha256_calculated": calculate_sha256,
                "sha256_matches": None if digest is None else digest == OFFICIAL_SHA256,
                "hash_seconds": hash_seconds,
            },
        },
        "metadata": metadata,
        "overview": {
            "tensor_count": len(tensors),
            "dtype_counts": _counter(value.get("dtype") for value in tensors.values()),
            "rank_counts": _counter(len(value.get("shape", [])) for value in tensors.values()),
            "expected_bytes_by_dtype": dict(sorted(dtype_expected_bytes.items())),
            "stored_bytes_by_dtype": dict(sorted(dtype_stored_bytes.items())),
            "top_level_name_counts": top_level_prefixes,
        },
        "namespaces": {
            "transformer_block_prefix_counts": block_prefixes,
            "lightx2v_expected_transformer_prefix": LIGHTX2V_MODEL_PREFIX,
            "unmatched_transformer_block_keys": unmatched,
        },
        "transformer_blocks": {
            "num_layers_source": num_layers_source,
            **_block_report(blocks, num_layers),
        },
        "contracts": {
            "gate_logits": _gate_contract(
                tensors, num_layers=num_layers, issues=issues
            ),
            "fp8": _fp8_contract(tensors, issues=issues),
            "explicit_runtime_keys": expected_key_rows,
        },
        "issues": issues,
        "issue_counts": _counter(issue["severity"] for issue in issues),
        "tensor_inventory": [
            {"name": name, **dict(descriptor)}
            for name, descriptor in sorted(tensors.items())
        ],
    }
    return report


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProbeError(f"cannot write report {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read only the LTX-2.3 Safetensors header and collect every key, "
            "dtype, shape, namespace, gate-logit and FP8-scale fact needed for diagnosis."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.getenv("LTX23_CHECKPOINT", str(DEFAULT_CHECKPOINT))),
        help="LTX-2.3 FP8 monolith (default: LTX23_CHECKPOINT or pinned B0 path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("LTX23_WEIGHT_PROBE_OUTPUT", str(DEFAULT_OUTPUT))),
        help="JSON report path (default: ./ltx23_weight_probe.json)",
    )
    parser.add_argument(
        "--expect-key",
        action="append",
        default=[],
        help="additional exact LightX2V runtime key to resolve; repeatable",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="also stream the complete file to verify SHA256 (slow; header probe is otherwise payload-free)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return status 1 after writing the report when any error-level issue is found",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = collect_report(
            args.checkpoint,
            calculate_sha256=args.sha256,
            expected_keys=args.expect_key,
        )
        write_report(args.output, report)
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    gate = report["contracts"]["gate_logits"]
    fp8 = report["contracts"]["fp8"]
    namespaces = report["namespaces"]["transformer_block_prefix_counts"]
    print(f"LTX-2.3 weight probe: {Path(args.output).expanduser().resolve()}")
    print(
        "Header-only: "
        f"{report['overview']['tensor_count']} tensors, "
        f"{report['checkpoint']['header_size_bytes']} header bytes"
    )
    print(f"Transformer namespaces: {json.dumps(namespaces, sort_keys=True)}")
    print(f"Gate status: {json.dumps(gate['status_counts'], sort_keys=True)}")
    print(
        "FP8: "
        f"{fp8['weight_count']} weights, "
        f"{fp8['invalid_or_missing_scale_count']} invalid/missing scales"
    )
    print(f"Issues: {json.dumps(report['issue_counts'], sort_keys=True)}")
    has_errors = any(issue["severity"] == "error" for issue in report["issues"])
    return 1 if args.strict and has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
