#!/usr/bin/env python3
"""Inspect one or both HVAT H3 Ref2VA ComfyUI weights without Torch or GPU allocation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import struct

from collect_ltx25_weight_census import inspect_asset, write_json

FILES = {
    "bf16": "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
    "int8_convrot": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
}
ASSET_CHOICES = {
    "both": tuple(FILES),
    "bf16": ("bf16",),
    "int8-convrot": ("int8_convrot",),
}
SUPPORTED_CONVROT_GROUPSIZES = {16, 64, 256}
EXPECTED_MAIN_BLOCKS = set(range(50))
EXPECTED_REFINER_BLOCKS = set(range(2))
H3_ANCHOR_KEYS = {
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "condition_proj.weight",
    "final_layer.video_out.weight",
    "final_layer.audio_out.weight",
}
EXPECTED_CORE_QUANT_BASES = {
    f"blocks.{block}.{suffix}"
    for block in EXPECTED_MAIN_BLOCKS
    for suffix in (
        "attn.qkv_proj",
        "attn.out_proj",
        "mlp.fc1",
        "mlp.fc2",
        "adaln_proj.linear",
    )
}
EXPECTED_CORE_QUANT_SHAPES = {
    "attn.qkv_proj": [21504, 5376],
    "attn.out_proj": [5376, 7168],
    "mlp.fc1": [28672, 5376],
    "mlp.fc2": [5376, 14336],
    "adaln_proj.linear": [96768, 2688],
}


def _layer_kind(base: str) -> str:
    if base.endswith(".attn.qkv_proj"):
        return "fused_qkv_split"
    if base.endswith(".mlp.fc1"):
        return "fused_fc1_half_swap"
    if base.startswith("blocks."):
        return "main_block_direct_rename"
    if base.startswith("token_refiner.blocks."):
        return "refiner_direct_rename"
    return "fixed_direct_rename"


def _merge_quant_sources(previous: dict | None, tensor_entry: dict) -> dict:
    """Match ComfyUI's legacy-metadata precedence while retaining both observations."""
    if previous is None:
        return {**tensor_entry, "source": "tensor", "effective_source": "tensor"}

    merged = dict(previous)
    merged["source"] = "tensor+safetensors_metadata"
    merged["effective_source"] = "safetensors_metadata"
    if previous.get("status") != "ok" or tensor_entry.get("status") != "ok":
        merged["tensor_representation"] = tensor_entry
        merged["representations_match"] = None
        return merged

    matches = previous.get("config") == tensor_entry.get("config")
    merged["representations_match"] = matches
    if not matches:
        # comfy.utils.convert_old_quants() writes metadata-derived
        # ``.comfy_quant`` entries after loading the tensor state dict, so the
        # safetensors metadata is the effective value when both are present.
        merged["shadowed_tensor_config"] = tensor_entry.get("config")
    return merged


def audit_int8_convrot(result: dict) -> dict:
    """Correlate ComfyUI quant metadata with its weight/scale header entries."""
    tensors = {item["name"]: item for item in result.get("tensor_inventory", [])}
    quant_layers = result.get("comfy_quant_layers", {})
    issues = []
    rows = []

    def issue(severity: str, code: str, message: str, **context) -> None:
        issues.append({"severity": severity, "code": code, "message": message, **context})

    for metadata_issue in result.get("quant_metadata_issues", []):
        issue("error", "invalid_quantization_metadata", metadata_issue)

    for quant_name, parsed in sorted(quant_layers.items()):
        base = quant_name.removesuffix(".comfy_quant")
        row = {"layer": base, "kind": _layer_kind(base), "metadata_tensor": quant_name}
        if parsed.get("status") != "ok":
            issue("error", "invalid_comfy_quant", "layer quantization JSON could not be parsed", layer=base, detail=parsed)
            row["status"] = "error"
            rows.append(row)
            continue
        config = parsed["config"]
        params = config.get("params", {})
        if not isinstance(params, dict):
            params = {}
        quant_format = config.get("format")
        convrot = config.get("convrot", params.get("convrot", False))
        groupsize = config.get("convrot_groupsize", params.get("convrot_groupsize", 256))
        weight = tensors.get(f"{base}.weight")
        scale = tensors.get(f"{base}.weight_scale")
        row.update(
            format=quant_format,
            convrot=convrot,
            convrot_groupsize=groupsize,
            full_precision_matrix_mult=config.get("full_precision_matrix_mult", False),
            metadata_source=parsed.get("source", "unknown"),
            effective_metadata_source=parsed.get("effective_source", parsed.get("source", "unknown")),
            metadata_representations_match=parsed.get("representations_match"),
            weight=None if weight is None else {"dtype": weight["dtype"], "shape": weight["shape"]},
            weight_scale=None if scale is None else {"dtype": scale["dtype"], "shape": scale["shape"]},
        )
        before = len(issues)
        if quant_format != "int8_tensorwise":
            issue("error", "unsupported_quant_format", "LightX2V H3 INT8 ConvRot expects int8_tensorwise", layer=base, actual=quant_format)
        if convrot is not True:
            issue("error", "convrot_not_enabled", "INT8 layer does not enable ConvRot", layer=base, actual=convrot)
        if not isinstance(groupsize, int) or isinstance(groupsize, bool) or groupsize not in SUPPORTED_CONVROT_GROUPSIZES:
            issue(
                "error", "unsupported_convrot_groupsize", "ConvRot group size is not supported by LightX2V",
                layer=base, actual=groupsize, supported=sorted(SUPPORTED_CONVROT_GROUPSIZES),
            )
        if weight is None:
            issue("error", "missing_quant_weight", "quantized layer is missing its weight tensor", layer=base)
        elif weight["dtype"] != "I8" or len(weight["shape"]) != 2:
            issue(
                "error", "invalid_quant_weight", "quantized layer weight must be a rank-2 I8 tensor",
                layer=base, dtype=weight["dtype"], shape=weight["shape"],
            )
        else:
            core_match = re.match(r"^blocks\.\d+\.(.+)$", base)
            suffix = core_match.group(1) if core_match is not None else None
            expected_shape = EXPECTED_CORE_QUANT_SHAPES.get(suffix)
            if expected_shape is not None and weight["shape"] != expected_shape:
                issue(
                    "error", "core_projection_shape", "Core projection shape does not match MiniMax-H3",
                    layer=base, expected=expected_shape, actual=weight["shape"],
                )
            if isinstance(groupsize, int) and not isinstance(groupsize, bool) and groupsize > 0 and weight["shape"][1] % groupsize:
                issue(
                    "error", "input_not_divisible_by_groupsize", "weight input width is not divisible by ConvRot group size",
                    layer=base, input_width=weight["shape"][1], groupsize=groupsize,
                )
            if row["kind"] == "fused_qkv_split" and weight["shape"][0] % 3:
                issue("error", "qkv_output_not_divisible", "fused QKV output width cannot be split into Q/K/V", layer=base, actual=weight["shape"][0])
            if row["kind"] == "fused_fc1_half_swap" and weight["shape"][0] % 2:
                issue("error", "fc1_output_not_divisible", "fused FC1 output width cannot be split into gate/value halves", layer=base, actual=weight["shape"][0])
        if scale is None:
            issue("error", "missing_weight_scale", "quantized layer is missing its weight_scale tensor", layer=base)
        elif scale["dtype"] != "F32":
            issue("error", "invalid_weight_scale_dtype", "ConvRot weight_scale must remain F32", layer=base, actual=scale["dtype"])
        elif weight is not None and len(weight["shape"]) == 2:
            output_width = weight["shape"][0]
            if not scale["shape"] or scale["shape"][0] != output_width or math.prod(scale["shape"]) != output_width:
                issue(
                    "error", "invalid_weight_scale_shape", "weight_scale must contain one value per output channel",
                    layer=base, output_width=output_width, actual=scale["shape"],
                )
        row["status"] = "ok" if len(issues) == before else "error"
        rows.append(row)

    quant_bases = {name.removesuffix(".comfy_quant") for name in quant_layers}
    missing_quant_bases = sorted(EXPECTED_CORE_QUANT_BASES - quant_bases)
    extra_quant_bases = sorted(quant_bases - EXPECTED_CORE_QUANT_BASES)
    if missing_quant_bases:
        issue(
            "error", "missing_core_quant_layers",
            "LightX2V int8-convrot instantiates all five Core projections as quantized weights",
            count=len(missing_quant_bases), layers=missing_quant_bases,
        )
    if extra_quant_bases:
        issue(
            "error", "unsupported_extra_quant_layers",
            "ComfyUI quantizes layers that LightX2V currently keeps in BF16/FP32",
            count=len(extra_quant_bases), layers=extra_quant_bases,
        )
    orphan_scales = sorted(
        name for name in tensors
        if name.endswith(".weight_scale") and name.removesuffix(".weight_scale") not in quant_bases
    )
    if orphan_scales:
        issue("error", "orphan_weight_scales", "weight_scale tensors have no matching comfy_quant metadata", tensors=orphan_scales)
    main_blocks = {int(match.group(1)) for name in tensors if (match := re.match(r"^blocks\.(\d+)\.", name))}
    refiner_blocks = {int(match.group(1)) for name in tensors if (match := re.match(r"^token_refiner\.blocks\.(\d+)\.", name))}
    missing_anchors = sorted(H3_ANCHOR_KEYS - tensors.keys())
    if main_blocks != EXPECTED_MAIN_BLOCKS:
        issue(
            "error", "main_block_set", "checkpoint does not expose the expected 50 H3 Core blocks",
            missing=sorted(EXPECTED_MAIN_BLOCKS - main_blocks), unexpected=sorted(main_blocks - EXPECTED_MAIN_BLOCKS),
        )
    if refiner_blocks != EXPECTED_REFINER_BLOCKS:
        issue(
            "error", "refiner_block_set", "checkpoint does not expose the expected two H3 Refiner blocks",
            missing=sorted(EXPECTED_REFINER_BLOCKS - refiner_blocks), unexpected=sorted(refiner_blocks - EXPECTED_REFINER_BLOCKS),
        )
    if missing_anchors:
        issue("error", "missing_h3_anchors", "checkpoint is missing H3 model-detection/IO anchors", tensors=missing_anchors)
    if not quant_layers:
        issue("error", "missing_comfy_quant", "checkpoint contains no ComfyUI per-layer quantization metadata")

    transformer_config = result.get("metadata", {}).get("config", {})
    if isinstance(transformer_config, dict):
        transformer_config = transformer_config.get("transformer", {})
    if not isinstance(transformer_config, dict) or not transformer_config:
        issue(
            "warning", "missing_transformer_metadata",
            "metadata.config.transformer is absent; converter must infer/freeze architecture from tensor shapes",
        )

    errors = [item for item in issues if item["severity"] == "error"]
    return {
        "status": "ready" if not errors else "blocked",
        "error_count": len(errors),
        "warning_count": sum(item["severity"] == "warning" for item in issues),
        "issue_code_counts": dict(Counter(item["code"] for item in issues)),
        "quantized_layer_count": len(rows),
        "layer_kind_counts": dict(Counter(row["kind"] for row in rows)),
        "valid_layer_count": sum(row.get("status") == "ok" for row in rows),
        "invalid_layer_count": sum(row.get("status") != "ok" for row in rows),
        "full_precision_matrix_mult_enabled_count": sum(row.get("full_precision_matrix_mult") is True for row in rows),
        "shadowed_tensor_metadata_count": sum(row.get("metadata_representations_match") is False for row in rows),
        "main_block_indices": sorted(main_blocks),
        "refiner_block_indices": sorted(refiner_blocks),
        "missing_anchor_keys": missing_anchors,
        "expected_core_quantized_layer_count": len(EXPECTED_CORE_QUANT_BASES),
        "expected_target_quantized_linear_count": len(EXPECTED_CORE_QUANT_BASES) + 2 * len(EXPECTED_MAIN_BLOCKS),
        "missing_core_quant_layers": missing_quant_bases,
        "unsupported_extra_quant_layers": extra_quant_bases,
        "orphan_weight_scales": orphan_scales,
        "issues": issues,
        "layers": rows,
    }


def inspect(root: Path, kind: str, relative: str) -> dict:
    path = root / relative
    result = inspect_asset(path, component=kind, relative_path=relative)
    result["runtime_validated"] = False
    if result["status"] != "ok":
        return result
    groupsizes = {}
    comfy_quant_layers = {}
    quant_metadata_issues = []
    quantization_formats = Counter()
    metadata_bytes = 0
    metadata_limit = 65536
    scalar_formats = {"I32": "<i", "I64": "<q"}
    legacy_quant = result.get("metadata", {}).get("_quantization_metadata")
    if isinstance(legacy_quant, str):
        try:
            legacy_quant = json.loads(legacy_quant)
        except json.JSONDecodeError as error:
            quant_metadata_issues.append(f"_quantization_metadata is invalid JSON: {error}")
            legacy_quant = None
    if legacy_quant is not None:
        layers = legacy_quant.get("layers") if isinstance(legacy_quant, dict) else None
        if not isinstance(layers, dict):
            quant_metadata_issues.append("_quantization_metadata.layers must be an object")
        else:
            for base, config in layers.items():
                name = f"{base}.comfy_quant"
                if not isinstance(base, str) or not isinstance(config, dict):
                    comfy_quant_layers[name] = {
                        "status": "error", "reason": "legacy layer name/config must be string/object",
                        "source": "safetensors_metadata",
                    }
                else:
                    comfy_quant_layers[name] = {
                        "status": "ok", "config": config, "source": "safetensors_metadata",
                    }
    # ConvRot metadata is either a legacy scalar sidecar or ComfyUI's small
    # uint8 JSON tensor. Never read matrix, embedding or scale payloads, and
    # never dequantize/rotate model weights.
    with path.open("rb") as stream:
        for tensor in result["tensor_inventory"]:
            name = tensor["name"]
            is_groupsize = name.endswith(".convrot_groupsize")
            is_comfy_quant = name.endswith(".comfy_quant")
            if not is_groupsize and not is_comfy_quant:
                continue
            start, end = tensor["data_offsets"]
            size = end - start
            if metadata_bytes + size > metadata_limit:
                if is_groupsize:
                    groupsizes[name] = {"value": None, "reason": "bounded metadata read limit exceeded"}
                else:
                    comfy_quant_layers[name] = _merge_quant_sources(
                        comfy_quant_layers.get(name),
                        {"status": "skipped", "reason": "bounded metadata read limit exceeded"},
                    )
                continue
            if is_groupsize:
                fmt = scalar_formats.get(tensor["dtype"])
                if fmt is None or tensor["shape"] not in ([], [1]) or size != struct.calcsize(fmt):
                    groupsizes[name] = {"value": None, "reason": "not a supported scalar"}
                    continue
            elif tensor["dtype"] != "U8" or len(tensor["shape"]) != 1 or size != tensor["shape"][0] or size > 4096:
                comfy_quant_layers[name] = _merge_quant_sources(
                    comfy_quant_layers.get(name),
                    {"status": "error", "reason": "not a bounded uint8 JSON tensor"},
                )
                continue
            stream.seek(result["safetensors"]["payload_start"] + start)
            data = stream.read(size)
            if len(data) != size:
                raise OSError(f"checkpoint changed while reading {path}")
            metadata_bytes += len(data)
            if is_groupsize:
                groupsizes[name] = {"value": struct.unpack(fmt, data)[0], "source": "tensor"}
                continue
            try:
                config = json.loads(data)
                if not isinstance(config, dict):
                    raise ValueError("JSON value is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                comfy_quant_layers[name] = _merge_quant_sources(
                    comfy_quant_layers.get(name),
                    {"status": "error", "reason": str(error)},
                )
                continue
            comfy_quant_layers[name] = _merge_quant_sources(
                comfy_quant_layers.get(name), {"status": "ok", "config": config}
            )
    for name, parsed in comfy_quant_layers.items():
        if parsed.get("status") != "ok":
            continue
        config = parsed["config"]
        quantization_formats[str(config.get("format"))] += 1
        params = config.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if config.get("convrot", params.get("convrot", False)):
            groupsize = config.get("convrot_groupsize", params.get("convrot_groupsize", 256))
            groupsize_name = name.removesuffix(".comfy_quant") + ".convrot_groupsize"
            existing = groupsizes.get(groupsize_name)
            if existing is not None and existing.get("value") != groupsize:
                parsed.update(
                    status="error",
                    reason="comfy_quant group size conflicts with convrot_groupsize tensor",
                    tensor_groupsize=existing.get("value"),
                )
                continue
            groupsizes[groupsize_name] = {"value": groupsize, "source": parsed.get("source", name)}
    after = path.stat()
    if after.st_size != result["identity"]["size_bytes"] or after.st_mtime_ns != result["identity"]["mtime_ns"]:
        raise OSError(f"checkpoint changed while inspecting {path}")
    result["convrot_groupsizes"] = groupsizes
    result["convrot_groupsize_counts"] = dict(Counter(str(item["value"]) for item in groupsizes.values()))
    result["comfy_quant_layers"] = comfy_quant_layers
    result["quant_metadata_issues"] = quant_metadata_issues
    result["comfy_quant_status_counts"] = dict(Counter(item["status"] for item in comfy_quant_layers.values()))
    result["quant_metadata_source_counts"] = dict(Counter(item.get("source", "unknown") for item in comfy_quant_layers.values()))
    result["quantization_format_counts"] = dict(quantization_formats)
    result["probe_contract"].update(
        header_only=metadata_bytes == 0,
        payload_read=metadata_bytes > 0,
        scalar_metadata_bytes_read=metadata_bytes,
        bounded_metadata_bytes_read=metadata_bytes,
        bounded_metadata_limit_bytes=metadata_limit,
        weight_matrix_payload_read=False,
    )
    if kind == "int8_convrot":
        result["conversion_readiness"] = audit_int8_convrot(result)
    return result


def compare_assets(assets: dict) -> dict | None:
    if not {"bf16", "int8_convrot"} <= assets.keys():
        return None
    if any(assets[k].get("status") != "ok" for k in ("bf16", "int8_convrot")):
        return None
    bf16 = {item["name"]: item for item in assets["bf16"].get("tensor_inventory", [])}
    int8 = {item["name"]: item for item in assets["int8_convrot"].get("tensor_inventory", [])}
    shared = sorted(bf16.keys() & int8.keys())
    return {
        "shared_key_count": len(shared), "bf16_only_keys": sorted(bf16.keys() - int8.keys()),
        "int8_only_keys": sorted(int8.keys() - bf16.keys()),
        "shape_changes": [{"name": n, "bf16": bf16[n]["shape"], "int8": int8[n]["shape"]} for n in shared if bf16[n]["shape"] != int8[n]["shape"]],
        "dtype_transitions": dict(Counter(f'{bf16[n]["dtype"]}->{int8[n]["dtype"]}' for n in shared)),
    }


def collect(root: Path, selected: tuple[str, ...] | None = None) -> dict:
    selected = tuple(FILES) if selected is None else selected
    invalid = sorted(set(selected) - FILES.keys())
    if invalid or not selected:
        raise ValueError(f"invalid H3 Ref2VA asset selection: {invalid or selected}")
    assets = {}
    for kind in selected:
        relative = FILES[kind]
        try:
            assets[kind] = inspect(root, kind, relative)
        except OSError as error:
            assets[kind] = {"status": "error", "input_path": str(root / relative), "error": str(error)}
    return {
        "schema": "lightx2v.minimax_h3.ref2va_weight_inventory", "schema_version": 2,
        "model_root": str(root), "runtime_validated": False,
        "selected_assets": list(selected),
        "status": "ok" if all(a["status"] == "ok" for a in assets.values()) else "error",
        "assets": assets,
        "comparison": compare_assets(assets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/disk/ssd1/difumodels/MiniMax-H3-ComfyUI"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--asset", choices=tuple(ASSET_CHOICES), default="both",
        help="inspect both checkpoints (default) or only the available BF16/INT8 ConvRot file",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = collect(args.model_root, ASSET_CHOICES[args.asset])
    write_json(args.output_dir / "h3_ref2va_weights.json", report)
    lines = ["# H3 Ref2VA weight inventory", "", "Header and bounded ConvRot metadata inspection; no GPU/runtime validation.", ""]
    for kind, asset in report["assets"].items():
        readiness = asset.get("conversion_readiness", {})
        lines.extend([f"## {kind}", "", f"Status: {asset['status']}", f"File: `{asset['input_path']}`",
                      f"Summary: `{json.dumps(asset.get('summary', {}), ensure_ascii=False)}`",
                      f"Quant formats: `{json.dumps(asset.get('quantization_format_counts', {}))}`",
                      f"ConvRot group sizes: `{json.dumps(asset.get('convrot_groupsize_counts', {}))}`",
                      f"Conversion readiness: `{json.dumps(readiness.get('status'))}`",
                      f"Full-precision override layers: `{readiness.get('full_precision_matrix_mult_enabled_count', 0)}`",
                      f"Shadowed tensor metadata: `{readiness.get('shadowed_tensor_metadata_count', 0)}`",
                      f"Conversion issue counts: `{json.dumps(readiness.get('issue_code_counts', {}), ensure_ascii=False)}`",
                      f"Conversion issue preview: `{json.dumps(readiness.get('issues', [])[:20], ensure_ascii=False)}`", ""])
    if report["comparison"] is not None:
        lines.append(f"Comparison: `{json.dumps(report['comparison'], ensure_ascii=False)}`")
    else:
        lines.append("Comparison: not requested (single-asset inventory).")
    (args.output_dir / "h3_ref2va_weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"H3 Ref2VA inventory: {report['status']} -> {args.output_dir}")
    for kind, asset in report["assets"].items():
        readiness = asset.get("conversion_readiness")
        if readiness is not None:
            print(
                f"{kind} conversion readiness: {readiness['status']} "
                f"({readiness['error_count']} errors, {readiness['warning_count']} warnings)"
            )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
