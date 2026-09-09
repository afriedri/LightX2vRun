#!/usr/bin/env python3
"""Inspect both HVAT H3 Ref2VA ComfyUI weights without Torch or GPU allocation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import struct

from collect_ltx25_weight_census import inspect_asset, write_json

FILES = {
    "bf16": "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
    "int8_convrot": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
}


def inspect(root: Path, kind: str, relative: str) -> dict:
    path = root / relative
    result = inspect_asset(path, component=kind, relative_path=relative)
    result["runtime_validated"] = False
    if result["status"] != "ok":
        return result
    groupsizes = {}
    scalar_bytes = 0
    scalar_formats = {"I32": "<i", "I64": "<q"}
    # ConvRot group sizes are tiny scalar metadata tensors. Never read matrix,
    # embedding or scale payloads, and never dequantize/rotate model weights.
    with path.open("rb") as stream:
        for tensor in result["tensor_inventory"]:
            if not tensor["name"].endswith(".convrot_groupsize"):
                continue
            fmt = scalar_formats.get(tensor["dtype"])
            if fmt is None or tensor["shape"] not in ([], [1]) or scalar_bytes >= 65536:
                groupsizes[tensor["name"]] = {"value": None, "reason": "not a supported scalar"}
                continue
            start, end = tensor["data_offsets"]
            if end - start != struct.calcsize(fmt):
                continue
            stream.seek(result["safetensors"]["payload_start"] + start)
            data = stream.read(end - start)
            if len(data) != end - start:
                raise OSError(f"checkpoint changed while reading {path}")
            groupsizes[tensor["name"]] = {"value": struct.unpack(fmt, data)[0]}
            scalar_bytes += len(data)
    after = path.stat()
    if after.st_size != result["identity"]["size_bytes"] or after.st_mtime_ns != result["identity"]["mtime_ns"]:
        raise OSError(f"checkpoint changed while inspecting {path}")
    result["convrot_groupsizes"] = groupsizes
    result["convrot_groupsize_counts"] = dict(Counter(str(item["value"]) for item in groupsizes.values()))
    result["probe_contract"].update(header_only=scalar_bytes == 0, payload_read=scalar_bytes > 0,
                                    scalar_metadata_bytes_read=scalar_bytes, weight_matrix_payload_read=False)
    return result


def collect(root: Path) -> dict:
    assets = {}
    for kind, relative in FILES.items():
        try:
            assets[kind] = inspect(root, kind, relative)
        except OSError as error:
            assets[kind] = {"status": "error", "input_path": str(root / relative), "error": str(error)}
    inventories = [{item["name"]: item for item in assets[k].get("tensor_inventory", [])} for k in FILES]
    bf16, int8 = inventories
    shared = sorted(bf16.keys() & int8.keys())
    return {
        "schema": "lightx2v.minimax_h3.ref2va_weight_inventory", "schema_version": 1,
        "model_root": str(root), "runtime_validated": False,
        "status": "ok" if all(a["status"] == "ok" for a in assets.values()) else "error",
        "assets": assets,
        "comparison": {
            "shared_key_count": len(shared), "bf16_only_keys": sorted(bf16.keys() - int8.keys()),
            "int8_only_keys": sorted(int8.keys() - bf16.keys()),
            "shape_changes": [{"name": n, "bf16": bf16[n]["shape"], "int8": int8[n]["shape"]} for n in shared if bf16[n]["shape"] != int8[n]["shape"]],
            "dtype_transitions": dict(Counter(f'{bf16[n]["dtype"]}->{int8[n]["dtype"]}' for n in shared)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/disk/ssd1/difumodels/MiniMax-H3-ComfyUI"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = collect(args.model_root)
    write_json(args.output_dir / "h3_ref2va_weights.json", report)
    lines = ["# H3 Ref2VA weight inventory", "", "Header and bounded ConvRot scalar inspection; no GPU/runtime validation.", ""]
    for kind, asset in report["assets"].items():
        lines.extend([f"## {kind}", "", f"Status: {asset['status']}", f"File: `{asset['input_path']}`",
                      f"Summary: `{json.dumps(asset.get('summary', {}), ensure_ascii=False)}`",
                      f"ConvRot group sizes: `{json.dumps(asset.get('convrot_groupsize_counts', {}))}`", ""])
    lines.append(f"Comparison: `{json.dumps(report['comparison'], ensure_ascii=False)}`")
    (args.output_dir / "h3_ref2va_weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"H3 Ref2VA inventory: {report['status']} -> {args.output_dir}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
