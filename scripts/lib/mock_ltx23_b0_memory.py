#!/usr/bin/env python3
"""Header-only, no-GPU memory gate for the LTX-2.3 FP8 B0."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "lightx2v.ltx23_fp8.b0.memory_mock"
SCHEMA_VERSION = 1
DEFAULT_CARD_BYTES = 34_208_743_424  # R9600D reported physical VRAM: 31.86 GiB.
DEFAULT_DYNAMIC_BUDGET_GIB = 16.0
DEFAULT_RUNTIME_RESERVE_MIB = 512
EXPECTED_FP8_BLOCKS = tuple(range(4, 47))
EXPECTED_BF16_BLOCKS = (0, 1, 2, 3, 47)
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(?P<block>[0-9]+)\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}
_REMOVED_COMPONENT_PATTERNS = (
    "text_embedding_projection",
    "audio_vae",
    "vae",
    "vocoder",
    "model.diffusion_model.audio_embeddings_connector",
    "model.diffusion_model.video_embeddings_connector",
)


class MemoryMockError(RuntimeError):
    pass


def _gib(value: int) -> float:
    return value / (1024**3)


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    if not path.is_file():
        raise MemoryMockError(f"checkpoint not found: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise MemoryMockError(f"truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= _MAX_HEADER_BYTES:
            raise MemoryMockError(
                f"invalid safetensors header size {header_size}: {path}"
            )
        encoded = stream.read(header_size)
    if len(encoded) != header_size:
        raise MemoryMockError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MemoryMockError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise MemoryMockError("safetensors header must be an object")
    return header, header_size


def _metadata(header: Mapping[str, Any]) -> dict[str, Any]:
    raw = header.get("__metadata__", {})
    if not isinstance(raw, Mapping):
        raise MemoryMockError("checkpoint __metadata__ must be an object")
    parsed = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        parsed[key] = value
    return parsed


def _tensor_nbytes(name: str, info: Mapping[str, Any]) -> int:
    dtype = str(info.get("dtype"))
    shape = info.get("shape")
    offsets = info.get("data_offsets")
    if dtype not in _DTYPE_BYTES:
        raise MemoryMockError(f"unsupported safetensors dtype {dtype!r}: {name}")
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or value < 0 for value in shape
    ):
        raise MemoryMockError(f"invalid tensor shape for {name}: {shape!r}")
    logical_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(value, int) or value < 0 for value in offsets)
        or offsets[1] < offsets[0]
    ):
        raise MemoryMockError(f"invalid data_offsets for {name}: {offsets!r}")
    stored_bytes = offsets[1] - offsets[0]
    if stored_bytes not in (0, logical_bytes):
        raise MemoryMockError(
            f"tensor byte span disagrees with dtype/shape for {name}: "
            f"span={stored_bytes}, logical={logical_bytes}"
        )
    return logical_bytes


def _block_signature(
    entries: list[tuple[str, Mapping[str, Any], int]],
) -> tuple[tuple[str, str, tuple[int, ...], int], ...]:
    signature = []
    for name, info, size in entries:
        normalized = _BLOCK_RE.sub(".transformer_blocks.*.", name, count=1)
        signature.append(
            (normalized, str(info["dtype"]), tuple(info["shape"]), size)
        )
    return tuple(sorted(signature))


def inspect_residency(path: Path) -> dict[str, Any]:
    header, header_size = read_safetensors_header(path)
    metadata = _metadata(header)
    if str(metadata.get("model_version", "")) != "2.3.0":
        raise MemoryMockError(
            f"expected model_version='2.3.0', got {metadata.get('model_version')!r}"
        )
    config = metadata.get("config")
    transformer = config.get("transformer") if isinstance(config, Mapping) else None
    if not isinstance(transformer, Mapping):
        raise MemoryMockError("checkpoint metadata is missing config.transformer")
    if transformer.get("num_layers") != 48:
        raise MemoryMockError(
            f"expected transformer.num_layers=48, got {transformer.get('num_layers')!r}"
        )

    blocks: dict[int, list[tuple[str, Mapping[str, Any], int]]] = {}
    resident_non_block = []
    removed_components = []
    checkpoint_tensor_bytes = 0
    max_data_end = 0
    for name, raw_info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw_info, Mapping):
            raise MemoryMockError(f"invalid tensor metadata for {name}")
        size = _tensor_nbytes(name, raw_info)
        checkpoint_tensor_bytes += size
        max_data_end = max(max_data_end, int(raw_info["data_offsets"][1]))
        match = _BLOCK_RE.search(name)
        if match is not None:
            blocks.setdefault(int(match.group("block")), []).append(
                (name, raw_info, size)
            )
        elif any(pattern in name for pattern in _REMOVED_COMPONENT_PATTERNS):
            removed_components.append((name, raw_info, size))
        elif name.startswith("model.diffusion_model."):
            resident_non_block.append((name, raw_info, size))
        else:
            raise MemoryMockError(f"unclassified checkpoint tensor: {name}")

    if tuple(sorted(blocks)) != tuple(range(48)):
        raise MemoryMockError(
            f"expected transformer blocks 0..47, got {sorted(blocks)}"
        )

    block_bytes = {index: sum(item[2] for item in blocks[index]) for index in blocks}
    block_schemes = {}
    for index, entries in blocks.items():
        weight_dtypes = {
            str(info["dtype"])
            for name, info, _ in entries
            if name.endswith(".weight")
        }
        if "F8_E4M3" in weight_dtypes:
            block_schemes[index] = "fp8"
        elif weight_dtypes == {"BF16"}:
            block_schemes[index] = "bf16"
        else:
            raise MemoryMockError(
                f"block {index} has unsupported weight dtypes: {sorted(weight_dtypes)}"
            )
    fp8_blocks = tuple(index for index in range(48) if block_schemes[index] == "fp8")
    bf16_blocks = tuple(index for index in range(48) if block_schemes[index] == "bf16")
    if fp8_blocks != EXPECTED_FP8_BLOCKS or bf16_blocks != EXPECTED_BF16_BLOCKS:
        raise MemoryMockError(
            "mixed block layout mismatch: "
            f"fp8={list(fp8_blocks)}, bf16={list(bf16_blocks)}"
        )

    scheme_block_bytes = {}
    for scheme, indices in (("bf16", bf16_blocks), ("fp8", fp8_blocks)):
        representative = indices[0]
        signature = _block_signature(blocks[representative])
        sizes = {block_bytes[index] for index in indices}
        if len(sizes) != 1:
            raise MemoryMockError(
                f"{scheme} blocks do not have one reusable byte size: {sorted(sizes)}"
            )
        for index in indices[1:]:
            if _block_signature(blocks[index]) != signature:
                raise MemoryMockError(
                    f"{scheme} block {index} does not match block {representative} schema"
                )
        scheme_block_bytes[scheme] = block_bytes[representative]

    non_block_bytes = sum(item[2] for item in resident_non_block)
    removed_bytes = sum(item[2] for item in removed_components)
    block_pool_bytes = 2 * sum(scheme_block_bytes.values())
    device_static_bytes = block_pool_bytes + non_block_bytes
    pinned_host_bytes = block_pool_bytes + non_block_bytes
    physical_static_bytes = device_static_bytes + pinned_host_bytes
    required_file_size = 8 + header_size + max_data_end

    return {
        "checkpoint": {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "header_size_bytes": header_size,
            "required_full_file_size_bytes": required_file_size,
            "header_only_input": path.stat().st_size < required_file_size,
            "tensor_count": len(header) - int("__metadata__" in header),
            "tensor_bytes": checkpoint_tensor_bytes,
        },
        "model_version": "2.3.0",
        "block_layout": {
            "fp8_blocks": list(fp8_blocks),
            "bf16_blocks": list(bf16_blocks),
            "bytes_per_fp8_block": scheme_block_bytes["fp8"],
            "bytes_per_bf16_block": scheme_block_bytes["bf16"],
        },
        "residency": {
            "strategy": "disk_streamed_mixed_bf16_fp8",
            "gpu_slots": {"bf16": 2, "fp8": 2},
            "pinned_host_slots": {"bf16": 2, "fp8": 2},
            "gpu_block_pool_bytes": block_pool_bytes,
            "pinned_host_block_pool_bytes": block_pool_bytes,
            "gpu_non_block_dit_bytes": non_block_bytes,
            "pinned_host_non_block_dit_bytes": non_block_bytes,
            "device_static_bytes": device_static_bytes,
            "pinned_host_bytes": pinned_host_bytes,
            "rocm_conservative_physical_static_bytes": physical_static_bytes,
            "canonical_block_bytes_not_resident": sum(block_bytes.values()),
            "phase_unloaded_component_bytes": removed_bytes,
        },
        "transformer": dict(transformer),
    }


def build_report(
    checkpoint: Path,
    *,
    card_bytes: int = DEFAULT_CARD_BYTES,
    dynamic_budget_gib: float = DEFAULT_DYNAMIC_BUDGET_GIB,
    runtime_reserve_mib: int = DEFAULT_RUNTIME_RESERVE_MIB,
) -> dict[str, Any]:
    if card_bytes <= 0:
        raise MemoryMockError("card_bytes must be positive")
    if not math.isfinite(dynamic_budget_gib) or dynamic_budget_gib < 0:
        raise MemoryMockError("dynamic_budget_gib must be finite and non-negative")
    if runtime_reserve_mib < 0:
        raise MemoryMockError("runtime_reserve_mib must be non-negative")

    report = inspect_residency(checkpoint)
    transformer = report.pop("transformer")
    num_heads = int(transformer["num_attention_heads"])
    head_dim = int(transformer["attention_head_dim"])
    hidden_dim = num_heads * head_dim
    latent_frames = (121 - 1) // 8 + 1
    stages = {}
    for name, height, width in (
        ("stage1", 352, 640),
        ("stage2", 704, 1280),
    ):
        global_tokens = latent_frames * (height // 32) * (width // 32)
        local_tokens = global_tokens // 2
        ulysses_heads = num_heads // 2
        stages[name] = {
            "height": height,
            "width": width,
            "global_video_tokens": global_tokens,
            "local_video_tokens": local_tokens,
            "local_video_hidden_bf16_bytes": local_tokens * hidden_dim * 2,
            "ulysses_attention_tokens_per_rank": global_tokens,
            "ulysses_attention_heads_per_rank": ulysses_heads,
            # Ulysses exchanges sequence shards for head shards before SDPA:
            # every rank attends the global sequence with half the heads.
            "math_sdpa_single_score_bytes": (
                global_tokens * global_tokens * ulysses_heads * 2
            ),
        }
    audio_tokens = round((121 / 24) * (16000 / 160 / 4))

    static_bytes = report["residency"]["rocm_conservative_physical_static_bytes"]
    dynamic_budget_bytes = round(dynamic_budget_gib * 1024**3)
    runtime_reserve_bytes = runtime_reserve_mib * 1024**2
    projected_bytes = static_bytes + dynamic_budget_bytes + runtime_reserve_bytes
    static_pass = static_bytes < card_bytes
    budget_pass = projected_bytes < card_bytes
    report.update(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if static_pass and budget_pass else "FAIL",
            "workload": {
                "gpus": 2,
                "per_gpu_capacity_bytes": card_bytes,
                "sequence_parallel_size": 2,
                "tensor_parallel_size": 1,
                "frames": 121,
                "fps": 24,
                "latent_frames": latent_frames,
                "video_hidden_dim": hidden_dim,
                "audio_tokens": audio_tokens,
                "stages": stages,
            },
            "gate": {
                "static_fit": "PASS" if static_pass else "FAIL",
                "budgeted_fit": "PASS" if budget_pass else "FAIL",
                "dynamic_activation_workspace_budget_bytes": dynamic_budget_bytes,
                "runtime_driver_reserve_bytes": runtime_reserve_bytes,
                "budgeted_peak_bytes": projected_bytes,
                "budgeted_headroom_bytes": card_bytes - projected_bytes,
                "maximum_dynamic_bytes_after_static_and_runtime_reserve": (
                    card_bytes - static_bytes - runtime_reserve_bytes
                ),
            },
            "limitations": [
                "Exact bytes cover checkpoint-derived resident tensors and configured staging slots only.",
                "Pinned host bytes are conservatively charged to physical VRAM for this R9600D/ROCm gate.",
                "The dynamic budget is a policy reserve, not a measured Torch/SDPA/scaled-mm workspace peak.",
                "A real two-rank collector run remains the final acceptance test.",
            ],
        }
    )
    return report


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def print_summary(report: Mapping[str, Any]) -> None:
    residency = report["residency"]
    gate = report["gate"]
    workload = report["workload"]
    print(f"LTX-2.3 FP8 B0 no-GPU memory mock: {report['status']}")
    print(
        "Workload: 2x32GB, 704x1280 final, 121f@24fps, "
        "SP=2 / TP=1"
    )
    print(
        "Tokens: Stage 1 global/local="
        f"{workload['stages']['stage1']['global_video_tokens']}/"
        f"{workload['stages']['stage1']['local_video_tokens']}; "
        "Stage 2 global/local="
        f"{workload['stages']['stage2']['global_video_tokens']}/"
        f"{workload['stages']['stage2']['local_video_tokens']}"
    )
    print(
        "Exact per-rank device tensors: "
        f"{_gib(residency['device_static_bytes']):.3f} GiB"
    )
    print(
        "Exact per-rank pinned host tensors: "
        f"{_gib(residency['pinned_host_bytes']):.3f} GiB"
    )
    print(
        "ROCm conservative physical static peak: "
        f"{_gib(residency['rocm_conservative_physical_static_bytes']):.3f} GiB / "
        f"{_gib(workload['per_gpu_capacity_bytes']):.3f} GiB "
        f"[{gate['static_fit']}]"
    )
    print(
        "Budgeted peak (static + runtime + dynamic policy reserve): "
        f"{_gib(gate['budgeted_peak_bytes']):.3f} GiB; "
        f"headroom={_gib(gate['budgeted_headroom_bytes']):.3f} GiB "
        f"[{gate['budgeted_fit']}]"
    )
    print(
        "NOTE: kernel/activation peak is budgeted, not measured; "
        "the two-rank collector run is still required."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--card-bytes", type=int, default=DEFAULT_CARD_BYTES)
    parser.add_argument(
        "--dynamic-budget-gib", type=float, default=DEFAULT_DYNAMIC_BUDGET_GIB
    )
    parser.add_argument(
        "--runtime-reserve-mib", type=int, default=DEFAULT_RUNTIME_RESERVE_MIB
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(
            args.checkpoint,
            card_bytes=args.card_bytes,
            dynamic_budget_gib=args.dynamic_budget_gib,
            runtime_reserve_mib=args.runtime_reserve_mib,
        )
    except (KeyError, TypeError, ValueError, MemoryMockError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.output is not None:
        write_json(args.output, report)
    print_summary(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
