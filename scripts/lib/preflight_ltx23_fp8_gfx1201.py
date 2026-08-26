#!/usr/bin/env python3
"""Strict host, image, and two-rank probes for the LTX-2.3 FP8 B0."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "lightx2v.ltx23_fp8.preflight"
SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_SIZE = 29_531_884_062
EXPECTED_CHECKPOINT_SHA256 = (
    "d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2"
)
EXPECTED_MODEL_VERSION = "2.3.0"
EXPECTED_FP8_BLOCKS = tuple(range(4, 47))
EXPECTED_BF16_BLOCKS = (0, 1, 2, 3, 47)
EXPECTED_LINEARS_PER_FP8_BLOCK = 34
EXPECTED_FP8_LINEAR_COUNT = 1462
B0_FPS = 24
B0_FRAMES = 121
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_BLOCK_WEIGHT_RE = re.compile(
    r"(?:^|\.)transformer_blocks\.(?P<block>[0-9]+)\.(?P<suffix>.+\.weight)$"
)


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            die(f"truncated safetensors prefix: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= _MAX_HEADER_BYTES:
            die(f"invalid safetensors header size {size}: {path}")
        encoded = stream.read(size)
    if len(encoded) != size:
        die(f"truncated safetensors header: {path}")
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        die(f"invalid safetensors header JSON in {path}: {exc}")
    if not isinstance(header, dict):
        die(f"safetensors header is not an object: {path}")
    return header


def _parsed_metadata(header: Mapping[str, Any]) -> dict[str, Any]:
    raw = header.get("__metadata__", {})
    if not isinstance(raw, Mapping):
        die("checkpoint __metadata__ must be an object")
    parsed = {}
    for key, value in raw.items():
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        parsed[key] = value
    return parsed


def inspect_checkpoint_header(path: Path) -> dict[str, Any]:
    header = read_safetensors_header(path)
    metadata = _parsed_metadata(header)
    if str(metadata.get("model_version", "")) != EXPECTED_MODEL_VERSION:
        die(
            f"expected model_version={EXPECTED_MODEL_VERSION}, "
            f"got {metadata.get('model_version')!r}"
        )
    config = metadata.get("config")
    transformer = config.get("transformer") if isinstance(config, Mapping) else None
    if not isinstance(transformer, Mapping):
        die("checkpoint metadata is missing config.transformer")
    if transformer.get("num_layers") != 48:
        die(f"expected transformer.num_layers=48, got {transformer.get('num_layers')!r}")

    tensors = {
        name: info
        for name, info in header.items()
        if name != "__metadata__" and isinstance(info, Mapping)
    }
    weights_by_block: dict[int, dict[str, tuple[str, str]]] = {}
    fp8_weights: list[tuple[str, int, str]] = []
    unsupported_fp8 = []
    for name, info in tensors.items():
        if not name.endswith(".weight"):
            continue
        match = _BLOCK_WEIGHT_RE.search(name)
        dtype = str(info.get("dtype"))
        if match is not None:
            block = int(match.group("block"))
            suffix = match.group("suffix")
            weights_by_block.setdefault(block, {})[suffix] = (name, dtype)
        if dtype != "F8_E4M3":
            continue
        if match is None:
            unsupported_fp8.append(name)
            continue
        block = int(match.group("block"))
        suffix = match.group("suffix")
        fp8_weights.append((name, block, suffix))
        layer = name.removesuffix(".weight")
        for scale_kind in ("input_scale", "weight_scale"):
            scale_name = f"{layer}.{scale_kind}"
            scale = tensors.get(scale_name)
            shape = scale.get("shape") if isinstance(scale, Mapping) else None
            scalar = isinstance(shape, list) and math.prod(shape) == 1
            if not isinstance(scale, Mapping) or scale.get("dtype") != "F32" or not scalar:
                die(
                    f"{scale_name} must be an F32 scalar, got "
                    f"{None if scale is None else dict(scale)}"
                )
    if unsupported_fp8:
        die(f"FP8 Linear outside transformer_blocks: {unsupported_fp8[0]}")
    suffixes = {suffix for _, _, suffix in fp8_weights}
    if len(suffixes) != EXPECTED_LINEARS_PER_FP8_BLOCK:
        die(
            f"expected {EXPECTED_LINEARS_PER_FP8_BLOCK} FP8 Linear suffixes, "
            f"got {len(suffixes)}"
        )

    fp8_blocks = []
    bf16_blocks = []
    for block in range(48):
        weights = weights_by_block.get(block, {})
        missing = suffixes - weights.keys()
        if missing:
            die(f"block {block} is missing FP8 schema weight {sorted(missing)[0]}")
        dtypes = [weights[suffix][1] for suffix in suffixes]
        fp8_count = dtypes.count("F8_E4M3")
        if fp8_count == EXPECTED_LINEARS_PER_FP8_BLOCK:
            fp8_blocks.append(block)
        elif fp8_count:
            die(f"block {block} mixes FP8 and BF16 Linears ({fp8_count}/34 FP8)")
        elif set(dtypes) == {"BF16"}:
            bf16_blocks.append(block)
        else:
            die(f"block {block} has unsupported non-FP8 Linear dtypes: {sorted(set(dtypes))}")

    if tuple(fp8_blocks) != EXPECTED_FP8_BLOCKS:
        die(f"expected FP8 blocks 4..46, got {fp8_blocks}")
    if tuple(bf16_blocks) != EXPECTED_BF16_BLOCKS:
        die(f"expected BF16 blocks {list(EXPECTED_BF16_BLOCKS)}, got {bf16_blocks}")
    if len(fp8_weights) != EXPECTED_FP8_LINEAR_COUNT:
        die(f"expected {EXPECTED_FP8_LINEAR_COUNT} FP8 Linears, got {len(fp8_weights)}")

    return {
        "model_version": EXPECTED_MODEL_VERSION,
        "num_layers": 48,
        "fp8_blocks": fp8_blocks,
        "bf16_blocks": bf16_blocks,
        "fp8_linears_per_block": len(suffixes),
        "fp8_linear_count": len(fp8_weights),
        "fp8_weight_dtype": "F8_E4M3",
        "scale_contract": "F32 scalar input_scale + weight_scale",
        "transformer": dict(transformer),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(
    path: Path,
    *,
    expected_size: int = EXPECTED_CHECKPOINT_SIZE,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    full_hash: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        die(f"checkpoint not found: {path}")
    size = path.stat().st_size
    if size != expected_size:
        die(f"checkpoint size is {size}, expected {expected_size}: {path}")
    started = time.perf_counter()
    digest = sha256_file(path) if full_hash else None
    if full_hash and digest != expected_sha256:
        die(f"checkpoint SHA256 is {digest}, expected {expected_sha256}: {path}")
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "sha256_expected": expected_sha256,
        "sha256_verified": full_hash,
        "hash_seconds": time.perf_counter() - started if full_hash else None,
        "layout": inspect_checkpoint_header(path),
    }


def check_artifacts(upscaler: Path, gemma_dir: Path) -> dict[str, Any]:
    if not upscaler.is_file():
        die(f"LTX spatial upscaler not found: {upscaler}")
    upscaler_header = read_safetensors_header(upscaler)
    upscaler_tensors = [key for key in upscaler_header if key != "__metadata__"]
    if not upscaler_tensors:
        die(f"LTX spatial upscaler has no tensors: {upscaler}")
    if not gemma_dir.is_dir():
        die(f"Gemma directory not found: {gemma_dir}")

    index_matches = [
        path
        for path in sorted(gemma_dir.rglob("model.safetensors.index.json"))
        if path.is_file()
    ]
    if len(index_matches) != 1:
        die(
            "Gemma directory must contain exactly one model.safetensors.index.json, "
            f"got {len(index_matches)}: {gemma_dir}"
        )
    model_root = index_matches[0].parent
    expected_shard_names = {
        f"model-{index:05d}-of-00005.safetensors"
        for index in range(1, 6)
    }
    model_files = [model_root / name for name in sorted(expected_shard_names)]
    missing_shards = [path.name for path in model_files if not path.is_file()]
    if missing_shards:
        die(f"Gemma directory is missing model shards: {missing_shards}: {model_root}")

    required_names = (
        "config.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "preprocessor_config.json",
    )
    missing = [name for name in required_names if not (model_root / name).is_file()]
    if missing:
        die(f"Gemma directory is missing required artifacts: {missing}: {model_root}")
    try:
        index_payload = json.loads(index_matches[0].read_text(encoding="utf-8"))
        config_payload = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
        json.loads((model_root / "tokenizer_config.json").read_text(encoding="utf-8"))
        json.loads((model_root / "preprocessor_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        die(f"Gemma JSON artifact is invalid: {exc}")
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        die("Gemma model.safetensors.index.json must contain a non-empty weight_map")
    referenced_shards = set(weight_map.values())
    if referenced_shards != expected_shard_names:
        die(
            "Gemma index shard set does not match the required five files: "
            f"{sorted(referenced_shards)}"
        )
    if not isinstance(config_payload, Mapping) or not config_payload:
        die("Gemma config.json must contain a non-empty object")
    required = {
        "model_root": str(model_root),
        "model_safetensors": [str(path) for path in model_files],
        "model.safetensors.index.json": str(index_matches[0]),
        **{name: str(model_root / name) for name in required_names},
    }
    return {
        "upscaler": {
            "path": str(upscaler),
            "size_bytes": upscaler.stat().st_size,
            "tensor_count": len(upscaler_tensors),
        },
        "gemma": {"path": str(gemma_dir), **required},
    }


def expected_local_video_tokens() -> dict[str, int]:
    latent_frames = (B0_FRAMES - 1) // 8 + 1
    return {
        "stage1": latent_frames * 11 * 20 // 2,
        "stage2": latent_frames * 22 * 40 // 2,
    }


def check_runtime_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"runtime config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "infer_steps": 8,
        "target_video_length": B0_FRAMES,
        "target_height": 704,
        "target_width": 1280,
        "fps": B0_FPS,
        "attn_type": "torch_sdpa",
        "rms_norm_type": "torch_native",
        "rope_type": "torch_real_rope",
        "layer_norm_type": "torch",
        "modulate_type": "torch",
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "lazy_load": True,
        "unload_modules": True,
        "gemma_cpu_offload": True,
        "ltx23_process_isolated_prompt_required": True,
        "vae_cpu_offload": True,
        "use_tiling_vae": True,
        "warmup": False,
        "dit_quantized": True,
        "dit_quant_scheme": "fp8-pertensor",
        "use_upsampler": True,
        "ltx_checkpoint_contract_required": True,
        "ltx23_b0_gfx1201": True,
        "sample_guide_scale": 1.0,
        "sample_shift": [2.05, 0.95],
        "num_channels_latents": 128,
        "vae_scale_factors": [8, 32, 32],
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_scale_factor": 4,
        "audio_fps": 24000,
        "audio_mel_bins": 16,
        "caption_proj_before_connector": True,
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
        "distilled_sigma_values": [
            1.0,
            0.99375,
            0.9875,
            0.98125,
            0.975,
            0.909375,
            0.725,
            0.421875,
            0.0,
        ],
        "distilled_sigma_values_upsample": [0.909375, 0.725, 0.421875, 0.0],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            die(f"runtime config {key} must be {value!r}, got {config.get(key)!r}")
    if config.get("skip_fp8_block_index") != list(EXPECTED_BF16_BLOCKS):
        die("runtime config skip_fp8_block_index must be [0,1,2,3,47]")
    parallel = config.get("parallel")
    expected_parallel = {
        "tensor_p_size": 1,
        "cfg_p_size": 1,
        "seq_p_size": 2,
        "seq_p_attn_type": "ulysses",
        "seq_p_fp8_comm": False,
        "seq_p_fp4_comm": False,
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
    }
    if not isinstance(parallel, Mapping):
        die("runtime config parallel must be an object")
    for key, value in expected_parallel.items():
        if parallel.get(key) != value:
            die(
                f"runtime config parallel.{key} must be {value!r}, "
                f"got {parallel.get(key)!r}"
            )
    expected_paths = {
        "dit_quantized_ckpt": "/models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors",
        "upsampler_original_ckpt": "/models/ltx23/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "gemma_original_ckpt": "/models/ltx23/gemma",
    }
    for key, value in expected_paths.items():
        if config.get(key) != value:
            die(f"runtime config {key} must be {value!r}, got {config.get(key)!r}")
    return config


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def check_runtime_dependencies() -> dict[str, Any]:
    try:
        import accelerate
        import aiter
        import lightx2v
        from lightx2v.models.input_encoders.hf.ltx2.model import LTX2TextEncoder
        from lightx2v.models.networks.ltx2.infer.offload.manager import (
            LTX2HeterogeneousOffloadManager,
        )
        from lightx2v.utils import ltx2_checkpoint
        from lightx2v.utils.ltx2_prompt_cache import (
            build_prompt_cache,
            load_prompt_cache,
        )
    except Exception as exc:
        die(f"image is missing the LTX-2.3/Aiter runtime: {exc}")
    if ltx2_checkpoint.LTX23_FP8_CHECKPOINT_SIZE != EXPECTED_CHECKPOINT_SIZE:
        die("LightX2V checkpoint identity constants disagree with the launcher")
    if ltx2_checkpoint.LTX23_FP8_CHECKPOINT_SHA256 != EXPECTED_CHECKPOINT_SHA256:
        die("LightX2V checkpoint SHA256 disagrees with the launcher")
    if not callable(getattr(LTX2HeterogeneousOffloadManager, "current_buffer", None)):
        die("image lacks the heterogeneous LTX BF16/FP8 offload manager")
    if not callable(
        getattr(LTX2HeterogeneousOffloadManager, "init_cpu_buffer_pools", None)
    ):
        die("image lacks mixed BF16/FP8 disk-streaming host pools")
    if "gemma_device_map" not in inspect.signature(LTX2TextEncoder).parameters:
        die("image lacks Accelerate device-map support for LTX-2.3 Gemma")
    if not callable(build_prompt_cache) or not callable(load_prompt_cache):
        die("image lacks the process-isolated LTX-2.3 prompt handoff")
    lightx2v_root = Path(lightx2v.__file__).resolve().parents[1]
    aiter_root = Path(aiter.__file__).resolve().parents[1]
    lightx2v_revision = os.getenv("LIGHTX2V_SOURCE_REVISION") or git_revision(lightx2v_root)
    aiter_revision = (
        os.getenv("AITER_SOURCE_REVISION")
        or os.getenv("AITER_COMMIT")
        or git_revision(aiter_root)
    )
    if not lightx2v_revision or not aiter_revision:
        die(
            "image source provenance is missing; retain git metadata or bake "
            "LIGHTX2V_SOURCE_REVISION and AITER_SOURCE_REVISION"
        )
    return {
        "lightx2v_root": str(lightx2v_root),
        "lightx2v_revision": lightx2v_revision,
        "lightx2v_dirty": git_dirty(lightx2v_root),
        "aiter_root": str(aiter_root),
        "aiter_revision": aiter_revision,
        "aiter_dirty": git_dirty(aiter_root),
        "offload_manager": LTX2HeterogeneousOffloadManager.__name__,
        "accelerate_version": accelerate.__version__,
        "prompt_handoff": "safetensors.process_isolated",
    }


def require_gfx1201(expected_devices: int = 2) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        die(f"PyTorch is required inside the runtime image: {exc}")
    if not torch.cuda.is_available():
        die("ROCm device is unavailable")
    count = torch.cuda.device_count()
    if count != expected_devices:
        die(f"expected exactly {expected_devices} visible GPUs, got {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        arch = str(getattr(properties, "gcnArchName", ""))
        if not arch.lower().split(":", 1)[0].startswith("gfx1201"):
            die(f"visible GPU {index} is {arch!r}, expected gfx1201")
        devices.append(
            {
                "visible_index": index,
                "name": properties.name,
                "arch": arch,
                "multiprocessor_count": int(properties.multi_processor_count),
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    rocm = str(torch.version.hip or "")
    expected_rocm = os.getenv("LTX23_EXPECTED_ROCM_PREFIX", "7.2")
    if not rocm.startswith(expected_rocm):
        die(f"expected ROCm {expected_rocm}.x, got torch.version.hip={rocm!r}")
    if os.getenv("GPU_ARCHS") != "gfx1201" or os.getenv("CU_NUM") != "24":
        die("LTX-2.3 B0 requires GPU_ARCHS=gfx1201 and CU_NUM=24")
    if not callable(getattr(torch, "_scaled_mm", None)):
        die("torch._scaled_mm is unavailable")
    return {
        "device_count": count,
        "devices": devices,
        "rocm": rocm,
        "torch": torch.__version__,
    }


def scaled_mm_layout_contract(m: int, n: int, k: int) -> dict[str, list[int]]:
    if min(m, n, k) <= 0:
        raise ValueError("scaled-mm dimensions must be positive")
    return {
        "lhs_shape": [m, k],
        "lhs_stride": [k, 1],
        "rhs_shape": [k, n],
        "rhs_stride": [1, k],
    }


def _scaled_mm_probe(torch, *, m: int, n: int, k: int, bias: bool) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(20260826 + m + n + k)
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator)
    # Match the real LightX2V checkpoint path exactly: weights are contiguous
    # [N, K] on disk, then exposed to torch._scaled_mm through a zero-copy
    # [K, N] transpose view. ROCm hipBLASLt rejects a row-major [K, N] RHS.
    b_checkpoint = torch.randn(
        (n, k), device=device, dtype=torch.bfloat16, generator=generator
    )
    scale_a = torch.clamp(a.abs().max().float() / 448.0, min=1.0e-8)
    scale_b = torch.clamp(
        b_checkpoint.abs().max().float() / 448.0, min=1.0e-8
    )
    a_fp8 = torch.clamp(a / scale_a, -448, 448).to(torch.float8_e4m3fn)
    b_fp8_checkpoint = torch.clamp(
        b_checkpoint / scale_b, -448, 448
    ).to(torch.float8_e4m3fn)
    b_fp8 = b_fp8_checkpoint.t()
    layout = scaled_mm_layout_contract(m, n, k)
    actual_layout = {
        "lhs_shape": list(a_fp8.shape),
        "lhs_stride": list(a_fp8.stride()),
        "rhs_shape": list(b_fp8.shape),
        "rhs_stride": list(b_fp8.stride()),
    }
    if actual_layout != layout:
        die(
            "torch._scaled_mm probe layout disagrees with the production TN "
            f"contract: expected={layout}, actual={actual_layout}"
        )
    bias_tensor = (
        torch.randn((n,), device=device, dtype=torch.bfloat16, generator=generator)
        if bias
        else None
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = torch._scaled_mm(
        a_fp8,
        b_fp8,
        scale_a=scale_a.reshape(1),
        scale_b=scale_b.reshape(1),
        bias=bias_tensor,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    end.record()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        die(f"torch._scaled_mm returned non-finite values for M/N/K={m}/{n}/{k}")
    result = {
        "m": m,
        "n": n,
        "k": k,
        "bias": bias,
        "elapsed_ms": float(start.elapsed_time(end)),
        "abs_max": float(out.abs().max()),
        "input_dtype": str(a_fp8.dtype),
        "scale_dtype": str(scale_a.dtype),
        "scale_shape": list(scale_a.shape),
        "output_dtype": str(out.dtype),
        "layout": actual_layout,
    }
    del a, b_checkpoint, a_fp8, b_fp8_checkpoint, b_fp8, bias_tensor, out
    torch.cuda.empty_cache()
    return result


def distributed_probe(output: Path, exact_shapes: bool) -> int:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            die(f"distributed LTX B0 probe requires world_size=2, got {world_size}")
        properties = torch.cuda.get_device_properties(local_rank)
        arch = str(getattr(properties, "gcnArchName", ""))
        if not arch.lower().split(":", 1)[0].startswith("gfx1201"):
            die(f"rank {rank} is bound to {arch!r}, expected gfx1201")

        collective = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(collective)
        if float(collective.item()) != 3.0:
            die(f"RCCL all-reduce returned {collective.item()}, expected 3")

        tokens = expected_local_video_tokens()
        shapes = [
            (tokens["stage1"], 4096, 4096, False),
            (tokens["stage2"], 4096, 4096, False),
            (128, 2048, 2048, True),
        ]
        if not exact_shapes:
            shapes = [(min(m, 64), min(n, 256), min(k, 256), bias) for m, n, k, bias in shapes]
        gemms = [
            _scaled_mm_probe(torch, m=m, n=n, k=k, bias=bias)
            for m, n, k, bias in shapes
        ]
        local = {
            "rank": rank,
            "local_rank": local_rank,
            "device_name": properties.name,
            "arch": arch,
            "rccl_all_reduce": float(collective.item()),
            "gemms": gemms,
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
        if rank == 0:
            write_json(
                output,
                {
                    "schema_name": SCHEMA_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "mode": "distributed",
                    "status": "ok",
                    "world_size": world_size,
                    "fps": B0_FPS,
                    "exact_shapes": exact_shapes,
                    "local_video_tokens": tokens,
                    "ranks": gathered,
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(encoded)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("host", "runtime", "distributed"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upscaler", type=Path, required=True)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick-gemm-probe", action="store_true")
    args = parser.parse_args()

    if args.mode == "distributed":
        if args.output is None:
            die("--output is required in distributed mode")
        return distributed_probe(args.output, not args.quick_gemm_probe)

    checkpoint = check_checkpoint(
        args.checkpoint,
        full_hash=args.mode == "host",
    )
    artifacts = check_artifacts(args.upscaler, args.gemma_dir)
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "status": "ok",
        "fps": B0_FPS,
        "checkpoint": checkpoint,
        "artifacts": artifacts,
    }
    if args.runtime_config is not None:
        payload["runtime_config"] = check_runtime_config(args.runtime_config)
    if args.mode == "runtime":
        payload["gpu"] = require_gfx1201(2)
        payload["source"] = check_runtime_dependencies()
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
