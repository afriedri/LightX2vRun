#!/usr/bin/env python3
"""C0 preflight for MiniMax-H3 on gfx1201. This runs on the GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_dirty(path: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def require_gfx1201() -> dict:
    try:
        import torch
    except ImportError as error:
        die(f"torch is required on the GPU host/container: {error}")
    if not torch.cuda.is_available():
        die("CUDA/ROCm device is unavailable; MiniMax-H3 Day0 must run on the gfx1201 card")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    gfx = arch.lower().split(":")[0]
    if not gfx.startswith("gfx1201"):
        die(f"expected gfx1201, got {arch}")
    rocm = str(torch.version.hip or "")
    if not rocm.startswith("7.2"):
        die(f"MiniMax-H3 Day0 is frozen to ROCm 7.2, got torch.version.hip={rocm!r}")
    if os.environ.get("GPU_ARCHS") != "gfx1201" or os.environ.get("CU_NUM") != "24":
        die(
            "MiniMax-H3 Day0 requires GPU_ARCHS=gfx1201 and CU_NUM=24 "
            f"(24 WGP / 48 CU), got GPU_ARCHS={os.environ.get('GPU_ARCHS')!r} "
            f"CU_NUM={os.environ.get('CU_NUM')!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "arch": arch,
        "name": properties.name,
        "multiprocessor_count": properties.multi_processor_count,
        "rocm": rocm,
        "torch": torch.__version__,
        "wgp_contract": 24,
        "cu_contract": 48,
    }


def check_runtime_dependencies(source_mode: str) -> dict:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            die(f"required executable is missing from the image: {executable}")

    try:
        import transformers
        from transformers import Qwen2TokenizerFast, Qwen3VLProcessor  # noqa: F401
    except ImportError as error:
        die(f"transformers with Qwen2TokenizerFast/Qwen3VLProcessor is required: {error}")

    try:
        import aiter
        from aiter import flash_attn_varlen_func as public_varlen
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func  # noqa: F401
    except ImportError as error:
        die(f"Aiter Triton MHA and Sage APIs are required by the Day0 runtime: {error}")
    if not callable(public_varlen):
        die("aiter.flash_attn_varlen_func is not callable")

    try:
        import lightx2v
        import lightx2v.infer  # noqa: F401
        import lightx2v_platform.ops.attn.amd_rocm  # noqa: F401
        from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER
        from lightx2v.utils.registry_factory import (
            ATTN_WEIGHT_REGISTER,
            RMS_WEIGHT_REGISTER,
            ROPE_REGISTER,
            RUNNER_REGISTER,
        )
    except Exception as error:
        die(f"LightX2V H3 runner/AMD registry cannot be imported: {error}")
    backend = "aiter_triton_bf16_flash_attn"
    if backend not in PLATFORM_ATTN_WEIGHT_REGISTER:
        die(f"LightX2V attention backend is not registered: {backend}")
    for registry, name in (
        (RUNNER_REGISTER, "minimax_h3"),
        (ATTN_WEIGHT_REGISTER, "torch_sdpa"),
        (RMS_WEIGHT_REGISTER, "torch_native"),
        (ROPE_REGISTER, "torch_real_rope"),
    ):
        if name not in registry:
            die(f"LightX2V runtime registry is missing: {name}")

    source_root = Path(lightx2v.__file__).resolve().parents[1]
    source_revision = os.environ.get("LIGHTX2V_SOURCE_REVISION") or git_revision(source_root)
    if not source_revision:
        die(
            "image source provenance is missing; bake LIGHTX2V_SOURCE_REVISION into the image "
            "or include readable git metadata"
        )
    return {
        "source_mode": source_mode,
        "lightx2v_root": str(source_root),
        "lightx2v_revision": source_revision,
        "lightx2v_dirty": git_dirty(source_root),
        "aiter_path": str(Path(aiter.__file__).resolve()),
        "aiter_revision": os.environ.get("AITER_SOURCE_REVISION")
        or os.environ.get("AITER_COMMIT")
        or git_revision(Path(aiter.__file__).resolve().parents[1]),
        "transformers": transformers.__version__,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "registered_backend": backend,
        "registered_runner": "minimax_h3",
        "registered_torch_ops": ["torch_sdpa", "torch_native", "torch_real_rope"],
        "sage_api_importable": True,
    }


def smoke_triton_mha() -> dict:
    import torch

    from aiter.ops.mha import flash_attn_varlen_func

    seq_len, heads, head_dim = 257, 56, 128
    device = torch.device("cuda")
    q = torch.randn(seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        dropout_p=0.0,
        causal=False,
        window_size=(-1, -1),
    )
    torch.cuda.synchronize()
    if tuple(out.shape) != (seq_len, heads, head_dim):
        die(f"Triton MHA output shape {tuple(out.shape)} != {(seq_len, heads, head_dim)}")
    if not torch.isfinite(out).all():
        die("Triton MHA produced non-finite values on gfx1201")
    peak = float(out.abs().max().item())
    if peak <= 0.0:
        die("Triton MHA returned an all-zero tensor")
    return {
        "backend": "flash_attn_varlen_func",
        "route": "gfx1201_triton_mha",
        "q_shape": [seq_len, heads, head_dim],
        "output_shape": list(out.shape),
        "dtype": str(out.dtype).replace("torch.", ""),
        "abs_max": peak,
        "device": str(out.device),
    }


def check_artifacts(model_dir: Path, turbo_dir: Path, first_frame: Path, last_frame: Path) -> dict:
    if not model_dir.is_dir():
        die(f"H3 model directory not found: {model_dir}")
    component_files = {
        # H3 DiT architecture comes from the frozen runtime config; the
        # released transformer directory only needs its safetensors shards.
        "transformer": (),
        "text_encoder": ("config.json",),
        "tokenizer": ("tokenizer.json",),
        "vae": ("config.json",),
        "audio_vae": ("config.json",),
    }
    for component, filenames in component_files.items():
        component_dir = model_dir / component
        if not component_dir.is_dir():
            die(f"H3 model component directory not found: {component_dir}")
        for filename in filenames:
            if not (component_dir / filename).is_file():
                die(f"H3 model component file not found: {component_dir / filename}")
    processor_dir = model_dir / "processor"
    if not processor_dir.is_dir():
        die(f"H3 model component directory not found: {processor_dir}")
    if not any((processor_dir / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        die(f"H3 processor config not found below: {processor_dir}")
    for component in ("transformer", "text_encoder", "vae", "audio_vae"):
        component_dir = model_dir / component
        if not any(component_dir.glob("*.safetensors")):
            die(f"H3 model component has no safetensors shard: {component_dir}")
    turbo = turbo_dir / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    if not turbo.is_file():
        die(f"Turbo LoRA not found: {turbo}")
    if not first_frame.is_file():
        die(f"first frame not found: {first_frame}")
    if not last_frame.is_file():
        die(f"last frame not found: {last_frame}")
    try:
        from PIL import Image

        image_info = {}
        for label, path in (("first_frame", first_frame), ("last_frame", last_frame)):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_info[label] = {
                    "path": str(path),
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                }
    except Exception as error:
        die(f"first/last frame decode failed: {error}")
    return {
        "model_dir": str(model_dir),
        "components": sorted((*component_files, "processor")),
        "turbo": str(turbo),
        **image_info,
    }


def check_runtime_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (
        ("video_flow_shift", 6.0),
        ("h3_step_update", "training_euler"),
        ("h3_refiner_attn_type", "aiter_triton_bf16_flash_attn"),
        ("h3_core_attn_type", "aiter_triton_bf16_flash_attn"),
        ("warmup", False),
    ):
        if config.get(key) != expected:
            die(f"runtime config {key} must be {expected!r}, got {config.get(key)!r}")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_PATH", "/models/minimax_h3")))
    parser.add_argument("--turbo-dir", type=Path, default=Path("/models/minimax_h3_turbo"))
    parser.add_argument("--first-frame", type=Path, default=Path("/inputs/minimax_h3_first_frame"))
    parser.add_argument("--last-frame", type=Path, default=Path("/inputs/minimax_h3_last_frame"))
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--source-mode", choices=("image",), default="image")
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()

    gpu = require_gfx1201()
    runtime = check_runtime_dependencies(args.source_mode)
    artifacts = {}
    if not args.skip_artifacts:
        artifacts = check_artifacts(args.model_dir, args.turbo_dir, args.first_frame, args.last_frame)
    config = check_runtime_config(args.runtime_config) if args.runtime_config else {}
    attn = smoke_triton_mha()
    payload = {
        "status": "ok",
        "gpu": gpu,
        "runtime": runtime,
        "artifacts": artifacts,
        "triton_mha": attn,
        "task": config.get("task", "fl2av"),
        "h3_core_attn_type": config.get("h3_core_attn_type", "aiter_triton_bf16_flash_attn"),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(
        f"[H3Preflight] gfx={gpu['arch']} rocm={gpu['rocm']} source={args.source_mode} "
        f"Triton MHA ran on {attn['device']} "
        f"q={tuple(attn['q_shape'])} abs_max={attn['abs_max']:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
