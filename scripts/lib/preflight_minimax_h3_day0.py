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


def require_gfx1201(expected_devices: int) -> dict:
    try:
        import torch
    except ImportError as error:
        die(f"torch is required on the GPU host/container: {error}")
    if not torch.cuda.is_available():
        die("CUDA/ROCm device is unavailable; MiniMax-H3 Day0 must run on the gfx1201 card")
    device_count = torch.cuda.device_count()
    if device_count != expected_devices:
        die(f"expected {expected_devices} visible gfx1201 GPU(s), got {device_count}")
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        arch = properties.gcnArchName
        gfx = arch.lower().split(":")[0]
        if not gfx.startswith("gfx1201"):
            die(f"expected gfx1201 at visible device {index}, got {arch}")
        devices.append(
            {
                "visible_index": index,
                "arch": arch,
                "name": properties.name,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    rocm = str(torch.version.hip or "")
    if not rocm.startswith("7.2"):
        die(f"MiniMax-H3 Day0 is frozen to ROCm 7.2, got torch.version.hip={rocm!r}")
    if os.environ.get("GPU_ARCHS") != "gfx1201" or os.environ.get("CU_NUM") != "24":
        die(
            "MiniMax-H3 Day0 requires GPU_ARCHS=gfx1201 and CU_NUM=24 "
            f"(24 WGP / 48 CU), got GPU_ARCHS={os.environ.get('GPU_ARCHS')!r} "
            f"CU_NUM={os.environ.get('CU_NUM')!r}"
        )
    return {
        **devices[0],
        "device_count": device_count,
        "devices": devices,
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
        (ATTN_WEIGHT_REGISTER, "ulysses"),
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


def smoke_triton_mha(device_index: int) -> dict:
    import torch

    from aiter.ops.mha import flash_attn_varlen_func

    seq_len, heads, head_dim = 257, 56, 128
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
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


def smoke_distributed_ulysses() -> int:
    import torch
    import torch.distributed as dist

    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v_platform.ops.attn.amd_rocm.triton_mha_attn import AiterTritonBf16FlashAttnWeight

    dist.init_process_group(backend="nccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if world_size != 2:
            raise RuntimeError(f"MiniMax-H3 distributed preflight requires world_size=2, got {world_size}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        aux_len = 3
        torch.manual_seed(20260817)
        aux_q = torch.randn(aux_len, 56, 128, device=device, dtype=torch.bfloat16)
        aux_k = torch.randn_like(aux_q)
        aux_v = torch.randn_like(aux_q)
        torch.manual_seed(20260817 + rank)
        local_len, heads, head_dim = 257, 56, 128
        q = torch.randn(local_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output, aux_output = UlyssesAttnWeight(a2a_backend="torch").apply_new(
            q=q,
            k=k,
            v=v,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            attention_module=AiterTritonBf16FlashAttnWeight(),
            seq_p_group=dist.group.WORLD,
            prepost_backend="torch",
            a2a_backend="torch",
            quant_scheme=None,
            tensor_fusion=False,
            head_parallel=False,
            aux_first=True,
            attention_kwargs={"causal": False},
        )
        torch.cuda.synchronize(device)
        expected_shape = (local_len, heads * head_dim)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"Ulysses output shape {tuple(output.shape)} != {expected_shape}")
        expected_aux_shape = (aux_len, heads * head_dim)
        if aux_output is None or tuple(aux_output.shape) != expected_aux_shape:
            raise RuntimeError(f"Ulysses auxiliary output shape is not {expected_aux_shape}: {None if aux_output is None else tuple(aux_output.shape)}")
        if not torch.isfinite(output).all() or output.abs().max().item() <= 0.0 or not torch.isfinite(aux_output).all():
            raise RuntimeError("Ulysses produced zero or non-finite output")
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "backend": "ulysses",
                        "attention": "aiter_triton_bf16_flash_attn",
                        "world_size": world_size,
                        "local_q_shape": [local_len, heads, head_dim],
                        "local_output_shape": list(output.shape),
                        "aux_q_shape": [aux_len, heads, head_dim],
                        "aux_output_shape": list(aux_output.shape),
                    },
                    sort_keys=True,
                )
            )
            print("[H3Preflight] RCCL Ulysses SP2 attention: OK", file=sys.stderr)
        return 0
    finally:
        dist.destroy_process_group()


def check_artifacts(model_dir: Path, turbo_dir: Path, first_frame: Path, last_frame: Path, expected_world_size: int = 1) -> dict:
    if not model_dir.is_dir():
        die(f"H3 model directory not found: {model_dir}")
    component_files = {
        # auto_calc_config reads the released transformer architecture before
        # the runner is constructed, even though runtime semantics are frozen.
        "transformer": ("config.json",),
        "text_encoder": ("config.json",),
        "tokenizer": ("tokenizer.json", "tokenizer_config.json"),
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
    try:
        transformer_config = json.loads((model_dir / "transformer/config.json").read_text(encoding="utf-8"))
        attention_heads = int(transformer_config.get("num_attention_heads", 56))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        die(f"H3 transformer config cannot be parsed: {error}")
    if attention_heads % expected_world_size:
        die(f"H3 Ulysses requires num_attention_heads={attention_heads} to be divisible by world_size={expected_world_size}")
    processor_dir = model_dir / "processor"
    if not processor_dir.is_dir():
        die(f"H3 model component directory not found: {processor_dir}")
    processor_files = (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for filename in processor_files:
        if not (processor_dir / filename).is_file():
            die(f"H3 processor file not found: {processor_dir / filename}")
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
        "attention_heads": attention_heads,
        **image_info,
    }


def check_runtime_config(path: Path, expected_world_size: int) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (
        ("enable_cfg", False),
        ("video_flow_shift", 6.0),
        ("h3_step_update", "training_euler"),
        ("h3_refiner_attn_type", "aiter_triton_bf16_flash_attn"),
        ("h3_core_attn_type", "aiter_triton_bf16_flash_attn"),
        ("warmup", False),
    ):
        if config.get(key) != expected:
            die(f"runtime config {key} must be {expected!r}, got {config.get(key)!r}")
    if config.get("tensor_parallel") is not False:
        die("runtime config tensor_parallel must remain false for MiniMax-H3 Day0")
    if config.get("text_encoder_tensor_parallel", False) is not False:
        die("runtime config text_encoder_tensor_parallel must remain false for MiniMax-H3 Day0")
    if config.get("cfg_parallel", False) is not False:
        die("runtime config cfg_parallel must remain false for MiniMax-H3 Day0")
    expected_sp = expected_world_size > 1
    if config.get("seq_parallel") is not expected_sp:
        die(f"runtime config seq_parallel must be {expected_sp} for world_size={expected_world_size}")
    if config.get("vae_decode_parallel") is not expected_sp:
        die(f"runtime config vae_decode_parallel must be {expected_sp} for world_size={expected_world_size}")
    parallel = config.get("parallel")
    if expected_world_size == 1:
        if parallel not in (False, None):
            die(f"single-GPU runtime config parallel must be false, got {parallel!r}")
    else:
        if not isinstance(parallel, dict):
            die("SP2 runtime config parallel must be a dictionary")
        if parallel.get("seq_p_size") != expected_world_size or parallel.get("seq_p_attn_type") != "ulysses":
            die(f"SP2 runtime config must select seq_p_size={expected_world_size} and Ulysses, got {parallel!r}")
        if parallel.get("tensor_p_size", 1) != 1 or parallel.get("cfg_p_size", 1) != 1:
            die(f"MiniMax-H3 Day0 SP2 must not enable TP or CFG parallel, got {parallel!r}")
        if parallel.get("seq_p_prepost_backend") != "torch" or parallel.get("seq_p_a2a_backend") != "torch":
            die(f"MiniMax-H3 Day0 SP2 requires the conservative torch layout/A2A backends, got {parallel!r}")
        if parallel.get("seq_p_quant_scheme") is not None or parallel.get("seq_p_fp8_comm", False) or parallel.get("seq_p_fp4_comm", False):
            die(f"MiniMax-H3 Day0 SP2 communication must remain unquantized, got {parallel!r}")
        if parallel.get("seq_p_tensor_fusion", False) is not False or parallel.get("seq_p_head_parallel", False) is not False:
            die(f"MiniMax-H3 Day0 SP2 must not enable tensor fusion or the head pipeline, got {parallel!r}")
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
    parser.add_argument("--expected-world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--distributed-attention-only", action="store_true")
    args = parser.parse_args()

    if args.distributed_attention_only:
        return smoke_distributed_ulysses()

    gpu = require_gfx1201(args.expected_world_size)
    runtime = check_runtime_dependencies(args.source_mode)
    artifacts = {}
    if not args.skip_artifacts:
        artifacts = check_artifacts(args.model_dir, args.turbo_dir, args.first_frame, args.last_frame, args.expected_world_size)
    config = check_runtime_config(args.runtime_config, args.expected_world_size) if args.runtime_config else {}
    attn_runs = [smoke_triton_mha(device_index) for device_index in range(args.expected_world_size)]
    import torch

    torch.cuda.set_device(0)
    attn = attn_runs[0]
    payload = {
        "status": "ok",
        "gpu": gpu,
        "runtime": runtime,
        "artifacts": artifacts,
        "triton_mha": attn,
        "triton_mha_devices": attn_runs,
        "world_size": args.expected_world_size,
        "task": config.get("task", "fl2av"),
        "h3_core_attn_type": config.get("h3_core_attn_type", "aiter_triton_bf16_flash_attn"),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(
        f"[H3Preflight] gfx={gpu['arch']} rocm={gpu['rocm']} source={args.source_mode} "
        f"Triton MHA ran on {args.expected_world_size} device(s) "
        f"q={tuple(attn['q_shape'])} abs_max={attn['abs_max']:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
