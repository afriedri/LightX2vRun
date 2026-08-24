#!/usr/bin/env python3
"""Production preflight for MiniMax-H3 on gfx1201. This runs on the GPU."""

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
        die("CUDA/ROCm device is unavailable; MiniMax-H3 must run on the gfx1201 card")
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
        die(f"MiniMax-H3 gfx1201 is frozen to ROCm 7.2, got torch.version.hip={rocm!r}")
    if os.environ.get("GPU_ARCHS") != "gfx1201" or os.environ.get("CU_NUM") != "24":
        die(
            "MiniMax-H3 gfx1201 requires GPU_ARCHS=gfx1201 and CU_NUM=24 "
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


def check_h1_patch_embed_runtime(patch_embed_class) -> dict:
    contract = {
        "name": getattr(patch_embed_class, "_lightx2v_patch_embed_backend", None),
        "frontend_api": getattr(
            patch_embed_class,
            "_lightx2v_collector_frontend_api",
            getattr(patch_embed_class, "_lightx2v_runcollector_frontend_api", None),
        ),
        "aten_operator": getattr(
            patch_embed_class,
            "_lightx2v_collector_aten_operator",
            getattr(patch_embed_class, "_lightx2v_runcollector_aten_operator", None),
        ),
        "sequence_sharded": False,
    }
    expected = {
        "name": "torch_linear",
        "frontend_api": "torch.nn.functional.linear",
        "aten_operator": "addmm",
        "sequence_sharded": False,
    }
    if contract != expected:
        die(f"LightX2V image does not contain the MiniMax-H3 T2a PatchEmbed contract: expected {expected}, got {contract}")
    return contract


def validate_runcollector_contract(
    schema_name: str,
    schema_version: int,
    mode: str,
    required_schema_version: int,
) -> dict:
    expected_name = "lightx2v.minimax_h3.runcollector"
    expected_mode = "h3_t1_lifecycle"
    contract = {
        "required": True,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "mode": mode,
    }
    if schema_name != expected_name or schema_version != required_schema_version or mode != expected_mode:
        die(
            "LightX2V image does not contain the required MiniMax-H3 T1 collector: "
            f"expected {expected_name} v{required_schema_version} mode={expected_mode!r}, got {contract}"
        )
    return contract


def check_runcollector_runtime(required_schema_version: int | None) -> dict:
    if required_schema_version is None:
        return {"required": False}
    try:
        from lightx2v.utils.minimax_h3_runcollector import (
            MINIMAX_H3_RUN_COLLECTOR_MODE,
            MINIMAX_H3_RUN_COLLECTOR_SCHEMA_NAME,
            MINIMAX_H3_RUN_COLLECTOR_SCHEMA_VERSION,
        )
    except (ImportError, AttributeError) as error:
        die(f"LightX2V image is missing the MiniMax-H3 T1 collector contract: {error}")
    return validate_runcollector_contract(
        MINIMAX_H3_RUN_COLLECTOR_SCHEMA_NAME,
        MINIMAX_H3_RUN_COLLECTOR_SCHEMA_VERSION,
        MINIMAX_H3_RUN_COLLECTOR_MODE,
        required_schema_version,
    )


def check_runtime_dependencies(source_mode: str, required_runcollector_schema: int | None = None) -> dict:
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
        die(f"Aiter Triton MHA and Sage APIs are required by the gfx1201 runtime: {error}")
    if not callable(public_varlen):
        die("aiter.flash_attn_varlen_func is not callable")

    try:
        import lightx2v
        import lightx2v.infer  # noqa: F401
        import lightx2v_platform.ops.attn.amd_rocm  # noqa: F401
        from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import _PatchEmbed
        from lightx2v_platform.ops.attn.amd_rocm.triton_mha_attn import (
            dense_flash_attn_func,
        )
        from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER
        from lightx2v.utils.registry_factory import (
            ATTN_WEIGHT_REGISTER,
            RMS_WEIGHT_REGISTER,
            ROPE_REGISTER,
            RUNNER_REGISTER,
        )
    except Exception as error:
        die(f"LightX2V H3 runner/AMD registry cannot be imported: {error}")
    attention_backends = (
        "aiter_triton_mha",
        "aiter_fav3_sage_bf16_attn",
    )
    for backend in attention_backends:
        if backend not in PLATFORM_ATTN_WEIGHT_REGISTER:
            die(f"LightX2V attention backend is not registered: {backend}")
        if backend not in ATTN_WEIGHT_REGISTER:
            die(f"LightX2V model-facing attention registry is missing: {backend}")
    sage_backend = PLATFORM_ATTN_WEIGHT_REGISTER["aiter_fav3_sage_bf16_attn"]
    if not callable(getattr(sage_backend, "apply_with_lse", None)):
        die("LightX2V Sage backend is missing apply_with_lse required by Ring SP16")
    for registry, name in (
        (RUNNER_REGISTER, "minimax_h3"),
        (ATTN_WEIGHT_REGISTER, "ulysses"),
        (ATTN_WEIGHT_REGISTER, "ring"),
        (ATTN_WEIGHT_REGISTER, "torch_sdpa"),
        (RMS_WEIGHT_REGISTER, "torch_native"),
        (RMS_WEIGHT_REGISTER, "minimax_h3_aiter_triton_qk"),
        (ROPE_REGISTER, "torch_real_rope"),
        (ROPE_REGISTER, "minimax_h3_triton_strict_rope"),
    ):
        if name not in registry:
            die(f"LightX2V runtime registry is missing: {name}")
    if not callable(dense_flash_attn_func):
        die("Aiter dense Triton MHA API required by H4 is unavailable")
    patch_embed_contract = check_h1_patch_embed_runtime(_PatchEmbed)

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
        "registered_backends": list(attention_backends),
        "registered_runner": "minimax_h3",
        "registered_routes": [
            "torch_sdpa",
            "torch_native",
            "torch_real_rope",
            "minimax_h3_aiter_triton_qk",
            "minimax_h3_triton_strict_rope",
            "aiter_triton_mha",
            "ulysses",
            "ring",
        ],
        "h1_patch_embed": patch_embed_contract,
        "h3_runcollector": check_runcollector_runtime(required_runcollector_schema),
        "sage_api_importable": True,
        "dense_gemm": check_dense_gemm_runtime(),
    }


def check_dense_gemm_runtime() -> dict:
    import torch

    enabled_env = os.environ.get("PYTORCH_TUNABLEOP_ENABLED")
    if enabled_env != "0":
        die(
            "MiniMax-H3 Dense GEMM/LoRA requires PYTORCH_TUNABLEOP_ENABLED=0; "
            f"got {enabled_env!r}"
        )
    forced_blas = {
        name: os.environ[name]
        for name in ("TORCH_BLAS_PREFER_CUBLASLT", "TORCH_BLAS_PREFER_HIPBLASLT")
        if name in os.environ
    }
    if forced_blas:
        die(
            "MiniMax-H3 Dense GEMM/LoRA uses the measured default BLAS heuristic; "
            f"remove forced BLAS environment variables: {forced_blas}"
        )
    tunable = getattr(torch.cuda, "tunable", None)
    if tunable is None or not callable(getattr(tunable, "is_enabled", None)):
        die("this PyTorch build does not expose torch.cuda.tunable.is_enabled")
    if tunable.is_enabled():
        die("MiniMax-H3 Dense GEMM/LoRA production route requires TunableOp disabled")
    return {
        "route": "torch.mm -> default ROCm BLAS heuristic",
        "measured_backend": "hipBLASLt",
        "tunableop_enabled": False,
        "tunableop_cache": False,
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


def smoke_sage_attention(device_index: int, heads: int) -> dict:
    import torch

    from lightx2v_platform.ops.attn.amd_rocm.sage_attn import (
        AiterFav3SageBf16AttnWeight,
    )

    if heads not in (7, 14, 28, 56):
        die(f"MiniMax-H3 Sage preflight expects 7, 14, 28, or 56 heads, got {heads}")
    seq_len, head_dim = 257, 128
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    q = torch.randn(seq_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    out = AiterFav3SageBf16AttnWeight().apply(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
        causal=False,
    )
    torch.cuda.synchronize()
    expected_shape = (seq_len, heads * head_dim)
    if tuple(out.shape) != expected_shape:
        die(f"SageAttention2 output shape {tuple(out.shape)} != {expected_shape}")
    if not torch.isfinite(out).all():
        die("SageAttention2 produced non-finite values on gfx1201")
    peak = float(out.abs().max().item())
    if peak <= 0.0:
        die("SageAttention2 returned an all-zero tensor")
    return {
        "backend": "aiter_fav3_sage_bf16_attn",
        "route": "sage_attn_v2_gfx1201",
        "numerics": "int8_qk_fp8_pv",
        "q_shape": [seq_len, heads, head_dim],
        "output_shape": list(out.shape),
        "dtype": str(out.dtype).replace("torch.", ""),
        "abs_max": peak,
        "device": str(out.device),
    }


def standalone_sage_smoke_plan(world_size: int) -> tuple[tuple[int, int], ...]:
    """Plan parent-process Sage coverage without cross-device reuse."""
    if world_size not in (1, 2, 4, 8, 16):
        raise ValueError(f"MiniMax-H3 Sage preflight does not support world_size={world_size}")
    # Multi-GPU runs are covered once, through the exact torchrun
    # Ulysses/Ring-to-Sage production path. A second parent-process Sage check would
    # be redundant and can reuse device-local FlyDSL state incorrectly.
    return ((0, 56),) if world_size == 1 else ()


def set_distributed_local_device() -> int:
    """Bind this torchrun worker before LightX2V/Aiter initializes a backend."""
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


def smoke_distributed_attention() -> int:
    local_rank = set_distributed_local_device()

    import torch
    import torch.distributed as dist

    # These imports can initialize Aiter/Triton/FlyDSL device-local state, so
    # they must happen only after the torchrun worker has selected LOCAL_RANK.
    from lightx2v.common.ops.attn.ring_attn import RingAttnWeight
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v_platform.ops.attn.amd_rocm.sage_attn import AiterFav3SageBf16AttnWeight

    dist.init_process_group(backend="nccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size not in (2, 4, 8, 16):
            raise RuntimeError(f"MiniMax-H3 distributed preflight requires world_size 2, 4, 8, or 16, got {world_size}")
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
        attention_kind = "ring" if world_size == 16 else "ulysses"
        distributed_attention = RingAttnWeight() if attention_kind == "ring" else UlyssesAttnWeight(a2a_backend="torch")
        output, aux_output = distributed_attention.apply_new(
            q=q,
            k=k,
            v=v,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            attention_module=AiterFav3SageBf16AttnWeight(),
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
                        "backend": attention_kind,
                        "attention": "aiter_fav3_sage_bf16_attn",
                        "attention_route": (
                            "sage_attn_v2_gfx1201_ring_hybrid"
                            if attention_kind == "ring"
                            else "sage_attn_v2_gfx1201"
                        ),
                        "world_size": world_size,
                        "local_q_shape": [local_len, heads, head_dim],
                        "local_output_shape": list(output.shape),
                        "aux_q_shape": [aux_len, heads, head_dim],
                        "aux_output_shape": list(aux_output.shape),
                    },
                    sort_keys=True,
                )
            )
            print(f"[H3Preflight] RCCL {attention_kind.title()} SP{world_size} attention: OK", file=sys.stderr)
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
    if expected_world_size != 16 and attention_heads % expected_world_size:
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
        ("h3_refiner_attn_type", "aiter_triton_mha"),
        ("h3_core_attn_type", "aiter_fav3_sage_bf16_attn"),
        ("attn_type", "aiter_triton_mha"),
        ("qwen3vl_attn_type", "aiter_triton_mha"),
        ("h3_core_qk_rms_type", "minimax_h3_aiter_triton_qk"),
        ("h3_core_rope_type", "minimax_h3_triton_strict_rope"),
        ("h4_video_attn_type", "aiter_triton_mha"),
        ("dit_quant_scheme", "Default"),
        ("lora_dynamic_apply", True),
        ("lazy_load", True),
        ("cpu_offload", True),
        ("offload_granularity", "block"),
        ("text_encoder_unload_after_infer", True),
        ("warmup", False),
    ):
        if config.get(key) != expected:
            die(f"runtime config {key} must be {expected!r}, got {config.get(key)!r}")
    if config.get("tensor_parallel") is not False:
        die("runtime config tensor_parallel must remain false for MiniMax-H3 gfx1201")
    if config.get("text_encoder_tensor_parallel", False) is not False:
        die("runtime config text_encoder_tensor_parallel must remain false for MiniMax-H3 gfx1201")
    expected_rank0_only = expected_world_size > 1
    if config.get("text_encoder_rank0_only") is not expected_rank0_only:
        die(
            "runtime config text_encoder_rank0_only must be "
            f"{expected_rank0_only} for world_size={expected_world_size}"
        )
    if config.get("cfg_parallel", False) is not False:
        die("runtime config cfg_parallel must remain false for MiniMax-H3 gfx1201")
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
            die(f"SP{expected_world_size} runtime config parallel must be a dictionary")
        expected_attention = "ring" if expected_world_size == 16 else "ulysses"
        if parallel.get("seq_p_size") != expected_world_size or parallel.get("seq_p_attn_type") != expected_attention:
            die(
                f"SP{expected_world_size} runtime config must select seq_p_size={expected_world_size} "
                f"and {expected_attention.title()}, got {parallel!r}"
            )
        if parallel.get("tensor_p_size", 1) != 1 or parallel.get("cfg_p_size", 1) != 1:
            die(f"MiniMax-H3 gfx1201 SP{expected_world_size} must not enable TP or CFG parallel, got {parallel!r}")
        if parallel.get("seq_p_prepost_backend") != "torch" or parallel.get("seq_p_a2a_backend") != "torch":
            die(f"MiniMax-H3 gfx1201 SP{expected_world_size} requires the conservative torch layout/A2A backends, got {parallel!r}")
        if parallel.get("seq_p_quant_scheme") is not None or parallel.get("seq_p_fp8_comm", False) or parallel.get("seq_p_fp4_comm", False):
            die(f"MiniMax-H3 gfx1201 SP{expected_world_size} communication must remain unquantized, got {parallel!r}")
        if parallel.get("seq_p_tensor_fusion", False) is not False or parallel.get("seq_p_head_parallel", False) is not False:
            die(f"MiniMax-H3 gfx1201 SP{expected_world_size} must not enable tensor fusion or the head pipeline, got {parallel!r}")
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
    parser.add_argument("--expected-world-size", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--required-runcollector-schema", type=int)
    parser.add_argument("--kernel-smoke", action="store_true", help="Run standalone Triton/Sage GPU kernels; reserved for explicit preflight-only runs")
    parser.add_argument("--distributed-attention-only", action="store_true")
    args = parser.parse_args()

    if args.distributed_attention_only:
        return smoke_distributed_attention()

    gpu = require_gfx1201(args.expected_world_size)
    import torch

    # The parent preflight process initializes Aiter once on logical cuda:0.
    # Do not execute native Sage on multiple devices from that one process:
    # FlyDSL owns device-local runtime state. Multi-GPU Sage coverage is
    # provided below by the one-process-per-device torchrun Ulysses/Ring smoke.
    torch.cuda.set_device(0)
    runtime = check_runtime_dependencies(args.source_mode, args.required_runcollector_schema)
    artifacts = {}
    if not args.skip_artifacts:
        artifacts = check_artifacts(args.model_dir, args.turbo_dir, args.first_frame, args.last_frame, args.expected_world_size)
    config = check_runtime_config(args.runtime_config, args.expected_world_size) if args.runtime_config else {}
    # Kernel execution is an explicit diagnostic. Normal inference immediately
    # exercises these same paths, so only retain the cheap static/runtime
    # contracts on every request.
    triton_runs = [smoke_triton_mha(0)] if args.kernel_smoke else []
    sage_runs = (
        [
            smoke_sage_attention(device_index, heads)
            for device_index, heads in standalone_sage_smoke_plan(args.expected_world_size)
        ]
        if args.kernel_smoke
        else []
    )

    torch.cuda.set_device(0)
    triton_attn = triton_runs[0] if triton_runs else None
    sage_attn = sage_runs[0] if sage_runs else None
    payload = {
        "status": "ok",
        "gpu": gpu,
        "runtime": runtime,
        "artifacts": artifacts,
        "triton_mha": triton_attn,
        "triton_mha_devices": triton_runs,
        "triton_mha_scope": "replicated Refiner kernel contract on cuda:0" if args.kernel_smoke else "skipped during normal inference startup",
        "sage_attention": sage_attn,
        "sage_attention_devices": sage_runs,
        "sage_attention_scope": (
            "standalone cuda:0"
            if args.kernel_smoke and args.expected_world_size == 1
            else (
                "standalone skipped; all devices use one torchrun worker in the following distributed attention smoke"
                if args.kernel_smoke
                else "skipped during normal inference startup"
            )
        ),
        "kernel_smoke_requested": args.kernel_smoke,
        "world_size": args.expected_world_size,
        "task": config.get("task", "fl2av"),
        "h3_core_attn_type": config.get("h3_core_attn_type", "aiter_fav3_sage_bf16_attn"),
        "dense_gemm": runtime["dense_gemm"],
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if not args.kernel_smoke:
        kernel_status = "kernel smoke skipped for normal inference"
    elif sage_attn is not None:
        sage_status = (
            "standalone Sage Core ran on cuda:0 "
            f"q={tuple(sage_attn['q_shape'])} abs_max={sage_attn['abs_max']:.4f}"
        )
        kernel_status = f"Triton Refiner ran on cuda:0; {sage_status}"
    else:
        kernel_status = "Triton Refiner ran on cuda:0; standalone Sage skipped; distributed Ulysses/Ring-to-Sage smoke follows"
    print(
        f"[H3Preflight] gfx={gpu['arch']} rocm={gpu['rocm']} source={args.source_mode} "
        f"{kernel_status}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
