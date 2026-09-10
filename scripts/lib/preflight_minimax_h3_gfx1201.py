#!/usr/bin/env python3
"""Production preflight for MiniMax-H3 on gfx1201. This runs on the GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    # Keep JSON evidence on stdout parseable.
    print(f"WARNING: {message}", file=sys.stderr)


def _normalize_h3_lora_key(key: str) -> str | None:
    for prefix in (
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
        "model.",
    ):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    suffixes = {
        ".lora_A.default.weight": ".lora_down.weight",
        ".lora_B.default.weight": ".lora_up.weight",
        ".lora_A.weight": ".lora_down.weight",
        ".lora_B.weight": ".lora_up.weight",
        ".lora.down.weight": ".lora_down.weight",
        ".lora.up.weight": ".lora_up.weight",
        ".lora_down.weight": ".lora_down.weight",
        ".lora_up.weight": ".lora_up.weight",
    }
    for suffix, replacement in suffixes.items():
        if key.endswith(suffix):
            return key[: -len(suffix)] + replacement
    return key if key.endswith(".alpha") else None


def inspect_h3_lora_header(path: Path) -> dict:
    """Validate the same LoRA key/pair contract as the runtime without loading tensors."""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError("missing safetensors header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > min(size - 8, 64 * 1024 * 1024):
                raise ValueError(f"invalid safetensors header length {header_length}")
            header = json.loads(stream.read(header_length))
    except (OSError, ValueError, TypeError, json.JSONDecodeError, struct.error) as error:
        die(f"Turbo LoRA header cannot be parsed: {path}: {error}")
    if not isinstance(header, dict):
        die(f"Turbo LoRA header must be an object: {path}")

    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    normalized = {}
    unsupported = []
    for source_key, descriptor in tensors.items():
        key = _normalize_h3_lora_key(source_key)
        if key is None:
            unsupported.append(source_key)
            continue
        if key in normalized:
            die(f"Turbo LoRA keys collide after normalization: {source_key} and {normalized[key][0]}")
        normalized[key] = (source_key, descriptor)
    if unsupported:
        die(f"Turbo LoRA contains {len(unsupported)} unsupported tensors: {unsupported[:4]}")

    down_names = {key for key in normalized if key.endswith(".lora_down.weight")}
    up_names = {key for key in normalized if key.endswith(".lora_up.weight")}
    expected_up = {key[: -len(".lora_down.weight")] + ".lora_up.weight" for key in down_names}
    if not down_names or up_names != expected_up:
        die(
            "Turbo LoRA has incomplete pairs: "
            f"missing_up={sorted(expected_up - up_names)[:3]}, "
            f"orphan_up={sorted(up_names - expected_up)[:3]}"
        )

    ranks = set()
    dtypes = set()
    for down_name in down_names:
        base = down_name[: -len(".lora_down.weight")]
        down = normalized[down_name][1]
        up = normalized[base + ".lora_up.weight"][1]
        try:
            down_shape = tuple(int(value) for value in down["shape"])
            up_shape = tuple(int(value) for value in up["shape"])
            dtypes.update((str(down["dtype"]), str(up["dtype"])))
        except (KeyError, TypeError, ValueError) as error:
            die(f"Turbo LoRA pair has an invalid tensor descriptor for {base}: {error}")
        if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1]:
            die(f"Turbo LoRA pair has incompatible shapes for {base}: down={down_shape}, up={up_shape}")
        ranks.add(down_shape[0])

    alpha_names = {key for key in normalized if key.endswith(".alpha")}
    expected_alpha = {key[: -len(".lora_down.weight")] + ".alpha" for key in down_names}
    orphan_alpha = alpha_names - expected_alpha
    if orphan_alpha:
        die(f"Turbo LoRA contains alpha tensors without matching pairs: {sorted(orphan_alpha)[:3]}")
    return {
        "path": str(path),
        "size_bytes": size,
        "tensor_count": len(tensors),
        "pair_count": len(down_names),
        "ranks": sorted(ranks),
        "dtypes": sorted(dtypes),
        "alpha_tensor_count": len(alpha_names),
        "config_alpha_required": bool(expected_alpha - alpha_names),
        "metadata": header.get("__metadata__", {}),
    }


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
    if not rocm:
        die("MiniMax-H3 gfx1201 requires a ROCm PyTorch build")
    if not rocm.startswith("7.2"):
        warn(f"MiniMax-H3 was validated on ROCm 7.2; testing torch.version.hip={rocm!r}")
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


def check_h1_patch_embed_runtime(patch_embed_class, *, required: bool = False) -> dict:
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
        message = f"LightX2V H1 PatchEmbed metadata differs from the measured T2a backend: expected {expected}, got {contract}"
        if required:
            die(message)
        warn(message)
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


def check_ck_conv3d_runtime(aiter_module) -> dict:
    expected_revision = "af7118e342580ecd3f71edce7b1d0ba465012ecf"
    expected_sha256 = "c05bd995fd752af344bb4850caf533b397f908292fccae66306574d1b1591b51"
    expected_fp16_rows = 17

    try:
        from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import (
            MiniMaxH3VideoCausalConv3d,
        )
    except ImportError as error:
        die(f"LightX2V image is missing the H3 video VAE CK route: {error}")

    if getattr(MiniMaxH3VideoCausalConv3d, "_lightx2v_ck_conv3d_backend", None) != "Aiter_CK_Conv3d_v1":
        die("LightX2V H3 video VAE does not declare the Aiter CK Conv3d v1 route")
    if getattr(MiniMaxH3VideoCausalConv3d, "_lightx2v_ck_conv3d_universal_hot", None) is not True:
        die("LightX2V H3 video VAE does not use Aiter's universal CK Conv3d HOT route")

    required_api = (
        "ck_conv3d_v1",
        "get_ck_conv3d_v1_config",
        "is_ck_conv3d_v1_prebuilt",
        "load_ck_conv3d_v1_configs",
    )
    missing_api = [name for name in required_api if not callable(getattr(aiter_module, name, None))]
    if missing_api:
        die(f"installed Aiter is missing CK Conv3d v1 APIs: {missing_api}")
    if not aiter_module.is_ck_conv3d_v1_prebuilt():
        die("Aiter CK Conv3d v1 AOT extension is absent from the installed wheel")
    fp16_aot_selectors = set(
        getattr(aiter_module, "CK_CONV3D_V1_FP16_HOT_KERNEL_KEYS", {})
    )
    universal_dtypes = set(
        getattr(aiter_module, "CK_CONV3D_V1_UNIVERSAL_DTYPES", ())
    )
    required_fp16_selectors = set(range(1, 12))
    if (
        not required_fp16_selectors.issubset(fp16_aot_selectors)
        or getattr(aiter_module, "CK_CONV3D_V1_FP16_UNIVERSAL_HOT", None) is not True
    ):
        die("installed Aiter does not contain the universal FP16 CK Conv3d HOT catalog")
    if universal_dtypes != {"bfloat16", "float16"}:
        warn(
            "installed Aiter does not declare universal CK Conv3d coverage for both "
            f"BF16 and FP16: {sorted(universal_dtypes)}"
        )

    observed_revision = getattr(aiter_module, "CK_CONV3D_V1_CK_REVISION", None)
    observed_sha256 = getattr(aiter_module, "CK_CONV3D_V1_TUNED_CONFIG_SHA256", None)
    observed_rows = getattr(aiter_module, "CK_CONV3D_V1_TUNED_CONFIG_ROWS", None)
    configured_catalog = os.environ.get("AITER_CONFIG_CONV3D_V1")
    config_path = Path(
        configured_catalog
        or getattr(aiter_module, "CK_CONV3D_V1_DEFAULT_CONFIG", "")
    )
    if observed_revision != expected_revision:
        warn(f"Aiter CK Conv3d revision is {observed_revision!r}, release provenance is {expected_revision}")
    if observed_sha256 != expected_sha256 or observed_rows != 545:
        warn(
            "Aiter default CK Conv3d catalog provenance differs from the H3 release; "
            f"validating the active catalog by FP16 coverage instead: sha256={observed_sha256!r}, "
            f"rows={observed_rows!r}"
        )
    actual_sha256 = None
    configs = {}
    if config_path.is_file():
        actual_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if configured_catalog is None and actual_sha256 != expected_sha256:
            warn(
                f"Aiter default CK Conv3d catalog content SHA256 is {actual_sha256}, "
                f"release provenance is {expected_sha256}"
            )
        try:
            configs = aiter_module.load_ck_conv3d_v1_configs(str(config_path))
        except (KeyError, OSError, TypeError, ValueError) as error:
            warn(
                "Aiter CK Conv3d preferred-selector catalog could not be loaded; "
                f"the universal HOT route remains available: {error}"
            )
    else:
        warn(
            f"Aiter CK Conv3d preferred-selector catalog is missing at {config_path}; "
            "the universal HOT route remains available"
        )
    fp16_rows = sum(1 for key in configs if key[2] == "float16")
    fp16_selectors = {
        int(config["selector_id"])
        for key, config in configs.items()
        if key[2] == "float16"
    }
    if (
        fp16_rows < expected_fp16_rows
        or not required_fp16_selectors.issubset(fp16_selectors)
    ):
        warn(
            "Aiter CK Conv3d tuned catalog has incomplete H3 preferred-selector coverage; "
            "the universal FP16 HOT route remains available: "
            f"rows={len(configs)}, fp16_rows={fp16_rows}, "
            f"fp16_selectors={sorted(fp16_selectors)}"
        )
    return {
        "backend": "Aiter_CK_Conv3d_v1",
        "ck_revision": observed_revision,
        "tuned_config": str(config_path),
        "tuned_config_sha256": actual_sha256,
        "tuned_config_rows": len(configs),
        "fp16_tuned_rows": fp16_rows,
        "fp16_selectors": sorted(fp16_selectors),
        "fp16_aot_selectors": sorted(fp16_aot_selectors),
        "aot_prebuilt": True,
        "universal_hot": True,
        "universal_dtypes": sorted(universal_dtypes),
    }


def check_runtime_dependencies(
    source_mode: str,
    required_runcollector_schema: int | None = None,
    expected_world_size: int = 1,
    require_ck_conv3d: bool = False,
) -> dict:
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
        from aiter.ops.flydsl.sage_attention import sage_attention_v2_gfx1201_support_reason  # noqa: F401
        from aiter.ops.triton.normalization.rmsnorm import rms_norm  # noqa: F401
    except ImportError as error:
        die(f"Aiter Triton MHA and Sage APIs are required by the gfx1201 runtime: {error}")
    if not callable(public_varlen):
        die("aiter.flash_attn_varlen_func is not callable")

    try:
        import lightx2v
        import lightx2v.infer  # noqa: F401
        import lightx2v_platform.ops.attn.amd_rocm  # noqa: F401
        from lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision import _PatchEmbed
        from lightx2v_platform.base.backend_logger import BackendLogger
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
    if expected_world_size == 16 and not callable(getattr(sage_backend, "apply_with_lse", None)):
        die("LightX2V Sage backend is missing apply_with_lse required by CP2xU8 SP16")
    for registry, name in (
        (RUNNER_REGISTER, "minimax_h3"),
        (ATTN_WEIGHT_REGISTER, "ulysses"),
        (ATTN_WEIGHT_REGISTER, "ring"),
        (ATTN_WEIGHT_REGISTER, "ulysses_ring"),
        (ATTN_WEIGHT_REGISTER, "torch_sdpa"),
        (RMS_WEIGHT_REGISTER, "torch_native"),
        (RMS_WEIGHT_REGISTER, "minimax_h3_aiter_triton_qk"),
        (ROPE_REGISTER, "torch_real_rope"),
        (ROPE_REGISTER, "minimax_h3_triton_strict_rope"),
    ):
        if name in ("ring", "ulysses_ring") and expected_world_size != 16:
            continue
        if name == "ulysses" and expected_world_size == 1:
            continue
        if name not in registry:
            die(f"LightX2V runtime registry is missing: {name}")
    if not callable(dense_flash_attn_func):
        die("Aiter dense Triton MHA API required by H4 is unavailable")
    if "minimax_h3" in os.environ.get("LIGHTX2V_NO_FALLBACK_MODELS", "").split(","):
        policy = getattr(BackendLogger("minimax_h3"), "fallback_is_disabled", None)
        if not callable(policy) or not policy():
            die("Rebuild the LightX2V image: it does not enforce required H3 kernels")
    patch_embed_contract = check_h1_patch_embed_runtime(_PatchEmbed, required=required_runcollector_schema is not None)

    source_root = Path(lightx2v.__file__).resolve().parents[1]
    source_revision = os.environ.get("LIGHTX2V_SOURCE_REVISION") or git_revision(source_root)
    if not source_revision:
        warn("image source provenance is missing; recording an unknown LightX2V revision")
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
            "ulysses_ring",
        ],
        "h1_patch_embed": patch_embed_contract,
        "h3_runcollector": check_runcollector_runtime(required_runcollector_schema),
        "ck_conv3d": check_ck_conv3d_runtime(aiter) if require_ck_conv3d else {"required": False},
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
    # distributed-to-Sage production path. A second parent-process Sage check would
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
    from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
    from lightx2v.common.ops.attn.ulysses_ring_attn import UlyssesRingAttnWeight
    from lightx2v.models.networks.minimax_h3.parallel import create_h3_2d_process_groups
    from lightx2v_platform.ops.attn.amd_rocm.sage_attn import AiterFav3SageBf16AttnWeight

    dist.init_process_group(backend="nccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size not in (2, 4, 8, 16):
            raise RuntimeError(f"MiniMax-H3 distributed preflight requires world_size 2, 4, 8, or 16, got {world_size}")
        device = torch.device(f"cuda:{local_rank}")
        aux_len = 4 if world_size == 16 else 3
        torch.manual_seed(20260817)
        aux_q = torch.randn(aux_len, 56, 128, device=device, dtype=torch.bfloat16)
        aux_k = torch.randn_like(aux_q)
        aux_v = torch.randn_like(aux_q)
        torch.manual_seed(20260817 + rank)
        local_len, heads, head_dim = 257, 56, 128
        q = torch.randn(local_len, heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        attention_kind = "ulysses_ring" if world_size == 16 else "ulysses"
        distributed_attention = (
            UlyssesRingAttnWeight(a2a_backend="torch")
            if attention_kind == "ulysses_ring"
            else UlyssesAttnWeight(a2a_backend="torch")
        )
        attention_kwargs = dict(
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
        if attention_kind == "ulysses_ring":
            groups = create_h3_2d_process_groups(
                dist.group.WORLD,
                context_size=2,
                ulysses_size=8,
            )
            attention_kwargs["ulysses_group"] = groups.ulysses_group
            attention_kwargs["context_group"] = groups.context_group
        output, aux_output = distributed_attention.apply_new(**attention_kwargs)
        torch.cuda.synchronize(device)
        expected_shape = (local_len, heads * head_dim)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"Distributed attention output shape {tuple(output.shape)} != {expected_shape}")
        expected_aux_shape = (aux_len, heads * head_dim)
        if aux_output is None or tuple(aux_output.shape) != expected_aux_shape:
            raise RuntimeError(f"Distributed auxiliary output shape is not {expected_aux_shape}: {None if aux_output is None else tuple(aux_output.shape)}")
        if not torch.isfinite(output).all() or output.abs().max().item() <= 0.0 or not torch.isfinite(aux_output).all():
            raise RuntimeError("Distributed attention produced zero or non-finite output")
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "backend": attention_kind,
                        "attention": "aiter_fav3_sage_bf16_attn",
                        "attention_route": (
                            "sage_attn_v2_gfx1201_cp2_u8_2d"
                            if attention_kind == "ulysses_ring"
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


def check_artifacts(model_dir: Path, turbo_dir: Path, first_frame: Path, last_frame: Path, expected_world_size: int = 1, *, task: str = "fl2av", lora_path: str | None = None) -> dict:
    transformer = "transformer_ref" if task == "ref2av" else "transformer"
    if not model_dir.is_dir():
        die(f"H3 model directory not found: {model_dir}")
    component_files = {
        # auto_calc_config reads the released transformer architecture before
        # the runner is constructed, even though runtime semantics are frozen.
        transformer: ("config.json",),
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
        transformer_config = json.loads((model_dir / transformer / "config.json").read_text(encoding="utf-8"))
        attention_heads = int(transformer_config.get("num_attention_heads", 56))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        die(f"H3 transformer config cannot be parsed: {error}")
    head_parallel_size = 8 if expected_world_size == 16 else expected_world_size
    if attention_heads % head_parallel_size:
        die(
            f"H3 Ulysses dimension requires num_attention_heads={attention_heads} "
            f"to be divisible by {head_parallel_size}"
        )
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
    for component in (transformer, "text_encoder", "vae", "audio_vae"):
        component_dir = model_dir / component
        if not any(component_dir.glob("*.safetensors")):
            die(f"H3 model component has no safetensors shard: {component_dir}")
    turbo = Path(lora_path) if lora_path else turbo_dir / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    if lora_path and not turbo.is_file():
        die(f"Turbo LoRA not found: {turbo}")
    if task != "ref2av" and not turbo.is_file():
        die(f"Turbo LoRA not found: {turbo}")
    lora_header = inspect_h3_lora_header(turbo) if lora_path else None
    images = [(f"reference_image_{i+1}", Path(path)) for i, path in enumerate(str(first_frame).split(","))] if task == "ref2av" else [("first_frame", first_frame), ("last_frame", last_frame)]
    if task == "ref2av" and not 1 <= len(images) <= 9:
        die("Ref2AV requires between 1 and 9 reference images")
    for label, path in images:
        if not path.is_file():
            die(f"{label.replace('_', ' ')} not found: {path}")
    try:
        from PIL import Image

        image_info = {}
        for label, path in images:
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
        "turbo": str(turbo) if lora_path or task != "ref2av" else None,
        "lora_header": lora_header,
        "attention_heads": attention_heads,
        **image_info,
    }


def check_runtime_config(path: Path, expected_world_size: int) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    task = config.get("task", "fl2av")
    profile = config.get("h3_profile")
    if profile is None:
        if task == "ref2av":
            profile = "ref2av_turbo_8step" if config.get("lora_configs") else "ref2av_bf16_base"
        else:
            profile = "fl2av_turbo_4step"
    expected_profiles = {
        "fl2av_turbo_4step": ("fl2av", True),
        "ref2av_bf16_base": ("ref2av", False),
        "ref2av_turbo_8step": ("ref2av", True),
    }
    if profile not in expected_profiles or expected_profiles[profile][0] != task:
        die(f"runtime config has invalid H3 profile/task combination: profile={profile!r}, task={task!r}")
    expects_lora = expected_profiles[profile][1]
    for key, expected in (
        ("enable_cfg", False),
        ("video_flow_shift", 12.0 if config.get("task") == "ref2av" else 6.0),
        ("h3_step_update", config.get("h3_step_update") if config.get("task") == "ref2av" else "training_euler"),
        ("h3_refiner_attn_type", "aiter_triton_mha"),
        ("h3_core_attn_type", "aiter_fav3_sage_bf16_attn"),
        ("attn_type", "aiter_triton_mha"),
        ("qwen3vl_attn_type", "aiter_triton_mha"),
        ("h3_core_qk_rms_type", "minimax_h3_aiter_triton_qk"),
        ("h3_core_rope_type", "minimax_h3_triton_strict_rope"),
        ("h4_video_attn_type", "aiter_triton_mha"),
        ("dit_quant_scheme", "Default"),
        ("lora_dynamic_apply", expects_lora),
        ("lazy_load", True),
        ("cpu_offload", True),
        ("offload_granularity", "block"),
        ("text_encoder_unload_after_infer", True),
        ("warmup", False),
    ):
        if config.get(key) != expected:
            die(f"runtime config {key} must be {expected!r}, got {config.get(key)!r}")
    if config.get("task") == "ref2av" and config.get("h3_step_update") not in ("comfy_res_multistep", "reference_blend", "training_euler"):
        die("Unsupported H3 Ref2AV sampling mode")
    if profile == "ref2av_bf16_base" and config.get("lora_configs"):
        die("Ref2AV BF16 base profile must not load LoRA")
    if profile == "ref2av_turbo_8step":
        lora = config.get("lora_configs") or []
        filename = "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
        if config.get("h3_step_update") != "training_euler" or config.get("infer_steps") != 9:
            die("Ref2AV Turbo 8-step requires training_euler with 9 grid points (8 model evaluations)")
        if (
            len(lora) != 1
            or lora[0].get("alpha") != 8
            or lora[0].get("strength") != 1.0
            or not str(lora[0].get("path") or "").endswith(filename)
        ):
            die(f"Ref2AV Turbo 8-step requires one {filename} LoRA with alpha=8 strength=1.0")
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
        expected_attention = "ulysses_ring" if expected_world_size == 16 else "ulysses"
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
        if expected_world_size == 16:
            if parallel.get("context_p_size") != 2 or parallel.get("ulysses_p_size") != 8:
                die(f"MiniMax-H3 gfx1201 SP16 requires CP2xU8, got {parallel!r}")
            if not bool(config.get("h3_ulysses_minimize_aux_prefix", True)):
                die("MiniMax-H3 gfx1201 CP2xU8 SP16 requires the minimal auxiliary prefix")
            workload = (
                config.get("target_video_length"),
                config.get("target_width"),
                config.get("target_height"),
                config.get("fps"),
            )
            frames, width, height, fps = workload
            if not (type(frames) is int and 124 <= frames <= 362 and frames % 17 == 5
                    and fps == 24 and (width, height) in {(1344, 768), (832, 480)}):
                die(
                    "MiniMax-H3 gfx1201 SP16 supports 17*n+5 frames within 124..362 at 24fps "
                    f"with 1344x768 or 832x480, got {workload!r}"
                )
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
    # provided below by the one-process-per-device torchrun distributed smoke.
    torch.cuda.set_device(0)
    config = check_runtime_config(args.runtime_config, args.expected_world_size) if args.runtime_config else {}
    runtime = check_runtime_dependencies(
        args.source_mode,
        args.required_runcollector_schema,
        args.expected_world_size,
        require_ck_conv3d=config.get("task") == "ref2av",
    )
    artifacts = {}
    if not args.skip_artifacts:
        artifacts = check_artifacts(args.model_dir, args.turbo_dir, args.first_frame, args.last_frame, args.expected_world_size,
                                    task=config.get("task", "fl2av"), lora_path=(config.get("lora_configs") or [{}])[0].get("path"))
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
        kernel_status = "Triton Refiner ran on cuda:0; standalone Sage skipped; distributed-to-Sage smoke follows"
    print(
        f"[H3Preflight] gfx={gpu['arch']} rocm={gpu['rocm']} source={args.source_mode} "
        f"{kernel_status}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
