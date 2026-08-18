#!/usr/bin/env python3
"""A/B the two existing Aiter Attention kernels at the collected H3 Core shape."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path


SEQUENCE_LENGTH = 42196
HEADS = 28
HEAD_DIM = 128


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": ordered[p95_index],
    }


def compare_outputs(reference, candidate, chunk_tokens: int = 2048) -> dict:
    import torch

    dot = reference_sq = candidate_sq = abs_sum = squared_error_sum = 0.0
    max_abs_error = 0.0
    elements = reference.numel()
    for start in range(0, reference.shape[0], chunk_tokens):
        ref = reference[start : start + chunk_tokens].float()
        cand = candidate[start : start + chunk_tokens].float()
        error = cand - ref
        dot += float((ref * cand).sum().item())
        reference_sq += float((ref * ref).sum().item())
        candidate_sq += float((cand * cand).sum().item())
        abs_sum += float(error.abs().sum().item())
        squared_error_sum += float((error * error).sum().item())
        max_abs_error = max(max_abs_error, float(error.abs().max().item()))
    return {
        "cosine_similarity": dot / math.sqrt(reference_sq * candidate_sq),
        "mean_absolute_error": abs_sum / elements,
        "root_mean_square_error": math.sqrt(squared_error_sum / elements),
        "max_absolute_error": max_abs_error,
    }


def timed_runs(function, warmup: int, iterations: int):
    import torch

    output = None
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return output, timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")

    import torch
    from aiter import flash_attn_varlen_func
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("H3 Core Attention A/B requires exactly one visible ROCm GPU")
    rocm = str(torch.version.hip or "")
    properties = torch.cuda.get_device_properties(0)
    arch = properties.gcnArchName
    if not rocm.startswith("7.2") or not arch.lower().startswith("gfx1201"):
        raise RuntimeError(f"expected ROCm 7.2 on gfx1201, got ROCm={rocm!r} arch={arch!r}")

    shape = (SEQUENCE_LENGTH, HEADS, HEAD_DIM)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25
    cu_seqlens = torch.tensor((0, SEQUENCE_LENGTH), device="cuda", dtype=torch.int32)

    def triton_flash():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
        )

    def native_sage():
        return fav3_sage_wrapper_func(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            return_lse=False,
            layout="bshd",
            config={"backend": "sage_attn_v2_gfx1201"},
            smooth_k=True,
        ).squeeze(0)

    with torch.inference_mode():
        flash_output, flash_timings = timed_runs(triton_flash, args.warmup, args.iterations)
        sage_output, sage_timings = timed_runs(native_sage, args.warmup, args.iterations)
        if not torch.isfinite(flash_output).all() or not torch.isfinite(sage_output).all():
            raise RuntimeError("Attention backend produced non-finite output")
        comparison = compare_outputs(flash_output, sage_output)

    flash_summary = summarize(flash_timings)
    sage_summary = summarize(sage_timings)
    result = {
        "schema": "lightx2v.minimax_h3.core_attention_ab.v1",
        "status": "ok",
        "scope": "isolated_h3_core_attention_only",
        "workload": {
            "source": "H3 runcollector 1344x768 124f SP2",
            "sequence_parallel_contract": "SP2",
            "qkv_shape_per_rank_after_ulysses": list(shape),
            "dtype": "bfloat16",
            "causal": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "device": {
            "name": properties.name,
            "arch": arch,
            "rocm": rocm,
            "torch": torch.__version__,
            "aiter_revision": os.environ.get("AITER_SOURCE_REVISION") or os.environ.get("AITER_COMMIT"),
        },
        "baseline": {
            "name": "aiter_triton_bf16_flash_attn",
            "numerics": "bf16_flash_attention",
            **flash_summary,
        },
        "candidate": {
            "name": "aiter_fav3_sage_bf16_attn",
            "route": "sage_attn_v2_gfx1201",
            "numerics": "int8_qk_fp8_pv",
            **sage_summary,
        },
        "candidate_vs_baseline": {
            "speedup_median": flash_summary["median_ms"] / sage_summary["median_ms"],
            **comparison,
        },
        "exclusions": [
            "Ulysses communication and layout",
            "QKV projections, RoPE, Norm, Dense GEMM, and dynamic LoRA",
            "end-to-end video quality",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
