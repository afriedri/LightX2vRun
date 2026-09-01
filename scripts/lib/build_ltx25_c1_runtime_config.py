#!/usr/bin/env python3
"""Build an LTX-2.5 C1 Ulysses runtime configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_ltx25_runtime_config import build_config as build_reference_config
from build_ltx25_runtime_config import write_json


SUPPORTED_SEQ_PARALLEL_SIZES = (2, 4, 8, 16)


def build_config(model_root: Path, seq_parallel_size: int = 2) -> dict:
    if seq_parallel_size not in SUPPORTED_SEQ_PARALLEL_SIZES:
        supported = ", ".join(str(size) for size in SUPPORTED_SEQ_PARALLEL_SIZES)
        raise ValueError(f"LTX-2.5 C1 seq parallel size must be one of {supported}")
    config = build_reference_config(model_root)
    config["parallel"]["seq_p_size"] = seq_parallel_size
    config["parallel"]["seq_p_attn_type"] = "ulysses"
    config.update(
        {
            # Keep the executable B0 split/offload lifecycle. C1 changes only
            # the isolated operator routes selected below.
            "ltx25_c1_gfx1201": True,
            "attn_type": "torch_sdpa",
            # SP2 plus the SP4/8/16 scale race covered post-A2A H16/H8/H4/H2;
            # native Sage v2 won every core. Keep dispatch role-level; Aiter
            # owns aligned/tail schedule selection for dynamic lengths.
            "ltx25_video_self_attn_type": "aiter_fav3_sage_bf16_attn",
            # Direct dense MHA won both collected video-to-text stages while
            # preserving one role-level route across their different lengths.
            "ltx25_video_text_attn_type": "aiter_triton_dense_bf16_attn",
            "ltx25_audio_self_attn_type": "torch_sdpa",
            "ltx25_audio_text_attn_type": "torch_sdpa",
            "ltx25_audio_to_video_attn_type": "torch_sdpa",
            # Both S1 and S2 video-to-audio contracts selected direct dense.
            "ltx25_video_to_audio_attn_type": "aiter_triton_dense_bf16_attn",
            # On AMD this key is backed by the Aiter compatibility layer; the
            # C1 launcher selects its explicit CK public RMSNorm route.
            "ltx25_qk_rms_norm_type": "sgl-kernel",
            # Block residual RMSNorm and the unpromoted families stay Torch.
            "rms_norm_type": "torch_native",
            # Structural LTX-2.5 BF16 D64/D128 contracts use the strict
            # split-half Triton route. Unsupported layouts fall back to the
            # bitwise reference inside the backend.
            "rope_type": "ltx25_triton_strict_rope",
            "layer_norm_type": "torch",
            "modulate_type": "torch",
        }
    )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seq-parallel-size",
        type=int,
        default=2,
        choices=SUPPORTED_SEQ_PARALLEL_SIZES,
    )
    args = parser.parse_args()
    write_json(
        args.output,
        build_config(args.model_root, seq_parallel_size=args.seq_parallel_size),
    )
    print(f"LTX-2.5 C1 runtime config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
