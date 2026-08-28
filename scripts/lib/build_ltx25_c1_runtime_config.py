#!/usr/bin/env python3
"""Build the two-rank LTX-2.5 C1 runtime configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_ltx25_runtime_config import build_config as build_reference_config
from build_ltx25_runtime_config import write_json


def build_config(model_root: Path) -> dict:
    config = build_reference_config(model_root)
    config.update(
        {
            # Keep the executable B0 split/offload lifecycle. C1 changes only
            # the isolated operator routes selected below.
            "ltx25_c1_gfx1201": True,
            "attn_type": "torch_sdpa",
            "ltx25_video_self_attn_type": "torch_sdpa",
            "ltx25_video_text_attn_type": "torch_sdpa",
            "ltx25_audio_self_attn_type": "torch_sdpa",
            "ltx25_audio_text_attn_type": "torch_sdpa",
            "ltx25_audio_to_video_attn_type": "torch_sdpa",
            # Both S1 and S2 exact contracts selected the direct Aiter Triton
            # dense backend for this semantic role.
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
    args = parser.parse_args()
    write_json(args.output, build_config(args.model_root))
    print(f"LTX-2.5 C1 runtime config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
