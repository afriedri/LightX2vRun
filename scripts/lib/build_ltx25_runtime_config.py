#!/usr/bin/env python3
"""Build the frozen LTX-2.5 gfx1201 B0-reference runtime configuration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def build_config(model_root: Path) -> dict:
    transformer = (
        model_root
        / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )
    text_encoder = (
        model_root
        / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    )
    official_video_vae = model_root / "vae/ltx-2.5-video-vae-bf16.safetensors"
    conv_video_vae = model_root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    audio_vae = model_root / "vae/ltx-2.5-audio-vae-bf16.safetensors"
    upscaler = (
        model_root
        / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )
    return {
        "infer_steps": 8,
        "target_video_length": 121,
        "target_height": 704,
        "target_width": 1280,
        "fps": 24,
        "attn_type": "torch_sdpa",
        "rms_norm_type": "torch_native",
        "rope_type": "torch_real_rope",
        "layer_norm_type": "torch",
        "modulate_type": "torch",
        "sample_guide_scale": 1.0,
        "sample_shift": [2.05, 0.95],
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "lazy_load": True,
        "unload_modules": True,
        "gemma_cpu_offload": False,
        "vae_cpu_offload": True,
        "use_tiling_vae": True,
        "warmup": False,
        "num_channels_latents": 128,
        "vae_scale_factors": [8, 32, 32],
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_scale_factor": 4,
        "audio_fps": 24000,
        # Transformer latent frequency bins. The Audio VAE decodes these 16
        # latent bins back to its 64-bin mel-spectrogram output.
        "audio_mel_bins": 16,
        "double_precision_rope": True,
        "dit_quantized": False,
        "dit_original_ckpt": os.fspath(transformer),
        "ltx_text_encoder_ckpt": os.fspath(text_encoder),
        "ltx_video_vae_ckpt": os.fspath(official_video_vae),
        "ltx_audio_vae_ckpt": os.fspath(audio_vae),
        "use_upsampler": True,
        "upsampler_original_ckpt": os.fspath(upscaler),
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
        "parallel": {
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
        },
        "ltx_model_generation": "2.5",
        "ltx25_split_contract_required": True,
        "ltx25_process_isolated_prompt_required": True,
        "ltx25_b0_gfx1201": True,
        # LightX2V's current executable L3 is the released Conv VAE. Keep the
        # official DiffVAE in the five-asset identity above, but label the
        # runtime override diagnostic-only so it can never qualify as B0.
        "ltx25_video_vae_route": "conv_diagnostic",
        "ltx25_video_vae_runtime_ckpt": os.fspath(conv_video_vae),
        "gemma_attn_implementation": "sdpa",
        "gemma_max_memory": {"0": "16GiB", "1": "16GiB"},
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.output,
        build_config(args.model_root),
    )
    print(f"LTX-2.5 runtime config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
