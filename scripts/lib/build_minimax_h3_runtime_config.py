#!/usr/bin/env python3
"""Build a MiniMax-H3 gfx1201 runtime JSON from the published config plus AMD overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PUBLISH_KEYS = (
    "infer_steps",
    "target_video_length",
    "target_height",
    "target_width",
    "fps",
    "video_flow_shift",
    "audio_flow_shift",
    "h3_step_update",
    "lora_dynamic_apply",
    "cpu_offload",
    "offload_granularity",
    "vae_cpu_offload",
    "keep_latents_dtype_in_scheduler",
)
FROZEN_VALUES = {
    "target_video_length": 124,
    "fps": 24,
    "target_fps": 24,
    "video_flow_shift": 6.0,
    "h3_step_update": "training_euler",
    "infer_steps": 5,
    "warmup": False,
    "h3_refiner_attn_type": "aiter_triton_mha",
    "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
    "attn_type": "aiter_triton_mha",
    "qwen3vl_attn_type": "aiter_triton_mha",
    "dit_quant_scheme": "Default",
    "rms_type": "torch_native",
    "rope_type": "torch_real_rope",
    "h3_core_qk_rms_type": "minimax_h3_aiter_triton_qk",
    "h3_core_rope_type": "minimax_h3_triton_strict_rope",
    "h4_video_attn_type": "aiter_triton_mha",
}
ALLOWED_OVERLAY_CHANGES = {
    "task",
    "attn_type",
    "h3_refiner_attn_type",
    "h3_core_attn_type",
    "qwen3vl_attn_type",
    "dit_quant_scheme",
    "rms_type",
    "rope_type",
    "h3_core_qk_rms_type",
    "h3_core_rope_type",
    "h4_video_attn_type",
    "warmup",
    "tensor_parallel",
    "seq_parallel",
    "vae_decode_parallel",
    "local_files_only",
    "lora_configs",
}
TURBO_NAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
TEST_WORKLOAD = {"target_video_length": 124, "fps": 24, "target_fps": 24}
SMOKE = {"target_height": 480, "target_width": 480}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(publish: dict, overlay: dict, *, smoke: bool, seq_parallel_size: int = 1) -> dict:
    if seq_parallel_size not in (1, 2, 4, 8, 16):
        raise SystemExit(f"MiniMax-H3 gfx1201 supports seq_parallel_size 1, 2, 4, 8, or 16, got {seq_parallel_size}")
    missing = [key for key in REQUIRED_PUBLISH_KEYS if key not in publish]
    if missing:
        raise SystemExit(f"published H3 JSON is missing keys: {missing}")
    unauthorized = {
        key: {"published": publish.get(key, "<missing>"), "overlay": value}
        for key, value in overlay.items()
        if publish.get(key, "<missing>") != value and key not in ALLOWED_OVERLAY_CHANGES
    }
    if unauthorized:
        raise SystemExit(f"AMD overlay changes non-whitelisted published semantics: {unauthorized}")
    config = dict(publish)
    config.update(overlay)
    # Run policy: every H3 backend test uses the shortest valid 17*n+5 shape.
    # Override older images whose baked published config still requests 362f.
    config.update(TEST_WORKLOAD)
    if smoke:
        config.update(SMOKE)
    for key, expected in FROZEN_VALUES.items():
        if config.get(key) != expected:
            raise SystemExit(f"{key} must be {expected!r}, got {config.get(key)!r}")
    lora = config.get("lora_configs") or []
    if len(lora) != 1 or lora[0].get("alpha") != 128 or lora[0].get("strength") != 1.0:
        raise SystemExit("H3 Turbo LoRA must be one entry with alpha=128 strength=1.0")
    path = str(lora[0].get("path") or "")
    if not path.endswith(TURBO_NAME):
        raise SystemExit(f"H3 Turbo LoRA path must end with {TURBO_NAME}, got {path}")
    lora[0]["path"] = f"/models/minimax_h3_turbo/{TURBO_NAME}"
    config["task"] = "fl2av"
    config["lora_dynamic_apply"] = True
    config["tensor_parallel"] = False
    config["text_encoder_tensor_parallel"] = False
    config["lazy_load"] = True
    config["cpu_offload"] = True
    config["offload_granularity"] = "block"
    config["text_encoder_rank0_only"] = seq_parallel_size > 1
    config["text_encoder_unload_after_infer"] = True
    config["seq_parallel"] = seq_parallel_size > 1
    config["vae_decode_parallel"] = seq_parallel_size > 1
    if seq_parallel_size > 1:
        config["parallel"] = {
            "seq_p_size": seq_parallel_size,
            "seq_p_attn_type": "ring" if seq_parallel_size == 16 else "ulysses",
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
            "seq_p_tensor_fusion": False,
            "seq_p_head_parallel": False,
        }
    else:
        # A truthy parallel dictionary would make the single-process entry try
        # to initialize torch.distributed without torchrun rendezvous variables.
        config["parallel"] = False
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-json", type=Path, required=True)
    parser.add_argument("--overlay-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seq-parallel-size", type=int, choices=(1, 2, 4, 8, 16), default=1)
    args = parser.parse_args()
    config = build(
        load_json(args.publish_json),
        load_json(args.overlay_json),
        smoke=args.smoke,
        seq_parallel_size=args.seq_parallel_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"H3 runtime config: {args.output_json}")
    print(
        f"geometry={config['target_height']}x{config['target_width']} "
        f"frames={config['target_video_length']} attn={config['h3_core_attn_type']} "
        f"seq_parallel_size={args.seq_parallel_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
