#!/usr/bin/env python3
"""Build a MiniMax-H3 Day0 runtime JSON from the published config plus AMD overlay."""

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
    "video_flow_shift": 6.0,
    "h3_step_update": "training_euler",
    "infer_steps": 5,
    "warmup": False,
    "h3_refiner_attn_type": "aiter_triton_bf16_flash_attn",
    "h3_core_attn_type": "aiter_triton_bf16_flash_attn",
    "attn_type": "aiter_triton_bf16_flash_attn",
    "qwen3vl_attn_type": "torch_sdpa",
    "dit_quant_scheme": "Default",
    "rms_type": "torch_native",
    "rope_type": "torch_real_rope",
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
    "warmup",
    "tensor_parallel",
    "seq_parallel",
    "vae_decode_parallel",
    "local_files_only",
    "lora_configs",
}
TURBO_NAME = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
SMOKE = {"target_height": 480, "target_width": 480, "target_video_length": 124, "fps": 24, "target_fps": 24}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(publish: dict, overlay: dict, *, smoke: bool) -> dict:
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
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-json", type=Path, required=True)
    parser.add_argument("--overlay-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = build(load_json(args.publish_json), load_json(args.overlay_json), smoke=args.smoke)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"H3 runtime config: {args.output_json}")
    print(
        f"geometry={config['target_height']}x{config['target_width']} "
        f"frames={config['target_video_length']} attn={config['h3_core_attn_type']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
