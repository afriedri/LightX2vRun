#!/usr/bin/env python3
"""Build a MiniMax-H3 gfx1201 runtime JSON from the published config plus AMD overlay."""

from __future__ import annotations

import argparse
import copy
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
PRODUCTION_GEOMETRIES = {(768, 1344), (480, 832)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(
    publish: dict,
    overlay: dict,
    *,
    smoke: bool,
    task: str = "fl2av",
    infer_steps: int | None = None,
    step_update: str | None = None,
    reference_image_resize_mode: str | None = None,
    seq_parallel_size: int = 1,
    target_height: int | None = None,
    target_width: int | None = None,
    target_video_length: int | None = None,
    output_duration: int | None = None,
) -> dict:
    if task not in ("fl2av", "ref2av"):
        raise SystemExit(f"unsupported H3 task: {task}")
    ref2av = task == "ref2av"
    frames = 124 if target_video_length is None else target_video_length
    if not isinstance(frames, int) or not 124 <= frames <= 362 or frames % 17 != 5:
        raise SystemExit("H3 frames must be 17*n+5 within 124..362")
    if not ref2av and (frames != 124 or output_duration is not None):
        raise SystemExit("FL2AV Turbo timing remains 124 frames")
    if output_duration is not None:
        if not isinstance(output_duration, int) or not 5 <= output_duration <= 15:
            raise SystemExit("H3 output duration must be an integer from 5 to 15")
        if frames != ((output_duration * 24 - 5 + 16) // 17) * 17 + 5:
            raise SystemExit("H3 output duration does not match the aligned native frame count")
    allowed = ALLOWED_OVERLAY_CHANGES | ({"infer_steps", "lazy_load", "offload_granularity", "text_encoder_offload_granularity", "lora_dynamic_apply", "h3_step_update", "reference_image_resize_mode", "target_height", "target_width"} if ref2av else set())
    if seq_parallel_size not in (1, 2, 4, 8, 16):
        raise SystemExit(f"MiniMax-H3 gfx1201 supports seq_parallel_size 1, 2, 4, 8, or 16, got {seq_parallel_size}")
    if seq_parallel_size == 16 and smoke:
        raise SystemExit("MiniMax-H3 CP2xU8 SP16 requires a supported production geometry")
    if (target_height is None) != (target_width is None):
        raise SystemExit("target_height and target_width must be provided together")
    missing = [key for key in REQUIRED_PUBLISH_KEYS if key not in publish and not (ref2av and key in ("lora_dynamic_apply", "h3_step_update"))]
    if missing:
        raise SystemExit(f"published H3 JSON is missing keys: {missing}")
    unauthorized = {
        key: {"published": publish.get(key, "<missing>"), "overlay": value}
        for key, value in overlay.items()
        if publish.get(key, "<missing>") != value and key not in allowed and key not in TEST_WORKLOAD
    }
    if unauthorized:
        raise SystemExit(f"AMD overlay changes non-whitelisted published semantics: {unauthorized}")
    config = copy.deepcopy(publish)
    config.update(copy.deepcopy(overlay))
    if step_update is not None or reference_image_resize_mode is not None:
        if not ref2av:
            raise SystemExit("Sampling/reference overrides apply only to Ref2AV")
        if step_update is not None:
            config["h3_step_update"] = step_update
        if reference_image_resize_mode is not None:
            config["reference_image_resize_mode"] = reference_image_resize_mode
    # Keep legacy FL2AV and unspecified ref workloads at 124f; Ref2AV can
    # explicitly request the native 5-15 second range.
    config.update(TEST_WORKLOAD)
    config["target_video_length"] = frames
    config.pop("h3_output_duration", None)
    if output_duration is not None:
        config["h3_output_duration"] = output_duration
    if smoke:
        config.update(SMOKE)
    else:
        geometry = (
            int(config["target_height"]) if target_height is None else target_height,
            int(config["target_width"]) if target_width is None else target_width,
        )
        if geometry not in PRODUCTION_GEOMETRIES:
            supported = ", ".join(f"{width}x{height}" for height, width in sorted(PRODUCTION_GEOMETRIES))
            raise SystemExit(f"unsupported MiniMax-H3 production geometry {geometry[1]}x{geometry[0]}; expected {supported}")
        config.update({"target_height": geometry[0], "target_width": geometry[1]})
    frozen = {**FROZEN_VALUES, "target_video_length": frames}
    if ref2av:
        frozen.update(video_flow_shift=12.0)
        frozen.pop("h3_step_update")
        if config.get("h3_step_update") not in ("comfy_res_multistep", "reference_blend", "training_euler"):
            raise SystemExit("Unsupported H3 Ref2AV sampling mode")
        if config.get("reference_image_resize_mode", "diffusers") not in ("match", "max", "diffusers"):
            raise SystemExit("Unsupported H3 reference image resize mode")
        frozen.pop("infer_steps")
        if infer_steps is not None:
            config["infer_steps"] = infer_steps
        minimum = 1 if config["h3_step_update"] == "comfy_res_multistep" else 2
        if type(config.get("infer_steps")) is not int or config["infer_steps"] < minimum:
            raise SystemExit(f"H3 infer_steps must be an integer >= {minimum}")
        if config["h3_step_update"] == "comfy_res_multistep" and config["infer_steps"] > 1000:
            raise SystemExit("H3 Comfy simple sampling requires 1..1000 model evaluations")
    elif infer_steps is not None:
        raise SystemExit("FL2AV Turbo uses its existing fixed step configuration")
    for key, expected in frozen.items():
        if config.get(key) != expected:
            raise SystemExit(f"{key} must be {expected!r}, got {config.get(key)!r}")
    if ref2av:
        if config.get("lora_configs") or config.get("lora_dynamic_apply", False):
            raise SystemExit("Ref2AV BF16 base profile does not load LoRA")
        config.pop("lora_configs", None)
        config["lora_dynamic_apply"] = False
    else:
        lora = config.get("lora_configs") or []
        if len(lora) != 1 or lora[0].get("alpha") != 128 or lora[0].get("strength") != 1.0:
            raise SystemExit("H3 Turbo LoRA must be one entry with alpha=128 strength=1.0")
        path = str(lora[0].get("path") or "")
        if not path.endswith(TURBO_NAME):
            raise SystemExit(f"H3 Turbo LoRA path must end with {TURBO_NAME}, got {path}")
        lora[0]["path"] = f"/models/minimax_h3_turbo/{TURBO_NAME}"
        config["lora_dynamic_apply"] = True
    config["task"] = task
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
        parallel = {
            "seq_p_size": seq_parallel_size,
            "seq_p_attn_type": "ulysses_ring" if seq_parallel_size == 16 else "ulysses",
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
            "seq_p_tensor_fusion": False,
            "seq_p_head_parallel": False,
        }
        if seq_parallel_size == 16:
            parallel.update(
                {
                    "context_p_size": 2,
                    "ulysses_p_size": 8,
                }
            )
        config["parallel"] = parallel
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
    parser.add_argument("--task", choices=("fl2av", "ref2av"), default="fl2av")
    parser.add_argument("--infer-steps", type=int)
    parser.add_argument("--step-update", choices=("comfy_res_multistep", "reference_blend", "training_euler"))
    parser.add_argument("--reference-image-resize-mode", choices=("match", "max", "diffusers"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seq-parallel-size", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-video-length", type=int)
    parser.add_argument("--output-duration", type=int)
    args = parser.parse_args()
    config = build(
        load_json(args.publish_json),
        load_json(args.overlay_json),
        smoke=args.smoke,
        task=args.task,
        infer_steps=args.infer_steps,
        step_update=args.step_update,
        reference_image_resize_mode=args.reference_image_resize_mode,
        seq_parallel_size=args.seq_parallel_size,
        target_height=args.target_height,
        target_width=args.target_width,
        target_video_length=args.target_video_length,
        output_duration=args.output_duration,
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
