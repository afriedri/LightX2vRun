from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts/lib/build_minimax_h3_runtime_config.py"
SPEC = importlib.util.spec_from_file_location("build_minimax_h3_runtime_config", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def published_config() -> dict:
    return {
        "infer_steps": 5,
        "target_video_length": 124,
        "target_height": 768,
        "target_width": 1344,
        "fps": 24,
        "target_fps": 24,
        "video_flow_shift": 6.0,
        "audio_flow_shift": 3.0,
        "h3_step_update": "training_euler",
        "lora_dynamic_apply": True,
        "cpu_offload": True,
        "offload_granularity": "block",
        "vae_cpu_offload": True,
        "keep_latents_dtype_in_scheduler": True,
        "attn_type": "flash_attn3",
        "rms_type": "torch_native",
        "rope_type": "torch_real_rope",
        "lora_configs": [
            {
                "path": "/published/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
                "strength": 1.0,
                "alpha": 128,
            }
        ],
    }


def amd_overlay() -> dict:
    return {
        "task": "fl2av",
        "attn_type": "aiter_triton_mha",
        "h3_refiner_attn_type": "aiter_triton_mha",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "qwen3vl_attn_type": "aiter_triton_mha",
        "dit_quant_scheme": "Default",
        "rms_type": "torch_native",
        "rope_type": "torch_real_rope",
        "h3_core_qk_rms_type": "minimax_h3_aiter_triton_qk",
        "h3_core_rope_type": "minimax_h3_triton_strict_rope",
        "h4_video_attn_type": "aiter_triton_mha",
        "warmup": False,
    }


def test_builder_freezes_full_and_smoke_geometry():
    full = BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=1)
    smoke = BUILDER.build(published_config(), amd_overlay(), smoke=True, seq_parallel_size=1)
    assert (full["target_height"], full["target_width"], full["target_video_length"]) == (768, 1344, 124)
    assert (smoke["target_height"], smoke["target_width"], smoke["target_video_length"]) == (480, 480, 124)
    assert full["lora_configs"][0]["path"].startswith("/models/minimax_h3_turbo/")
    assert full["parallel"] is False
    assert full["tensor_parallel"] is False
    assert full["text_encoder_tensor_parallel"] is False
    assert full["seq_parallel"] is False
    assert full["vae_decode_parallel"] is False
    assert full["h3_core_attn_type"] == "aiter_fav3_sage_bf16_attn"
    assert full["h3_refiner_attn_type"] == "aiter_triton_mha"
    assert full["qwen3vl_attn_type"] == "aiter_triton_mha"
    assert full["h3_core_qk_rms_type"] == "minimax_h3_aiter_triton_qk"
    assert full["h3_core_rope_type"] == "minimax_h3_triton_strict_rope"
    assert full["h4_video_attn_type"] == "aiter_triton_mha"
    assert full["dit_quant_scheme"] == "Default"
    assert full["lora_dynamic_apply"] is True
    assert full["lazy_load"] is True
    assert full["cpu_offload"] is True
    assert full["offload_granularity"] == "block"
    assert full["text_encoder_rank0_only"] is False
    assert full["text_encoder_unload_after_infer"] is True


def test_builder_enables_pure_ulysses_sp2_and_parallel_vae_decode():
    config = BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=2)

    assert config["tensor_parallel"] is False
    assert config["text_encoder_tensor_parallel"] is False
    assert config["seq_parallel"] is True
    assert config["vae_decode_parallel"] is True
    assert config["text_encoder_rank0_only"] is True
    assert config["parallel"] == {
        "seq_p_size": 2,
        "seq_p_attn_type": "ulysses",
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
    }


def test_builder_enables_pure_ulysses_sp4_and_parallel_vae_decode():
    config = BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=4)

    assert config["tensor_parallel"] is False
    assert config["text_encoder_tensor_parallel"] is False
    assert config["seq_parallel"] is True
    assert config["vae_decode_parallel"] is True
    assert config["text_encoder_rank0_only"] is True
    assert config["parallel"] == {
        "seq_p_size": 4,
        "seq_p_attn_type": "ulysses",
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
    }


def test_builder_enables_pure_ulysses_sp8_and_parallel_vae_decode():
    config = BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=8)

    assert config["tensor_parallel"] is False
    assert config["text_encoder_tensor_parallel"] is False
    assert config["seq_parallel"] is True
    assert config["vae_decode_parallel"] is True
    assert config["text_encoder_rank0_only"] is True
    assert config["parallel"] == {
        "seq_p_size": 8,
        "seq_p_attn_type": "ulysses",
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
    }


def test_builder_enables_cp2_ulysses8_for_sp16():
    config = BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=16)

    assert config["tensor_parallel"] is False
    assert config["text_encoder_tensor_parallel"] is False
    assert config["text_encoder_rank0_only"] is True
    assert config["text_encoder_unload_after_infer"] is True
    assert config["lazy_load"] is True
    assert config["seq_parallel"] is True
    assert config["vae_decode_parallel"] is True
    assert config["parallel"] == {
        "seq_p_size": 16,
        "seq_p_attn_type": "ulysses_ring",
        "seq_p_prepost_backend": "torch",
        "seq_p_a2a_backend": "torch",
        "seq_p_tensor_fusion": False,
        "seq_p_head_parallel": False,
        "context_p_size": 2,
        "ulysses_p_size": 8,
    }


def test_builder_rejects_sp16_smoke_geometry():
    with pytest.raises(SystemExit, match="full 124f 1344x768 workload"):
        BUILDER.build(published_config(), amd_overlay(), smoke=True, seq_parallel_size=16)


def test_builder_rejects_unsupported_sequence_parallel_size():
    with pytest.raises(SystemExit, match="seq_parallel_size 1, 2, 4, 8, or 16"):
        BUILDER.build(published_config(), amd_overlay(), smoke=False, seq_parallel_size=3)


def test_builder_rejects_non_whitelisted_semantic_change():
    overlay = amd_overlay()
    overlay["audio_flow_shift"] = 99.0
    with pytest.raises(SystemExit, match="non-whitelisted"):
        BUILDER.build(published_config(), overlay, smoke=False, seq_parallel_size=1)


def test_builder_overrides_older_image_workload_to_124_frames():
    publish = published_config()
    publish["target_video_length"] = 362
    config = BUILDER.build(publish, amd_overlay(), smoke=False, seq_parallel_size=1)
    assert config["target_video_length"] == 124
