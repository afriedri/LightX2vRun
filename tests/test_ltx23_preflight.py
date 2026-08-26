import importlib.util
import json
import struct
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/preflight_ltx23_fp8_gfx1201.py"
SPEC = importlib.util.spec_from_file_location("preflight_ltx23_fp8_gfx1201", SCRIPT)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREFLIGHT)


def write_checkpoint_header(path: Path, *, missing_scale: bool = False) -> None:
    header = {
        "__metadata__": {
            "model_version": "2.3.0",
            "config": json.dumps({"transformer": {"num_layers": 48}}),
        }
    }
    for block in range(48):
        fp8 = 4 <= block <= 46
        for index in range(34):
            layer = f"model.diffusion_model.transformer_blocks.{block}.linear_{index:02d}"
            header[f"{layer}.weight"] = {
                "dtype": "F8_E4M3" if fp8 else "BF16",
                "shape": [8, 8],
                "data_offsets": [0, 0],
            }
            if fp8:
                header[f"{layer}.input_scale"] = {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 0],
                }
                if not (missing_scale and block == 4 and index == 0):
                    header[f"{layer}.weight_scale"] = {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 0],
                    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def runtime_config() -> dict:
    return {
        "infer_steps": 8,
        "target_video_length": 121,
        "target_height": 704,
        "target_width": 1280,
        "fps": 24,
        "sample_guide_scale": 1.0,
        "sample_shift": [2.05, 0.95],
        "num_channels_latents": 128,
        "vae_scale_factors": [8, 32, 32],
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_scale_factor": 4,
        "audio_fps": 24000,
        "audio_mel_bins": 16,
        "caption_proj_before_connector": True,
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
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
        "attn_type": "torch_sdpa",
        "rms_norm_type": "torch_native",
        "rope_type": "torch_real_rope",
        "layer_norm_type": "torch",
        "modulate_type": "torch",
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "lazy_load": False,
        "unload_modules": True,
        "gemma_cpu_offload": True,
        "vae_cpu_offload": True,
        "use_tiling_vae": True,
        "warmup": False,
        "dit_quantized": True,
        "dit_quant_scheme": "fp8-pertensor",
        "dit_quantized_ckpt": "/models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors",
        "gemma_original_ckpt": "/models/ltx23/gemma",
        "skip_fp8_block_index": [0, 1, 2, 3, 47],
        "use_upsampler": True,
        "upsampler_original_ckpt": "/models/ltx23/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx_checkpoint_contract_required": True,
        "ltx23_b0_gfx1201": True,
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
    }


def test_header_contract_discovers_exact_mixed_layout(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    write_checkpoint_header(checkpoint)
    layout = PREFLIGHT.inspect_checkpoint_header(checkpoint)
    assert layout["fp8_blocks"] == list(range(4, 47))
    assert layout["bf16_blocks"] == [0, 1, 2, 3, 47]
    assert layout["fp8_linears_per_block"] == 34
    assert layout["fp8_linear_count"] == 1462
    assert layout["scale_contract"] == "F32 scalar input_scale + weight_scale"


def test_header_contract_rejects_missing_scale(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    write_checkpoint_header(checkpoint, missing_scale=True)
    with pytest.raises(SystemExit):
        PREFLIGHT.inspect_checkpoint_header(checkpoint)


def test_artifact_contract_accepts_the_downloaded_five_shard_layout(tmp_path):
    upscaler = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    encoded = json.dumps(
        {
            "weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 0],
            }
        },
        separators=(",", ":"),
    ).encode()
    upscaler.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    gemma = tmp_path / "gemma"
    gemma.mkdir()
    shard_names = [
        f"model-{index:05d}-of-00005.safetensors"
        for index in range(1, 6)
    ]
    for name in shard_names:
        (gemma / name).write_bytes(b"test")
    (gemma / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"layer.{index}": name
                    for index, name in enumerate(shard_names)
                }
            }
        ),
        encoding="utf-8",
    )
    for name, payload in (
        ("config.json", {"model_type": "gemma3"}),
        ("tokenizer_config.json", {}),
        ("preprocessor_config.json", {}),
    ):
        (gemma / name).write_text(json.dumps(payload), encoding="utf-8")
    (gemma / "tokenizer.model").write_bytes(b"tokenizer")
    artifacts = PREFLIGHT.check_artifacts(upscaler, gemma)
    assert len(artifacts["gemma"]["model_safetensors"]) == 5


def test_exact_local_video_token_contract():
    assert PREFLIGHT.expected_local_video_tokens() == {
        "stage1": 1760,
        "stage2": 7040,
    }


def test_scaled_mm_probe_uses_the_production_tn_layout_contract():
    assert PREFLIGHT.scaled_mm_layout_contract(1760, 4096, 4096) == {
        "lhs_shape": [1760, 4096],
        "lhs_stride": [4096, 1],
        "rhs_shape": [4096, 4096],
        "rhs_stride": [1, 4096],
    }
    with pytest.raises(ValueError, match="positive"):
        PREFLIGHT.scaled_mm_layout_contract(0, 4096, 4096)


def test_runtime_config_accepts_only_frozen_b0(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(runtime_config()), encoding="utf-8")
    assert PREFLIGHT.check_runtime_config(path)["target_height"] == 704
    broken = runtime_config()
    broken["target_width"] = 2560
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_config(path)
