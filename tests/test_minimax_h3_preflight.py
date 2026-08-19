from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "scripts/lib/preflight_minimax_h3_day0.py"
SPEC = importlib.util.spec_from_file_location("preflight_minimax_h3_day0", PREFLIGHT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_h1_patch_embed_contract_separates_frontend_from_aten_operator():
    patch_embed = SimpleNamespace(
        _lightx2v_patch_embed_backend="torch_linear",
        _lightx2v_runcollector_frontend_api="torch.nn.functional.linear",
        _lightx2v_runcollector_aten_operator="addmm",
    )

    assert PREFLIGHT.check_h1_patch_embed_runtime(patch_embed) == {
        "name": "torch_linear",
        "frontend_api": "torch.nn.functional.linear",
        "aten_operator": "addmm",
        "sequence_sharded": False,
    }


def test_h1_patch_embed_contract_rejects_an_old_image():
    with pytest.raises(SystemExit):
        PREFLIGHT.check_h1_patch_embed_runtime(SimpleNamespace())


def test_dense_gemm_contract_requires_direct_torch_mm_without_tunableop(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(tunable=SimpleNamespace(is_enabled=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.delenv("TORCH_BLAS_PREFER_CUBLASLT", raising=False)
    monkeypatch.delenv("TORCH_BLAS_PREFER_HIPBLASLT", raising=False)

    contract = PREFLIGHT.check_dense_gemm_runtime()

    assert contract == {
        "measured_backend": "hipBLASLt",
        "route": "torch.mm -> default ROCm BLAS heuristic",
        "tunableop_cache": False,
        "tunableop_enabled": False,
    }


def test_dense_gemm_contract_rejects_tunableop_or_forced_blas(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(tunable=SimpleNamespace(is_enabled=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "1")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_dense_gemm_runtime()

    monkeypatch.setenv("PYTORCH_TUNABLEOP_ENABLED", "0")
    monkeypatch.setenv("TORCH_BLAS_PREFER_HIPBLASLT", "1")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_dense_gemm_runtime()


def test_distributed_worker_selects_local_device_before_backend_import(monkeypatch):
    selected = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=selected.append)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LOCAL_RANK", "3")

    assert PREFLIGHT.set_distributed_local_device() == 3
    assert selected == [3]


@pytest.mark.parametrize(
    ("world_size", "expected"),
    ((1, ((0, 56),)), (2, ()), (4, ())),
)
def test_parent_sage_smoke_is_only_used_for_single_gpu(world_size, expected):
    assert PREFLIGHT.standalone_sage_smoke_plan(world_size) == expected


def make_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model_dir = tmp_path / "MiniMax-H3"
    required = {
        "transformer": ("config.json", "model.safetensors"),
        "text_encoder": ("config.json", "model.safetensors"),
        "tokenizer": ("tokenizer.json", "tokenizer_config.json"),
        "vae": ("config.json", "model.safetensors"),
        "audio_vae": ("config.json", "model.safetensors"),
        "processor": (
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    }
    for component, filenames in required.items():
        component_dir = model_dir / component
        component_dir.mkdir(parents=True)
        for filename in filenames:
            (component_dir / filename).write_text("{}\n", encoding="utf-8")

    turbo_dir = tmp_path / "Minimax-h3-Turbo"
    turbo_dir.mkdir()
    (turbo_dir / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors").write_bytes(b"stub")
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    Image.new("RGB", (32, 32)).save(first_frame)
    Image.new("RGB", (32, 32)).save(last_frame)
    return model_dir, turbo_dir, first_frame, last_frame


def test_artifact_contract_accepts_published_component_layout(tmp_path):
    artifacts = PREFLIGHT.check_artifacts(*make_artifacts(tmp_path), expected_world_size=2)
    assert artifacts["components"] == ["audio_vae", "processor", "text_encoder", "tokenizer", "transformer", "vae"]
    assert artifacts["turbo"].endswith("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")


def test_artifact_contract_rejects_incomplete_processor(tmp_path):
    model_dir, turbo_dir, first_frame, last_frame = make_artifacts(tmp_path)
    (model_dir / "processor/video_preprocessor_config.json").unlink()
    with pytest.raises(SystemExit):
        PREFLIGHT.check_artifacts(model_dir, turbo_dir, first_frame, last_frame, expected_world_size=2)


@pytest.mark.parametrize("world_size", (2, 4))
def test_runtime_contract_accepts_only_pure_ulysses(tmp_path, world_size):
    config = {
        "video_flow_shift": 6.0,
        "h3_step_update": "training_euler",
        "h3_refiner_attn_type": "aiter_triton_bf16_flash_attn",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "attn_type": "aiter_triton_bf16_flash_attn",
        "qwen3vl_attn_type": "aiter_triton_bf16_flash_attn",
        "dit_quant_scheme": "Default",
        "lora_dynamic_apply": True,
        "warmup": False,
        "enable_cfg": False,
        "tensor_parallel": False,
        "text_encoder_tensor_parallel": False,
        "seq_parallel": True,
        "vae_decode_parallel": True,
        "parallel": {
            "seq_p_size": world_size,
            "seq_p_attn_type": "ulysses",
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert PREFLIGHT.check_runtime_config(path, world_size)["parallel"]["seq_p_attn_type"] == "ulysses"

    config["tensor_parallel"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_config(path, world_size)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("seq_p_quant_scheme", "fp8"),
        ("seq_p_fp8_comm", True),
        ("seq_p_fp4_comm", True),
        ("seq_p_tensor_fusion", True),
        ("seq_p_head_parallel", True),
    ),
)
def test_runtime_contract_rejects_non_day0_ulysses_options(tmp_path, key, value):
    config = {
        "enable_cfg": False,
        "video_flow_shift": 6.0,
        "h3_step_update": "training_euler",
        "h3_refiner_attn_type": "aiter_triton_bf16_flash_attn",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "attn_type": "aiter_triton_bf16_flash_attn",
        "qwen3vl_attn_type": "aiter_triton_bf16_flash_attn",
        "dit_quant_scheme": "Default",
        "lora_dynamic_apply": True,
        "warmup": False,
        "tensor_parallel": False,
        "text_encoder_tensor_parallel": False,
        "seq_parallel": True,
        "vae_decode_parallel": True,
        "parallel": {
            "seq_p_size": 2,
            "seq_p_attn_type": "ulysses",
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
            key: value,
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_config(path, 2)
