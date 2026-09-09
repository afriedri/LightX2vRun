from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = REPO_ROOT / "scripts/lib/preflight_minimax_h3_gfx1201.py"
SPEC = importlib.util.spec_from_file_location("preflight_minimax_h3_gfx1201", PREFLIGHT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_h1_patch_embed_contract_separates_frontend_from_aten_operator():
    patch_embed = SimpleNamespace(
        _lightx2v_patch_embed_backend="torch_linear",
        _lightx2v_collector_frontend_api="torch.nn.functional.linear",
        _lightx2v_collector_aten_operator="addmm",
    )

    assert PREFLIGHT.check_h1_patch_embed_runtime(patch_embed) == {
        "name": "torch_linear",
        "frontend_api": "torch.nn.functional.linear",
        "aten_operator": "addmm",
        "sequence_sharded": False,
    }


def test_h1_patch_embed_metadata_is_optional_without_collection(capsys):
    contract = PREFLIGHT.check_h1_patch_embed_runtime(SimpleNamespace())
    assert contract["name"] is None
    assert "WARNING:" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        PREFLIGHT.check_h1_patch_embed_runtime(SimpleNamespace(), required=True)


def test_h3_t1_runcollector_contract_accepts_schema2():
    assert PREFLIGHT.validate_runcollector_contract(
        "lightx2v.minimax_h3.runcollector",
        2,
        "h3_t1_lifecycle",
        2,
    ) == {
        "required": True,
        "schema_name": "lightx2v.minimax_h3.runcollector",
        "schema_version": 2,
        "mode": "h3_t1_lifecycle",
    }


@pytest.mark.parametrize(
    ("schema_name", "schema_version", "mode"),
    (
        ("lightx2v.minimax_h3.runcollector", 1, "h3_t1_lifecycle"),
        ("lightx2v.minimax_h3.runcollector", 2, "legacy"),
        ("wrong", 2, "h3_t1_lifecycle"),
    ),
)
def test_h3_t1_runcollector_contract_rejects_old_or_wrong_images(schema_name, schema_version, mode):
    with pytest.raises(SystemExit):
        PREFLIGHT.validate_runcollector_contract(schema_name, schema_version, mode, 2)


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
    ((1, ((0, 56),)), (2, ()), (4, ()), (8, ()), (16, ())),
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


def test_artifact_contract_uses_u8_head_divisibility_for_sp16(tmp_path):
    artifacts = PREFLIGHT.check_artifacts(*make_artifacts(tmp_path), expected_world_size=16)
    assert artifacts["attention_heads"] == 56


def test_artifact_contract_rejects_incomplete_processor(tmp_path):
    model_dir, turbo_dir, first_frame, last_frame = make_artifacts(tmp_path)
    (model_dir / "processor/video_preprocessor_config.json").unlink()
    with pytest.raises(SystemExit):
        PREFLIGHT.check_artifacts(model_dir, turbo_dir, first_frame, last_frame, expected_world_size=2)


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_runtime_contract_accepts_only_pure_ulysses(tmp_path, world_size):
    config = {
        "video_flow_shift": 6.0,
        "h3_step_update": "training_euler",
        "h3_refiner_attn_type": "aiter_triton_mha",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "attn_type": "aiter_triton_mha",
        "qwen3vl_attn_type": "aiter_triton_mha",
        "h3_core_qk_rms_type": "minimax_h3_aiter_triton_qk",
        "h3_core_rope_type": "minimax_h3_triton_strict_rope",
        "h4_video_attn_type": "aiter_triton_mha",
        "dit_quant_scheme": "Default",
        "lora_dynamic_apply": True,
        "lazy_load": True,
        "cpu_offload": True,
        "offload_granularity": "block",
        "text_encoder_rank0_only": True,
        "text_encoder_unload_after_infer": True,
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


@pytest.mark.parametrize(("target_width", "target_height"), ((1344, 768), (832, 480)))
def test_runtime_contract_accepts_cp2_ulysses8_lazy_rank0_encoder(
    tmp_path,
    target_width,
    target_height,
):
    config = {
        "video_flow_shift": 6.0,
        "h3_step_update": "training_euler",
        "h3_refiner_attn_type": "aiter_triton_mha",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "attn_type": "aiter_triton_mha",
        "qwen3vl_attn_type": "aiter_triton_mha",
        "h3_core_qk_rms_type": "minimax_h3_aiter_triton_qk",
        "h3_core_rope_type": "minimax_h3_triton_strict_rope",
        "h4_video_attn_type": "aiter_triton_mha",
        "dit_quant_scheme": "Default",
        "lora_dynamic_apply": True,
        "lazy_load": True,
        "cpu_offload": True,
        "offload_granularity": "block",
        "text_encoder_rank0_only": True,
        "text_encoder_unload_after_infer": True,
        "warmup": False,
        "enable_cfg": False,
        "target_video_length": 124,
        "target_width": target_width,
        "target_height": target_height,
        "fps": 24,
        "tensor_parallel": False,
        "text_encoder_tensor_parallel": False,
        "seq_parallel": True,
        "vae_decode_parallel": True,
        "parallel": {
            "seq_p_size": 16,
            "seq_p_attn_type": "ulysses_ring",
            "context_p_size": 2,
            "ulysses_p_size": 8,
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
        },
    }
    path = tmp_path / "runtime-cp2-u8-sp16.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert PREFLIGHT.check_runtime_config(path, 16)["parallel"]["seq_p_attn_type"] == "ulysses_ring"

    config["h3_ulysses_minimize_aux_prefix"] = False
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_config(path, 16)


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
def test_runtime_contract_rejects_unsupported_ulysses_options(tmp_path, key, value):
    config = {
        "enable_cfg": False,
        "video_flow_shift": 6.0,
        "h3_step_update": "training_euler",
        "h3_refiner_attn_type": "aiter_triton_mha",
        "h3_core_attn_type": "aiter_fav3_sage_bf16_attn",
        "attn_type": "aiter_triton_mha",
        "qwen3vl_attn_type": "aiter_triton_mha",
        "h3_core_qk_rms_type": "minimax_h3_aiter_triton_qk",
        "h3_core_rope_type": "minimax_h3_triton_strict_rope",
        "h4_video_attn_type": "aiter_triton_mha",
        "dit_quant_scheme": "Default",
        "lora_dynamic_apply": True,
        "lazy_load": True,
        "cpu_offload": True,
        "offload_granularity": "block",
        "text_encoder_rank0_only": True,
        "text_encoder_unload_after_infer": True,
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


@pytest.mark.parametrize("rocm", ("7.2.0", "7.3.0"))
def test_rocm_version_difference_is_diagnostic(monkeypatch, capsys, rocm):
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=rocm), __version__="test",
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_properties=lambda _: SimpleNamespace(
                gcnArchName="gfx1201", name="test", multi_processor_count=48),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("GPU_ARCHS", "gfx1201")
    monkeypatch.setenv("CU_NUM", "24")
    assert PREFLIGHT.require_gfx1201(1)["rocm"] == rocm
    assert ("WARNING:" in capsys.readouterr().err) == (rocm != "7.2.0")
    # A ROCm version difference is allowed; a wrong world size still fails.
    with pytest.raises(SystemExit):
        PREFLIGHT.require_gfx1201(2)


def test_dependency_gate_only_requires_ring_for_sp16(monkeypatch, tmp_path, capsys):
    from types import ModuleType

    def module(name, **values):
        parent, _, leaf = name.rpartition(".")
        if parent and parent not in sys.modules:
            module(parent)
        value = ModuleType(name)
        value.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, value)
        if parent:
            monkeypatch.setattr(sys.modules[parent], leaf, value, raising=False)
        return value

    module("transformers", __version__="test", Qwen2TokenizerFast=object, Qwen3VLProcessor=object)
    module("aiter", __file__=str(tmp_path / "aiter/__init__.py"), flash_attn_varlen_func=lambda: None)
    module("aiter.ops.triton.attention.fav3_sage", fav3_sage_wrapper_func=lambda: None)
    module("aiter.ops.flydsl.sage_attention", sage_attention_v2_gfx1201_support_reason=lambda: None)
    module("aiter.ops.triton.normalization.rmsnorm", rms_norm=lambda: None)
    backend_logger = module("lightx2v_platform.base.backend_logger", BackendLogger=lambda name: SimpleNamespace(fallback_is_disabled=lambda: True))
    module("lightx2v", __file__=str(tmp_path / "lightx2v/__init__.py"))
    module("lightx2v.infer")
    module("lightx2v_platform.ops.attn.amd_rocm")
    module("lightx2v.models.input_encoders.hf.minimax_h3.qwen3vl_vision", _PatchEmbed=SimpleNamespace())
    module("lightx2v_platform.ops.attn.amd_rocm.triton_mha_attn", dense_flash_attn_func=lambda: None)
    attention = {name: object for name in ("aiter_triton_mha", "aiter_fav3_sage_bf16_attn", "ulysses", "torch_sdpa")}
    module("lightx2v_platform.registry_factory", PLATFORM_ATTN_WEIGHT_REGISTER=attention)
    module("lightx2v.utils.registry_factory",
           ATTN_WEIGHT_REGISTER=attention, RUNNER_REGISTER={"minimax_h3": object},
           RMS_WEIGHT_REGISTER={"torch_native": object, "minimax_h3_aiter_triton_qk": object},
           ROPE_REGISTER={"torch_real_rope": object, "minimax_h3_triton_strict_rope": object})
    monkeypatch.setattr(PREFLIGHT.shutil, "which", lambda name: "/bin/" + name)
    monkeypatch.setattr(PREFLIGHT, "git_revision", lambda path: None)
    monkeypatch.setattr(PREFLIGHT, "git_dirty", lambda path: None)
    monkeypatch.setattr(PREFLIGHT, "check_dense_gemm_runtime", lambda: {})
    monkeypatch.delenv("LIGHTX2V_SOURCE_REVISION", raising=False)
    for count in (1, 2, 4, 8):
        result = PREFLIGHT.check_runtime_dependencies("image", expected_world_size=count)
        assert result["lightx2v_revision"] is None
    assert "provenance is missing" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_dependencies("image", expected_world_size=16)
    attention["aiter_fav3_sage_bf16_attn"] = SimpleNamespace(apply_with_lse=lambda: None)
    attention.update(ring=object, ulysses_ring=object)
    assert PREFLIGHT.check_runtime_dependencies("image", expected_world_size=16)["source_mode"] == "image"
    monkeypatch.setenv("LIGHTX2V_NO_FALLBACK_MODELS", "minimax_h3")
    assert PREFLIGHT.check_runtime_dependencies("image", expected_world_size=2)["source_mode"] == "image"
    backend_logger.BackendLogger = lambda name: SimpleNamespace()
    with pytest.raises(SystemExit):
        PREFLIGHT.check_runtime_dependencies("image", expected_world_size=2)
