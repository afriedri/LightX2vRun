"""Check Run's generated configuration against a real LightX2V source tree."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

RUN_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("LIGHTX2V_SOURCE", RUN_ROOT.parent / "LightX2V-ROCm-GFX1201/LightX2V"))


def load_builder(name):
    spec = importlib.util.spec_from_file_location(name, RUN_ROOT / f"scripts/lib/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source():
    if not (SOURCE / "lightx2v/infer.py").is_file():
        pytest.skip("Set LIGHTX2V_SOURCE to run cross-repository entry checks")
    return SOURCE


def model_choices(source):
    for node in ast.walk(ast.parse((source / "lightx2v/infer.py").read_text())):
        if isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "--model_cls" for a in node.args):
            values = next(kw.value for kw in node.keywords if kw.arg == "choices")
            return {item.value for item in values.elts if isinstance(item, ast.Constant)}
    raise AssertionError("LightX2V model CLI not found")


@pytest.mark.parametrize("name,model", [("wan21", "wan2.1"), ("wan22", "wan2.2_moe")])
@pytest.mark.parametrize("warmup", ["0", "1"])
def test_wan_new_entry_preserves_distillation_and_warmup(source, tmp_path, name, model, warmup):
    script = RUN_ROOT / f"scripts/lib/run_{name}_in_container.sh"
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, env={
        **os.environ, "LIGHTX2V_PATH": str(source), "HIP_VISIBLE_DEVICES": "0,1", "CONFIG_ONLY": "1",
        "AITER_JIT_DIR": str(tmp_path / "aiter"), "RUNTIME_CONFIG_DIR": str(tmp_path), "LIGHTX2V_WARMUP": warmup,
    })
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(result.stdout[result.stdout.index("{\n"):])
    assert config["model_cls"] == model and model in model_choices(source)
    assert config["distill_method"] == "dmd2"
    assert config["warmup"] == (warmup == "1")
    assert config["enable_cfg"] is False and config["infer_steps"] == 4
    assert f"--model_cls {model}\n" in script.read_text()
    assert "infer_args+=(--warmup)" not in script.read_text()
    # Execute the upstream factory's real selection logic with constructor sentinels.
    tree = ast.parse((source / "lightx2v/models/schedulers/wan/scheduler_factory.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    scope = {"logger": type("Log", (), {"info": lambda *a: None})(), "WanStepDistillScheduler": lambda cfg: ("DMD2", cfg)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "scheduler_factory.py", "exec"), scope)
    assert scope["create_wan_scheduler"](config)[0] == "DMD2"


def test_ltx_entries_keep_split_assets_and_use_generation_specific_runner(source, tmp_path):
    config = load_builder("build_ltx25_runtime_config").build_config(tmp_path)
    assert config["model_cls"] == "ltx2_5" and "ltx2_5" in model_choices(source)
    assert "duration_head_original_ckpt" not in config
    assert config["lazy_load"] and config["unload_modules"]
    assert config["ltx25_process_isolated_prompt_required"]
    assert config["ltx25_video_vae_runtime_ckpt"].endswith("ltx-2.5-video-vae-conv-bf16.safetensors")
    assert config["ltx_video_vae_ckpt"].endswith("ltx-2.5-video-vae-bf16.safetensors")
    # Exercise the final LightX2V path gate, not just Run's config builder.
    for key in ("dit_original_ckpt", "ltx_text_encoder_ckpt", "ltx_video_vae_ckpt", "ltx_audio_vae_ckpt", "ltx25_video_vae_runtime_ckpt", "upsampler_original_ckpt"):
        path = Path(config[key]); path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    tree = ast.parse((source / "lightx2v/utils/utils.py").read_text())
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_config_paths")
    def exists(path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    scope = {"check_path_exists": exists, "logger": type("Log", (), {"debug": lambda *a: None, "info": lambda *a: None})()}
    exec(compile(ast.Module(body=[validate], type_ignores=[]), "validate_config_paths.py", "exec"), scope)
    scope[validate.name](config)
    Path(config["ltx25_video_vae_runtime_ckpt"]).unlink()
    with pytest.raises(FileNotFoundError):
        scope[validate.name](config)
    for name in ("run_ltx25_in_container.sh", "run_ltx25_c1_in_container.sh"):
        assert "--model_cls ltx2_5 " in (RUN_ROOT / "scripts/lib" / name).read_text()
    assert "--model_cls ltx2\n" in (RUN_ROOT / "scripts/lib/run_ltx23_in_container.sh").read_text()
    assert "ltx2" in model_choices(source)


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_h3_current_source_configs_keep_routes(source, size):
    config_dir = source / "configs"
    published = json.loads((config_dir / "minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json").read_text())
    overlay = json.loads((config_dir / "platforms/amd_rocm/minimax_h3_fl2av_4step_768p_gfx1201.json").read_text())
    config = load_builder("build_minimax_h3_runtime_config").build(published, overlay, smoke=False, seq_parallel_size=size)
    assert "minimax_h3" in model_choices(source)
    assert config["h3_core_attn_type"] == "aiter_fav3_sage_bf16_attn"
    assert config["h4_video_attn_type"] == "aiter_triton_mha"
    assert config["lazy_load"]
