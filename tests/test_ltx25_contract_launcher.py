from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts/collector/run_ltx25_contract_collector.sh"
IN_CONTAINER = REPO_ROOT / "scripts/lib/run_ltx25_in_container.sh"
BUILDER_PATH = REPO_ROOT / "scripts/lib/build_ltx25_runtime_config.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_ltx25_runtime_config", BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ltx25_contract_wrapper_has_zero_argument_automatic_entry():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "an existing LTX-2.5 B0 command is required" not in text
    assert "run_ltx25_in_container.sh" in text
    assert "/disk/ssd/difumodels-Dis/LTX-2.5" in text
    assert "HIP_VISIBLE_DEVICES:-0,1" in text

    for script in (WRAPPER, IN_CONTAINER):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_ltx25_runtime_config_freezes_b0ref_and_labels_conv_diagnostic(tmp_path):
    builder = _load_builder()
    model_root = tmp_path / "LTX-2.5"
    census = tmp_path / "ltx25_weight_census.json"
    config = builder.build_config(model_root, census, "a" * 64)

    assert config["target_height"] == 704
    assert config["target_width"] == 1280
    assert config["target_video_length"] == 121
    assert config["fps"] == 24
    assert len(config["distilled_sigma_values"]) - 1 == 8
    assert len(config["distilled_sigma_values_upsample"]) - 1 == 3
    assert config["parallel"]["seq_p_size"] == 2
    assert config["ltx25_video_vae_route"] == "conv_diagnostic"
    assert config["ltx_video_vae_ckpt"].endswith(
        "ltx-2.5-video-vae-bf16.safetensors"
    )
    assert config["ltx25_video_vae_runtime_ckpt"].endswith(
        "ltx-2.5-video-vae-conv-bf16.safetensors"
    )
    assert config["audio_mel_bins"] == 64


def test_ltx25_runtime_config_cli_writes_json(tmp_path):
    output = tmp_path / "runtime.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--model-root",
            str(tmp_path / "model"),
            "--census",
            str(tmp_path / "census.json"),
            "--census-sha256",
            "b" * 64,
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ltx25_census_sha256"] == "b" * 64
    assert payload["ltx25_b0_gfx1201"] is True
