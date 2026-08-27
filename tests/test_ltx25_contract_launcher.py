from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts/collector/run_ltx25_contract_collector.sh"
ULTRA_WRAPPER = REPO_ROOT / "scripts/collector/LTX25-CollectorUltra.sh"
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
    assert "lightx2vModelrun" in text
    assert "--run-ltx25-contract-collector" in text
    assert 'bash "${run_root}/scripts/lib/run_ltx25_in_container.sh"' not in text
    assert "python3 -m lightx2v" not in text
    assert "/disk/ssd/difumodels-Dis/LTX-2.5" in text
    assert "HIP_VISIBLE_DEVICES:-0,1" in text
    in_container_text = IN_CONTAINER.read_text(encoding="utf-8")
    assert "LIGHTX2V_LTX25_IN_CONTAINER" in in_container_text
    assert "/workspace/LightX2V" in in_container_text
    assert "/disk/ssd/difumodels-Dis/LTX-2.5" in in_container_text
    assert "/workspace/.cache/aiter" in in_container_text
    model_path_assignment = in_container_text.index('model_path="${model_root}"')
    base_source = in_container_text.index(
        'source "${lightx2v_path}/scripts/base/base.sh"'
    )
    assert model_path_assignment < base_source

    ultra_text = ULTRA_WRAPPER.read_text(encoding="utf-8")
    assert "LIGHTX2V_LTX25_CONTRACT_ULTRA=1" in ultra_text
    assert "run_ltx25_contract_collector.sh" in ultra_text
    assert "Official DiffVAE remains an explicitly unexecuted" in ultra_text
    assert "git clone --no-checkout" in ultra_text
    assert "FETCH_HEAD^{commit}" in ultra_text
    assert "submodule update" not in ultra_text
    assert "submodule sync" not in ultra_text
    assert "--ignore-submodules=all" in ultra_text

    for script in (WRAPPER, ULTRA_WRAPPER, IN_CONTAINER):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_ltx25_image_dry_run_starts_container_with_exact_mounts(tmp_path):
    model_root = tmp_path / "models" / "LTX-2.5"
    model_root.mkdir(parents=True)
    output_dir = tmp_path / "evidence"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "LTX25_MODEL_ROOT": str(model_root),
        "LTX25_CONTRACT_OUTPUT_DIR": str(output_dir),
    }

    completed = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--image",
            "example.invalid/ltx25:collector",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.replace("\\,", ",")
    assert "Image:          example.invalid/ltx25:collector" in stdout
    assert "Mode:           ltx25_contract" in stdout
    assert "GPU devices:    0,1" in stdout
    assert f"source={model_root},target={model_root},readonly" in stdout
    assert f"source={output_dir},target=/workspace/LightX2V/save_results" in stdout
    assert "LIGHTX2V_LTX25_IN_CONTAINER=1" in stdout
    assert "/workspace/LightX2VRun/scripts/lib/run_ltx25_in_container.sh" in stdout


def test_ltx25_inner_helper_rejects_direct_host_execution():
    completed = subprocess.run(
        ["bash", str(IN_CONTAINER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "inside the ROCm image" in completed.stderr


def test_ltx25_collector_ultra_dry_run_reaches_the_container(tmp_path):
    model_root = tmp_path / "models" / "LTX-2.5"
    model_root.mkdir(parents=True)
    output_dir = tmp_path / "ultra-evidence"
    lightx2v_source = tmp_path / "source-cache" / "LightX2V"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_RUN": str(REPO_ROOT / "lightx2vModelrun"),
        "LTX25_MODEL_ROOT": str(model_root),
        "LTX25_CONTRACT_OUTPUT_DIR": str(output_dir),
        "LTX25_LIGHTX2V_GIT_DIR": str(lightx2v_source),
    }

    completed = subprocess.run(
        [
            "bash",
            str(ULTRA_WRAPPER),
            "--image",
            "example.invalid/ltx25:ultra",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout.replace("\\,", ",")
    assert "Image:          example.invalid/ltx25:ultra" in stdout
    assert "LTX-2.5 route:  CollectorUltra evidence; C1 probe disabled" in stdout
    assert "LIGHTX2V_LTX25_CONTRACT_ULTRA=1" in stdout
    assert f"source={lightx2v_source},target=/workspace/LightX2V,readonly" in stdout
    assert "CollectorUltra source plan: fetch hvat-gfx1201" in stdout


def test_ltx25_runtime_config_freezes_b0ref_and_labels_conv_diagnostic(tmp_path):
    builder = _load_builder()
    model_root = tmp_path / "LTX-2.5"
    config = builder.build_config(model_root)

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
    assert config["audio_mel_bins"] == 16
    # The transformer flattens 8 audio latent channels x 16 latent mel bins.
    # Its audio_patchify_proj checkpoint therefore accepts 128 features.
    assert 8 * config["audio_mel_bins"] == 128


def test_ltx25_runtime_config_cli_writes_json(tmp_path):
    output = tmp_path / "runtime.json"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--model-root",
            str(tmp_path / "model"),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "ltx25_census_path" not in payload
    assert "ltx25_census_sha256" not in payload
    assert payload["ltx25_b0_gfx1201"] is True
