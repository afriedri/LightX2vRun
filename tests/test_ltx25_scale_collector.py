from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "scripts/collector/LTX25-ScaleCollector.sh"


def _git_repo(path: Path, files: list[str]) -> str:
    for name in files:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_scale_collector_mounts_lightx2v_and_aiter_for_task10(tmp_path: Path) -> None:
    lightx2v = tmp_path / "LightX2V"
    lightx2v_revision = _git_repo(
        lightx2v,
        [
            "lightx2v/__init__.py",
            "lightx2v/models/video_encoders/hf/ltx2/video_vae/distributed_decoder.py",
        ],
    )
    aiter = tmp_path / "aiter"
    aiter_revision = _git_repo(aiter, ["aiter/__init__.py"])
    result_dir = tmp_path / "results"

    env = {
        **os.environ,
        "DRY_RUN": "1",
        "SOURCE_OVERLAY": "1",
        "MEMORY_MONITOR": "0",
        "LIGHTX2V_SRC": str(lightx2v),
        "AITER_SRC": str(aiter),
        "OUTPUT_DIR": str(tmp_path / "outputs"),
    }
    completed = subprocess.run(
        [
            "bash",
            str(COLLECTOR),
            "--image",
            "example.invalid/ltx25:scale",
            "--output-dir",
            str(result_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout.replace("\\,", ",")
    assert "GPU counts: 4,8,16" in output
    assert f"LightX2V={lightx2v_revision}" in output
    assert f"Aiter={aiter_revision}" in output
    assert f"source={lightx2v},target=/workspace/LightX2V,readonly" in output
    assert f"source={aiter},target=/workspace/aiter-source,readonly" in output
    assert "PYTHONPATH=/workspace/aiter-source" in output
    assert f"AITER_JIT_DIR=/workspace/.cache/aiter/ltx25-source-{aiter_revision[:12]}" in output
    assert "ScaleCollector container: SP4" in output
    assert "ScaleCollector container: SP8" in output
    assert "ScaleCollector container: SP16" in output


def test_scale_collector_defaults_to_image_installed_sources(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MEMORY_MONITOR": "0",
        "OUTPUT_DIR": str(tmp_path / "outputs"),
    }
    completed = subprocess.run(
        [
            "bash",
            str(COLLECTOR),
            "--image",
            "example.invalid/ltx25:scale",
            "--gpu-count",
            "4",
            "--output-dir",
            str(result_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout.replace("\\,", ",")
    assert "Source overlay: disabled" in output
    assert "image-installed LightX2V + Aiter AOT wheel" in output
    assert "target=/workspace/aiter-source" not in output
    assert "target=/workspace/LightX2V,readonly" not in output
    assert "PYTHONPATH=/workspace/aiter-source" not in output
    assert "AITER_JIT_DIR=" not in output


def test_scale_collector_keeps_source_fetch_as_explicit_debug_mode() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "git@gitcode.com:hvat-ai/LightX2V.git" in source
    assert "git@gitcode.com:hvat-ai/aiter.git" in source
    assert "submodule update --init --recursive" in source
    assert 'export MODEL_RUN="${script_dir}/LTX25-ScaleCollector.sh"' in source
    assert 'source_overlay="${SOURCE_OVERLAY:-0}"' in source
    assert "HF_TOKEN" not in source
    assert "HTTP_PROXY" not in source
    assert "aiter-tune-gfx1201" in source


def test_scale_collector_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(COLLECTOR)], check=True)
