"""Exercise the container launcher with a Docker substitute, without a daemon."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from test_h3_ref2va_inventory import FILES, write

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collector/run_minimax_h3_ref2v_weights.sh"


def launch(tmp_path, *args, dry_run=False):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["DOCKER_CAPTURE"]).write_text(json.dumps(args))
mounts = {}
for i, arg in enumerate(args):
    if arg == "--volume":
        host, target, mode = args[i + 1].rsplit(":", 2)
        mounts[target] = host
start = args.index("/workspace/LightX2VRun/scripts/collector/collect_minimax_h3_ref2v_weights.py")
command = args[start:]
for i, arg in enumerate(command):
    for target, host in mounts.items():
        if arg == target or arg.startswith(target + "/"):
            command[i] = host + arg[len(target):]
            break
# -S verifies the actual collector can run without site packages.
sys.exit(subprocess.call([sys.executable, "-B", "-S", *command]))
''')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}",
           "DOCKER_CAPTURE": str(tmp_path / "docker.json"), "DRY_RUN": "1" if dry_run else "0"}
    env.pop("IMAGE", None)
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path, env=env, text=True, capture_output=True)


def test_container_launch_uses_current_run_and_keeps_reports(tmp_path):
    models = tmp_path / "model files"
    output = tmp_path / "report files"
    write(models / FILES["bf16"], [("w", "BF16", [1], bytes(2))])
    write(models / FILES["int8_convrot"], [("w", "I8", [1], bytes(1))])
    result = launch(tmp_path, "--model-root", str(models), "--output-dir", str(output), "--image", "test:old")
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "docker.json").read_text())
    assert f"{ROOT}:/workspace/LightX2VRun:ro" in args
    assert f"{models}:/models/minimax_h3_comfyui:ro" in args
    assert f"{output}:/output:rw" in args
    assert args[args.index("--entrypoint") + 1] == "python3"
    assert "test:old" in args
    assert "--privileged" not in args and "--device" not in args and "--gpus" not in args
    assert json.loads((output / "h3_ref2va_weights.json").read_text())["status"] == "ok"
    assert (output / "h3_ref2va_weights.md").is_file()


def test_container_failure_retains_diagnostics_and_exit_status(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    output = tmp_path / "reports"
    result = launch(tmp_path, "--model-root", str(models), "--output-dir", str(output))
    assert result.returncode == 1, result.stderr
    report = json.loads((output / "h3_ref2va_weights.json").read_text())
    assert report["status"] == "error" and len(report["assets"]) == 2


def test_dry_run_needs_no_weights_and_does_not_create_output(tmp_path):
    result = launch(tmp_path, "--model-root", "missing models", "--output-dir", "reports", dry_run=True)
    assert result.returncode == 0, result.stderr
    args = shlex.split(result.stdout)
    assert "compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260901-LTX25-Scale-ReleaseFinal" in args
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "docker.json").exists()


def test_missing_option_value_is_actionable(tmp_path):
    result = launch(tmp_path, "--image")
    assert result.returncode == 2
    assert "missing value for --image" in result.stderr
