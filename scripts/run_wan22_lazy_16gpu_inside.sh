#!/usr/bin/env bash
set -Eeuo pipefail

readonly run_root=/workspace/LightX2VRun
readonly compatibility_root="${run_root}/python_compat/wan22_lazy"
readonly profile_launcher="${run_root}/scripts/run_wan22_ckconv3d_v1_profile_16gpu.sh"

[[ -f "${compatibility_root}/sitecustomize.py" ]] || {
    echo "ERROR: lazy compatibility shim not found: ${compatibility_root}/sitecustomize.py" >&2
    exit 1
}
[[ -f "${profile_launcher}" ]] || {
    echo "ERROR: Wan2.2 16-GPU profile launcher not found: ${profile_launcher}" >&2
    exit 1
}

export PYTHONPATH="${compatibility_root}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Wan2.2 lazy compatibility shim: ${compatibility_root}/sitecustomize.py"

exec bash "${profile_launcher}"
