#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 1 || "$1" != "--profile-wan22-ckconv3d-v1" ]]; then
    echo "ERROR: dedicated lazy model launcher only accepts --profile-wan22-ckconv3d-v1" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
real_model_run="${LIGHTX2V_LAZY_REAL_MODEL_RUN:-${repo_root}/lightx2vModelrun}"

[[ -x "${real_model_run}" ]] || {
    echo "ERROR: real LightX2V model launcher is not executable: ${real_model_run}" >&2
    exit 1
}

exec "${real_model_run}" \
    /bin/bash \
    /workspace/LightX2VRun/scripts/run_wan22_lazy_16gpu_inside.sh
