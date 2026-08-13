#!/usr/bin/env bash
set -Eeuo pipefail

readonly run_root=/workspace/LightX2VRun
readonly compatibility_root="${run_root}/python_compat/wan22_lazy"
if (( $# > 1 )); then
    echo "ERROR: lazy container launcher accepts at most one GPU-count argument" >&2
    exit 2
fi
readonly gpu_count="${1:-16}"
case "${gpu_count}" in
    2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: lazy container launcher supports 2, 4, 8, or 16 GPUs, got ${gpu_count}" >&2
        exit 2
        ;;
esac
readonly profile_launcher="${run_root}/scripts/run_wan22_ckconv3d_v1_profile_${gpu_count}gpu.sh"

[[ -f "${compatibility_root}/sitecustomize.py" ]] || {
    echo "ERROR: lazy compatibility shim not found: ${compatibility_root}/sitecustomize.py" >&2
    exit 1
}
[[ -f "${profile_launcher}" ]] || {
    echo "ERROR: Wan2.2 ${gpu_count}-GPU profile launcher not found: ${profile_launcher}" >&2
    exit 1
}

export PYTHONPATH="${compatibility_root}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Wan2.2 lazy compatibility shim: ${compatibility_root}/sitecustomize.py"
echo "Wan2.2 lazy GPU count: ${gpu_count}"

exec bash "${profile_launcher}"
