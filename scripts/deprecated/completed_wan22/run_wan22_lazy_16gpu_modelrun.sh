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

gpu_devices="${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=',' read -r -a selected_gpus <<<"${gpu_devices}"
gpu_count="${WAN22_GPU_COUNT:-${#selected_gpus[@]}}"
case "${gpu_count}" in
    2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: lazy model launcher supports 2, 4, 8, or 16 GPUs, got ${gpu_count}" >&2
        exit 2
        ;;
esac
if [[ "${#selected_gpus[@]}" -ne "${gpu_count}" ]]; then
    echo "ERROR: GPU_DEVICES count ${#selected_gpus[@]} does not match WAN22_GPU_COUNT=${gpu_count}" >&2
    exit 2
fi

exec "${real_model_run}" \
    /bin/bash \
    /workspace/LightX2VRun/scripts/run_wan22_lazy_16gpu_inside.sh \
    "${gpu_count}"
