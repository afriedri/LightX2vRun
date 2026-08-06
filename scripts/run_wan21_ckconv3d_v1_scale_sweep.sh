#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
gpu_devices="${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
sweep_counts="${SWEEP_COUNTS:-1,2,4,8,16}"
sweep_tag="${SWEEP_TAG:-wan21_ckconv3d_v1_scale_$(date +%Y%m%d_%H%M%S)}"

[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
[[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "ERROR: GPU_DEVICES must be comma-separated ids" >&2; exit 1; }
IFS=',' read -r -a available_gpus <<<"${gpu_devices}"

IFS=',' read -r -a counts <<<"${sweep_counts}"
for count in "${counts[@]}"; do
    case "${count}" in
        1 | 2 | 4 | 8 | 16) ;;
        *) echo "ERROR: SWEEP_COUNTS contains unsupported count ${count}" >&2; exit 1 ;;
    esac
    if (( ${#available_gpus[@]} < count )); then
        echo "ERROR: ${count}-GPU run requested but only ${#available_gpus[@]} GPU ids were provided" >&2
        exit 1
    fi

    selected="$(IFS=','; echo "${available_gpus[*]:0:count}")"
    echo "[$(date +%H:%M:%S)] Wan2.1 CK Conv3d profile: ${count} GPUs (${selected})"
    env \
        GPU_DEVICES="${selected}" \
        CONTAINER_NAME="ck-conv3d-profile-${count}gpu-${sweep_tag}" \
        PROFILE_TAG="${sweep_tag}_${count}gpu" \
        "${model_run}" --profile-wan21-ckconv3d-v1
done

echo "Wan2.1 CK Conv3d scale sweep complete: ${sweep_tag}"
