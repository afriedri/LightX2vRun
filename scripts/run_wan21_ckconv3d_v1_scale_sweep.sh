#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
sweep_counts="${SWEEP_COUNTS:-1,2,4,8,16}"
sweep_tag="${SWEEP_TAG:-wan21_ckconv3d_v1_release_$(date +%Y%m%d_%H%M%S)}"
scale_gpu_groups="${SCALE_GPU_GROUPS:-0;1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
unset LIGHTX2V_CK_CONV3D_PROFILE_DIR PROFILE_TAG

[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
IFS=';' read -r -a gpu_groups <<<"${scale_gpu_groups}"
if [[ "${#gpu_groups[@]}" -ne 5 ]]; then
    echo "ERROR: SCALE_GPU_GROUPS must contain five semicolon-separated groups for 1/2/4/8/16 GPUs" >&2
    exit 1
fi
declare -A gpu_group_by_count=(
    [1]="${gpu_groups[0]}"
    [2]="${gpu_groups[1]}"
    [4]="${gpu_groups[2]}"
    [8]="${gpu_groups[3]}"
    [16]="${gpu_groups[4]}"
)
for expected_count in 1 2 4 8 16; do
    group="${gpu_group_by_count[${expected_count}]}"
    [[ "${group}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        echo "ERROR: invalid ${expected_count}-GPU group: ${group}" >&2
        exit 1
    }
    IFS=',' read -r -a group_gpus <<<"${group}"
    if [[ "${#group_gpus[@]}" -ne "${expected_count}" ]]; then
        echo "ERROR: ${expected_count}-GPU group contains ${#group_gpus[@]} ids: ${group}" >&2
        exit 1
    fi
    declare -A seen_group_gpus=()
    for gpu in "${group_gpus[@]}"; do
        [[ -z "${seen_group_gpus[${gpu}]:-}" ]] || {
            echo "ERROR: ${expected_count}-GPU group contains duplicate id ${gpu}" >&2
            exit 1
        }
        seen_group_gpus["${gpu}"]=1
    done
    unset seen_group_gpus
done

IFS=',' read -r -a counts <<<"${sweep_counts}"
for count in "${counts[@]}"; do
    case "${count}" in
        1 | 2 | 4 | 8 | 16) ;;
        *) echo "ERROR: SWEEP_COUNTS contains unsupported count ${count}" >&2; exit 1 ;;
    esac
    selected="${gpu_group_by_count[${count}]}"
    output_path="/workspace/LightX2V/save_results/${sweep_tag}_${count}gpu.mp4"
    echo "[$(date +%H:%M:%S)] Wan2.1 CK Conv3d release: ${count} GPUs (${selected})"
    env \
        GPU_DEVICES="${selected}" \
        CONTAINER_NAME="ck-conv3d-release-${count}gpu-${sweep_tag}" \
        SAVE_RESULT_PATH="${output_path}" \
        "${model_run}" --wan21-ckconv3d-v1-release
done

echo "Wan2.1 CK Conv3d production scale sweep complete: ${sweep_tag}"
