#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_scale_sweep.sh [--image IMAGE]

IMAGE can also be supplied as an exported environment variable or as:
  IMAGE=registry/image:tag bash scripts/run_wan22_ckconv3d_v1_scale_sweep.sh
EOF
}

image_override="${IMAGE:-}"
while (( $# > 0 )); do
    case "$1" in
        --image)
            (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }
            image_override="$2"
            shift 2
            ;;
        --image=*)
            image_override="${1#--image=}"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unsupported argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
[[ -z "${image_override}" || "${image_override}" != *[[:space:]]* ]] || {
    echo "ERROR: image reference may not contain whitespace" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
summary_script="${SUMMARY_SCRIPT:-${script_dir}/summarize_wan22_ckconv3d_scale.py}"
sweep_counts="${SWEEP_COUNTS:-2,4,8,16}"
sweep_tag="${SWEEP_TAG:-wan22_ckconv3d_v1_sweep_$(date +%Y%m%d_%H%M%S)}"
result_dir="${RESULT_DIR:-${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${sweep_tag}}"
host_output_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
scale_gpu_groups="${SCALE_GPU_GROUPS:-0;1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
[[ -f "${summary_script}" ]] || { echo "ERROR: Wan2.2 sweep summary not found: ${summary_script}" >&2; exit 1; }
if [[ -d "${result_dir}" ]] && [[ -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}"
if [[ -n "${image_override}" ]]; then
    echo "Wan2.2 CK Conv3d image override: ${image_override}"
fi

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
profile_roots=()
declare -A seen_counts=()
for count in "${counts[@]}"; do
    case "${count}" in
        1 | 2 | 4 | 8 | 16) ;;
        *) echo "ERROR: SWEEP_COUNTS contains unsupported count ${count}" >&2; exit 1 ;;
    esac
    [[ -z "${seen_counts[${count}]:-}" ]] || {
        echo "ERROR: SWEEP_COUNTS contains duplicate count ${count}" >&2
        exit 1
    }
    seen_counts["${count}"]=1
    selected="${gpu_group_by_count[${count}]}"
    profile_tag="${sweep_tag}_${count}gpu"
    profile_dir="${host_output_dir}/ck_conv3d_profiles/${profile_tag}"
    output_path="/workspace/LightX2V/save_results/${profile_tag}.mp4"
    log_path="${result_dir}/${count}gpu.log"
    if [[ "${DRY_RUN:-0}" != "1" && -e "${profile_dir}" ]]; then
        echo "ERROR: profile directory already exists: ${profile_dir}" >&2
        echo "Use a fresh SWEEP_TAG so old rank snapshots cannot contaminate this sweep." >&2
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] Wan2.2 CK Conv3d profile: ${count} GPUs (${selected})"
    run_environment=(
        GPU_DEVICES="${selected}"
        PROFILE_TAG="${profile_tag}"
        LIGHTX2V_CK_CONV3D_PROFILE_DIR="/workspace/LightX2V/save_results/ck_conv3d_profiles/${profile_tag}"
        SAVE_RESULT_PATH="${output_path}"
    )
    if [[ -n "${image_override}" ]]; then
        run_environment+=(IMAGE="${image_override}")
    fi
    set +e
    env "${run_environment[@]}" \
        "${model_run}" --profile-wan22-ckconv3d-v1 2>&1 | tee "${log_path}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[0] != 0 )); then
        echo "ERROR: ${count}-GPU profile failed; see ${log_path}" >&2
        exit "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        echo "ERROR: could not write ${log_path}" >&2
        exit "${pipeline_status[1]}"
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi
    [[ -f "${profile_dir}/manifest.json" ]] || {
        echo "ERROR: profile manifest missing: ${profile_dir}/manifest.json" >&2
        exit 1
    }
    profile_roots+=("${profile_dir}")
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Wan2.2 CK Conv3d scale sweep dry-run complete: ${sweep_tag}"
    exit 0
fi

summary_dir="${SUMMARY_DIR:-${result_dir}/sweep_summary}"
PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" \
    --inputs "${profile_roots[@]}" \
    --output-dir "${summary_dir}"
echo "Wan2.2 CK Conv3d scale sweep complete: ${sweep_tag}"
echo "Logs: ${result_dir}"
echo "Profiles: ${host_output_dir}/ck_conv3d_profiles/${sweep_tag}_*gpu"
echo "Summary: ${summary_dir}"
