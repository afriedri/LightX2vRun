#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_scale_sweep.sh [--image IMAGE]

IMAGE can also be supplied as an exported environment variable or as:
  IMAGE=registry/image:tag bash scripts/run_wan22_ckconv3d_v1_scale_sweep.sh

SWEEP_COUNTS may select from 2,4,8,16. A legacy 1-GPU entry is ignored.
SWEEP_WORKLOADS may select unique values from 720p,480p,480p121f.
The default is 480p121f so completed 81f workloads are not repeated.
480p121f is 480x832, 121 frames, and 24 fps (5 seconds).
Use SWEEP_RESULT_DIR and SWEEP_SUMMARY_DIR for task-specific output overrides.
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
requested_sweep_counts="${SWEEP_COUNTS:-2,4,8,16}"
IFS=',' read -r -a requested_counts <<<"${requested_sweep_counts}"
counts=()
ignored_single_gpu=0
for count in "${requested_counts[@]}"; do
    case "${count}" in
        1)
            ignored_single_gpu=1
            ;;
        2 | 4 | 8 | 16)
            counts+=("${count}")
            ;;
        *)
            echo "ERROR: SWEEP_COUNTS contains unsupported count ${count}" >&2
            exit 1
            ;;
    esac
done
if (( ignored_single_gpu )); then
    echo "WARNING: ignoring legacy 1-GPU SWEEP_COUNTS entry; Wan2.2 CK Conv3d sweep starts at 2 GPUs" >&2
fi

default_sweep_tag="wan22_ckconv3d_v1_sweep_$(date +%Y%m%d_%H%M%S)"
if (( ${#counts[@]} == 0 )); then
    counts=(2 4 8 16)
    sweep_tag="${default_sweep_tag}"
    if [[ -n "${SWEEP_TAG:-}" ]]; then
        echo "WARNING: ignoring stale SWEEP_TAG=${SWEEP_TAG} from the legacy 1-GPU-only sweep" >&2
    fi
else
    sweep_tag="${SWEEP_TAG:-${default_sweep_tag}}"
fi
requested_sweep_workloads="${SWEEP_WORKLOADS:-480p121f}"
IFS=',' read -r -a requested_workloads <<<"${requested_sweep_workloads}"
workloads=()
declare -A seen_workloads=()
for workload in "${requested_workloads[@]}"; do
    case "${workload}" in
        720p | 480p | 480p121f) ;;
        *)
            echo "ERROR: SWEEP_WORKLOADS contains unsupported workload ${workload}" >&2
            exit 1
            ;;
    esac
    [[ -z "${seen_workloads[${workload}]:-}" ]] || {
        echo "ERROR: SWEEP_WORKLOADS contains duplicate workload ${workload}" >&2
        exit 1
    }
    seen_workloads["${workload}"]=1
    workloads+=("${workload}")
done
declare -A target_height_by_workload=(
    [720p]=720
    [480p]=480
    [480p121f]=480
)
declare -A target_width_by_workload=(
    [720p]=1280
    [480p]=832
    [480p121f]=832
)
declare -A target_video_length_by_workload=(
    [720p]=81
    [480p]=81
    [480p121f]=121
)
declare -A target_fps_by_workload=(
    [720p]=16
    [480p]=16
    [480p121f]=24
)
host_output_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
result_dir="${SWEEP_RESULT_DIR:-${host_output_dir}/${sweep_tag}}"
scale_gpu_groups="${SCALE_GPU_GROUPS:-1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

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
if [[ "${#gpu_groups[@]}" -eq 5 ]]; then
    echo "WARNING: ignoring legacy 1-GPU entry in SCALE_GPU_GROUPS" >&2
    gpu_groups=("${gpu_groups[@]:1}")
elif [[ "${#gpu_groups[@]}" -ne 4 ]]; then
    echo "ERROR: SCALE_GPU_GROUPS must contain four semicolon-separated groups for 2/4/8/16 GPUs" >&2
    exit 1
fi
declare -A gpu_group_by_count=(
    [2]="${gpu_groups[0]}"
    [4]="${gpu_groups[1]}"
    [8]="${gpu_groups[2]}"
    [16]="${gpu_groups[3]}"
)
for expected_count in 2 4 8 16; do
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

declare -A seen_counts=()
for count in "${counts[@]}"; do
    case "${count}" in
        2 | 4 | 8 | 16) ;;
        *) echo "ERROR: SWEEP_COUNTS contains unsupported count ${count}" >&2; exit 1 ;;
    esac
    [[ -z "${seen_counts[${count}]:-}" ]] || {
        echo "ERROR: SWEEP_COUNTS contains duplicate count ${count}" >&2
        exit 1
    }
    seen_counts["${count}"]=1
done

profile_roots=()
for workload in "${workloads[@]}"; do
    target_height="${target_height_by_workload[${workload}]}"
    target_width="${target_width_by_workload[${workload}]}"
    target_video_length="${target_video_length_by_workload[${workload}]}"
    target_fps="${target_fps_by_workload[${workload}]}"
    for count in "${counts[@]}"; do
        selected="${gpu_group_by_count[${count}]}"
        profile_tag="${sweep_tag}_${workload}_${count}gpu"
        profile_dir="${host_output_dir}/ck_conv3d_profiles/${profile_tag}"
        output_path="/workspace/LightX2V/save_results/${profile_tag}.mp4"
        log_path="${result_dir}/${workload}_${count}gpu.log"
        if [[ "${DRY_RUN:-0}" != "1" && -e "${profile_dir}" ]]; then
            echo "ERROR: profile directory already exists: ${profile_dir}" >&2
            echo "Use a fresh SWEEP_TAG so old rank snapshots cannot contaminate this sweep." >&2
            exit 1
        fi
        echo "[$(date +%H:%M:%S)] Wan2.2 CK Conv3d profile: ${workload} ${target_height}x${target_width}, ${target_video_length} frames @ ${target_fps} fps; ${count} GPUs (${selected})"
        run_environment=(
            GPU_DEVICES="${selected}"
            PROFILE_TAG="${profile_tag}"
            PROFILE_WORKLOAD="${workload}"
            TARGET_VIDEO_LENGTH="${target_video_length}"
            TARGET_HEIGHT="${target_height}"
            TARGET_WIDTH="${target_width}"
            TARGET_FPS="${target_fps}"
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
            echo "ERROR: ${workload}/${count}-GPU profile failed; see ${log_path}" >&2
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
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Wan2.2 CK Conv3d scale sweep dry-run complete: ${sweep_tag}"
    exit 0
fi

summary_dir="${SWEEP_SUMMARY_DIR:-${result_dir}/sweep_summary}"
PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" \
    --inputs "${profile_roots[@]}" \
    --output-dir "${summary_dir}"
echo "Wan2.2 CK Conv3d scale sweep complete: ${sweep_tag}"
echo "Logs: ${result_dir}"
echo "Profiles: ${host_output_dir}/ck_conv3d_profiles/${sweep_tag}_<workload>_*gpu"
echo "Summary: ${summary_dir}"
