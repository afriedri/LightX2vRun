#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_720p_5s_timing_sweep.sh [--image IMAGE]

Runs the 720P five-second timing workload on 2, 4, 8, and 16 GPUs:
  720x1280, 24 FPS, 121 frames ((121 - 1) / 24 = 5 seconds)

This records stage timings only and explicitly disables CK Conv3D shape
collection. TIMING_SWEEP_COUNTS, TIMING_GPU_GROUPS, TIMING_SWEEP_TAG, and
TIMING_RESULT_DIR override the timing sweep.
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
summary_script="${TIMING_SUMMARY_SCRIPT:-${script_dir}/summarize_wan22_ckconv3d_timing.py}"
sweep_counts="${TIMING_SWEEP_COUNTS:-2,4,8,16}"
sweep_tag="${TIMING_SWEEP_TAG:-wan22_ckconv3d_v1_720p_5s_timing_$(date +%Y%m%d_%H%M%S)}"
result_dir="${TIMING_RESULT_DIR:-${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${sweep_tag}}"
gpu_groups_value="${TIMING_GPU_GROUPS:-1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

readonly workload=720p_24fps_121f
readonly target_height=720
readonly target_width=1280
readonly target_video_length=121
readonly target_fps=24
readonly duration_seconds=5

unset LIGHTX2V_CK_CONV3D_PROFILE_DIR PROFILE_TAG PROFILE_WORKLOAD
[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
[[ -f "${summary_script}" ]] || { echo "ERROR: timing summary not found: ${summary_script}" >&2; exit 1; }
if [[ -d "${result_dir}" ]] && [[ -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}"

IFS=';' read -r -a gpu_groups <<<"${gpu_groups_value}"
if [[ "${#gpu_groups[@]}" -ne 4 ]]; then
    echo "ERROR: TIMING_GPU_GROUPS must contain four groups for 2/4/8/16 GPUs" >&2
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
    [[ "${#group_gpus[@]}" -eq "${expected_count}" ]] || {
        echo "ERROR: ${expected_count}-GPU group contains ${#group_gpus[@]} ids: ${group}" >&2
        exit 1
    }
done

IFS=',' read -r -a counts <<<"${sweep_counts}"
declare -A seen_counts=()
for count in "${counts[@]}"; do
    case "${count}" in
        2 | 4 | 8 | 16) ;;
        *) echo "ERROR: TIMING_SWEEP_COUNTS contains unsupported count ${count}" >&2; exit 1 ;;
    esac
    [[ -z "${seen_counts[${count}]:-}" ]] || {
        echo "ERROR: TIMING_SWEEP_COUNTS contains duplicate count ${count}" >&2
        exit 1
    }
    seen_counts["${count}"]=1
done

cat > "${result_dir}/workloads.csv" <<EOF
workload,width,height,fps,frames,duration_seconds
${workload},${target_width},${target_height},${target_fps},${target_video_length},${duration_seconds}
EOF

for count in "${counts[@]}"; do
    selected="${gpu_group_by_count[${count}]}"
    output_path="/workspace/LightX2V/save_results/${sweep_tag}_${count}gpu.mp4"
    log_path="${result_dir}/${count}gpu.log"
    echo "[$(date +%H:%M:%S)] Wan2.2 timing: 720P ${target_fps} FPS, ${target_video_length} frames, ${duration_seconds}s; ${count} GPUs (${selected})"
    run_environment=(
        GPU_DEVICES="${selected}"
        TARGET_HEIGHT="${target_height}"
        TARGET_WIDTH="${target_width}"
        TARGET_VIDEO_LENGTH="${target_video_length}"
        TARGET_FPS="${target_fps}"
        SAVE_RESULT_PATH="${output_path}"
    )
    if [[ -n "${image_override}" ]]; then
        run_environment+=(IMAGE="${image_override}")
    fi
    set +e
    env "${run_environment[@]}" "${model_run}" --wan22-ckconv3d-v1-release 2>&1 | tee "${log_path}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[0] != 0 )); then
        echo "ERROR: ${count}-GPU timing failed; see ${log_path}" >&2
        exit "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        echo "ERROR: could not write ${log_path}" >&2
        exit "${pipeline_status[1]}"
    fi
done

PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" \
    --log-dir "${result_dir}" \
    --counts "${sweep_counts}"

cat > "${result_dir}/timing_manifest.json" <<EOF
{
  "kind": "wan22_720p_24fps_5s_timing_sweep",
  "workload": "${workload}",
  "width": ${target_width},
  "height": ${target_height},
  "fps": ${target_fps},
  "frames": ${target_video_length},
  "duration_seconds": ${duration_seconds},
  "gpu_counts": [${sweep_counts//,/, }]
}
EOF

echo "Wan2.2 720P 24FPS five-second timing sweep complete: ${sweep_tag}"
echo "Results : ${result_dir}"
echo "Timing  : ${result_dir}/timings.md"
