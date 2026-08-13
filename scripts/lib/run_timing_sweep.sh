#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<EOF
Usage:
  bash scripts/${RUN_ENTRY_NAME} [GPU_COUNT]
  bash scripts/${RUN_ENTRY_NAME} [--gpu-count GPU_COUNT | --counts LIST] [--image IMAGE]

Task: ${RUN_TASK_LABEL}
Default GPU counts: ${RUN_DEFAULT_COUNTS}
Supported GPU counts: ${RUN_SUPPORTED_COUNTS}

Examples:
  bash scripts/${RUN_ENTRY_NAME}
  bash scripts/${RUN_ENTRY_NAME} --gpu-count ${RUN_EXAMPLE_COUNT}
  bash scripts/${RUN_ENTRY_NAME} ${RUN_EXAMPLE_COUNT}

Overrides:
  WAN_GPU_GROUPS       Semicolon-separated physical GPU groups in supported-count order.
  WAN_RUN_TAG          Result directory name.
  WAN_RESULT_DIR       Exact result directory.
  MODEL_RUN            Host container launcher.

The inherited IMAGE environment variable is intentionally ignored. Use the
explicit --image option when a numbered task must run another image.
EOF
}

for required_name in \
    RUN_ENTRY_NAME RUN_TASK_ID RUN_TASK_LABEL RUN_MODEL_VERSION \
    RUN_DEFAULT_COUNTS RUN_SUPPORTED_COUNTS RUN_DEFAULT_GPU_GROUPS \
    RUN_TARGET_HEIGHT RUN_TARGET_WIDTH RUN_TARGET_FRAMES RUN_TARGET_FPS; do
    [[ -n "${!required_name:-}" ]] || {
        echo "ERROR: shared timing launcher requires ${required_name}" >&2
        exit 2
    }
done

case "${RUN_MODEL_VERSION}" in
    wan21)
        model_run_mode=--run-wan21-gfx1201
        summary_model=wan21
        ;;
    wan22)
        model_run_mode=--run-wan22-gfx1201
        summary_model=wan22
        ;;
    *)
        echo "ERROR: RUN_MODEL_VERSION must be wan21 or wan22" >&2
        exit 2
        ;;
esac

selected_counts="${WAN_GPU_COUNT:-${RUN_DEFAULT_COUNTS}}"
image_override=
while (( $# > 0 )); do
    case "$1" in
        --gpu-count)
            (( $# >= 2 )) || { echo "ERROR: --gpu-count requires a value" >&2; exit 2; }
            selected_counts="$2"
            shift 2
            ;;
        --gpu-count=*)
            selected_counts="${1#--gpu-count=}"
            shift
            ;;
        --counts)
            (( $# >= 2 )) || { echo "ERROR: --counts requires a value" >&2; exit 2; }
            selected_counts="$2"
            shift 2
            ;;
        --counts=*)
            selected_counts="${1#--counts=}"
            shift
            ;;
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
        1 | 2 | 4 | 8 | 16)
            [[ "${selected_counts}" == "${RUN_DEFAULT_COUNTS}" ]] || {
                echo "ERROR: positional GPU count cannot be combined with another count selection" >&2
                exit 2
            }
            selected_counts="$1"
            shift
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
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
model_run="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
summary_script="${WAN_TIMING_SUMMARY_SCRIPT:-${script_dir}/summarize_timings.py}"
run_tag="${WAN_RUN_TAG:-${RUN_TASK_ID}_$(date +%Y%m%d_%H%M%S)}"
result_dir="${WAN_RESULT_DIR:-${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}}"
gpu_groups_value="${WAN_GPU_GROUPS:-${RUN_DEFAULT_GPU_GROUPS}}"

[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
[[ -f "${summary_script}" ]] || { echo "ERROR: timing summary not found: ${summary_script}" >&2; exit 1; }
if [[ -d "${result_dir}" ]] && [[ -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}"

IFS=',' read -r -a supported_counts <<<"${RUN_SUPPORTED_COUNTS}"
IFS=';' read -r -a gpu_groups <<<"${gpu_groups_value}"
if [[ "${#gpu_groups[@]}" -ne "${#supported_counts[@]}" ]]; then
    echo "ERROR: WAN_GPU_GROUPS must provide ${#supported_counts[@]} groups for ${RUN_SUPPORTED_COUNTS}" >&2
    exit 2
fi

declare -A supported=()
declare -A gpu_group_by_count=()
for index in "${!supported_counts[@]}"; do
    count="${supported_counts[${index}]}"
    group="${gpu_groups[${index}]}"
    supported["${count}"]=1
    [[ "${group}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        echo "ERROR: invalid ${count}-GPU group: ${group}" >&2
        exit 2
    }
    IFS=',' read -r -a group_devices <<<"${group}"
    [[ "${#group_devices[@]}" -eq "${count}" ]] || {
        echo "ERROR: ${count}-GPU group contains ${#group_devices[@]} ids: ${group}" >&2
        exit 2
    }
    declare -A group_seen=()
    for device in "${group_devices[@]}"; do
        [[ -z "${group_seen[${device}]:-}" ]] || {
            echo "ERROR: ${count}-GPU group contains duplicate id ${device}" >&2
            exit 2
        }
        group_seen["${device}"]=1
    done
    unset group_seen
    gpu_group_by_count["${count}"]="${group}"
done

IFS=',' read -r -a counts <<<"${selected_counts}"
declare -A count_seen=()
for count in "${counts[@]}"; do
    [[ -n "${supported[${count}]:-}" ]] || {
        echo "ERROR: GPU count ${count} is not supported; choose from ${RUN_SUPPORTED_COUNTS}" >&2
        exit 2
    }
    [[ -z "${count_seen[${count}]:-}" ]] || {
        echo "ERROR: duplicate GPU count ${count}" >&2
        exit 2
    }
    count_seen["${count}"]=1
done

duration_seconds="$(( (RUN_TARGET_FRAMES - 1) / RUN_TARGET_FPS ))"
python3 - "${result_dir}/run_manifest.json" <<PY
import json
import pathlib

payload = {
    "kind": "lightx2v_gfx1201_timing_sweep",
    "task_id": "${RUN_TASK_ID}",
    "label": "${RUN_TASK_LABEL}",
    "model": "${RUN_MODEL_VERSION}",
    "width": ${RUN_TARGET_WIDTH},
    "height": ${RUN_TARGET_HEIGHT},
    "frames": ${RUN_TARGET_FRAMES},
    "fps": ${RUN_TARGET_FPS},
    "duration_seconds": ${duration_seconds},
    "gpu_counts": [${selected_counts//,/, }],
    "video_pattern": "/workspace/LightX2V/save_results/${run_tag}/<count>gpu.mp4",
    "b1_policy": "2/4 baseline, 8/16 b1" if "${RUN_MODEL_VERSION}" == "wan22" else None,
}
pathlib.Path("${result_dir}/run_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "Task: ${RUN_TASK_LABEL}"
echo "Workload: ${RUN_TARGET_HEIGHT}x${RUN_TARGET_WIDTH}, ${RUN_TARGET_FRAMES} frames @ ${RUN_TARGET_FPS} FPS (${duration_seconds}s)"
echo "GPU counts: ${selected_counts}"
echo "Results: ${result_dir}"
if [[ "${RUN_MODEL_VERSION}" == wan22 ]]; then
    echo "Wan2.2 B1 policy: auto (2/4=baseline, 8/16=b1; explicit override remains available)"
fi

for count in "${counts[@]}"; do
    selected="${gpu_group_by_count[${count}]}"
    output_path="/workspace/LightX2V/save_results/${run_tag}/${count}gpu.mp4"
    log_path="${result_dir}/${count}gpu.log"
    echo "[$(date +%H:%M:%S)] ${RUN_MODEL_VERSION}: ${count} GPUs (${selected})"
    run_environment=(
        GPU_DEVICES="${selected}"
        IMAGE="${image_override}"
        TARGET_HEIGHT="${RUN_TARGET_HEIGHT}"
        TARGET_WIDTH="${RUN_TARGET_WIDTH}"
        TARGET_VIDEO_LENGTH="${RUN_TARGET_FRAMES}"
        TARGET_FPS="${RUN_TARGET_FPS}"
        PROFILE_WORKLOAD="${RUN_TASK_ID}"
        SAVE_RESULT_PATH="${output_path}"
    )
    if [[ "${RUN_MODEL_VERSION}" == wan22 ]]; then
        run_environment+=(LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE="${LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE:-auto}")
    fi
    set +e
    env "${run_environment[@]}" "${model_run}" "${model_run_mode}" 2>&1 | tee "${log_path}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[0] != 0 )); then
        echo "ERROR: ${count}-GPU inference failed; see ${log_path}" >&2
        exit "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        echo "ERROR: could not write ${log_path}" >&2
        exit "${pipeline_status[1]}"
    fi
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Dry run complete; timing summary skipped."
    exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" \
    --model "${summary_model}" \
    --log-dir "${result_dir}" \
    --counts "${selected_counts}"

echo "Timing sweep complete: ${run_tag}"
echo "Raw logs : ${result_dir}/<count>gpu.log"
echo "Videos   : ${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}/<count>gpu.mp4"
echo "CSV      : ${result_dir}/timings.csv"
echo "Table    : ${result_dir}/timings.md"
echo "Manifest : ${result_dir}/run_manifest.json"
