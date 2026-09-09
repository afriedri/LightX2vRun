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
  TIMING_GPU_GROUPS    Semicolon-separated physical GPU groups in supported-count order.
  TIMING_RUN_TAG       Result directory name.
  TIMING_RESULT_DIR    Exact result directory.
  WAN_GPU_GROUPS / WAN_RUN_TAG / WAN_RESULT_DIR remain supported aliases.
  MODEL_RUN            Host container launcher.
  MEMORY_MONITOR       1 (default) enables host RAM/rank RSS/PSS/GPU VRAM sampling; 0 disables it.
  MEMORY_MONITOR_INTERVAL        JSONL sampling interval in seconds (default: 2).
  MEMORY_MONITOR_PRINT_INTERVAL  Terminal heartbeat interval in seconds (default: 30).

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
        memory_phase_schema=wan
        ;;
    wan22)
        model_run_mode=--run-wan22-gfx1201
        summary_model=wan22
        memory_phase_schema=wan
        ;;
    minimax_h3)
        model_run_mode=--run-minimax-h3-gfx1201
        summary_model=minimax_h3
        memory_phase_schema=minimax_h3
        ;;
    ltx25_c1)
        model_run_mode=--run-ltx25-c1-gfx1201
        summary_model=ltx25_c1
        memory_phase_schema=ltx2
        ;;
    *)
        echo "ERROR: RUN_MODEL_VERSION must be wan21, wan22, minimax_h3, or ltx25_c1" >&2
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
if [[ "${RUN_MODEL_VERSION}" == minimax_h3 && -z "${image_override}" ]]; then
    echo "ERROR: MiniMax-H3 timing requires --image with the rebuilt production image" >&2
    exit 2
fi
if [[ "${RUN_MODEL_VERSION}" == ltx25_c1 && -z "${image_override}" ]]; then
    echo "ERROR: LTX-2.5 C1 timing requires --image with the rebuilt production image" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
model_run="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
summary_script="${TIMING_SUMMARY_SCRIPT:-${WAN_TIMING_SUMMARY_SCRIPT:-${script_dir}/summarize_timings.py}}"
memory_monitor_script="${MEMORY_MONITOR_SCRIPT:-${script_dir}/monitor_run_memory.py}"
memory_monitor_enabled="${MEMORY_MONITOR:-1}"
memory_monitor_interval="${MEMORY_MONITOR_INTERVAL:-2.0}"
memory_monitor_print_interval="${MEMORY_MONITOR_PRINT_INTERVAL:-30.0}"
memory_monitor_start_timeout="${MEMORY_MONITOR_START_TIMEOUT:-900}"
memory_monitor_host_limit_gib="${MEMORY_MONITOR_HOST_LIMIT_GIB:-512}"
memory_monitor_gpu_limit_gib="${MEMORY_MONITOR_GPU_LIMIT_GIB:-32}"
memory_monitor_container="${MEMORY_MONITOR_CONTAINER:-${CONTAINER_NAME:-lightx2v-gfx1201}}"
run_tag="${TIMING_RUN_TAG:-${WAN_RUN_TAG:-${RUN_TASK_ID}_$(date +%Y%m%d_%H%M%S)}}"
result_dir="${TIMING_RESULT_DIR:-${WAN_RESULT_DIR:-${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}}}"
gpu_groups_value="${TIMING_GPU_GROUPS:-${WAN_GPU_GROUPS:-${RUN_DEFAULT_GPU_GROUPS}}}"
h3_first_frame=
h3_last_frame=
if [[ "${RUN_MODEL_VERSION}" == minimax_h3 ]]; then
    # Numbered tasks are fixed workloads. Do not let frame settings inherited
    # from a previous shell silently replace the task's declared keyframes.
    h3_first_frame="${RUN_H3_FIRST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png}"
    h3_last_frame="${RUN_H3_LAST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png}"
fi

[[ -x "${model_run}" ]] || { echo "ERROR: launcher is not executable: ${model_run}" >&2; exit 1; }
[[ -f "${summary_script}" ]] || { echo "ERROR: timing summary not found: ${summary_script}" >&2; exit 1; }
case "${memory_monitor_enabled}" in
    0 | 1) ;;
    *)
        echo "ERROR: MEMORY_MONITOR must be 0 or 1, got ${memory_monitor_enabled}" >&2
        exit 2
        ;;
esac
if [[ "${memory_monitor_enabled}" == "1" ]]; then
    [[ -f "${memory_monitor_script}" ]] || {
        echo "ERROR: memory monitor not found: ${memory_monitor_script}" >&2
        exit 1
    }
fi
if [[ -d "${result_dir}" ]] && [[ -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}"

IFS=',' read -r -a supported_counts <<<"${RUN_SUPPORTED_COUNTS}"
IFS=';' read -r -a gpu_groups <<<"${gpu_groups_value}"
if [[ "${#gpu_groups[@]}" -ne "${#supported_counts[@]}" ]]; then
    echo "ERROR: timing GPU groups must provide ${#supported_counts[@]} groups for ${RUN_SUPPORTED_COUNTS}" >&2
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

memory_monitor_pid=
stop_memory_monitor() {
    local monitor_pid="${memory_monitor_pid:-}"
    if [[ -n "${monitor_pid}" ]]; then
        if kill -0 "${monitor_pid}" 2>/dev/null; then
            kill "${monitor_pid}" 2>/dev/null || true
        fi
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    memory_monitor_pid=
}
trap stop_memory_monitor EXIT

start_memory_monitor() {
    local ranks="$1"
    local gpu_indices="$2"
    local inference_log="$3"
    local memory_output="$4"
    if [[ "${memory_monitor_enabled}" != "1" || "${DRY_RUN:-0}" == "1" ]]; then
        return
    fi
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${memory_monitor_script}" \
        --container "${memory_monitor_container}" \
        --log "${inference_log}" \
        --output "${memory_output}" \
        --interval "${memory_monitor_interval}" \
        --print-interval "${memory_monitor_print_interval}" \
        --start-timeout "${memory_monitor_start_timeout}" \
        --expected-ranks "${ranks}" \
        --gpu-indices "${gpu_indices}" \
        --host-limit-gib "${memory_monitor_host_limit_gib}" \
        --gpu-limit-gib "${memory_monitor_gpu_limit_gib}" \
        --phase-schema "${memory_phase_schema}" &
    memory_monitor_pid=$!
    echo "Memory monitor: ${memory_output} (sample=${memory_monitor_interval}s, terminal=${memory_monitor_print_interval}s + phase changes)"
}

wait_memory_monitor() {
    local monitor_pid="${memory_monitor_pid:-}"
    local monitor_status=0
    if [[ -z "${monitor_pid}" ]]; then
        return
    fi
    set +e
    wait "${monitor_pid}"
    monitor_status=$?
    set -e
    memory_monitor_pid=
    if (( monitor_status != 0 )); then
        echo "ERROR: memory monitor failed with status ${monitor_status}" >&2
        return "${monitor_status}"
    fi
}

duration_seconds="${RUN_DURATION_SECONDS:-$(( (RUN_TARGET_FRAMES - 1) / RUN_TARGET_FPS ))}"
python3 - "${result_dir}/run_manifest.json" "${h3_first_frame}" "${h3_last_frame}" <<PY
import json
import pathlib
import sys

gpu_counts = [${selected_counts//,/, }]
supported_counts = [${RUN_SUPPORTED_COUNTS//,/, }]
gpu_groups = "${gpu_groups_value}".split(";")
gpu_group_by_count = {
    str(count): group for count, group in zip(supported_counts, gpu_groups)
}
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
    "gpu_counts": gpu_counts,
    "gpu_groups": {
        str(count): gpu_group_by_count[str(count)] for count in gpu_counts
    },
    "parallel_topologies": (
        {
            str(count): ("ulysses_ring_2d" if count == 16 else "ulysses")
            for count in gpu_counts
        }
        if "${RUN_MODEL_VERSION}" == "minimax_h3"
        else ({str(count): "ulysses" for count in gpu_counts}
              if "${RUN_MODEL_VERSION}" == "ltx25_c1" else None)
    ),
    "first_frame": sys.argv[2] or None,
    "last_frame": sys.argv[3] or None,
    "prompt_preset": "${H3_PROMPT_PRESET:-bird-stable}" if "${RUN_MODEL_VERSION}" == "minimax_h3" else None,
    "video_pattern": "/workspace/LightX2V/save_results/${run_tag}/<count>gpu.mp4",
    "b1_policy": "2/4 baseline, 8/16 b1" if "${RUN_MODEL_VERSION}" == "wan22" else None,
}
pathlib.Path("${result_dir}/run_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "Task: ${RUN_TASK_LABEL}"
echo "Workload: ${RUN_TARGET_WIDTH}x${RUN_TARGET_HEIGHT}, ${RUN_TARGET_FRAMES} frames @ ${RUN_TARGET_FPS} FPS (${duration_seconds}s)"
echo "GPU counts: ${selected_counts}"
echo "Results: ${result_dir}"
if [[ "${memory_monitor_enabled}" == "1" ]]; then
    echo "Memory monitor: enabled (sample=${memory_monitor_interval}s, terminal=${memory_monitor_print_interval}s + phase changes)"
else
    echo "Memory monitor: disabled"
fi
if [[ "${RUN_MODEL_VERSION}" == wan22 ]]; then
    echo "Wan2.2 B1 policy: auto (2/4=baseline, 8/16=b1; explicit override remains available)"
elif [[ "${RUN_MODEL_VERSION}" == minimax_h3 ]]; then
    echo "MiniMax-H3 topology: SP2/SP4/SP8=Ulysses, SP16=CP2xU8"
elif [[ "${RUN_MODEL_VERSION}" == ltx25_c1 ]]; then
    echo "LTX-2.5 topology: SP2/SP4/SP8/SP16=pure Ulysses"
fi

for count in "${counts[@]}"; do
    selected="${gpu_group_by_count[${count}]}"
    output_path="/workspace/LightX2V/save_results/${run_tag}/${count}gpu.mp4"
    log_path="${result_dir}/${count}gpu.log"
    memory_output_path="${result_dir}/${count}gpu_memory.jsonl"
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
    elif [[ "${RUN_MODEL_VERSION}" == minimax_h3 ]]; then
        run_environment+=(
            LIGHTX2V_H3_EXPLICIT_IMAGE=1
            H3_SMOKE=0
            H3_PREFLIGHT_ONLY=0
            CONFIG_ONLY=0
            H3_FRAME_SOURCE=host
            H3_FIRST_FRAME="${h3_first_frame}"
            H3_LAST_FRAME="${h3_last_frame}"
            H3_PROMPT_PRESET="${H3_PROMPT_PRESET:-bird-stable}"
        )
    elif [[ "${RUN_MODEL_VERSION}" == ltx25_c1 ]]; then
        run_environment+=(LIGHTX2V_LTX25_C1_EXPLICIT=1)
    fi
    start_memory_monitor "${count}" "${selected}" "${log_path}" "${memory_output_path}"
    set +e
    env "${run_environment[@]}" "${model_run}" "${model_run_mode}" 2>&1 | tee "${log_path}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if (( pipeline_status[0] != 0 )); then
        stop_memory_monitor
        echo "ERROR: ${count}-GPU inference failed; see ${log_path}" >&2
        exit "${pipeline_status[0]}"
    fi
    if (( pipeline_status[1] != 0 )); then
        stop_memory_monitor
        echo "ERROR: could not write ${log_path}" >&2
        exit "${pipeline_status[1]}"
    fi
    wait_memory_monitor
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Dry run complete; timing summary skipped."
    exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" \
    --model "${summary_model}" \
    --log-dir "${result_dir}" \
    --counts "${selected_counts}" \
    --width "${RUN_TARGET_WIDTH}" \
    --height "${RUN_TARGET_HEIGHT}" \
    --frames "${RUN_TARGET_FRAMES}" \
    --fps "${RUN_TARGET_FPS}"

echo "Timing sweep complete: ${run_tag}"
echo "Raw logs : ${result_dir}/<count>gpu.log"
echo "Videos   : ${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}/<count>gpu.mp4"
echo "CSV      : ${result_dir}/timings.csv"
echo "Table    : ${result_dir}/timings.md"
echo "Manifest : ${result_dir}/run_manifest.json"
if [[ "${memory_monitor_enabled}" == "1" ]]; then
    echo "Memory   : ${result_dir}/<count>gpu_memory.jsonl (+ .summary.json)"
fi
