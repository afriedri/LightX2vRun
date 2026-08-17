#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${run_root}/lightx2vModelrun}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_minimax_h3_fl2av_4step_768p_day0.sh --image IMAGE \
      --model-dir DIR --turbo-dir DIR --first-frame FILE --last-frame FILE

Runs MiniMax-H3 FL2V on one gfx1201 GPU or two gfx1201 GPUs with pure Ulysses
sequence parallelism. --image is required and must contain
the complete Day0 LightX2V source at /workspace/LightX2V. Host source overlays
are deliberately unsupported; rebuild the image before C0.

Options:
  --image IMAGE          Rebuilt gfx1201 H3 image containing Day0 source
  --model-dir DIR        Host MiniMax-H3 checkpoint directory
  --turbo-dir DIR        Host directory containing the Turbo LoRA
  --first-frame FILE     Host first-frame image
  --last-frame FILE      Host last-frame image
  --result-dir DIR       Exact host result directory (default: timestamped)
  --smoke                480x480 / 124 frames instead of 1344x768 / 362
  --preflight-only       Run per-GPU MHA and SP2 RCCL/Ulysses smoke, then exit
  --gpus IDS             One GPU id or two comma-separated ids (default: 0)
  --gpu IDS              Backward-compatible alias for --gpus

Environment:
  PROMPT / SEED          Frozen request values (defaults: built-in prompt / 42)
  GPU_DEVICES            One id for single GPU or two ids for Ulysses SP2
  MEMORY_MONITOR         1 (default) records host RAM, process RSS/PSS and VRAM
EOF
}

image=
model_dir="${H3_MODEL_DIR:-}"
turbo_dir="${H3_TURBO_DIR:-}"
first_frame="${H3_FIRST_FRAME:-}"
last_frame="${H3_LAST_FRAME:-}"
gpu_devices="${GPU_DEVICES:-0}"
result_dir="${H3_RESULT_DIR:-}"
smoke=0
preflight_only=0

while (( $# > 0 )); do
    case "$1" in
        --help | -h)
            usage
            exit 0
            ;;
        --image)
            (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }
            image="$2"
            shift 2
            ;;
        --image=*)
            image="${1#--image=}"
            shift
            ;;
        --model-dir)
            (( $# >= 2 )) || { echo "ERROR: --model-dir requires a value" >&2; exit 2; }
            model_dir="$2"
            shift 2
            ;;
        --turbo-dir)
            (( $# >= 2 )) || { echo "ERROR: --turbo-dir requires a value" >&2; exit 2; }
            turbo_dir="$2"
            shift 2
            ;;
        --first-frame)
            (( $# >= 2 )) || { echo "ERROR: --first-frame requires a value" >&2; exit 2; }
            first_frame="$2"
            shift 2
            ;;
        --last-frame)
            (( $# >= 2 )) || { echo "ERROR: --last-frame requires a value" >&2; exit 2; }
            last_frame="$2"
            shift 2
            ;;
        --result-dir)
            (( $# >= 2 )) || { echo "ERROR: --result-dir requires a value" >&2; exit 2; }
            result_dir="$2"
            shift 2
            ;;
        --gpu | --gpus)
            (( $# >= 2 )) || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            gpu_devices="$2"
            shift 2
            ;;
        --smoke)
            smoke=1
            shift
            ;;
        --preflight-only)
            preflight_only=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${image}" ]] || { echo "ERROR: MiniMax-H3 requires --image; do not fall back to the Wan default image" >&2; exit 2; }
[[ -n "${model_dir}" && -n "${turbo_dir}" && -n "${first_frame}" && -n "${last_frame}" ]] || {
    echo "ERROR: --model-dir, --turbo-dir, --first-frame, and --last-frame are required" >&2
    exit 2
}
[[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+)?$ ]] || {
    echo "ERROR: MiniMax-H3 Day0 requires one GPU id or two comma-separated ids" >&2
    exit 2
}
IFS=',' read -r -a gpu_device_ids <<<"${gpu_devices}"
gpu_count="${#gpu_device_ids[@]}"
if [[ "${gpu_count}" == 2 && "${gpu_device_ids[0]}" == "${gpu_device_ids[1]}" ]]; then
    echo "ERROR: MiniMax-H3 GPU ids must be unique" >&2
    exit 2
fi

memory_monitor_enabled="${MEMORY_MONITOR:-1}"
case "${memory_monitor_enabled}" in
    0 | 1) ;;
    *) echo "ERROR: MEMORY_MONITOR must be 0 or 1" >&2; exit 2 ;;
esac
run_tag="${H3_RUN_TAG:-minimax_h3_fl2av_4step_768p_day0_$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${result_dir}" ]]; then
    result_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}"
fi
if [[ -d "${result_dir}" && -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

readonly source_mode=image
default_prompt="A cinematic tracking shot between the first and last frames, natural motion, photorealistic lighting."
prompt="${PROMPT:-${default_prompt}}"
seed="${SEED:-42}"
[[ "${seed}" =~ ^-?[0-9]+$ ]] || { echo "ERROR: SEED must be an integer" >&2; exit 2; }
log_path="${result_dir}/run.log"
memory_path="${result_dir}/memory.jsonl"
video_path="${result_dir}/minimax_h3_fl2av.mp4"
summary_path="${result_dir}/summary.json"
container_evidence_dir=/workspace/LightX2V/save_results

python3 - "${result_dir}/run_manifest.json" "${image}" "${source_mode}" \
    "${model_dir}" "${turbo_dir}" "${first_frame}" "${last_frame}" \
    "${gpu_devices}" "${smoke}" "${preflight_only}" "${prompt}" "${seed}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output,
    image,
    source_mode,
    model_dir,
    turbo_dir,
    first_frame,
    last_frame,
    gpu_devices,
    smoke,
    preflight_only,
    prompt,
    seed,
) = sys.argv[1:]

def sha256(path):
    candidate = pathlib.Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

gpus = [int(item) for item in gpu_devices.split(",")]
payload = {
    "kind": "minimax_h3_gfx1201_day0",
    "task": "fl2av",
    "image": image,
    "source_mode": source_mode,
    "model_dir": model_dir,
    "turbo_dir": turbo_dir,
    "turbo": {
        "path": str(pathlib.Path(turbo_dir) / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"),
        "sha256": sha256(pathlib.Path(turbo_dir) / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"),
    },
    "first_frame": {"path": first_frame, "sha256": sha256(first_frame)},
    "last_frame": {"path": last_frame, "sha256": sha256(last_frame)},
    "gpu": gpus[0],
    "gpus": gpus,
    "gpu_count": len(gpus),
    "parallel": {
        "tensor_parallel": False,
        "seq_parallel": len(gpus) == 2,
        "seq_p_size": len(gpus),
        "seq_p_attn_type": "ulysses" if len(gpus) == 2 else None,
        "vae_decode_parallel": len(gpus) == 2,
    },
    "smoke": smoke == "1",
    "preflight_only": preflight_only == "1",
    "seed": int(seed),
    "prompt": prompt,
    "workload": {
        "height": 480 if smoke == "1" else 768,
        "width": 480 if smoke == "1" else 1344,
        "frames": 124 if smoke == "1" else 362,
        "fps": 24,
        "model_evaluations": 4,
    },
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

export IMAGE="${image}"
export LIGHTX2V_H3_EXPLICIT_IMAGE=1
unset LIGHTX2V_DIR
export GPU_DEVICES="${gpu_devices}"
export H3_MODEL_DIR="${model_dir}"
export H3_TURBO_DIR="${turbo_dir}"
export H3_FIRST_FRAME="${first_frame}"
export H3_LAST_FRAME="${last_frame}"
export H3_SMOKE="${smoke}"
export H3_PREFLIGHT_ONLY="${preflight_only}"
export PROMPT="${prompt}"
export SEED="${seed}"
export H3_EVIDENCE_DIR="${container_evidence_dir}"
export SAVE_RESULT_PATH="${container_evidence_dir}/minimax_h3_fl2av.mp4"
export OUTPUT_DIR="${result_dir}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-minimax-h3}"

monitor_pid=
stop_monitor() {
    if [[ -n "${monitor_pid:-}" ]]; then
        if kill -0 "${monitor_pid}" 2>/dev/null; then
            kill "${monitor_pid}" 2>/dev/null || true
        fi
        wait "${monitor_pid}" 2>/dev/null || true
        monitor_pid=
    fi
}
wait_monitor() {
    local status=0
    if [[ -z "${monitor_pid:-}" ]]; then
        return
    fi
    set +e
    wait "${monitor_pid}"
    status=$?
    set -e
    monitor_pid=
    if (( status != 0 )); then
        echo "ERROR: memory monitor failed with status ${status}" >&2
        return "${status}"
    fi
}
trap stop_monitor EXIT

if [[ "${memory_monitor_enabled}" == 1 && "${DRY_RUN:-0}" != 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${run_root}/scripts/lib/monitor_run_memory.py" \
        --container "${CONTAINER_NAME}" \
        --log "${log_path}" \
        --output "${memory_path}" \
        --interval "${MEMORY_MONITOR_INTERVAL:-2.0}" \
        --print-interval "${MEMORY_MONITOR_PRINT_INTERVAL:-30.0}" \
        --start-timeout "${MEMORY_MONITOR_START_TIMEOUT:-900}" \
        --expected-ranks "${gpu_count}" \
        --gpu-indices "${gpu_devices}" \
        --host-limit-gib "${MEMORY_MONITOR_HOST_LIMIT_GIB:-512}" \
        --gpu-limit-gib "${MEMORY_MONITOR_GPU_LIMIT_GIB:-32}" \
        --phase-schema minimax_h3 &
    monitor_pid=$!
fi

echo "MiniMax-H3 Day0 results: ${result_dir}"
echo "Source mode: ${source_mode}"
set +e
"${model_run}" --run-minimax-h3-gfx1201 2>&1 | tee "${log_path}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if (( pipeline_status[0] != 0 )); then
    stop_monitor
    echo "ERROR: MiniMax-H3 run failed; evidence remains in ${result_dir}" >&2
    exit "${pipeline_status[0]}"
fi
if (( pipeline_status[1] != 0 )); then
    stop_monitor
    echo "ERROR: could not write ${log_path}" >&2
    exit "${pipeline_status[1]}"
fi
wait_monitor

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Dry run complete: ${result_dir}"
    exit 0
fi
if [[ "${preflight_only}" == 1 ]]; then
    echo "C0 preflight complete: ${result_dir}/preflight.json"
    exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B "${run_root}/scripts/lib/summarize_minimax_h3_run.py" \
    --log "${log_path}" \
    --video "${video_path}" \
    --probe-json "${result_dir}/ffprobe.json" \
    --output-json "${summary_path}"

echo "MiniMax-H3 Day0 complete: ${result_dir}"
echo "Video: ${video_path}"
echo "Summary: ${summary_path}"
if [[ "${memory_monitor_enabled}" == 1 ]]; then
    echo "Memory: ${memory_path} (+ .summary.json)"
fi
