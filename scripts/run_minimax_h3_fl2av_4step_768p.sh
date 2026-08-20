#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${run_root}/lightx2vModelrun}"
source "${run_root}/scripts/lib/minimax_h3_prompt_presets.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_minimax_h3_fl2av_4step_768p.sh --image IMAGE

Runs MiniMax-H3 FL2V on one, two, or four gfx1201 GPUs. Multi-GPU runs use pure
Ulysses sequence parallelism. --image is required and must contain
the complete production LightX2V source at /workspace/LightX2V. Host source overlays
are deliberately unsupported; rebuild the image before production runs.

Options:
  --image IMAGE          Rebuilt gfx1201 H3 image containing production source
  --model-dir DIR        Host MiniMax-H3 checkpoint directory
                         (default: /disk/ssd1/difumodels/MiniMax-h3)
  --turbo-dir DIR        Host directory containing the Turbo LoRA
                         (default: /disk/ssd/difumodels-Dis/Minimax-h3-Turbo)
  --first-frame FILE     Custom host first-frame image; must be used with --last-frame
  --last-frame FILE      Custom host last-frame image; must be used with --first-frame
                         (defaults: /disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png
                                    /disk/ssd/lightx2v-runtime/inputs/END_FL2V.png)
  --result-dir DIR       Exact host result directory (default: timestamped)
  --prompt-preset NAME   bird-stable (default), bird-dynamic, or bird-audio
  --prompt-file FILE     Custom H3-structured prompt; overrides the preset
  --smoke                Use 480x480 instead of 1344x768; both run 124 frames
  --preflight-only       Run per-GPU MHA and RCCL/Ulysses smoke, then exit
  --collect              Record schema-2 H3 T1 lifecycle/operator evidence
  --gpus IDS             One, two, or four comma-separated GPU ids
                         (default: 0,1,2,3; four-GPU pure Ulysses SP4)
  --gpu IDS              Backward-compatible alias for --gpus

Environment:
  PROMPT / SEED          PROMPT overrides preset/file; seed defaults to 42
  H3_PROMPT_PRESET       Default prompt preset (default: bird-stable)
  H3_PROMPT_FILE         Custom prompt file, overridden by PROMPT
  GPU_DEVICES            One, two, or four ids; multi-GPU uses pure Ulysses
  H3_MODEL_DIR           Override the default complete H3 checkpoint directory
  H3_TURBO_DIR           Override the default Turbo LoRA directory
  MEMORY_MONITOR         1 (default) records host RAM, process RSS/PSS and VRAM
  LIGHTX2V_H3_RUN_COLLECTOR
                         Prefer --collect; direct environment use accepts 0/1
EOF
}

image=
model_dir="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
turbo_dir="${H3_TURBO_DIR:-/disk/ssd/difumodels-Dis/Minimax-h3-Turbo}"
first_frame="${H3_FIRST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png}"
last_frame="${H3_LAST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png}"
frame_source=host
gpu_devices="${GPU_DEVICES:-0,1,2,3}"
result_dir="${H3_RESULT_DIR:-}"
prompt_preset="${H3_PROMPT_PRESET:-bird-stable}"
prompt_file="${H3_PROMPT_FILE:-}"
smoke=0
preflight_only=0
collect="${LIGHTX2V_H3_RUN_COLLECTOR:-0}"

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
        --prompt-preset)
            (( $# >= 2 )) || { echo "ERROR: --prompt-preset requires a value" >&2; exit 2; }
            prompt_preset="$2"
            shift 2
            ;;
        --prompt-file)
            (( $# >= 2 )) || { echo "ERROR: --prompt-file requires a value" >&2; exit 2; }
            prompt_file="$2"
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
        --collect)
            collect=1
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
if [[ -z "${first_frame}" || -z "${last_frame}" ]]; then
    echo "ERROR: --first-frame and --last-frame must be provided together" >&2
    exit 2
fi
[[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+){0,3}$ ]] || {
    echo "ERROR: MiniMax-H3 gfx1201 requires one, two, or four comma-separated GPU ids" >&2
    exit 2
}
IFS=',' read -r -a gpu_device_ids <<<"${gpu_devices}"
gpu_count="${#gpu_device_ids[@]}"
case "${gpu_count}" in
    1 | 2 | 4) ;;
    *) echo "ERROR: MiniMax-H3 gfx1201 supports exactly 1, 2, or 4 GPUs" >&2; exit 2 ;;
esac
declare -A seen_gpu_device_ids=()
for gpu_device_id in "${gpu_device_ids[@]}"; do
    [[ -z "${seen_gpu_device_ids[${gpu_device_id}]:-}" ]] || {
        echo "ERROR: MiniMax-H3 GPU ids must be unique" >&2
        exit 2
    }
    seen_gpu_device_ids["${gpu_device_id}"]=1
done

memory_monitor_enabled="${MEMORY_MONITOR:-1}"
case "${memory_monitor_enabled}" in
    0 | 1) ;;
    *) echo "ERROR: MEMORY_MONITOR must be 0 or 1" >&2; exit 2 ;;
esac
case "${collect}" in
    0 | 1) ;;
    *) echo "ERROR: LIGHTX2V_H3_RUN_COLLECTOR must be 0 or 1" >&2; exit 2 ;;
esac
prompt_duration_seconds=5.17
resolve_minimax_h3_prompt "${run_root}" "${prompt_preset}" "${prompt_file}" "${prompt_duration_seconds}" || exit $?
prompt="${MINIMAX_H3_PROMPT}"
prompt_source="${MINIMAX_H3_PROMPT_SOURCE}"
run_tag="${H3_RUN_TAG:-minimax_h3_fl2av_4step_768p_$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${result_dir}" ]]; then
    result_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}"
fi
if [[ -d "${result_dir}" && -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

readonly source_mode=image
seed="${SEED:-42}"
[[ "${seed}" =~ ^-?[0-9]+$ ]] || { echo "ERROR: SEED must be an integer" >&2; exit 2; }
log_path="${result_dir}/run.log"
memory_path="${result_dir}/memory.jsonl"
video_path="${result_dir}/minimax_h3_fl2av.mp4"
summary_path="${result_dir}/summary.json"
container_evidence_dir=/workspace/LightX2V/save_results

python3 - "${result_dir}/run_manifest.json" "${image}" "${source_mode}" \
    "${model_dir}" "${turbo_dir}" "${first_frame}" "${last_frame}" \
    "${frame_source}" "${gpu_devices}" "${smoke}" "${preflight_only}" \
    "${prompt}" "${prompt_source}" "${prompt_preset}" "${seed}" "${collect}" <<'PY'
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
    frame_source,
    gpu_devices,
    smoke,
    preflight_only,
    prompt,
    prompt_source,
    prompt_preset,
    seed,
    collect,
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
    "kind": "minimax_h3_gfx1201_production",
    "task": "fl2av",
    "image": image,
    "source_mode": source_mode,
    "model_dir": model_dir,
    "turbo_dir": turbo_dir,
    "turbo": {
        "path": str(pathlib.Path(turbo_dir) / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"),
        "sha256": sha256(pathlib.Path(turbo_dir) / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"),
    },
    "first_frame": {"path": first_frame, "source": frame_source, "sha256": sha256(first_frame)},
    "last_frame": {"path": last_frame, "source": frame_source, "sha256": sha256(last_frame)},
    "gpu": gpus[0],
    "gpus": gpus,
    "gpu_count": len(gpus),
    "parallel": {
        "tensor_parallel": False,
        "seq_parallel": len(gpus) > 1,
        "seq_p_size": len(gpus),
        "seq_p_attn_type": "ulysses" if len(gpus) > 1 else None,
        "vae_decode_parallel": len(gpus) > 1,
    },
    "backends": {
        "h1_patch_embed": {
            "name": "torch_linear",
            "frontend_api": "torch.nn.functional.linear",
            "aten_operator": "addmm",
            "measured_device_backend": "hipBLASLt",
            "matrix_mkn_per_rank": [1800 if smoke == "1" else 8064, 1536, 1152],
            "calls_per_h1": 1,
            "sequence_sharded": False,
            "tunableop_enabled": False,
        },
        "h3_core_attention": {
            "name": "aiter_fav3_sage_bf16_attn",
            "route": "sage_attn_v2_gfx1201",
            "numerics": "int8_qk_fp8_pv",
        },
        "h3_refiner_attention": "aiter_triton_mha",
        "h3_core_qk_rms": "minimax_h3_aiter_triton_qk",
        "h3_core_rope": "minimax_h3_triton_strict_rope",
        "h4_video_attention": "aiter_triton_mha",
        "dense_gemm_lora": {
            "route": "torch.mm -> default ROCm BLAS heuristic",
            "measured_backend": "hipBLASLt",
            "tunableop_enabled": False,
            "tunableop_cache": False,
        },
    },
    "smoke": smoke == "1",
    "preflight_only": preflight_only == "1",
    "runcollector": {
        "enabled": collect == "1",
        "schema_name": "lightx2v.minimax_h3.runcollector",
        "schema_version": 2,
        "mode": "h3_t1_lifecycle",
        "output_pattern": "h3_runcollector.rank{rank}.json",
        "cross_rank_summary": "h3_t1_summary.json",
        "profiling_only": True,
        "numerical_path_changed": False,
    },
    "seed": int(seed),
    "prompt": prompt,
    "prompt_source": prompt_source,
    "prompt_preset": prompt_preset if prompt_source.startswith("preset:") else None,
    "workload": {
        "height": 480 if smoke == "1" else 768,
        "width": 480 if smoke == "1" else 1344,
        "frames": 124,
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
export H3_FRAME_SOURCE="${frame_source}"
export H3_SMOKE="${smoke}"
export H3_PREFLIGHT_ONLY="${preflight_only}"
export LIGHTX2V_H3_RUN_COLLECTOR="${collect}"
export LIGHTX2V_H3_RUN_COLLECTOR_OUTPUT="${container_evidence_dir}/h3_runcollector.rank{rank}.json"
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

echo "MiniMax-H3 gfx1201 results: ${result_dir}"
echo "Source mode: ${source_mode}"
echo "Prompt: ${prompt_source} (${prompt_duration_seconds}s alignment)"
if [[ "${collect}" == 1 ]]; then
    echo "H3 T1 collector: ${result_dir}/h3_runcollector.rankNNN.json (schema-2, profiling only)"
fi
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
    echo "gfx1201 production preflight complete: ${result_dir}/preflight.json"
    exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B "${run_root}/scripts/lib/summarize_minimax_h3_run.py" \
    --log "${log_path}" \
    --video "${video_path}" \
    --probe-json "${result_dir}/ffprobe.json" \
    --output-json "${summary_path}"

if [[ "${collect}" == 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${run_root}/scripts/lib/summarize_minimax_h3_t1.py" \
        --result-dir "${result_dir}" \
        --expected-ranks "${gpu_count}"
fi

echo "MiniMax-H3 gfx1201 complete: ${result_dir}"
echo "Video: ${video_path}"
echo "Summary: ${summary_path}"
if [[ "${memory_monitor_enabled}" == 1 ]]; then
    echo "Memory: ${memory_path} (+ .summary.json)"
fi
if [[ "${collect}" == 1 ]]; then
    echo "Runcollector: ${result_dir}/h3_runcollector.rankNNN.json"
    echo "T1 summary: ${result_dir}/h3_t1_summary.json"
fi
