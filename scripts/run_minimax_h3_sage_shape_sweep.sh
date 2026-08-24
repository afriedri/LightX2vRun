#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
single_run="${H3_SINGLE_RUN_ENTRY:-${script_dir}/run_minimax_h3_fl2av_4step_768p.sh}"
collector="${H3_SAGE_SHAPE_COLLECTOR:-${script_dir}/lib/collect_minimax_h3_sage_shapes.py}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_minimax_h3_sage_shape_sweep.sh --image IMAGE [options]

Runs the frozen 1344x768/124-frame MiniMax-H3 workload sequentially at SP4 and
SP8, then extracts only the first real Sage dispatch from every rank. The
schema-2 H3 runcollector remains disabled.

Options:
  --image IMAGE          Production gfx1201 H3 image with LightX2V/Aiter source
  --counts LIST          4,8 (default), 4, or 8
  --gpus IDS             Ordered physical GPU pool (default: 0,1,2,3,4,5,6,7)
  --result-dir DIR       Exact sweep result directory (default: timestamped)
  --model-dir DIR        Host MiniMax-H3 checkpoint directory
  --turbo-dir DIR        Host Turbo LoRA directory
  --first-frame FILE     Host first frame; must be paired with --last-frame
  --last-frame FILE      Host last frame; must be paired with --first-frame
  --prompt-preset NAME   bird-stable (default), bird-dynamic, or bird-audio
  --prompt-file FILE     Custom H3-structured prompt; overrides the preset

The first N ids from --gpus are used for SPN. For example, the default pool
uses 0,1,2,3 for SP4 and 0,1,2,3,4,5,6,7 for SP8.
EOF
}

image=
counts="${H3_SAGE_SHAPE_COUNTS:-4,8}"
gpu_pool="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
result_dir="${H3_SAGE_SHAPE_RESULT_DIR:-}"
model_dir="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
turbo_dir="${H3_TURBO_DIR:-/disk/ssd/difumodels-Dis/Minimax-h3-Turbo}"
first_frame="${H3_FIRST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png}"
last_frame="${H3_LAST_FRAME:-/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png}"
prompt_preset="${H3_PROMPT_PRESET:-bird-stable}"
prompt_file="${H3_PROMPT_FILE:-}"

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
        --image=*) image="${1#--image=}"; shift ;;
        --counts)
            (( $# >= 2 )) || { echo "ERROR: --counts requires a value" >&2; exit 2; }
            counts="$2"
            shift 2
            ;;
        --counts=*) counts="${1#--counts=}"; shift ;;
        --gpus)
            (( $# >= 2 )) || { echo "ERROR: --gpus requires a value" >&2; exit 2; }
            gpu_pool="$2"
            shift 2
            ;;
        --gpus=*) gpu_pool="${1#--gpus=}"; shift ;;
        --result-dir)
            (( $# >= 2 )) || { echo "ERROR: --result-dir requires a value" >&2; exit 2; }
            result_dir="$2"
            shift 2
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
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${image}" ]] || { echo "ERROR: --image is required" >&2; exit 2; }
[[ -f "${single_run}" ]] || { echo "ERROR: H3 launcher not found: ${single_run}" >&2; exit 1; }
[[ -f "${collector}" ]] || { echo "ERROR: Sage shape collector not found: ${collector}" >&2; exit 1; }
[[ -n "${first_frame}" && -n "${last_frame}" ]] || {
    echo "ERROR: --first-frame and --last-frame must be provided together" >&2
    exit 2
}
[[ "${gpu_pool}" =~ ^[0-9]+(,[0-9]+){0,7}$ ]] || {
    echo "ERROR: --gpus must contain up to eight comma-separated GPU ids" >&2
    exit 2
}

IFS=',' read -r -a gpu_ids <<<"${gpu_pool}"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    [[ -z "${seen_gpu_ids[${gpu_id}]:-}" ]] || {
        echo "ERROR: --gpus contains duplicate id ${gpu_id}" >&2
        exit 2
    }
    seen_gpu_ids["${gpu_id}"]=1
done

IFS=',' read -r -a selected_counts <<<"${counts}"
declare -A seen_counts=()
max_count=0
for count in "${selected_counts[@]}"; do
    case "${count}" in
        4 | 8) ;;
        *) echo "ERROR: --counts supports only 4 and 8, got ${count}" >&2; exit 2 ;;
    esac
    [[ -z "${seen_counts[${count}]:-}" ]] || {
        echo "ERROR: --counts contains duplicate SP${count}" >&2
        exit 2
    }
    seen_counts["${count}"]=1
    (( count > max_count )) && max_count="${count}"
done
(( ${#gpu_ids[@]} >= max_count )) || {
    echo "ERROR: SP${max_count} requires ${max_count} GPU ids, got ${#gpu_ids[@]}" >&2
    exit 2
}

run_tag="${H3_SAGE_SHAPE_RUN_TAG:-minimax_h3_sage_shapes_$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${result_dir}" ]]; then
    result_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}"
fi
if [[ -d "${result_dir}" && -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

python3 - "${result_dir}/run_manifest.json" "${image}" "${counts}" "${gpu_pool}" <<'PY'
import json
import pathlib
import sys

output, image, counts, gpu_pool = sys.argv[1:]
payload = {
    "kind": "minimax_h3_sage_shape_sweep",
    "image": image,
    "sp_sizes": [int(value) for value in counts.split(",")],
    "gpu_pool": [int(value) for value in gpu_pool.split(",")],
    "collector": {
        "source": "normalized runtime dispatch",
        "full_runcollector_enabled": False,
        "output": "sage_shapes.json",
    },
    "workload": {
        "height": 768,
        "width": 1344,
        "frames": 124,
        "fps": 24,
        "model_evaluations": 4,
    },
}
pathlib.Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "MiniMax-H3 Sage-only shape sweep: ${result_dir}"
echo "SP sizes: ${counts}"
echo "GPU pool: ${gpu_pool}"
echo "Full H3 collector: disabled"

collector_args=()
for count in "${selected_counts[@]}"; do
    selected="$(IFS=,; printf '%s' "${gpu_ids[*]:0:${count}}")"
    child_dir="${result_dir}/sp${count}"
    child_args=(
        --image "${image}"
        --model-dir "${model_dir}"
        --turbo-dir "${turbo_dir}"
        --first-frame "${first_frame}"
        --last-frame "${last_frame}"
        --result-dir "${child_dir}"
        --prompt-preset "${prompt_preset}"
        --gpus "${selected}"
    )
    if [[ -n "${prompt_file}" ]]; then
        child_args+=(--prompt-file "${prompt_file}")
    fi
    echo "[$(date +%H:%M:%S)] SP${count}: GPUs ${selected}"
    env LIGHTX2V_H3_RUN_COLLECTOR=0 \
        bash "${single_run}" "${child_args[@]}"
    collector_args+=(--run "${count}=${child_dir}/run.log")
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "Dry run complete; Sage shape extraction skipped."
    exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B "${collector}" \
    "${collector_args[@]}" \
    --output "${result_dir}/sage_shapes.json"

echo "Sage shape sweep complete: ${result_dir}/sage_shapes.json"
