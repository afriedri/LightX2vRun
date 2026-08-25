#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
entry="${run_root}/scripts/run_minimax_h3_fl2av_4step_768p.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_minimax_h3_sp8_supercollect.sh --image IMAGE [OPTIONS]

Runs the frozen MiniMax-H3 1344x768, 124-frame, 24 FPS, four-step workload on
exactly eight GPUs with the schema-2 runcollector and system-wide read-only
evidence enabled. All regular production launcher path/prompt/result options
are forwarded.

Options:
  --gpus IDS   Exactly eight comma-separated physical GPU ids (default: 0..7)

The wrapper preserves BF16 DiT, dynamic LoRA, Ulysses SP8, lazy B1, two CPU
block buffers, two GPU block buffers, default ROCm BLAS, and all tuned kernels.
EOF
}

gpu_devices="${H3_SUPERCOLLECT_GPUS:-0,1,2,3,4,5,6,7}"
forwarded=()
while (( $# > 0 )); do
    case "$1" in
        --help | -h)
            usage
            exit 0
            ;;
        --gpu | --gpus)
            (( $# >= 2 )) || { echo "ERROR: $1 requires a value" >&2; exit 2; }
            gpu_devices="$2"
            shift 2
            ;;
        --gpu=* | --gpus=*)
            gpu_devices="${1#*=}"
            shift
            ;;
        --smoke | --preflight-only)
            echo "ERROR: SP8 SuperCollect requires the full production workload" >&2
            exit 2
            ;;
        --collect)
            shift
            ;;
        *)
            forwarded+=("$1")
            shift
            ;;
    esac
done

[[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+){7}$ ]] || {
    echo "ERROR: SP8 SuperCollect requires exactly eight comma-separated GPU ids" >&2
    exit 2
}
IFS=',' read -r -a gpu_ids <<<"${gpu_devices}"
declare -A seen=()
for gpu_id in "${gpu_ids[@]}"; do
    [[ -z "${seen[${gpu_id}]:-}" ]] || {
        echo "ERROR: SP8 SuperCollect GPU ids must be unique" >&2
        exit 2
    }
    seen["${gpu_id}"]=1
done

export H3_SUPERCOLLECT=1
export MEMORY_MONITOR=1
export MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-1.0}"
export MEMORY_MONITOR_PRINT_INTERVAL="${MEMORY_MONITOR_PRINT_INTERVAL:-15.0}"
export H3_RUN_TAG="${H3_RUN_TAG:-minimax_h3_sp8_supercollect_$(date +%Y%m%d_%H%M%S)}"

exec bash "${entry}" "${forwarded[@]}" --collect --gpus "${gpu_devices}"
