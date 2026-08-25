#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
entry="${run_root}/scripts/run_minimax_h3_fl2av_4step_768p.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_minimax_h3_megadiscovery.sh --image IMAGE [OPTIONS]

Runs one frozen MiniMax-H3 production workload with the schema-2 runtime
collector, exact-shape collective microbenchmarks, RCCL initialization logs,
host/KFD/PCIe/NUMA snapshots, process/cgroup/VRAM monitoring, and a unified
JSON/Markdown diagnosis. The default is Ring16; SP8/Ulysses is also supported.

Options:
  --gpus IDS   Exactly 8 or 16 ordered physical GPU ids (default: 0..15)

All regular production path, prompt, and result options are forwarded. Smoke
and preflight-only runs are rejected because they do not expose the production
communication contract.

MegaDiscovery does not change BF16, dynamic LoRA, lazy B1, two CPU buffers,
two GPU buffers, TunableOp, Sage kernels, Ulysses SP8, or Ring16.
EOF
}

gpu_devices="${H3_MEGADISCOVERY_GPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
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
            echo "ERROR: MegaDiscovery requires the full production workload" >&2
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

[[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+){7}$ || "${gpu_devices}" =~ ^[0-9]+(,[0-9]+){15}$ ]] || {
    echo "ERROR: MegaDiscovery requires exactly eight or sixteen comma-separated GPU ids" >&2
    exit 2
}
IFS=',' read -r -a gpu_ids <<<"${gpu_devices}"
declare -A seen=()
for gpu_id in "${gpu_ids[@]}"; do
    [[ -z "${seen[${gpu_id}]:-}" ]] || {
        echo "ERROR: MegaDiscovery GPU ids must be unique" >&2
        exit 2
    }
    seen["${gpu_id}"]=1
done

gpu_count="${#gpu_ids[@]}"
profile="ring16"
if (( gpu_count == 8 )); then
    profile="sp8"
fi

export H3_MEGADISCOVERY=1
export MEMORY_MONITOR=1
export MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-0.5}"
export MEMORY_MONITOR_PRINT_INTERVAL="${MEMORY_MONITOR_PRINT_INTERVAL:-15.0}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-minimax-h3-megadiscovery}"
export H3_RUN_TAG="${H3_RUN_TAG:-minimax_h3_${profile}_megadiscovery_$(date +%Y%m%d_%H%M%S)}"

exec bash "${entry}" "${forwarded[@]}" --collect --gpus "${gpu_devices}"
