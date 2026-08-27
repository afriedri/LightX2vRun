#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"

output_dir="${LTX25_CONTRACT_OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/ltx25-contract-$(date +%Y%m%d_%H%M%S)}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
census="${LTX25_CENSUS:-/disk/ssd/lightx2v-runtime/ltx25-day0/ltx25_weight_census.json}"
world_size=2
run_id=""
require_upsample=1

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_ltx25_contract_collector.sh [OPTIONS]

Run this with no arguments inside the LightX2V image. It automatically uses
/disk/ssd/difumodels-Dis/LTX-2.5, GPUs 0,1, a timestamped evidence directory,
and the in-image LTX-2.5 B0-reference entry. If the Day-0 census is absent it
is generated once with full SHA verification before any GPU model process.

Options override defaults only:
  --output-dir DIR    Evidence directory
  --model-root DIR    LTX-2.5 model root
  --census FILE       Existing successful Day-0 census JSON
  --world-size N      Expected denoise ranks (automatic route requires 2)
  --run-id ID         Stable collector run id
  --no-require-upsample

Advanced diagnostic use may still provide a command after --. The automatic
route is used when no command is supplied. Collection does not select or
probe a C1 backend.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --output-dir) output_dir="$2"; shift 2 ;;
        --model-root) model_root="$2"; shift 2 ;;
        --census) census="$2"; shift 2 ;;
        --world-size) world_size="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --no-require-upsample) require_upsample=0; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${world_size}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --world-size must be positive" >&2; exit 2; }
if [[ -z "${run_id}" ]]; then
    run_id="ltx25-contract-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: --run-id contains unsafe characters" >&2; exit 2; }

mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"
shopt -s nullglob
existing=(
    "${output_dir}"/ltx25_collector*.json
    "${output_dir}"/ltx25_contract_bundle*.json
)
shopt -u nullglob
(( ${#existing[@]} == 0 )) || { echo "ERROR: collector outputs already exist in ${output_dir}" >&2; exit 1; }

export LIGHTX2V_LTX25_CONTRACT_COLLECTOR=1
export LIGHTX2V_LTX25_CONTRACT_RUN_ID="${run_id}"
export LIGHTX2V_LTX25_CONTRACT_COLLECTOR_L1_OUTPUT="${output_dir}/ltx25_collector.l1.json"
export LIGHTX2V_LTX25_CONTRACT_COLLECTOR_OUTPUT="${output_dir}/ltx25_collector.rank{rank}.json"

echo "LTX-2.5 contract discovery: run_id=${run_id} output_dir=${output_dir}"
if (( $# == 0 )); then
    [[ "${world_size}" == 2 ]] || {
        echo "ERROR: the automatic LTX-2.5 B0-reference route requires --world-size 2" >&2
        exit 2
    }
    if [[ ! -f "${census}" ]]; then
        echo "LTX-2.5 Day-0 census missing; running automatic full-SHA E0/E1"
        bash "${run_root}/scripts/collector/run_ltx25_weight_census.sh" \
            --model-root "${model_root}" \
            --output-dir "$(dirname -- "${census}")" \
            --sha256
    fi
    export LTX25_MODEL_ROOT="${model_root}"
    export LTX25_CENSUS="${census}"
    export LTX25_EVIDENCE_DIR="${output_dir}"
    export LTX25_WORLD_SIZE="${world_size}"
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
    bash "${run_root}/scripts/lib/run_ltx25_in_container.sh"
else
    "$@"
fi

validator_args=(
    --output-dir "${output_dir}"
    --expected-ranks "${world_size}"
)
if [[ "${require_upsample}" == 0 ]]; then
    validator_args+=(--no-require-upsample)
fi
PYTHONDONTWRITEBYTECODE=1 python3 -B \
    "${run_root}/scripts/collector/validate_ltx25_contract_bundle.py" \
    "${validator_args[@]}"
