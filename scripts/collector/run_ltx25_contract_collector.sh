#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"

output_dir="${PWD}/ltx25-contract-$(date -u +%Y%m%dT%H%M%SZ)"
world_size=2
run_id=""
require_upsample=1

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_ltx25_contract_collector.sh \
    --output-dir /path/to/evidence \
    --world-size 2 \
    -- EXISTING_LTX25_B0_COMMAND [ARG...]

Run this inside the new LightX2V image. It enables discovery-only observation
of the existing LTX-2.5 B0 reference path, assigns one run-id to L1 and all
denoise ranks, and validates the emitted JSON files. It does not select or
probe a C1 backend.

The wrapped B0 command must execute both the L1 prompt producer and L2/L3. The
L1 CLI may omit --collector-output; the wrapper supplies it through the
collector environment.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --output-dir) output_dir="$2"; shift 2 ;;
        --world-size) world_size="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --no-require-upsample) require_upsample=0; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

(( $# > 0 )) || { echo "ERROR: an existing LTX-2.5 B0 command is required after --" >&2; exit 2; }
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
"$@"

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
