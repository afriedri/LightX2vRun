#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"

model_root="/disk/ssd/difumodels-Dis/LTX-2.5"
output_dir="${PWD}/ltx25-day0"
sha256=0
card_bytes=34208743424
dynamic_budget_gib=16
runtime_reserve_mib=512
min_margin_gib=8

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_ltx25_weight_census.sh \
    --model-root /disk/ssd/difumodels-Dis/LTX-2.5 \
    --output-dir /disk/ssd/lightx2v-runtime/ltx25-day0 \
    --sha256

Runs the no-GPU LTX-2.5 E0/E1 gates. Safetensors inspection reads headers
only. --sha256 is required for CENSUS_OK and streams each full file through a
fixed 16 MiB buffer. The command writes:

  ltx25_weight_census.json
  ltx25_weight_census.md
  ltx25_mock_memory.json
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model-root) model_root="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --sha256) sha256=1; shift ;;
        --card-bytes) card_bytes="$2"; shift 2 ;;
        --dynamic-budget-gib) dynamic_budget_gib="$2"; shift 2 ;;
        --runtime-reserve-mib) runtime_reserve_mib="$2"; shift 2 ;;
        --min-margin-gib) min_margin_gib="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p -- "${output_dir}"

census_args=(
    --model-root "${model_root}"
    --output-dir "${output_dir}"
)
if (( sha256 )); then
    census_args+=(--sha256)
fi

if ! python3 "${run_root}/scripts/collector/collect_ltx25_weight_census.py" "${census_args[@]}"; then
    echo "LTX-2.5 E0 failed; evidence was retained in ${output_dir}. E1 was not run." >&2
    exit 1
fi

python3 "${run_root}/scripts/lib/mock_ltx25_b0_memory.py" \
    --census "${output_dir}/ltx25_weight_census.json" \
    --card-bytes "${card_bytes}" \
    --dynamic-budget-gib "${dynamic_budget_gib}" \
    --runtime-reserve-mib "${runtime_reserve_mib}" \
    --min-margin-gib "${min_margin_gib}" \
    --output "${output_dir}/ltx25_mock_memory.json"

echo "LTX25_DAY0_OK output_dir=${output_dir}"
