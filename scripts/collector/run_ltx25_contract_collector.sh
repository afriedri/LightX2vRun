#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
model_run="${MODEL_RUN:-${run_root}/lightx2vModelrun}"

output_dir="${LTX25_CONTRACT_OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/ltx25-contract-$(date +%Y%m%d_%H%M%S)}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
gpu_devices="${GPU_DEVICES:-${HIP_VISIBLE_DEVICES:-0,1}}"
image="${LTX25_IMAGE:-${IMAGE:-}}"
run_id="${LIGHTX2V_LTX25_CONTRACT_RUN_ID:-}"
require_upsample="${LTX25_REQUIRE_UPSAMPLE:-1}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_ltx25_contract_collector.sh [OPTIONS]

Run this with no arguments on the Docker host. It launches the standard
LightX2V ROCm image and performs L1 prompt collection, SP2 L2/L3 collection,
and bundle validation
inside that image. The default image comes from lightx2vModelrun.

Options override defaults only:
  --image IMAGE        Runtime image override
  --output-dir DIR     Host evidence directory
  --model-root DIR     Host LTX-2.5 model root
  --gpus IDS           Exactly two physical GPU ids (default: 0,1)
  --run-id ID          Stable collector run id
  --no-require-upsample

Environment:
  DRY_RUN=1            Print the complete Docker command without starting it

Collection does not select or probe a C1 backend. A complete bundle remains
B0-reference evidence and keeps b0_accepted=false and c1_pass=false.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --image) image="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --model-root) model_root="$2"; shift 2 ;;
        --gpus|--gpu) gpu_devices="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --no-require-upsample) require_upsample=0; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${gpu_devices}" =~ ^[0-9]+,[0-9]+$ ]] || {
    echo "ERROR: --gpus requires exactly two comma-separated ids" >&2
    exit 2
}
IFS=',' read -r gpu0 gpu1 <<<"${gpu_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: --gpus ids must be unique" >&2; exit 2; }
case "${require_upsample}" in 0|1) ;; *) echo "ERROR: LTX25_REQUIRE_UPSAMPLE must be 0 or 1" >&2; exit 2 ;; esac
if [[ -z "${run_id}" ]]; then
    run_id="ltx25-contract-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: --run-id contains unsafe characters" >&2; exit 2; }
[[ "${output_dir}" == /* ]] || { echo "ERROR: --output-dir must be an absolute host path" >&2; exit 2; }
[[ "${model_root}" == /* ]] || { echo "ERROR: --model-root must be an absolute host path" >&2; exit 2; }

mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"
model_root="$(cd -- "$(dirname -- "${model_root}")" && pwd -P)/$(basename -- "${model_root}")"

shopt -s nullglob
existing=(
    "${output_dir}"/ltx25_collector*.json
    "${output_dir}"/ltx25_contract_bundle*.json
)
shopt -u nullglob
(( ${#existing[@]} == 0 )) || { echo "ERROR: collector outputs already exist in ${output_dir}" >&2; exit 1; }
[[ -f "${model_run}" ]] || { echo "ERROR: container launcher not found: ${model_run}" >&2; exit 1; }

export RUN_REPO_DIR="${run_root}"
export OUTPUT_DIR="${output_dir}"
export GPU_DEVICES="${gpu_devices}"
export LTX25_MODEL_ROOT="${model_root}"
export LIGHTX2V_LTX25_EXPLICIT_COLLECTOR=1
export LIGHTX2V_LTX25_CONTRACT_RUN_ID="${run_id}"
export LTX25_REQUIRE_UPSAMPLE="${require_upsample}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-ltx25-contract}"
if [[ -n "${image}" ]]; then
    export IMAGE="${image}"
fi

echo "LTX-2.5 contract discovery: run_id=${run_id} output_dir=${output_dir}"
echo "LTX-2.5 container launch: GPUs=${gpu_devices} model=${model_root}"
set +e
"${model_run}" --run-ltx25-contract-collector 2>&1 | tee "${output_dir}/run.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if (( pipeline_status[0] != 0 )); then
    echo "ERROR: LTX-2.5 contract collection failed; evidence remains in ${output_dir}" >&2
    exit "${pipeline_status[0]}"
fi
if (( pipeline_status[1] != 0 )); then
    echo "ERROR: could not write ${output_dir}/run.log" >&2
    exit "${pipeline_status[1]}"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "LTX-2.5 contract collector dry run complete: ${output_dir}"
    exit 0
fi

bundle="${output_dir}/ltx25_contract_bundle.json"
[[ -s "${bundle}" ]] || { echo "ERROR: container did not write ${bundle}" >&2; exit 1; }
echo "LTX-2.5 contract bundle: ${bundle}"
