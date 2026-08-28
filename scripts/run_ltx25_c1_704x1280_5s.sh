#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${run_root}/lightx2vModelrun}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_ltx25_c1_704x1280_5s.sh \
    --image IMAGE [OPTIONS]

Runs the fixed LTX-2.5 704x1280 / 121-frame / 24-FPS SP2 workload with the
currently promoted C1 operator families. B0 CollectorUltra is not modified.

Options:
  --image IMAGE             ROCm gfx1201 image
  --tunableop-cache FILE    Optional host override for the cache baked into IMAGE
  --conv3d-csv FILE         Optional HotCraft Conv3D tuned CSV
  --lightx2v-source DIR     Optional current LightX2V checkout mounted read-only
  --model-root DIR          LTX-2.5 split asset root
  --result-dir DIR          Empty result directory
  --gpus IDS                Exactly two physical GPU ids (default: 0,1)
  --prompt TEXT             T2AV prompt
  --no-package              Do not create result-dir.tar.gz

Use --lightx2v-source during source iteration; omit it only when the image
already contains the exact LightX2V revision being tested.
EOF
}

image="${IMAGE:-}"
tunableop_cache="${LTX25_TUNABLEOP_CACHE:-}"
conv3d_csv="${CK_CONV3D_TUNED_CSV:-}"
lightx2v_source="${LTX25_LIGHTX2V_SOURCE:-}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
result_dir="${LTX25_C1_RESULT_DIR:-}"
gpu_devices="${GPU_DEVICES:-0,1}"
prompt="${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
package_result=1

while (( $# > 0 )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --image) (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }; image="$2"; shift 2 ;;
        --image=*) image="${1#--image=}"; shift ;;
        --tunableop-cache) (( $# >= 2 )) || { echo "ERROR: --tunableop-cache requires a value" >&2; exit 2; }; tunableop_cache="$2"; shift 2 ;;
        --conv3d-csv) (( $# >= 2 )) || { echo "ERROR: --conv3d-csv requires a value" >&2; exit 2; }; conv3d_csv="$2"; shift 2 ;;
        --lightx2v-source) (( $# >= 2 )) || { echo "ERROR: --lightx2v-source requires a value" >&2; exit 2; }; lightx2v_source="$2"; shift 2 ;;
        --model-root) (( $# >= 2 )) || { echo "ERROR: --model-root requires a value" >&2; exit 2; }; model_root="$2"; shift 2 ;;
        --result-dir) (( $# >= 2 )) || { echo "ERROR: --result-dir requires a value" >&2; exit 2; }; result_dir="$2"; shift 2 ;;
        --gpus|--gpu) (( $# >= 2 )) || { echo "ERROR: $1 requires a value" >&2; exit 2; }; gpu_devices="$2"; shift 2 ;;
        --prompt) (( $# >= 2 )) || { echo "ERROR: --prompt requires a value" >&2; exit 2; }; prompt="$2"; shift 2 ;;
        --no-package) package_result=0; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${image}" ]] || { echo "ERROR: --image is required" >&2; exit 2; }
[[ "${gpu_devices}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "ERROR: --gpus requires exactly two ids" >&2; exit 2; }
IFS=',' read -r gpu0 gpu1 <<<"${gpu_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: GPU ids must be unique" >&2; exit 2; }

if [[ -z "${result_dir}" ]]; then
    result_dir="/disk/ssd/lightx2v-runtime/ltx25-c1-$(date +%Y%m%d_%H%M%S)"
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -z "${tunableop_cache}" || -f "${tunableop_cache}" ]] || { echo "ERROR: TunableOp cache not found: ${tunableop_cache}" >&2; exit 1; }
    [[ -d "${model_root}" ]] || { echo "ERROR: model root not found: ${model_root}" >&2; exit 1; }
    [[ -z "${conv3d_csv}" || -f "${conv3d_csv}" ]] || { echo "ERROR: Conv3D CSV not found: ${conv3d_csv}" >&2; exit 1; }
    [[ -z "${lightx2v_source}" || -d "${lightx2v_source}/lightx2v" ]] || { echo "ERROR: LightX2V source is invalid: ${lightx2v_source}" >&2; exit 1; }
fi
if [[ -d "${result_dir}" && -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 2
fi
mkdir -p -- "${result_dir}"

export IMAGE="${image}"
export GPU_DEVICES="${gpu_devices}"
export OUTPUT_DIR="${result_dir}"
export LTX25_C1_RESULT_DIR="${result_dir}"
export LTX25_MODEL_ROOT="${model_root}"
export LIGHTX2V_LTX25_C1_EXPLICIT=1
export PROMPT="${prompt}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-ltx25-c1}"
if [[ -n "${conv3d_csv}" ]]; then export CK_CONV3D_TUNED_CSV="${conv3d_csv}"; fi
if [[ -n "${lightx2v_source}" ]]; then export LTX25_LIGHTX2V_SOURCE="${lightx2v_source}"; fi
if [[ -n "${tunableop_cache}" ]]; then export LTX25_TUNABLEOP_CACHE="${tunableop_cache}"; else unset LTX25_TUNABLEOP_CACHE; fi

set +e
"${model_run}" --run-ltx25-c1-gfx1201 2>&1 | tee "${result_dir}/run.log"
status=${PIPESTATUS[0]}
set -e
(( status == 0 )) || { echo "ERROR: LTX-2.5 C1 run failed; results remain in ${result_dir}" >&2; exit "${status}"; }

if [[ "${DRY_RUN:-0}" != 1 && "${package_result}" == 1 ]]; then
    tar -czf "${result_dir}.tar.gz" -C "$(dirname "${result_dir}")" "$(basename "${result_dir}")"
    echo "LTX-2.5 C1 result package: ${result_dir}.tar.gz"
fi
