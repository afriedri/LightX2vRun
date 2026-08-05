#!/usr/bin/env bash
# Two-card Wan2.1 VAE A/B: PyTorch offload versus prebuilt CK v1 resident.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_repo_dir="$(cd -- "${script_dir}/.." && pwd)"

if [[ -n "${LIGHTX2V_DIR:-}" ]]; then
    lightx2v_dir="${LIGHTX2V_DIR}"
else
    candidates=(
        "${run_repo_dir}/../LightX2V-ROCm-GFX1201/LightX2V"
        "${run_repo_dir}/../LightX2V"
        "/workspace/LightX2V"
    )
    lightx2v_dir=""
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            lightx2v_dir="${candidate}"
            break
        fi
    done
fi

[[ -n "${lightx2v_dir}" && -d "${lightx2v_dir}" ]] || {
    echo "ERROR: LightX2V checkout not found; set LIGHTX2V_DIR" >&2
    exit 1
}
lightx2v_dir="$(cd -- "${lightx2v_dir}" && pwd)"
launcher="${lightx2v_dir}/scripts/platforms/amd_rocm/run_ck_conv3d_v1_runtune.sh"
[[ -f "${launcher}" ]] || {
    echo "ERROR: CK Conv3d v1 launcher not found: ${launcher}" >&2
    exit 1
}

runtime_root="${RUNTIME_ROOT:-/disk/ssd/lightx2v-runtime}"
output_dir="${OUTPUT_DIR:-${runtime_root}/outputs}"
result_dir="${RESULT_DIR:-${output_dir}/ck_conv3d_v1_runtune_$(date +%Y%m%d_%H%M%S)}"
legacy_image="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260804-SageUtimateWithMM-T1"
ck_aot_image="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260805-CKConv3dV1-AOT"
if [[ -z "${IMAGE:-}" || "${IMAGE}" == "${legacy_image}" ]]; then
    IMAGE="${ck_aot_image}"
fi

export LIGHTX2V_DIR="${lightx2v_dir}"
export GPU_IDS="${GPU_IDS:-${GPU_DEVICES:-0,1}}"
export RESULT_DIR="${result_dir}"
export IMAGE MODEL_STORE WAN21_DISTILL_DIR

exec bash "${launcher}"
