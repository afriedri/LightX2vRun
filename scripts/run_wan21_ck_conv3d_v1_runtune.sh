#!/usr/bin/env bash
# Two-card Wan2.1 VAE A/B: PyTorch offload versus prebuilt CK v1 resident.
set -Eeuo pipefail

runtime_root="${RUNTIME_ROOT:-/disk/ssd/lightx2v-runtime}"
output_dir="${OUTPUT_DIR:-${runtime_root}/outputs}"
result_dir="${RESULT_DIR:-${output_dir}/ck_conv3d_v1_runtune_$(date +%Y%m%d_%H%M%S)}"
legacy_image="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260804-SageUtimateWithMM-T1"
ck_aot_image="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260805-CKConv3dV1-AOT"
if [[ -z "${IMAGE:-}" || "${IMAGE}" == "${legacy_image}" ]]; then
    IMAGE="${ck_aot_image}"
fi

lightx2v_dir=""
if [[ -n "${LIGHTX2V_DIR:-}" ]]; then
    if [[ -f "${LIGHTX2V_DIR}/scripts/platforms/amd_rocm/run_ck_conv3d_v1_runtune.sh" ]]; then
        lightx2v_dir="$(cd -- "${LIGHTX2V_DIR}" && pwd)"
    else
        echo "WARN: LIGHTX2V_DIR is unavailable; falling back to image-bundled LightX2V: ${LIGHTX2V_DIR}" >&2
    fi
fi

launcher_tmp_dir=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${launcher_tmp_dir}" && -d "${launcher_tmp_dir}" ]]; then
        rm -rf -- "${launcher_tmp_dir}"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [[ -n "${lightx2v_dir}" ]]; then
    launcher="${lightx2v_dir}/scripts/platforms/amd_rocm/run_ck_conv3d_v1_runtune.sh"
    export LIGHTX2V_DIR="${lightx2v_dir}"
    echo "INFO: using host LightX2V checkout: ${lightx2v_dir}"
else
    command -v docker >/dev/null || {
        echo "ERROR: docker is required to use image-bundled LightX2V" >&2
        exit 1
    }
    docker image inspect "${IMAGE}" >/dev/null
    launcher_tmp_dir="$(mktemp -d /tmp/lightx2v-ck-conv3d-launcher.XXXXXX)"
    launcher="${launcher_tmp_dir}/run_ck_conv3d_v1_runtune.sh"
    docker run --rm --network=none --entrypoint=cat "${IMAGE}" \
        /workspace/LightX2V/scripts/platforms/amd_rocm/run_ck_conv3d_v1_runtune.sh \
        >"${launcher}"
    [[ -s "${launcher}" ]] || {
        echo "ERROR: CK Conv3d v1 launcher is missing from image: ${IMAGE}" >&2
        exit 1
    }
    bash -n "${launcher}"
    unset LIGHTX2V_DIR
    echo "INFO: using LightX2V and runtune runner bundled in image: ${IMAGE}"
fi

export GPU_IDS="${GPU_IDS:-${GPU_DEVICES:-0,1}}"
export RESULT_DIR="${result_dir}"
export IMAGE MODEL_STORE WAN21_DISTILL_DIR

bash "${launcher}"
