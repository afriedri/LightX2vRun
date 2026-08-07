#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${EXPECTED_GPU_COUNT:-}" ]]; then
    visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
    [[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        echo "ERROR: HIP_VISIBLE_DEVICES must be set for CK Conv3d release inference" >&2
        exit 1
    }
    IFS=',' read -r -a release_gpus <<<"${visible_devices}"
    export EXPECTED_GPU_COUNT="${#release_gpus[@]}"
fi
case "${EXPECTED_GPU_COUNT}" in
    1 | 2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: CK Conv3d release supports 1, 2, 4, 8, or 16 GPUs" >&2
        exit 1
        ;;
esac
export SELF_ATTN_TYPE=aiter_fav3_sage_bf16_attn
export ROPE_TYPE="${ROPE_TYPE:-triton_fused_qk_rope}"
export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export LIGHTX2V_AMD_MIOPEN=0
export USE_TILING_VAE=0
export VAE_CPU_OFFLOAD=0
export T5_CPU_OFFLOAD="${T5_CPU_OFFLOAD:-1}"
export CLIP_CPU_OFFLOAD="${CLIP_CPU_OFFLOAD:-1}"
export USE_CHANNELS_LAST_3D=0
export EXPECTED_CK_REV="${EXPECTED_CK_REV:-af7118e342580ecd3f71edce7b1d0ba465012ecf}"
export SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-/workspace/LightX2V/save_results/output_wan21_i2v_720p_81f_ckconv3d_v1_release_${EXPECTED_GPU_COUNT}gpu.mp4}"

echo "Wan2.1 CK Conv3d v1 release: ${EXPECTED_GPU_COUNT} GPU(s), VAE resident, tiling=0, MIOpen=0"

if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
    python3 -c 'import importlib, os; import aiter; assert callable(aiter.ck_conv3d_v1); assert aiter.is_ck_conv3d_v1_prebuilt(); assert aiter.CK_CONV3D_V1_CK_REVISION == os.environ["EXPECTED_CK_REV"]; module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'
fi

exec bash "${script_dir}/run_wan21_i2v_distill_fp8_4step_gfx1201_1gpu_benchmark_4090.sh" "$@"
