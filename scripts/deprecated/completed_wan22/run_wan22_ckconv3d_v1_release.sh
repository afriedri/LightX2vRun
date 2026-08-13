#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"

if [[ -z "${EXPECTED_GPU_COUNT:-}" ]]; then
    visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
    [[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        echo "ERROR: HIP_VISIBLE_DEVICES must be set for Wan2.2 CK Conv3d release inference" >&2
        exit 1
    }
    IFS=',' read -r -a release_gpus <<<"${visible_devices}"
    export EXPECTED_GPU_COUNT="${#release_gpus[@]}"
fi
case "${EXPECTED_GPU_COUNT}" in
    2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: Wan2.2 CK Conv3d release supports 2, 4, 8, or 16 GPUs" >&2
        exit 1
        ;;
esac

# Match the Wan2.2 shape sweep workload without enabling its shape collector.
unset LIGHTX2V_CK_CONV3D_PROFILE_DIR PROFILE_TAG
export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export CPU_OFFLOAD=1
export OFFLOAD_GRANULARITY=block
export OFFLOAD_RATIO="${OFFLOAD_RATIO:-1.0}"
export T5_CPU_OFFLOAD="${T5_CPU_OFFLOAD:-1}"
export VAE_CPU_OFFLOAD=0
export USE_TILING_VAE=0
export LIGHTX2V_AMD_MIOPEN=0
export USE_CHANNELS_LAST_3D=0
export LIGHTX2V_WARMUP="${LIGHTX2V_WARMUP:-0}"
export SELF_ATTN_TYPE=aiter_fav3_sage_bf16_attn
export ROPE_TYPE="${ROPE_TYPE:-triton_fused_qk_rope}"
export BASE_CONFIG_JSON="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201.json}"
export SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_wan22_i2v_ckconv3d_v1_release_${EXPECTED_GPU_COUNT}gpu.mp4}"

echo "Wan2.2 CK Conv3d v1 release: ${EXPECTED_GPU_COUNT} GPU(s)"
echo "Offload: DiT=block/${OFFLOAD_RATIO}, T5=${T5_CPU_OFFLOAD}; VAE resident, tiling=0"
echo "Shape collection: disabled"

if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
    python3 -c 'import importlib; import aiter; assert callable(aiter.ck_conv3d_v1); assert aiter.is_ck_conv3d_v1_prebuilt(); module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'
fi

exec bash "${script_dir}/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_common.sh" "$@"
