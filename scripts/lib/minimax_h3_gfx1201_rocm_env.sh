# shellcheck shell=bash
# ROCm/LightX2V environment for the MiniMax-H3 TP2 Ref2AV entry on gfx1201
# (AMD Radeon R9600D, 24 CUs). Sourced inside the container by
# scripts/lib/run_minimax_h3_ref2av_tp2_in_container.sh.
#
# The rocm/vllm-dev:gfx1201-h3-* images ship ROCm as TheRock Python wheels in
# /opt/venv (_rocm_sdk_core / _rocm_sdk_devel). ROCM_HOME may be overridden;
# otherwise the wheel SDK is used when present and /opt/rocm is the fallback.
# LIGHTX2V_PATH selects the LightX2V source tree (default: /app/LightX2V).

if [[ -z "${ROCM_HOME:-}" ]]; then
    for rocm_sdk_core in /opt/venv/lib/python3*/site-packages/_rocm_sdk_core; do
        if [[ -d "${rocm_sdk_core}" ]]; then
            ROCM_HOME="${rocm_sdk_core}"
            break
        fi
    done
    unset rocm_sdk_core
    ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
fi
export ROCM_HOME
export ROCM_PATH="${ROCM_HOME}"
export PATH="${ROCM_HOME}/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:${PATH}}"
export HIP_CLANG_PATH="${ROCM_HOME}/lib/llvm/bin"
export HIP_DEVICE_LIB_PATH="${ROCM_HOME}/lib/llvm/amdgcn/bitcode"
export DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH}"
export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=${ROCM_HOME} --rocm-device-lib-path=${HIP_DEVICE_LIB_PATH}"
rocm_sdk_devel_include="${ROCM_HOME%/_rocm_sdk_core}/_rocm_sdk_devel/include"
if [[ -d "${rocm_sdk_devel_include}" ]]; then
    export CPLUS_INCLUDE_PATH="${rocm_sdk_devel_include}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
unset rocm_sdk_devel_include

export GPU_ARCHS=gfx1201
export CU_NUM=24
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
export AITER_USE_SYSTEM_TRITON=1
export PLATFORM=amd_rocm
export LIGHTX2V_PATH="${LIGHTX2V_PATH:-/app/LightX2V}"
export PYTHONPATH="${LIGHTX2V_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
unset CUDA_VISIBLE_DEVICES
