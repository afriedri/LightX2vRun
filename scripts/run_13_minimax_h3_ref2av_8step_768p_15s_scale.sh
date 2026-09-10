#!/usr/bin/env bash
# MiniMax-H3 Ref2AV Turbo 8-step 2/4/8/16-GPU timing sweep.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=minimax_h3_ref2av_turbo_8step_768p_15s_scale
export RUN_TASK_LABEL="13 MiniMax-H3 Ref2AV Turbo 8-step 2/4/8/16-GPU 1344x768 15s"
export RUN_MODEL_VERSION=minimax_h3 RUN_H3_TASK=ref2av
export H3_PROFILE=ref2av_turbo_8step
export RUN_DEFAULT_COUNTS=2,4,8,16 RUN_SUPPORTED_COUNTS=2,4,8,16
# Rotate the partial-scale arms and keep each CP2xU8 row within a consecutive
# eight-GPU group for the SP16 arm.
export RUN_DEFAULT_GPU_GROUPS='1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RUN_EXAMPLE_COUNT=16
export RUN_TARGET_HEIGHT=768 RUN_TARGET_WIDTH=1344
export RUN_TARGET_FRAMES=362 RUN_TARGET_FPS=24 RUN_DURATION_SECONDS=15
export RUN_DIT_STEPS=8
export H3_STEP_UPDATE=training_euler
export H3_INFER_STEPS=9
export H3_OUTPUT_DURATION=15
export H3_MODEL_DIR="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
export H3_TURBO_DIR="${H3_TURBO_DIR:-/disk/ssd/difumodels-Dis/Minimax-h3-Turbo}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/disk/ssd1/lightx2v-runtime}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNTIME_ROOT}/outputs}"
export LIGHTX2V_H3_RUN_COLLECTOR="${LIGHTX2V_H3_RUN_COLLECTOR:-1}"
export LIGHTX2V_VAE_CONV3D_BACKEND="${LIGHTX2V_VAE_CONV3D_BACKEND:-Aiter_CK_Conv3d_v1}"
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_routes}"

exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
