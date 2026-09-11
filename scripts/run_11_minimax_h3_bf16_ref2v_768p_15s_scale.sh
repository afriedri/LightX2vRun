#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=minimax_h3_bf16_ref2v_768p_15s_scale
export RUN_TASK_LABEL="11 MiniMax-H3 BF16 Ref2V 2/4/8/16-GPU 1344x768 15s (no LoRA)"
export RUN_MODEL_VERSION=minimax_h3 RUN_H3_TASK=ref2av
export H3_PROFILE=ref2av_bf16_base
export RUN_DEFAULT_COUNTS=2,4,8,16 RUN_SUPPORTED_COUNTS=2,4,8,16
# Same physical GPU rotation as the FL2AV scale entry; each U8 row stays
# within one consecutive eight-GPU group at SP16.
export RUN_DEFAULT_GPU_GROUPS='1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RUN_EXAMPLE_COUNT=2
export RUN_TARGET_HEIGHT=768 RUN_TARGET_WIDTH=1344
export RUN_TARGET_FRAMES=362 RUN_TARGET_FPS=24 RUN_DURATION_SECONDS=15
export RUNTIME_ROOT="${RUNTIME_ROOT:-/disk/ssd1/lightx2v-runtime}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUNTIME_ROOT}/outputs}"
# Full-size, short denoise runs can override H3_INFER_STEPS=3; geometry stays fixed.
export H3_STEP_UPDATE="${H3_STEP_UPDATE:-comfy_res_multistep}"
default_steps=20
if [[ "$H3_STEP_UPDATE" != comfy_res_multistep ]]; then default_steps=30; fi
export H3_INFER_STEPS="${H3_INFER_STEPS:-$default_steps}"
export H3_OUTPUT_DURATION=15
export LIGHTX2V_H3_RUN_COLLECTOR="${LIGHTX2V_H3_RUN_COLLECTOR:-1}"
export LIGHTX2V_VAE_CONV3D_BACKEND="${LIGHTX2V_VAE_CONV3D_BACKEND:-Aiter_CK_Conv3d_v1}"
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_routes}"
exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
