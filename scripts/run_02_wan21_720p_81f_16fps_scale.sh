#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=wan21_720p_81f_16fps_scale
export RUN_TASK_LABEL="02 Wan2.1 1/2/4/8/16-GPU 720P 81f 16 FPS timing"
export RUN_MODEL_VERSION=wan21
export RUN_DEFAULT_COUNTS=1,2,4,8,16
export RUN_SUPPORTED_COUNTS=1,2,4,8,16
export RUN_DEFAULT_GPU_GROUPS='0;0,1;0,1,2,3;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RUN_EXAMPLE_COUNT=2
export RUN_TARGET_HEIGHT=720
export RUN_TARGET_WIDTH=1280
export RUN_TARGET_FRAMES=81
export RUN_TARGET_FPS=16
exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
