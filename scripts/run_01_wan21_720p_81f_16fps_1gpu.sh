#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=wan21_720p_81f_16fps_1gpu
export RUN_TASK_LABEL="01 Wan2.1 single-GPU 720P 81f 16 FPS timing"
export RUN_MODEL_VERSION=wan21
export RUN_DEFAULT_COUNTS=1
export RUN_SUPPORTED_COUNTS=1
export RUN_DEFAULT_GPU_GROUPS=0
export RUN_EXAMPLE_COUNT=1
export RUN_TARGET_HEIGHT=720
export RUN_TARGET_WIDTH=1280
export RUN_TARGET_FRAMES=81
export RUN_TARGET_FPS=16
exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
