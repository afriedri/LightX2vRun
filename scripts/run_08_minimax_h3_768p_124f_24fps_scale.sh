#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=minimax_h3_768p_124f_24fps_scale
export RUN_TASK_LABEL="08 MiniMax-H3 2/4/8/16-GPU 1344x768 124f 24 FPS timing"
export RUN_MODEL_VERSION=minimax_h3
export RUN_DEFAULT_COUNTS=2,4,8,16
export RUN_SUPPORTED_COUNTS=2,4,8,16
# Preserve the original Wan timing rotation so each partial-scale arm lets the
# previously active cards cool before they are reused by a larger arm.
export RUN_DEFAULT_GPU_GROUPS='1,2;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RUN_EXAMPLE_COUNT=2
export RUN_TARGET_HEIGHT=768
export RUN_TARGET_WIDTH=1344
export RUN_TARGET_FRAMES=124
export RUN_TARGET_FPS=24
export RUN_DURATION_SECONDS=5.17
export RUN_H3_FIRST_FRAME=/disk/ssd/lightx2v-runtime/inputs/STRAT_FL2V.png
export RUN_H3_LAST_FRAME=/disk/ssd/lightx2v-runtime/inputs/END_FL2V.png
exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
