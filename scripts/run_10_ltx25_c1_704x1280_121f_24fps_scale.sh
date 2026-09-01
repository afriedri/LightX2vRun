#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ENTRY_NAME="$(basename -- "${BASH_SOURCE[0]}")"
export RUN_TASK_ID=ltx25_c1_704x1280_121f_24fps_scale
export RUN_TASK_LABEL="10 LTX-2.5 C1 2/4/8/16-GPU 1280x704 121f 24 FPS timing"
export RUN_MODEL_VERSION=ltx25_c1
export RUN_DEFAULT_COUNTS=2,4,8,16
export RUN_SUPPORTED_COUNTS=2,4,8,16
# Pin SP2 to the established LTX-2.5 comparison pair; retain H3 rotation above SP2.
export RUN_DEFAULT_GPU_GROUPS='2,3;15,14,13,12;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RUN_EXAMPLE_COUNT=2
export RUN_TARGET_HEIGHT=704
export RUN_TARGET_WIDTH=1280
export RUN_TARGET_FRAMES=121
export RUN_TARGET_FPS=24
export RUN_DURATION_SECONDS=5.0
exec bash "${script_dir}/lib/run_timing_sweep.sh" "$@"
