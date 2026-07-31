#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_GPU_COUNT=4
exec bash "${script_dir}/run_wan22_moe_i2v_distill_lora_fp8_4step_gfx1201_common.sh" "$@"
