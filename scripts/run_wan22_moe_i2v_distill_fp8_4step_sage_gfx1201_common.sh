#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"

export SELF_ATTN_TYPE=aiter_fav3_sage_bf16_attn
export BASE_CONFIG_JSON="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201.json}"

exec bash "${script_dir}/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_common.sh" "$@"
