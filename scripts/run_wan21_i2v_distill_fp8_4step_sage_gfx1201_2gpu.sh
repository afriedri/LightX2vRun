#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_GPU_COUNT=2
export SELF_ATTN_TYPE=aiter_fav3_sage_bf16_attn
export ROPE_TYPE="${ROPE_TYPE:-triton_fused_qk_rope}"
exec bash "${script_dir}/run_wan21_i2v_distill_fp8_4step_gfx1201_1gpu_benchmark_4090.sh" "$@"
