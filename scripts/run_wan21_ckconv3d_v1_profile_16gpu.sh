#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPECTED_GPU_COUNT=16
exec bash "${script_dir}/run_wan21_ckconv3d_v1_profile_common.sh" "$@"
