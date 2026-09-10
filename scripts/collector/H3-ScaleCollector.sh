#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
image="${IMAGE:-}"
output="${RESULT_DIR:-}"
[[ -n "$image" ]] || { echo 'IMAGE or --image is required' >&2; exit 2; }
args=()
while (($#)); do
  case "$1" in
    --image) image="$2"; shift 2;;
    --image=*) image="${1#*=}"; shift;;
    --output-dir) output="$2"; shift 2;;
    --output-dir=*) output="${1#*=}"; shift;;
    --counts|--gpu-count) args+=("$1" "$2"); shift 2;;
    --counts=*|--gpu-count=*) args+=("$1"); shift;;
    2|4|8|16) args+=("$1"); shift;;
    -h|--help) echo 'Usage: IMAGE=... bash scripts/collector/H3-ScaleCollector.sh [--counts 2,4,8,16]'; exit 0;;
    *) echo "unsupported argument: $1" >&2; exit 2;;
  esac
done
export IMAGE="$image" RUN_H3_TASK=ref2av
export LIGHTX2V_VAE_CONV3D_BACKEND="${LIGHTX2V_VAE_CONV3D_BACKEND:-Aiter_CK_Conv3d_v1}"
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_routes}"
[[ -n "$output" ]] && export RESULT_DIR="$output"
exec bash "$run_root/scripts/run_08_minimax_h3_768p_124f_24fps_scale.sh" "${args[@]}"
