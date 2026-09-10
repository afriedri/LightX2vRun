#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
image="${IMAGE:-}"
output="${RESULT_DIR:-}"
steps="${H3_INFER_STEPS:-}"
[[ -n "$image" ]] || { echo 'IMAGE or --image is required' >&2; exit 2; }
args=()
while (($#)); do
  case "$1" in
    --image) image="$2"; shift 2;;
    --image=*) image="${1#*=}"; shift;;
    --output-dir) output="$2"; shift 2;;
    --steps) steps="$2"; shift 2;;
    --steps=*) steps="${1#*=}"; shift;;
    --output-dir=*) output="${1#*=}"; shift;;
    --counts|--gpu-count) args+=("$1" "$2"); shift 2;;
    --counts=*|--gpu-count=*) args+=("$1"); shift;;
    2|4|8|16) args+=("$1"); shift;;
    -h|--help) echo 'Usage: IMAGE=... bash scripts/collector/H3-ScaleCollector.sh [--counts 2,4,8,16] [--steps N]'; exit 0;;
    *) echo "unsupported argument: $1" >&2; exit 2;;
  esac
done
export IMAGE="$image" RUN_H3_TASK=ref2av
[[ -z "$steps" || "$steps" =~ ^[1-9][0-9]*$ ]] || { echo "--steps must be a positive integer" >&2; exit 2; }
[[ -z "$steps" ]] || export H3_INFER_STEPS="$steps"

# Fail before launching 2/4/8/16 workers when an old image still contains the
# former 124-frame-only UlyssesRing gate. The current H3 source accepts the
# native 17*n+5 range through 362 frames; this check is diagnostic only.
if [[ "${H3_IMAGE_COMPAT_CHECK:-1}" == 1 ]] && command -v docker >/dev/null 2>&1; then
  if ! docker run --rm --entrypoint sh "$image" -c \
      'p=/workspace/LightX2V/lightx2v/models/networks/minimax_h3/model.py; test -f "$p" && grep -q "17\\*n+5 frames within 124..362" "$p"'; then
    echo "ERROR: image lacks long-video H3 UlyssesRing support (17*n+5, 124..362 frames). Rebuild from LightX2V hvat-gfx1201-R9600 at b826c99a or newer." >&2
    echo "       Set H3_IMAGE_COMPAT_CHECK=0 only for a 124-frame run." >&2
    exit 3
  fi
fi
export LIGHTX2V_VAE_CONV3D_BACKEND="${LIGHTX2V_VAE_CONV3D_BACKEND:-Aiter_CK_Conv3d_v1}"
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_routes}"
[[ -n "$output" ]] && export RESULT_DIR="$output"
exec bash "$run_root/scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh" --image "$image" "${args[@]}"
