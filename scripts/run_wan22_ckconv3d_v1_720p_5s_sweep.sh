#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_720p_5s_sweep.sh [--mode MODE] [--image IMAGE]

MODE is shape (default) or timing:
  shape   collect 720P/24fps/121f Conv3D shapes and summarize hot contracts
  timing  measure the same workload's stage timings with shape collection off

Use --mode shape, --mode timing, --shape, or --timing. SWEEP_MODE is an
environment alternative to --mode.
EOF
}

mode="${SWEEP_MODE:-shape}"
forward_args=()
while (( $# > 0 )); do
    case "$1" in
        --mode)
            (( $# >= 2 )) || { echo "ERROR: --mode requires shape or timing" >&2; exit 2; }
            mode="$2"
            shift 2
            ;;
        --mode=*)
            mode="${1#--mode=}"
            shift
            ;;
        --shape)
            mode=shape
            shift
            ;;
        --timing)
            mode=timing
            shift
            ;;
        --image)
            (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }
            forward_args+=(--image "$2")
            shift 2
            ;;
        --image=*)
            forward_args+=("$1")
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unsupported argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${mode}" in
    shape)
        exec bash "${script_dir}/run_wan22_ckconv3d_v1_720p_5s_shape_sweep.sh" "${forward_args[@]}"
        ;;
    timing)
        exec bash "${script_dir}/run_wan22_ckconv3d_v1_720p_5s_timing_sweep.sh" "${forward_args[@]}"
        ;;
    *)
        echo "ERROR: mode must be shape or timing, got ${mode}" >&2
        exit 2
        ;;
esac
