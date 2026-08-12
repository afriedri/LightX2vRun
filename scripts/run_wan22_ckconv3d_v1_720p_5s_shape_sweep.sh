#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_720p_5s_shape_sweep.sh [--image IMAGE]

Collects one explicit five-second 720P Conv3D shape workload on 2, 4, 8,
and 16 GPUs:
  720x1280, 24 FPS, 121 frames ((121 - 1) / 24 = 5 seconds)

This is a workload-specific wrapper around the Wan2.2 shape sweep. It reuses
the existing GPU groups, profile collection, and rank-critical hot-contract
summary. SWEEP_COUNTS, SCALE_GPU_GROUPS, SWEEP_TAG, SWEEP_RESULT_DIR, and
SWEEP_SUMMARY_DIR are passed through to the shared sweep.
EOF
}

image_override="${IMAGE:-}"
while (( $# > 0 )); do
    case "$1" in
        --image)
            (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }
            image_override="$2"
            shift 2
            ;;
        --image=*)
            image_override="${1#--image=}"
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
[[ -z "${image_override}" || "${image_override}" != *[[:space:]]* ]] || {
    echo "ERROR: image reference may not contain whitespace" >&2
    exit 2
}

if [[ -n "${SWEEP_WORKLOADS:-}" && "${SWEEP_WORKLOADS}" != "720p121f" ]]; then
    echo "ERROR: this launcher only supports SWEEP_WORKLOADS=720p121f" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shared_sweep="${script_dir}/run_wan22_ckconv3d_v1_scale_sweep.sh"
[[ -f "${shared_sweep}" ]] || {
    echo "ERROR: shared Wan2.2 shape sweep not found: ${shared_sweep}" >&2
    exit 1
}

export SWEEP_WORKLOADS=720p121f
if [[ -n "${image_override}" ]]; then
    exec bash "${shared_sweep}" --image "${image_override}"
fi
exec bash "${shared_sweep}" "$@"
