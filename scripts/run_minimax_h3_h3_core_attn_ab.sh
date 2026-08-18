#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
benchmark_script="${run_root}/scripts/lib/benchmark_minimax_h3_core_attention.py"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_minimax_h3_h3_core_attn_ab.sh --image IMAGE [options]

Runs one isolated H3 Core Attention A/B on one gfx1201 GPU. It does not load
model weights or change the production MiniMax-H3 runner.

Options:
  --image IMAGE       Existing H3 gfx1201 image (required; no rebuild needed)
  --gpu ID            Physical GPU id (default: 0)
  --warmup N          Warmup calls per backend (default: 1)
  --iterations N      Timed calls per backend (default: 3)
  --result-dir DIR    Host output directory (default: timestamped runtime path)

Environment:
  CACHE_DIR           Persistent Aiter/Triton cache
  DRY_RUN=1           Print the Docker command without running it
EOF
}

image=
gpu=0
warmup=1
iterations=3
result_dir=
runtime_root="${RUNTIME_ROOT:-/disk/ssd/lightx2v-runtime}"
cache_dir="${CACHE_DIR:-${runtime_root}/cache}"

while (( $# > 0 )); do
    case "$1" in
        --help | -h) usage; exit 0 ;;
        --image) (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }; image="$2"; shift 2 ;;
        --gpu) (( $# >= 2 )) || { echo "ERROR: --gpu requires a value" >&2; exit 2; }; gpu="$2"; shift 2 ;;
        --warmup) (( $# >= 2 )) || { echo "ERROR: --warmup requires a value" >&2; exit 2; }; warmup="$2"; shift 2 ;;
        --iterations) (( $# >= 2 )) || { echo "ERROR: --iterations requires a value" >&2; exit 2; }; iterations="$2"; shift 2 ;;
        --result-dir) (( $# >= 2 )) || { echo "ERROR: --result-dir requires a value" >&2; exit 2; }; result_dir="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${image}" ]] || { echo "ERROR: --image is required" >&2; exit 2; }
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu must be a non-negative integer" >&2; exit 2; }
[[ "${warmup}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --warmup must be positive" >&2; exit 2; }
[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --iterations must be positive" >&2; exit 2; }
[[ -f "${benchmark_script}" ]] || { echo "ERROR: benchmark script not found: ${benchmark_script}" >&2; exit 2; }

if [[ -z "${result_dir}" ]]; then
    result_dir="${runtime_root}/outputs/minimax_h3_h3_core_attn_ab_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${result_dir}" "${cache_dir}/aiter" "${cache_dir}/triton"
result_dir="$(realpath "${result_dir}")"
cache_dir="$(realpath "${cache_dir}")"
benchmark_script="$(realpath "${benchmark_script}")"

docker_args=(
    run --rm
    --name "minimax-h3-core-attn-ab-${gpu}"
    --ipc=host
    --cap-add=SYS_PTRACE
    --security-opt=seccomp=unconfined
    --device=/dev/kfd
    --device=/dev/dri
    --ulimit=memlock=-1:-1
    --workdir=/workspace/LightX2V
    --env "HIP_VISIBLE_DEVICES=${gpu}"
    --env PLATFORM=amd_rocm
    --env GPU_ARCHS=gfx1201
    --env CU_NUM=24
    --env AITER_JIT_DIR=/workspace/.cache/aiter
    --env TRITON_CACHE_DIR=/workspace/.cache/triton
    --mount "type=bind,source=${benchmark_script},target=/opt/h3-core-attn-ab.py,readonly"
    --mount "type=bind,source=${result_dir},target=/results"
    --mount "type=bind,source=${cache_dir}/aiter,target=/workspace/.cache/aiter"
    --mount "type=bind,source=${cache_dir}/triton,target=/workspace/.cache/triton"
    "${image}"
    python3 /opt/h3-core-attn-ab.py
    --warmup "${warmup}"
    --iterations "${iterations}"
    --output /results/h3_core_attention_ab.json
)

echo "H3 Core Attention A/B: S=42196 H=28 D=128 (collected SP2 contract)"
echo "Result: ${result_dir}/h3_core_attention_ab.json"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' docker "${docker_args[@]}"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed" >&2; exit 1; }
[[ -c /dev/kfd && -d /dev/dri ]] || { echo "ERROR: ROCm devices are unavailable" >&2; exit 1; }
docker image inspect "${image}" >/dev/null 2>&1 || { echo "ERROR: Docker image is unavailable: ${image}" >&2; exit 1; }
exec docker "${docker_args[@]}"
