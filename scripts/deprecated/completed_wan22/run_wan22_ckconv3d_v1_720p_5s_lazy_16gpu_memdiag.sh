#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  WAN22_GPU_COUNT=8 GPU_DEVICES=0,1,2,3,4,5,6,7 \
    bash scripts/run_wan22_ckconv3d_v1_720p_5s_lazy_16gpu_memdiag.sh [--image IMAGE]

Runs one Wan2.2 720p, 121-frame, 24-FPS workload on 2, 4, 8, or 16 GPUs
with DiT safetensors lazy loading. RSS/PSS/private/cgroup and physical VRAM
sampling is always enabled. The default remains the original 16-GPU run.

Overrides:
  WAN22_GPU_COUNT                2, 4, 8, or 16 (default: 16)
  GPU_DEVICES                    Exactly WAN22_GPU_COUNT ids (default: 0..N-1)
  MEMORY_MONITOR_INTERVAL        Sampling interval in seconds (default: 1.0)
  MEMORY_MONITOR_HOST_LIMIT_GIB  Diagnostic warning threshold (default: 512)
  MEMORY_MONITOR_GPU_LIMIT_GIB   Per-GPU warning threshold (default: 32)
  SWEEP_TAG                      Result tag
  SWEEP_RESULT_DIR               Explicit result directory
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
lazy_config_host="${repo_root}/configs/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201_lazy.json"
lazy_config_container="/workspace/LightX2VRun/configs/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201_lazy.json"
shared_launcher="${script_dir}/run_wan22_ckconv3d_v1_720p_5s_shape_sweep.sh"
lazy_model_run="${script_dir}/run_wan22_lazy_16gpu_modelrun.sh"

[[ -f "${lazy_config_host}" ]] || {
    echo "ERROR: lazy-load config not found: ${lazy_config_host}" >&2
    exit 1
}
[[ -f "${shared_launcher}" ]] || {
    echo "ERROR: shared Wan2.2 launcher not found: ${shared_launcher}" >&2
    exit 1
}
[[ -x "${lazy_model_run}" ]] || {
    echo "ERROR: dedicated lazy model launcher is not executable: ${lazy_model_run}" >&2
    exit 1
}

gpu_count="${WAN22_GPU_COUNT:-16}"
case "${gpu_count}" in
    2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: WAN22_GPU_COUNT must be 2, 4, 8, or 16, got ${gpu_count}" >&2
        exit 2
        ;;
esac
default_gpu_devices=""
for ((gpu_index = 0; gpu_index < gpu_count; gpu_index++)); do
    default_gpu_devices+="${default_gpu_devices:+,}${gpu_index}"
done
gpu_devices="${GPU_DEVICES:-${default_gpu_devices}}"
if [[ ! "${gpu_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: GPU_DEVICES must be comma-separated physical GPU ids" >&2
    exit 2
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_devices}"
if [[ "${#selected_gpus[@]}" -ne "${gpu_count}" ]]; then
    echo "ERROR: lazy memory diagnostic requires exactly ${gpu_count} GPU_DEVICES, got ${#selected_gpus[@]}: ${gpu_devices}" >&2
    exit 2
fi
declare -A seen_gpus=()
for gpu in "${selected_gpus[@]}"; do
    if [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
        echo "ERROR: GPU_DEVICES contains duplicate id ${gpu}" >&2
        exit 2
    fi
    seen_gpus["${gpu}"]=1
done

# The shared sweep validates all four slots. Replace only the selected slot;
# the other groups are inert defaults.
gpu_groups=(
    "0,1"
    "0,1,2,3"
    "0,1,2,3,4,5,6,7"
    "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
)
case "${gpu_count}" in
    2) gpu_groups[0]="${gpu_devices}" ;;
    4) gpu_groups[1]="${gpu_devices}" ;;
    8) gpu_groups[2]="${gpu_devices}" ;;
    16) gpu_groups[3]="${gpu_devices}" ;;
esac
export SCALE_GPU_GROUPS="${gpu_groups[0]};${gpu_groups[1]};${gpu_groups[2]};${gpu_groups[3]}"
export SWEEP_COUNTS="${gpu_count}"
export SWEEP_WORKLOADS=720p121f
export SWEEP_TAG="${SWEEP_TAG:-wan22_lazy${gpu_count}_memdiag_$(date +%Y%m%d_%H%M%S)}"
export WAN22_GPU_COUNT="${gpu_count}"
export BASE_CONFIG_JSON="${lazy_config_container}"
export LIGHTX2V_LAZY_REAL_MODEL_RUN="${MODEL_RUN:-${repo_root}/lightx2vModelrun}"
export MODEL_RUN="${lazy_model_run}"
export IMAGE="${IMAGE:-compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260812-CKConv3dV1-Hot28-AOTFinal}"

# Keep this diagnostic isolated from inherited production offload settings.
export CPU_OFFLOAD=1
export OFFLOAD_GRANULARITY=block
export OFFLOAD_RATIO=1.0
export T5_CPU_OFFLOAD=1

# Monitoring is mandatory for this launcher. The limits are warnings, not an
# OOM-inducing hard cgroup cap.
export MEMORY_MONITOR=1
export MEMORY_MONITOR_INTERVAL="${MEMORY_MONITOR_INTERVAL:-1.0}"
export MEMORY_MONITOR_HOST_LIMIT_GIB="${MEMORY_MONITOR_HOST_LIMIT_GIB:-512}"
export MEMORY_MONITOR_GPU_LIMIT_GIB="${MEMORY_MONITOR_GPU_LIMIT_GIB:-32}"

echo "Wan2.2 dedicated lazy-load memory diagnostic"
echo "Workload: 720x1280, 121 frames @ 24 FPS; ${gpu_count} GPUs (${gpu_devices})"
echo "Memory policy: DiT lazy=1, block offload=1.0; T5 CPU/lazy=1/0; VAE GPU phase-load"
echo "Lazy compatibility: Run-only attention interface shim enabled"
echo "Monitor thresholds: host=${MEMORY_MONITOR_HOST_LIMIT_GIB} GiB, per-GPU=${MEMORY_MONITOR_GPU_LIMIT_GIB} GiB"
echo "Result tag: ${SWEEP_TAG}"

exec bash "${shared_launcher}" "$@"
