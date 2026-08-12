#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_wan22_ckconv3d_v1_720p_5s_lazy_16gpu_memdiag.sh [--image IMAGE]

Runs only the Wan2.2 720p, 121-frame, 24-FPS workload on 16 GPUs with
DiT safetensors lazy loading. RSS/PSS/private/cgroup and physical VRAM
sampling is always enabled. The regular 2/4/8/16-GPU sweep is unchanged.

Overrides:
  GPU_DEVICES                    Exactly 16 physical GPU ids (default: 0..15)
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

[[ -f "${lazy_config_host}" ]] || {
    echo "ERROR: lazy-load config not found: ${lazy_config_host}" >&2
    exit 1
}
[[ -f "${shared_launcher}" ]] || {
    echo "ERROR: shared Wan2.2 launcher not found: ${shared_launcher}" >&2
    exit 1
}

gpu_devices="${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
if [[ ! "${gpu_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: GPU_DEVICES must be comma-separated physical GPU ids" >&2
    exit 2
fi
IFS=',' read -r -a selected_gpus <<<"${gpu_devices}"
if [[ "${#selected_gpus[@]}" -ne 16 ]]; then
    echo "ERROR: lazy memory diagnostic requires exactly 16 GPU_DEVICES, got ${#selected_gpus[@]}: ${gpu_devices}" >&2
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

# The shared sweep still validates its four GPU-group slots even though this
# launcher selects only the 16-GPU slot. The first three groups are inert.
export SCALE_GPU_GROUPS="0,1;0,1,2,3;0,1,2,3,4,5,6,7;${gpu_devices}"
export SWEEP_COUNTS=16
export SWEEP_WORKLOADS=720p121f
export SWEEP_TAG="${SWEEP_TAG:-wan22_lazy16_memdiag_$(date +%Y%m%d_%H%M%S)}"
export BASE_CONFIG_JSON="${lazy_config_container}"

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
echo "Workload: 720x1280, 121 frames @ 24 FPS; 16 GPUs (${gpu_devices})"
echo "Memory policy: DiT lazy=1, block offload=1.0; T5 CPU/lazy=1/0; VAE GPU phase-load"
echo "Monitor thresholds: host=${MEMORY_MONITOR_HOST_LIMIT_GIB} GiB, per-GPU=${MEMORY_MONITOR_GPU_LIMIT_GIB} GiB"
echo "Result tag: ${SWEEP_TAG}"

exec bash "${shared_launcher}" "$@"
