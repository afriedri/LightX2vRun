#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_lightx2v_fp8.sh [gpu_id]

Environment overrides:
  DOCKER_IMAGE      LightX2V image tag
  HOST_MODELS_DIR   Directory containing Wan-AI, lightx2v, and encoders
  HOST_OUTPUT_DIR   Host directory for generated videos
  GPU_ID            Physical ROCm GPU id (default: 0)
  SHM_SIZE          Container shared memory size (default: 32g)
  USE_SUDO          Set to 1 to run Docker through sudo
  DRY_RUN           Set to 1 to print the Docker command only

Expected model layout:
  HOST_MODELS_DIR/Wan-AI/Wan2.2-I2V-A14B
  HOST_MODELS_DIR/lightx2v/Wan2.2-Distill-Models
  HOST_MODELS_DIR/encoders/t5/models_t5_umt5-xxl-enc-fp8.pth
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
gpu_id="${1:-${GPU_ID:-0}}"
docker_image="${DOCKER_IMAGE:-ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260801-v2}"
host_models_dir="${HOST_MODELS_DIR:-/disk/ssd/difumodels}"
host_output_dir="${HOST_OUTPUT_DIR:-${script_dir}/outputs}"
shm_size="${SHM_SIZE:-32g}"
container_name="${CONTAINER_NAME:-lightx2v_wan22_fp8_gpu${gpu_id}}"

if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: gpu_id must be one physical GPU number, got: ${gpu_id}" >&2
    exit 2
fi

if [[ ! -d "${host_models_dir}" ]]; then
    echo "ERROR: model directory does not exist: ${host_models_dir}" >&2
    echo "Set HOST_MODELS_DIR to the directory containing Wan-AI, lightx2v, and encoders." >&2
    exit 2
fi

required_paths=(
    "Wan-AI/Wan2.2-I2V-A14B"
    "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
    "lightx2v/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
    "encoders/t5/models_t5_umt5-xxl-enc-fp8.pth"
)

missing_paths=()
for relative_path in "${required_paths[@]}"; do
    if [[ ! -e "${host_models_dir}/${relative_path}" ]]; then
        missing_paths+=("${relative_path}")
    fi
done
if (( ${#missing_paths[@]} > 0 )); then
    echo "ERROR: required model files are missing under ${host_models_dir}:" >&2
    printf '  %s\n' "${missing_paths[@]}" >&2
    exit 2
fi

mkdir -p "${host_output_dir}"

docker_command=(docker)
if [[ "${USE_SUDO:-0}" == "1" ]]; then
    docker_command=(sudo docker)
fi

if ! "${docker_command[@]}" image inspect "${docker_image}" >/dev/null 2>&1; then
    echo "ERROR: Docker image is not available locally: ${docker_image}" >&2
    exit 2
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
fi

docker_args=(
    run --rm "${tty_args[@]}"
    --name "${container_name}"
    --network host
    --group-add video
    --group-add render
    --ipc host
    --shm-size "${shm_size}"
    --security-opt seccomp=unconfined
    --device /dev/kfd
    --device /dev/dri
    -v "${host_models_dir}:/models:ro"
    -v "${host_output_dir}:/workspace/LightX2V/save_results"
    -e "ROCR_VISIBLE_DEVICES=${gpu_id}"
    -e "HIP_VISIBLE_DEVICES=0"
    -e "LIGHTX2V_PATH=/workspace/LightX2V"
    -e "MODELS_ROOT=/models"
    -e "MODEL_PATH=/models/Wan-AI/Wan2.2-I2V-A14B"
    -e "TOKENIZERS_PARALLELISM=false"
    "${docker_image}"
    bash /workspace/LightX2V/scripts/platforms/amd_rocm/run_wan22_moe_i2v_distill_fp8_4step_gfx1201.sh
)

full_command=("${docker_command[@]}" "${docker_args[@]}")

echo "Image:   ${docker_image}"
echo "GPU:     physical ${gpu_id} -> container logical 0"
echo "Models:  ${host_models_dir} -> /models (read-only)"
echo "Output:  ${host_output_dir}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${full_command[@]}"
    printf '\n'
    exit 0
fi

"${full_command[@]}"
echo "Generated video: ${host_output_dir}/output_lightx2v_wan22_moe_i2v_distill_fp8_4step_gfx1201.mp4"
