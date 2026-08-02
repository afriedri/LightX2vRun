#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/Wan-AI/Wan2.2-I2V-A14B}"
expected_gpu_count="${EXPECTED_GPU_COUNT:?EXPECTED_GPU_COUNT must be set by a GPU-count launcher}"
base_config_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_gfx1201.json}"

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export AITER_LOG_TUNED_CONFIG="${AITER_LOG_TUNED_CONFIG:-1}"
if [[ "${GPU_ARCHS}" != "gfx1201" || "${CU_NUM}" != "24" ]]; then
    echo "ERROR: gfx1201 FP8 requires GPU_ARCHS=gfx1201 and CU_NUM=24" >&2
    exit 1
fi

aiter_jit_root="${AITER_JIT_DIR:-/workspace/.cache/aiter}"
export AITER_JIT_DIR="${aiter_jit_root%/}/gfx1201-cu24-rowwise-v2"
mkdir -p "${AITER_JIT_DIR}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
unset CUDA_VISIBLE_DEVICES
if [[ ! "${HIP_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: HIP_VISIBLE_DEVICES must be comma-separated GPU ids" >&2
    exit 1
fi
IFS=',' read -r -a gpu_devices <<<"${HIP_VISIBLE_DEVICES}"
if [[ "${#gpu_devices[@]}" -ne "${expected_gpu_count}" ]]; then
    echo "ERROR: this launcher requires ${expected_gpu_count} GPUs, got ${HIP_VISIBLE_DEVICES}" >&2
    exit 1
fi
declare -A seen_gpu_devices=()
for gpu_device in "${gpu_devices[@]}"; do
    if [[ -n "${seen_gpu_devices[${gpu_device}]:-}" ]]; then
        echo "ERROR: HIP_VISIBLE_DEVICES contains duplicate id ${gpu_device}" >&2
        exit 1
    fi
    seen_gpu_devices["${gpu_device}"]=1
done

case "${expected_gpu_count}" in
    1)
        seq_p_size=1
        seq_p_attn_type=ulysses
        self_attn_type=aiter_flydsl_bf16_flash_attn
        ;;
    2 | 4 | 8)
        seq_p_size="${expected_gpu_count}"
        seq_p_attn_type=ulysses
        self_attn_type=aiter_flydsl_bf16_flash_attn
        ;;
    16)
        seq_p_size=16
        seq_p_attn_type=ring
        self_attn_type=aiter_triton_bf16_flash_attn
        ;;
    *)
        echo "ERROR: supported GPU counts are 1, 2, 4, 8, and 16" >&2
        exit 1
        ;;
esac

if [[ -n "${SELF_ATTN_TYPE:-}" ]]; then
    case "${SELF_ATTN_TYPE}" in
        aiter_flydsl_bf16_flash_attn | aiter_triton_bf16_flash_attn | aiter_fav3_sage_bf16_attn)
            self_attn_type="${SELF_ATTN_TYPE}"
            ;;
        *)
            echo "ERROR: unsupported SELF_ATTN_TYPE=${SELF_ATTN_TYPE}" >&2
            exit 1
            ;;
    esac
fi

[[ -f "${base_config_json}" ]] || {
    echo "ERROR: base config not found: ${base_config_json}" >&2
    exit 1
}

runtime_config_dir="${RUNTIME_CONFIG_DIR:-/tmp/lightx2v-run}"
mkdir -p "${runtime_config_dir}"
runtime_config_json="${runtime_config_dir}/wan22-i2v-fp8-${expected_gpu_count}gpu-$$.json"
trap 'rm -f "${runtime_config_json}"' EXIT

python3 - "${base_config_json}" "${runtime_config_json}" "${seq_p_size}" "${seq_p_attn_type}" "${self_attn_type}" <<'PY'
import json
import os
import sys

source_path, output_path, seq_p_size, seq_p_attn_type, self_attn_type = sys.argv[1:]
with open(source_path, encoding="utf-8") as source_file:
    config = json.load(source_file)

config["enable_cfg"] = False
config["self_attn_1_type"] = self_attn_type
config["parallel"] = {
    "seq_p_size": int(seq_p_size),
    "seq_p_attn_type": seq_p_attn_type,
    "seq_p_prepost_backend": "torch",
    "seq_p_a2a_backend": "torch",
}

integer_overrides = {
    "INFER_STEPS": "infer_steps",
    "TARGET_VIDEO_LENGTH": "target_video_length",
    "TARGET_HEIGHT": "target_height",
    "TARGET_WIDTH": "target_width",
}
for environment_name, config_name in integer_overrides.items():
    value = os.environ.get(environment_name)
    if value:
        config[config_name] = int(value)

checkpoint_overrides = {
    "HIGH_NOISE_QUANTIZED_CKPT": "high_noise_quantized_ckpt",
    "LOW_NOISE_QUANTIZED_CKPT": "low_noise_quantized_ckpt",
}
for environment_name, config_name in checkpoint_overrides.items():
    value = os.environ.get(environment_name)
    if value:
        config[config_name] = value

with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=4, ensure_ascii=False)
    output_file.write("\n")
PY

prompt="${PROMPT:-Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights intricate feline details and the refreshing atmosphere of the seaside.}"
negative_prompt="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill_fp8_4step_gfx1201_${expected_gpu_count}gpu.mp4}"

echo "GPUs: ${HIP_VISIBLE_DEVICES} (${expected_gpu_count} ranks)"
echo "Parallel: SP=${seq_p_size}, attention=${seq_p_attn_type}, CFG=disabled"
echo "Config: ${runtime_config_json}"
echo "Output: ${save_result_path}"
warmup="${LIGHTX2V_WARMUP:-1}"
case "${warmup}" in
    0 | 1) ;;
    *)
        echo "ERROR: LIGHTX2V_WARMUP must be 0 or 1" >&2
        exit 1
        ;;
esac
if [[ "${warmup}" == 1 ]]; then
    echo "Warmup: enabled"
else
    echo "Warmup: disabled"
fi
dit_progress="${LIGHTX2V_DIT_PROGRESS:-1}"
case "${dit_progress}" in
    0 | 1) ;;
    *)
        echo "ERROR: LIGHTX2V_DIT_PROGRESS must be 0 or 1" >&2
        exit 1
        ;;
esac
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    cat "${runtime_config_json}"
    exit 0
fi

export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

infer_args=(
    --model_cls wan2.2_moe_distill
    --task i2v
    --model_path "${model_path}"
    --config_json "${runtime_config_json}"
    --prompt "${prompt}"
    --negative_prompt "${negative_prompt}"
    --image_path "${image_path}"
    --save_result_path "${save_result_path}"
)
if [[ "${warmup}" == 1 ]]; then
    infer_args+=(--warmup)
fi
if [[ -n "${SEED:-}" ]]; then
    infer_args+=(--seed "${SEED}")
fi
if [[ -n "${LIGHTX2V_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_infer_args <<<"${LIGHTX2V_EXTRA_ARGS}"
    infer_args+=("${extra_infer_args[@]}")
fi

cd "${models_root}"
torchrun_args=(
    --nnodes=1
    --nproc_per_node="${expected_gpu_count}"
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-29500}"
    -m lightx2v.infer
    "${infer_args[@]}"
)
if [[ "${dit_progress}" == 1 ]]; then
    progress_reporter="${script_dir}/report_dit_progress.py"
    if [[ ! -f "${progress_reporter}" ]]; then
        echo "ERROR: DiT progress reporter not found: ${progress_reporter}" >&2
        exit 1
    fi
    torchrun "${torchrun_args[@]}" 2>&1 | \
        python3 "${progress_reporter}" --ranks "${expected_gpu_count}"
else
    torchrun "${torchrun_args[@]}"
fi
