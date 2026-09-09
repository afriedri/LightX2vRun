#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v/distill_fp8}"
distill_checkpoint="${DIT_QUANTIZED_CKPT:-${model_path}}"
base_config_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/distill/wan21/wan_i2v_distill_fp8_4step_cfg.json}"

visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "ERROR: HIP_VISIBLE_DEVICES must be comma-separated GPU ids" >&2
    exit 1
}
IFS=',' read -r -a gpu_devices <<<"${visible_devices}"
expected_gpu_count="${EXPECTED_GPU_COUNT:-${#gpu_devices[@]}}"
case "${expected_gpu_count}" in
    1 | 2 | 4 | 8 | 16) ;;
    *) echo "ERROR: Wan2.1 supports 1, 2, 4, 8, or 16 GPUs" >&2; exit 1 ;;
esac
[[ "${#gpu_devices[@]}" -eq "${expected_gpu_count}" ]] || {
    echo "ERROR: expected ${expected_gpu_count} GPUs, got ${visible_devices}" >&2
    exit 1
}
declare -A seen_gpu_devices=()
for gpu_device in "${gpu_devices[@]}"; do
    [[ -z "${seen_gpu_devices[${gpu_device}]:-}" ]] || {
        echo "ERROR: HIP_VISIBLE_DEVICES contains duplicate id ${gpu_device}" >&2
        exit 1
    }
    seen_gpu_devices["${gpu_device}"]=1
done
export HIP_VISIBLE_DEVICES="${visible_devices}"
unset CUDA_VISIBLE_DEVICES

target_height="${TARGET_HEIGHT:-720}"
target_width="${TARGET_WIDTH:-1280}"
target_video_length="${TARGET_VIDEO_LENGTH:-81}"
target_fps="${TARGET_FPS:-16}"
for target_value in "${target_height}" "${target_width}" "${target_video_length}" "${target_fps}"; do
    [[ "${target_value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: Wan2.1 target dimensions, frames, and FPS must be positive integers" >&2
        exit 1
    }
done
(( (target_video_length - 1) % 4 == 0 )) || {
    echo "ERROR: TARGET_VIDEO_LENGTH must satisfy the Wan VAE 4n+1 contract" >&2
    exit 1
}

self_attn_type="${SELF_ATTN_TYPE:-aiter_fav3_sage_bf16_attn}"
seq_p_attn_type="${SEQ_P_ATTN_TYPE:-ulysses}"
if [[ "${expected_gpu_count}" == 16 ]]; then
    seq_p_attn_type="${SEQ_P_ATTN_TYPE:-ring}"
fi
case "${self_attn_type}" in
    aiter_flydsl_bf16_flash_attn | aiter_triton_mha | aiter_fav3_sage_bf16_attn) ;;
    *) echo "ERROR: unsupported SELF_ATTN_TYPE=${self_attn_type}" >&2; exit 1 ;;
esac
case "${seq_p_attn_type}" in
    ulysses | ring) ;;
    *) echo "ERROR: unsupported SEQ_P_ATTN_TYPE=${seq_p_attn_type}" >&2; exit 1 ;;
esac
if [[ "${expected_gpu_count}" == 16 && "${seq_p_attn_type}" == ulysses ]]; then
    echo "ERROR: Wan2.1 has 40 attention heads; SP16 requires ring attention" >&2
    exit 1
fi

rope_type="${ROPE_TYPE:-triton_fused_qk_rope}"
case "${rope_type}" in
    torch_real_rope | triton_fused_qk_rope) ;;
    *) echo "ERROR: unsupported ROPE_TYPE=${rope_type}" >&2; exit 1 ;;
esac
offload_ratio="${OFFLOAD_RATIO:-1.0}"
use_tiling_vae="${USE_TILING_VAE:-0}"
vae_cpu_offload="${VAE_CPU_OFFLOAD:-0}"
t5_cpu_offload="${T5_CPU_OFFLOAD:-1}"
clip_cpu_offload="${CLIP_CPU_OFFLOAD:-1}"
for flag_name in use_tiling_vae vae_cpu_offload t5_cpu_offload clip_cpu_offload; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == 0 || "${flag_value}" == 1 ]] || {
        echo "ERROR: ${flag_name} must be 0 or 1" >&2
        exit 1
    }
done

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export AITER_LOG_TUNED_CONFIG="${AITER_LOG_TUNED_CONFIG:-1}"
export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export LIGHTX2V_AMD_MIOPEN=0
export USE_CHANNELS_LAST_3D=0
[[ "${GPU_ARCHS}" == gfx1201 && "${CU_NUM}" == 24 ]] || {
    echo "ERROR: this entry requires GPU_ARCHS=gfx1201 and CU_NUM=24" >&2
    exit 1
}

aiter_jit_root="${AITER_JIT_DIR:-/workspace/.cache/aiter}"
export AITER_JIT_DIR="${aiter_jit_root%/}/gfx1201-cu24-rowwise-v2"
mkdir -p "${AITER_JIT_DIR}"
[[ -f "${base_config_json}" ]] || { echo "ERROR: base config not found: ${base_config_json}" >&2; exit 1; }

runtime_config_dir="${RUNTIME_CONFIG_DIR:-/tmp/lightx2v-run}"
mkdir -p "${runtime_config_dir}"
runtime_config_json="${runtime_config_dir}/wan21-${expected_gpu_count}gpu-$$.json"
trap 'rm -f "${runtime_config_json}"' EXIT

python3 - "${base_config_json}" "${runtime_config_json}" "${model_path}" "${distill_checkpoint}" \
    "${offload_ratio}" "${self_attn_type}" "${rope_type}" "${expected_gpu_count}" \
    "${seq_p_attn_type}" "${use_tiling_vae}" "${vae_cpu_offload}" \
    "${t5_cpu_offload}" "${clip_cpu_offload}" "${target_height}" "${target_width}" \
    "${target_video_length}" "${target_fps}" <<'PY'
import json
import os
import sys

(
    source_path,
    output_path,
    model_path,
    distill_checkpoint,
    offload_ratio_text,
    self_attn_type,
    rope_type,
    seq_p_size_text,
    seq_p_attn_type,
    use_tiling_vae_text,
    vae_cpu_offload_text,
    t5_cpu_offload_text,
    clip_cpu_offload_text,
    target_height_text,
    target_width_text,
    target_video_length_text,
    target_fps_text,
) = sys.argv[1:]

offload_ratio = float(offload_ratio_text)
if not 0.0 <= offload_ratio <= 1.0:
    raise SystemExit(f"OFFLOAD_RATIO must be in [0, 1], got {offload_ratio_text!r}")
with open(source_path, encoding="utf-8") as source_file:
    config = json.load(source_file)

seq_p_size = int(seq_p_size_text)
config.update(
    {
        "model_cls": "wan2.1",
        "distill_method": "dmd2",
        "warmup": os.environ.get("LIGHTX2V_WARMUP", "0") == "1",
        "infer_steps": 4,
        "target_video_length": int(target_video_length_text),
        "target_height": int(target_height_text),
        "target_width": int(target_width_text),
        "fps": int(target_fps_text),
        "text_len": 512,
        "self_attn_1_type": self_attn_type,
        "cross_attn_1_type": "aiter_triton_mha",
        "cross_attn_2_type": "aiter_triton_mha",
        "sample_guide_scale": 5,
        "sample_shift": 5,
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "offload_ratio": offload_ratio,
        "t5_cpu_offload": t5_cpu_offload_text == "1",
        "clip_cpu_offload": clip_cpu_offload_text == "1",
        "vae_cpu_offload": vae_cpu_offload_text == "1",
        "use_tiling_vae": use_tiling_vae_text == "1",
        "denoising_step_list": [1000, 750, 500, 250],
        "dit_quantized": True,
        "dit_quantized_ckpt": distill_checkpoint,
        "dit_quant_scheme": "fp8-sgl",
        "t5_quantized": True,
        "t5_quantized_ckpt": f"{model_path}/models_t5_umt5-xxl-enc-fp8.pth",
        "t5_quant_scheme": "fp8-sgl",
        "clip_quantized": True,
        "clip_quantized_ckpt": f"{model_path}/clip-fp8.pth",
        "clip_quant_scheme": "fp8-sgl",
        "rms_norm_type": "sgl-kernel",
        "rope_type": rope_type,
        "rope_chunk": False,
        "modulate_type": "torch",
        "feature_caching": "NoCaching",
        "parallel": {
            "seq_p_size": seq_p_size,
            "seq_p_attn_type": seq_p_attn_type,
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
            "vae_parallel": seq_p_size > 1,
        },
    }
)
config.pop("rope_chunk_size", None)
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=4, ensure_ascii=False)
    output_file.write("\n")
PY

prompt="${PROMPT:-A close-up cinematic view of a person cooking in a warm, sunlit kitchen, using a wooden spatula to stir-fry colorful fresh vegetables in a black frying pan.}"
negative_prompt="${NEGATIVE_PROMPT:-镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，最差质量，低质量，畸形，杂乱的背景}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_2.jpg}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/wan21_${target_height}p_${target_video_length}f_${target_fps}fps_${expected_gpu_count}gpu.mp4}"
seed="${SEED:-1234}"

echo "Wan2.1 I2V 14B: ${target_height}x${target_width}, ${target_video_length} frames @ ${target_fps} FPS, 4 steps"
echo "GPUs: ${visible_devices} (${expected_gpu_count} ranks)"
echo "Parallel: SP=${expected_gpu_count}, attention=${seq_p_attn_type}"
echo "Offload: block/${offload_ratio}; T5=${t5_cpu_offload}, CLIP=${clip_cpu_offload}"
echo "VAE: CK Conv3d v1, resident=$((1 - vae_cpu_offload)), tiling=${use_tiling_vae}, parallel=$((expected_gpu_count > 1 ? 1 : 0))"
echo "Config: ${runtime_config_json}"
echo "Output: ${save_result_path}"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
    cat "${runtime_config_json}"
    exit 0
fi
python3 -c 'import importlib; import aiter; assert callable(aiter.ck_conv3d_v1); assert aiter.is_ck_conv3d_v1_prebuilt(); module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'

export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
infer_args=(
    --model_cls wan2.1
    --task i2v
    --model_path "${model_path}"
    --config_json "${runtime_config_json}"
    --prompt "${prompt}"
    --negative_prompt "${negative_prompt}"
    --image_path "${image_path}"
    --save_result_path "${save_result_path}"
    --seed "${seed}"
)

cd "${models_root}"
torchrun_args=(
    --nnodes=1
    --nproc_per_node="${expected_gpu_count}"
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-29500}"
    -m lightx2v.infer
    "${infer_args[@]}"
)
if [[ "${LIGHTX2V_DIT_PROGRESS:-1}" == 1 ]]; then
    torchrun "${torchrun_args[@]}" 2>&1 | python3 "${script_dir}/../report_dit_progress.py" --ranks "${expected_gpu_count}"
else
    torchrun "${torchrun_args[@]}"
fi
