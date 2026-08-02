#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v/distill_fp8}"
distill_checkpoint="${DIT_QUANTIZED_CKPT:-${model_path}}"
base_config_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/distill/wan21/wan_i2v_distill_fp8_4step_cfg.json}"
self_attn_type="${SELF_ATTN_TYPE:-aiter_flydsl_bf16_flash_attn}"
case "${self_attn_type}" in
    aiter_flydsl_bf16_flash_attn | aiter_fav3_sage_bf16_attn) ;;
    *)
        echo "ERROR: unsupported SELF_ATTN_TYPE=${self_attn_type}" >&2
        exit 1
        ;;
esac

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
if [[ ! "${HIP_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: the Wan2.1 RTX 4090 comparison requires exactly one HIP_VISIBLE_DEVICES id" >&2
    exit 1
fi

[[ -f "${base_config_json}" ]] || {
    echo "ERROR: base config not found: ${base_config_json}" >&2
    exit 1
}

runtime_config_dir="${RUNTIME_CONFIG_DIR:-/tmp/lightx2v-run}"
mkdir -p "${runtime_config_dir}"
runtime_config_json="${runtime_config_dir}/wan21-i2v-4090-comparison-$$.json"
trap 'rm -f "${runtime_config_json}"' EXIT
offload_ratio="${OFFLOAD_RATIO:-1.0}"
if [[ -n "${ROPE_CHUNK_SIZE:-}" ]]; then
    rope_chunk_size="${ROPE_CHUNK_SIZE}"
elif [[ "${self_attn_type}" == "aiter_fav3_sage_bf16_attn" ]]; then
    rope_chunk_size=8192
else
    rope_chunk_size=0
fi
if [[ ! "${rope_chunk_size}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROPE_CHUNK_SIZE must be a non-negative integer, got ${rope_chunk_size}" >&2
    exit 1
fi

python3 - "${base_config_json}" "${runtime_config_json}" "${model_path}" "${distill_checkpoint}" "${offload_ratio}" "${self_attn_type}" "${rope_chunk_size}" <<'PY'
import json
import sys

source_path, output_path, model_path, distill_checkpoint, offload_ratio_text, self_attn_type, rope_chunk_size_text = sys.argv[1:]
try:
    offload_ratio = float(offload_ratio_text)
except ValueError as exc:
    raise SystemExit(f"OFFLOAD_RATIO must be a number in [0, 1], got {offload_ratio_text!r}") from exc
if not 0.0 <= offload_ratio <= 1.0:
    raise SystemExit(f"OFFLOAD_RATIO must be in [0, 1], got {offload_ratio_text!r}")
rope_chunk_size = int(rope_chunk_size_text)

with open(source_path, encoding="utf-8") as source_file:
    config = json.load(source_file)

# Match the published RTX 4090 LightX2V_6-Distill 720p workload while
# retaining the GFX1201 ROCm attention and FP8 implementations.
config.update(
    {
        "infer_steps": 4,
        "target_video_length": 81,
        "text_len": 512,
        "target_height": 720,
        "target_width": 1280,
        "self_attn_1_type": self_attn_type,
        "cross_attn_1_type": "aiter_triton_bf16_flash_attn",
        "cross_attn_2_type": "aiter_triton_bf16_flash_attn",
        "sample_guide_scale": 5,
        "sample_shift": 5,
        "enable_cfg": False,
        "cpu_offload": True,
        "offload_granularity": "block",
        "offload_ratio": offload_ratio,
        "t5_cpu_offload": True,
        "clip_cpu_offload": True,
        "vae_cpu_offload": True,
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
        "use_tiling_vae": True,
        "rms_norm_type": "sgl-kernel",
        "rope_type": "torch_real_rope",
        "rope_chunk": rope_chunk_size > 0,
        "modulate_type": "torch",
        "feature_caching": "NoCaching",
        "parallel": {
            "seq_p_size": 1,
            "seq_p_attn_type": "ulysses",
            "seq_p_prepost_backend": "torch",
            "seq_p_a2a_backend": "torch",
        },
    }
)
if rope_chunk_size > 0:
    config["rope_chunk_size"] = rope_chunk_size
else:
    config.pop("rope_chunk_size", None)

with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=4, ensure_ascii=False)
    output_file.write("\n")
PY

prompt="${PROMPT:-A close-up cinematic view of a person cooking in a warm,sunlit kitchen, using a wooden spatula to stir-fry a colorful mix of freshvegetables—carrots, broccoli, and bell peppers—in a black frying pan on amodern induction stove. The scene captures the glistening texture of thevegetables, steam gently rising, and subtle reflections on the stove surface.In the background, soft-focus jars, fruits, and a window with natural daylightcreate a cozy atmosphere. The hand motions are smooth and rhythmic, with a realisticsense of motion blur and lighting.}"
negative_prompt="${NEGATIVE_PROMPT:-镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_2.jpg}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/benchmark_wan21_i2v_720p_81f_fp8_gfx1201_1gpu.mp4}"
seed="${SEED:-1234}"

echo "Benchmark: Wan2.1 I2V 14B 720p, 81 frames, 4 steps, CFG disabled"
echo "Reference: upstream RTX 4090 LightX2V_6-Distill = 171s"
echo "GPU: ${HIP_VISIBLE_DEVICES} (single rank, SP=1)"
echo "Self attention: ${self_attn_type}"
if (( rope_chunk_size > 0 )); then
    echo "RoPE: torch_real_rope, chunked at ${rope_chunk_size} sequence rows"
else
    echo "RoPE: torch_real_rope, unchunked"
fi
echo "Weights: ${distill_checkpoint} (4-step Distill per-channel FP8, Aiter fp8-sgl runtime)"
resident_blocks="$(python3 -c 'import math, sys; ratio=float(sys.argv[1]); print(40-math.ceil(40*ratio))' "${offload_ratio}")"
streamed_blocks="$((40 - resident_blocks))"
echo "Offload: block; ratio=${offload_ratio} (${streamed_blocks} streamed, ${resident_blocks} GPU-resident)"
echo "Encoder offload: T5/CLIP/VAE CPU offload enabled"
echo "Seed: ${seed}"
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
    --model_cls wan2.1_distill
    --task i2v
    --model_path "${model_path}"
    --config_json "${runtime_config_json}"
    --prompt "${prompt}"
    --negative_prompt "${negative_prompt}"
    --image_path "${image_path}"
    --save_result_path "${save_result_path}"
    --seed "${seed}"
)
if [[ "${warmup}" == "1" ]]; then
    infer_args+=(--warmup)
fi
if [[ -n "${LIGHTX2V_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_infer_args <<<"${LIGHTX2V_EXTRA_ARGS}"
    infer_args+=("${extra_infer_args[@]}")
fi

cd "${models_root}"
torchrun_args=(
    --nnodes=1
    --nproc_per_node=1
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-29500}"
    -m lightx2v.infer
    "${infer_args[@]}"
)
if [[ "${dit_progress}" == "1" ]]; then
    progress_reporter="${script_dir}/report_dit_progress.py"
    [[ -f "${progress_reporter}" ]] || {
        echo "ERROR: DiT progress reporter not found: ${progress_reporter}" >&2
        exit 1
    }
    torchrun "${torchrun_args[@]}" 2>&1 | \
        python3 "${progress_reporter}" --ranks 1
else
    torchrun "${torchrun_args[@]}"
fi
