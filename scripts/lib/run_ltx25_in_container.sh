#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
evidence_dir="${LTX25_EVIDENCE_DIR:-/disk/ssd/lightx2v-runtime/ltx25-contract}"
census="${LTX25_CENSUS:-/disk/ssd/lightx2v-runtime/ltx25-day0/ltx25_weight_census.json}"
visible_devices="${HIP_VISIBLE_DEVICES:-0,1}"
world_size="${LTX25_WORLD_SIZE:-2}"
prompt="${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
negative_prompt="${NEGATIVE_PROMPT:-}"

[[ "${world_size}" == 2 ]] || { echo "ERROR: LTX-2.5 B0-reference requires exactly two ranks" >&2; exit 2; }
[[ "${visible_devices}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "ERROR: HIP_VISIBLE_DEVICES must contain exactly two GPU ids" >&2; exit 2; }
IFS=',' read -r gpu0 gpu1 <<<"${visible_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: GPU ids must be unique" >&2; exit 2; }

transformer="${model_root}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
text_encoder="${model_root}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
official_video_vae="${model_root}/vae/ltx-2.5-video-vae-bf16.safetensors"
conv_video_vae="${model_root}/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
audio_vae="${model_root}/vae/ltx-2.5-audio-vae-bf16.safetensors"
upscaler="${model_root}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
config_builder="${run_root}/scripts/lib/build_ltx25_runtime_config.py"
handoff_probe="${run_root}/scripts/lib/check_ltx23_phase_handoff.py"

mkdir -p -- "${evidence_dir}"
for required in "${lightx2v_path}/scripts/base/base.sh" "${transformer}" "${text_encoder}" \
    "${official_video_vae}" "${conv_video_vae}" "${audio_vae}" "${upscaler}" \
    "${census}" "${config_builder}" "${handoff_probe}"; do
    [[ -e "${required}" ]] || { echo "ERROR: required LTX-2.5 file not found: ${required}" >&2; exit 1; }
done

census_sha256="${LTX25_CENSUS_SHA256:-$(sha256sum -- "${census}" | awk '{print $1}')}"
runtime_config="${evidence_dir}/ltx25_runtime_config.json"
prompt_cache="${evidence_dir}/ltx25_prompt_embeddings.safetensors"
video_path="${evidence_dir}/ltx25_b0ref_conv_704x1280_24fps.mp4"
[[ ! -e "${prompt_cache}" ]] || { echo "ERROR: prompt cache already exists: ${prompt_cache}" >&2; exit 1; }

PYTHONDONTWRITEBYTECODE=1 python3 -B "${config_builder}" \
    --model-root "${model_root}" \
    --census "${census}" \
    --census-sha256 "${census_sha256}" \
    --output "${runtime_config}"

export PLATFORM=amd_rocm
export DTYPE=BF16
export GPU_ARCHS=gfx1201
export CU_NUM=24
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
export TOKENIZERS_PARALLELISM=false
export PYTORCH_TUNABLEOP_ENABLED=0
export HIP_VISIBLE_DEVICES="${visible_devices}"
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-/workspace/.cache/aiter}/gfx1201-cu24-ltx25-b0ref"
mkdir -p -- "${AITER_JIT_DIR}"

source "${lightx2v_path}/scripts/base/base.sh"

echo "LTX-2.5 B0-reference contract discovery"
echo "Assets: ${model_root}"
echo "GPUs: ${visible_devices}; SP2; BF16 two-slot disk stream"
echo "Workload: 704x1280, 121 frames @ 24 FPS, Stage 1=8 eval, Stage 2=3 eval"
echo "L3 route: conv_diagnostic (not official DiffVAE; not B0-acceptance eligible)"

python3 -m lightx2v.utils.ltx2_prompt_cache \
    --model-generation 2.5 \
    --checkpoint "${transformer}" \
    --text-encoder "${text_encoder}" \
    --video-vae "${official_video_vae}" \
    --audio-vae "${audio_vae}" \
    --upscaler "${upscaler}" \
    --census "${census}" \
    --census-sha256 "${census_sha256}" \
    --gemma-max-memory-json '{"0":"16GiB","1":"16GiB"}' \
    --gemma-attn-implementation sdpa \
    --output "${prompt_cache}" \
    --prompt "${prompt}" \
    --negative-prompt "${negative_prompt}"

python3 "${handoff_probe}" \
    --gpu-indices "${visible_devices}" \
    --prompt-cache "${prompt_cache}" \
    --output "${evidence_dir}/ltx25_handoff.l1_to_l2.json" \
    --limit-mib "${LTX25_HANDOFF_VRAM_LIMIT_MIB:-1024}" \
    --timeout "${LTX25_HANDOFF_TIMEOUT_SECONDS:-60}"

export LIGHTX2V_LTX25_PROMPT_CACHE="${prompt_cache}"
torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
    --seed "${SEED:-42}" \
    --model_cls ltx2 \
    --task t2av \
    --model_path "${model_root}" \
    --config_json "${runtime_config}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${video_path}"
