#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${LIGHTX2V_LTX25_IN_CONTAINER:-0}" == 1 ]] || {
    echo "ERROR: this helper must be launched by lightx2vModelrun inside the ROCm image" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
evidence_dir="${LTX25_EVIDENCE_DIR:-/workspace/LightX2V/save_results}"
visible_devices="${HIP_VISIBLE_DEVICES:-0,1}"
world_size="${LTX25_WORLD_SIZE:-2}"
run_id="${LIGHTX2V_LTX25_CONTRACT_RUN_ID:-ltx25-contract-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
require_upsample="${LTX25_REQUIRE_UPSAMPLE:-1}"
ultra="${LIGHTX2V_LTX25_CONTRACT_ULTRA:-0}"
prompt="${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
negative_prompt="${NEGATIVE_PROMPT:-}"

[[ "${world_size}" == 2 ]] || { echo "ERROR: LTX-2.5 B0-reference requires exactly two ranks" >&2; exit 2; }
[[ "${visible_devices}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "ERROR: HIP_VISIBLE_DEVICES must contain exactly two GPU ids" >&2; exit 2; }
IFS=',' read -r gpu0 gpu1 <<<"${visible_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: GPU ids must be unique" >&2; exit 2; }
case "${require_upsample}" in 0|1) ;; *) echo "ERROR: LTX25_REQUIRE_UPSAMPLE must be 0 or 1" >&2; exit 2 ;; esac
case "${ultra}" in 0|1) ;; *) echo "ERROR: LIGHTX2V_LTX25_CONTRACT_ULTRA must be 0 or 1" >&2; exit 2 ;; esac
[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: collector run id contains unsafe characters" >&2; exit 2; }

transformer="${model_root}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
text_encoder="${model_root}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
official_video_vae="${model_root}/vae/ltx-2.5-video-vae-bf16.safetensors"
conv_video_vae="${model_root}/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
audio_vae="${model_root}/vae/ltx-2.5-audio-vae-bf16.safetensors"
upscaler="${model_root}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
config_builder="${run_root}/scripts/lib/build_ltx25_runtime_config.py"
validator="${run_root}/scripts/collector/validate_ltx25_contract_bundle.py"

mkdir -p -- "${evidence_dir}"
for required in "${lightx2v_path}/scripts/base/base.sh" "${transformer}" "${text_encoder}" \
    "${official_video_vae}" "${conv_video_vae}" "${audio_vae}" "${upscaler}" \
    "${config_builder}" "${validator}"; do
    [[ -e "${required}" ]] || { echo "ERROR: required LTX-2.5 file not found: ${required}" >&2; exit 1; }
done

shopt -s nullglob
existing=(
    "${evidence_dir}"/ltx25_collector*.json
    "${evidence_dir}"/ltx25_contract_bundle*.json
)
shopt -u nullglob
(( ${#existing[@]} == 0 )) || { echo "ERROR: collector outputs already exist in ${evidence_dir}" >&2; exit 1; }

export LIGHTX2V_LTX25_CONTRACT_COLLECTOR=1
export LIGHTX2V_LTX25_CONTRACT_RUN_ID="${run_id}"
export LIGHTX2V_LTX25_CONTRACT_COLLECTOR_L1_OUTPUT="${evidence_dir}/ltx25_collector.l1.json"
export LIGHTX2V_LTX25_CONTRACT_COLLECTOR_OUTPUT="${evidence_dir}/ltx25_collector.rank{rank}.json"

runtime_config="${evidence_dir}/ltx25_runtime_config.json"
prompt_cache="${evidence_dir}/ltx25_prompt_embeddings.safetensors"
video_path="${evidence_dir}/ltx25_b0ref_conv_704x1280_24fps.mp4"
[[ ! -e "${prompt_cache}" ]] || { echo "ERROR: prompt cache already exists: ${prompt_cache}" >&2; exit 1; }

PYTHONDONTWRITEBYTECODE=1 python3 -B "${config_builder}" \
    --model-root "${model_root}" \
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

# scripts/base/base.sh consumes these legacy shell variable names directly.
model_path="${model_root}"
source "${lightx2v_path}/scripts/base/base.sh"

echo "LTX-2.5 B0-reference contract discovery (inside ROCm image)"
echo "Contract detail: $([[ "${ultra}" == 1 ]] && echo ultra || echo standard)"
echo "LightX2V source: ${lightx2v_path}"
echo "Assets: ${model_root}"
echo "Evidence: ${evidence_dir}"
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
    --gemma-max-memory-json '{"0":"16GiB","1":"16GiB"}' \
    --gemma-attn-implementation sdpa \
    --output "${prompt_cache}" \
    --prompt "${prompt}" \
    --negative-prompt "${negative_prompt}"

export LIGHTX2V_LTX25_PROMPT_CACHE="${prompt_cache}"
torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
    --seed "${SEED:-42}" \
    --model_cls ltx2_5 \
    --task t2av \
    --model_path "${model_root}" \
    --config_json "${runtime_config}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${video_path}"

validator_args=(
    --output-dir "${evidence_dir}"
    --expected-ranks "${world_size}"
)
if [[ "${require_upsample}" == 0 ]]; then
    validator_args+=(--no-require-upsample)
fi
if [[ "${ultra}" == 1 ]]; then
    validator_args+=(--require-ultra)
fi
PYTHONDONTWRITEBYTECODE=1 python3 -B "${validator}" "${validator_args[@]}"
