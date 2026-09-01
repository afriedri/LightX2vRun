#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${LIGHTX2V_LTX25_C1_IN_CONTAINER:-0}" == 1 ]] || {
    echo "ERROR: this C1 helper must be launched by lightx2vModelrun inside the ROCm image" >&2
    exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
result_dir="${LTX25_C1_RESULT_DIR:-/workspace/LightX2V/save_results}"
visible_devices="${HIP_VISIBLE_DEVICES:-0,1}"
world_size="${LTX25_WORLD_SIZE:-2}"
prompt="${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
negative_prompt="${NEGATIVE_PROMPT:-}"
tunableop_cache_pattern="${PYTORCH_TUNABLEOP_FILENAME:-}"
tunableop_cache_rank0="${LTX25_TUNABLEOP_CACHE_RANK0:-}"
tunableop_cache_rank1="${LTX25_TUNABLEOP_CACHE_RANK1:-}"

case "${world_size}" in
    2 | 4 | 8 | 16) ;;
    *) echo "ERROR: LTX-2.5 C1 supports exactly 2, 4, 8, or 16 ranks" >&2; exit 2 ;;
esac
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "ERROR: HIP_VISIBLE_DEVICES must contain comma-separated GPU ids" >&2; exit 2; }
IFS=',' read -r -a visible_device_ids <<<"${visible_devices}"
[[ "${#visible_device_ids[@]}" -eq "${world_size}" ]] || {
    echo "ERROR: LTX25_WORLD_SIZE=${world_size}, but HIP_VISIBLE_DEVICES contains ${#visible_device_ids[@]} ids" >&2
    exit 2
}
declare -A visible_device_seen=()
for device_id in "${visible_device_ids[@]}"; do
    [[ -z "${visible_device_seen[${device_id}]:-}" ]] || { echo "ERROR: duplicate GPU id ${device_id}" >&2; exit 2; }
    visible_device_seen["${device_id}"]=1
done
[[ "${PYTORCH_TUNABLEOP_ENABLED:-0}" == 1 ]] || { echo "ERROR: C1 requires TunableOp replay enabled" >&2; exit 2; }
[[ "${PYTORCH_TUNABLEOP_TUNING:-0}" == 0 ]] || { echo "ERROR: C1 runtime must not tune GEMMs" >&2; exit 2; }
[[ "${PYTORCH_TUNABLEOP_RECORD_UNTUNED:-1}" == 0 ]] || { echo "ERROR: C1 runtime must not record untuned GEMMs" >&2; exit 2; }
[[ "${tunableop_cache_pattern}" == *'%d'* ]] || { echo "ERROR: C1 TunableOp cache must use a rank-local %d pattern" >&2; exit 2; }
for rank_cache in "${tunableop_cache_rank0}" "${tunableop_cache_rank1}"; do
    [[ -n "${rank_cache}" && -f "${rank_cache}" ]] || { echo "ERROR: C1 TunableOp rank cache is missing: ${rank_cache:-unset}" >&2; exit 1; }
done
cmp -s -- "${tunableop_cache_rank0}" "${tunableop_cache_rank1}" || {
    echo "ERROR: LTX-2.5 C1 requires one unified SP2/4/8/16 TunableOp cache on every rank" >&2
    exit 1
}

transformer="${model_root}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
text_encoder="${model_root}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
official_video_vae="${model_root}/vae/ltx-2.5-video-vae-bf16.safetensors"
audio_vae="${model_root}/vae/ltx-2.5-audio-vae-bf16.safetensors"
upscaler="${model_root}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
config_builder="${run_root}/scripts/lib/build_ltx25_c1_runtime_config.py"
strict_rope_impl="${lightx2v_path}/lightx2v/models/networks/ltx2/infer/strict_rope.py"

video_path="${SAVE_RESULT_PATH:-${result_dir}/ltx25_c1_conv_704x1280_24fps.mp4}"
if [[ -n "${LTX25_C1_ARTIFACT_DIR:-}" ]]; then
    artifact_dir="${LTX25_C1_ARTIFACT_DIR}"
elif [[ -n "${SAVE_RESULT_PATH:-}" ]]; then
    artifact_dir="${video_path%.mp4}_artifacts"
else
    artifact_dir="${result_dir}"
fi
mkdir -p -- "${result_dir}" "$(dirname -- "${video_path}")" "${artifact_dir}"
for required in "${lightx2v_path}/scripts/base/base.sh" "${transformer}" "${text_encoder}" \
    "${official_video_vae}" "${audio_vae}" "${upscaler}" "${config_builder}" \
    "${strict_rope_impl}"; do
    [[ -e "${required}" ]] || { echo "ERROR: required LTX-2.5 C1 file not found: ${required}" >&2; exit 1; }
done

runtime_config="${artifact_dir}/ltx25_c1_runtime_config.json"
prompt_cache="${artifact_dir}/ltx25_c1_prompt_embeddings.safetensors"
prompt_collector="${artifact_dir}/ltx25_c1_prompt_collector.json"
for output in "${runtime_config}" "${prompt_cache}" "${prompt_collector}" "${video_path}"; do
    [[ ! -e "${output}" ]] || { echo "ERROR: C1 output already exists: ${output}" >&2; exit 1; }
done

PYTHONDONTWRITEBYTECODE=1 python3 -B "${config_builder}" \
    --model-root "${model_root}" \
    --seq-parallel-size "${world_size}" \
    --output "${runtime_config}"

PYTHONDONTWRITEBYTECODE=1 python3 -B - "${runtime_config}" "${world_size}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
world_size = int(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {
    "rope_type": "ltx25_triton_strict_rope",
    "ltx25_video_self_attn_type": "aiter_fav3_sage_bf16_attn",
    "ltx25_video_text_attn_type": "aiter_triton_dense_bf16_attn",
    "ltx25_distributed_video_vae": world_size >= 4,
}
for key, expected_value in expected.items():
    actual = config.get(key)
    if actual != expected_value:
        raise SystemExit(
            "ERROR: LTX-2.5 C1 runtime config selected "
            f"{key}={actual!r}; expected {expected_value!r}"
        )
parallel = config.get("parallel", {})
if parallel.get("seq_p_attn_type") != "ulysses" or parallel.get("seq_p_size") != world_size:
    raise SystemExit(
        "ERROR: LTX-2.5 C1 runtime config selected invalid parallel route: "
        f"{parallel!r}; expected pure Ulysses SP{world_size}"
    )
print(
    "LTX-2.5 C1 routes verified: "
    f"RoPE={expected['rope_type']}; "
    f"video_self={expected['ltx25_video_self_attn_type']}; "
    f"video_to_text={expected['ltx25_video_text_attn_type']}; "
    f"parallel=Ulysses SP{world_size}"
)
PY

export PLATFORM=amd_rocm
export DTYPE=BF16
export GPU_ARCHS=gfx1201
export CU_NUM=24
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
export TOKENIZERS_PARALLELISM=false
export PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
export LIGHTX2V_AMD_RMSNORM_BACKEND=ck
export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${artifact_dir}/ck_conv3d_routes"
export HIP_VISIBLE_DEVICES="${visible_devices}"
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-/workspace/.cache/aiter}/gfx1201-cu24-ltx25-c1"
mkdir -p -- "${AITER_JIT_DIR}"

# The integrated image contains one verified SP2/4/8/16 union cache, duplicated
# as rank0/rank1 only because PyTorch's filename contract is rank-indexed.  For
# larger Ulysses sizes materialize that same immutable evidence into rank-local
# writable paths; runtime tuning and untuned recording remain disabled.
if (( world_size > 2 )); then
    tunableop_scale_dir="${AITER_JIT_DIR}/tunableop-sp${world_size}"
    mkdir -p -- "${tunableop_scale_dir}"
    for (( rank = 0; rank < world_size; rank++ )); do
        cp -f -- "${tunableop_cache_rank0}" "${tunableop_scale_dir}/tunableop_results_full${rank}.csv"
    done
    tunableop_cache_pattern="${tunableop_scale_dir}/tunableop_results_full%d.csv"
    export PYTORCH_TUNABLEOP_FILENAME="${tunableop_cache_pattern}"
fi

model_path="${model_root}"
source "${lightx2v_path}/scripts/base/base.sh"

echo "LTX-2.5 C1 scale runtime"
echo "LightX2V source: ${lightx2v_path}"
echo "Assets: ${model_root}"
echo "Results: ${result_dir}"
echo "GPUs: ${visible_devices}; pure Ulysses SP${world_size}"
echo "GEMM: TunableOp read-only rank cache pattern ${tunableop_cache_pattern}"
if (( world_size > 2 )); then
    echo "GEMM scale policy: one verified SP2/4/8/16 union cache materialized read-only for every rank"
fi
echo "Attention: video_self=Aiter native Sage v2 (Aiter schedule selector; aligned LTX cores use r32_kv); video_to_text/video_to_audio=Aiter Triton dense; other roles=Torch SDPA"
echo "Q/K RMSNorm: Aiter CK; block RMSNorm=Torch"
echo "RoPE: LTX-2.5 strict/no-FMA Triton; structural miss falls back to Torch"
echo "Conv3D: shared Aiter CK adapter; CSV hit or Aiter AOT fallback Hot selection"
if (( world_size >= 4 )); then
    echo "Video VAE: distributed ${world_size}-tile spatial decode; RCCL P2P gather; rank0 BF16 blend"
else
    echo "Video VAE: validated rank0 tiled decode (SP2 baseline)"
fi

python3 -m lightx2v.utils.ltx2_prompt_cache \
    --model-generation 2.5 \
    --checkpoint "${transformer}" \
    --text-encoder "${text_encoder}" \
    --video-vae "${official_video_vae}" \
    --audio-vae "${audio_vae}" \
    --upscaler "${upscaler}" \
    --gemma-max-memory-json '{"0":"16GiB","1":"16GiB"}' \
    --gemma-attn-implementation sdpa \
    --collector-output "${prompt_collector}" \
    --output "${prompt_cache}" \
    --prompt "${prompt}" \
    --negative-prompt "${negative_prompt}"

export LIGHTX2V_LTX25_PROMPT_CACHE="${prompt_cache}"
torchrun --standalone --nproc_per_node="${world_size}" -m lightx2v.infer \
    --seed "${SEED:-42}" \
    --model_cls ltx2 \
    --task t2av \
    --model_path "${model_root}" \
    --config_json "${runtime_config}" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --save_result_path "${video_path}"

echo "LTX-2.5 C1 run complete: ${video_path}"
