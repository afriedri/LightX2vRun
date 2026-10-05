#!/usr/bin/env bash
# In-container entry for MiniMax-H3 Ref2AV, BF16, TP2 on two gfx1201 GPUs.
# Started by scripts/run_14_minimax_h3_ref2av_tp2_768p_15s.sh inside a
# rocm/vllm-dev:gfx1201-h3-* image (LightX2V source in /app/LightX2V).
#
# Workload: 1344x768, 362 frames at 24 FPS (15.08 s) plus 32 kHz audio,
# seed 42, default 20 steps, tensor parallel over two ranks. No quantization
# beyond the Sage attention kernel's own int8 QK / fp8 PV.
#
# Inputs (environment):
#   MODEL_PATH              MiniMax-H3 model directory (HF layout)
#   H3_PROMPT_FILE          prompt text file
#   H3_REFERENCE_IMAGES     newline-separated reference images (1-9, prompt order)
#   H3_INFER_STEPS          denoising steps (default 20; the image ships AdaLN
#                           caches for 1, 3 and 20)
#   H3_BASE_RUNTIME_CONFIG  base runtime config JSON (default: the image's
#                           rooftop_15s runtime_config.json; 768x1344, 362 frames)
#   SAVE_RESULT_PATH, H3_RUN_LOG, SEED, NPROC_PER_NODE
# Switches (1/0) are documented in docs/minimax-h3-ref2av-tp2-gfx1201.md.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/minimax_h3_gfx1201_rocm_env.sh
source "${script_dir}/minimax_h3_gfx1201_rocm_env.sh"

steps="${H3_INFER_STEPS:-20}"
model="${MODEL_PATH:-/models/minimax_h3}"
lightx2v_path="${LIGHTX2V_PATH}"
base_config="${H3_BASE_RUNTIME_CONFIG:-${lightx2v_path}/assets/inputs/minimax_h3/rooftop_15s/runtime_config.json}"
prompt_file="${H3_PROMPT_FILE:?H3_PROMPT_FILE is required}"
runtime_config="${H3_RUNTIME_CONFIG:-/tmp/minimax_h3_ref2av_tp2_runtime_config.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_ref2av_tp2_${steps}.mp4}"
run_log="${H3_RUN_LOG:-${save_result_path%.mp4}.log}"

[[ "${steps}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: H3_INFER_STEPS must be a positive integer" >&2; exit 2; }
[[ -d "${model}" ]] || { echo "ERROR: model directory not found: ${model}" >&2; exit 1; }
[[ -f "${prompt_file}" ]] || { echo "ERROR: prompt file not found: ${prompt_file}" >&2; exit 1; }
[[ -f "${base_config}" ]] || { echo "ERROR: base runtime config not found: ${base_config}" >&2; exit 1; }
[[ -n "${H3_REFERENCE_IMAGES:-}" ]] || { echo "ERROR: H3_REFERENCE_IMAGES is required" >&2; exit 2; }
mapfile -t reference_images <<<"${H3_REFERENCE_IMAGES}"
(( ${#reference_images[@]} >= 1 && ${#reference_images[@]} <= 9 )) || {
    echo "ERROR: Ref2AV accepts 1 to 9 reference images" >&2; exit 2;
}
for reference in "${reference_images[@]}"; do
    [[ -f "${reference}" ]] || { echo "ERROR: reference image not found: ${reference}" >&2; exit 1; }
done
image_path_arg="$(IFS=,; echo "${reference_images[*]}")"

# Attention core and per-call schedule. The launcher resolves H3_SAGE_PRESET
# into these two variables; the image default is the exact esp321w core.
export AITER_GFX1201_SAGE_ASM="${AITER_GFX1201_SAGE_ASM:-esp321w}"
export AITER_GFX1201_SAGE_ASM_SCHED="${AITER_GFX1201_SAGE_ASM_SCHED-}"
# Fused TP2 block tail with the IPC pull reducer (bitwise equal to the unfused tail).
export RADEON_CORESW_FUSED_TAIL="${FUSED_TAIL:-${RADEON_CORESW_FUSED_TAIL:-1}}"
# FFN down GEMM through hipBLASLt (0 = aiter Triton kernel).
export AITER_GFX1201_DOWN_BLAS="${AITER_GFX1201_DOWN_BLAS:-1}"
# No FP8 GEMM or VAE quantization.
export RADEON_CORESW_FP8=0 LIGHTX2V_H3_VAE_FP8=0
# Pageable text-encoder staging and VAE encode split over both ranks.
export TE_PAGEABLE="${TE_PAGEABLE:-1}" VAE_ENC_PAR="${VAE_ENC_PAR:-1}"
# Non-DiT pipeline items (each 0 = off; VAE_DEC_BATCH 0/1 = one tile per call).
export LIGHTX2V_H3_VAE_DEC_BATCH="${LIGHTX2V_H3_VAE_DEC_BATCH:-2}"
export LIGHTX2V_H3_VAE_TRITON_ATTN="${LIGHTX2V_H3_VAE_TRITON_ATTN:-1}"
export LIGHTX2V_H3_TE_PREFETCH_LATE="${LIGHTX2V_H3_TE_PREFETCH_LATE:-1}"
export LIGHTX2V_H3_TE_ANON_HOST="${LIGHTX2V_H3_TE_ANON_HOST:-1}"
export LIGHTX2V_H3_GC_FREEZE="${LIGHTX2V_H3_GC_FREEZE:-1}"
export LIGHTX2V_H3_ASYNC_VAE_LOAD="${LIGHTX2V_H3_ASYNC_VAE_LOAD:-1}"
export LIGHTX2V_H3_VAE_ENC_PREWARM="${LIGHTX2V_H3_VAE_ENC_PREWARM:-1}"
export LIGHTX2V_H3_DIT_PIN_SLAB="${LIGHTX2V_H3_DIT_PIN_SLAB:-1}"
export LIGHTX2V_H3_AUDIO_POLY_UPSAMPLE="${LIGHTX2V_H3_AUDIO_POLY_UPSAMPLE:-1}"
export LIGHTX2V_H3_VAE_STREAM_SAVE="${LIGHTX2V_H3_VAE_STREAM_SAVE:-1}"

python3 - "${base_config}" "${runtime_config}" "${steps}" <<'PY'
import json, os, pathlib, sys

src, dst, steps = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
cfg = json.loads(src.read_text())
cfg["infer_steps"] = steps
cfg["parallel"] = {"tensor_p_size": 2, "seq_p_size": 1}
cfg["tensor_parallel"] = True
cfg["seq_parallel"] = False
# Qwen3-VL text encoder TP2 over the two ranks (H3_TEXT_ENCODER_TP=0: replicated).
cfg["text_encoder_tensor_parallel"] = os.environ.get("H3_TEXT_ENCODER_TP", "1") != "0"
cfg["text_encoder_host_pinned"] = os.environ["TE_PAGEABLE"] == "0"
cfg["vae_encode_parallel"] = os.environ["VAE_ENC_PAR"] != "0"
cfg["attn_type"] = "gfx1201_sage_attn"
cfg["radeon_gfx1201_h3"] = True
cfg["radeon_gfx1201_qk_norm_rope"] = True
cfg["use_fused_qkv"] = False
cfg["use_fused_qkv_norm_rope"] = False
cfg["refiner_attn_type"] = "aiter_attn"
cfg["vae_cpu_offload"] = False
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"infer_steps={steps} config={dst} " + " ".join(f"{k}={cfg[k]}" for k in (
    "text_encoder_tensor_parallel", "text_encoder_host_pinned", "vae_encode_parallel")))
PY

adaln_cache="${lightx2v_path}/cache/adaln/minimax_h3/ref2av_$(printf '%02d' "${steps}")steps_shift_12.0_3.0"
if [[ ! -d "${adaln_cache}" ]]; then
    echo "AdaLN cache missing: ${adaln_cache}; generating (the image ships caches for 1, 3 and 20 steps)"
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES%%,*}" python3 \
        "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
        --model_path "${model}" --config_json "${runtime_config}" --model-variant ref2av
fi

echo "MiniMax-H3 Ref2AV TP2 gfx1201: 1344x768, 362 frames @ 24 FPS, ${steps} steps, seed ${SEED:-42}"
echo "Sage core: AITER_GFX1201_SAGE_ASM=${AITER_GFX1201_SAGE_ASM} AITER_GFX1201_SAGE_ASM_SCHED=${AITER_GFX1201_SAGE_ASM_SCHED}"
echo "Model: ${model}"
echo "Prompt: ${prompt_file}"
echo "References (${#reference_images[@]}): ${image_path_arg}"
echo "Output: ${save_result_path}  log: ${run_log}"

mkdir -p "$(dirname -- "${save_result_path}")" "$(dirname -- "${run_log}")"
cd "${lightx2v_path}"
set +e
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m lightx2v.infer \
    --seed "${SEED:-42}" \
    --model_cls minimax_h3 \
    --model-variant ref2av \
    --task ref2av \
    --model_path "${model}" \
    --config_json "${runtime_config}" \
    --prompt "$(<"${prompt_file}")" \
    --image_path "${image_path_arg}" \
    --size 768 1344 \
    --save_result_path "${save_result_path}" 2>&1 | tee "${run_log}"
rc=${PIPESTATUS[0]}
set -e

echo "=================== summary (${steps} steps, rc=${rc}) ==================="
grep -a -E 'pin slab|stream|gc.freeze|anonymous memory|MIOpen prewarm|sage ASM variant|FUSED_TAIL|\[ACTIVE\] TP2|route=ASM|Video VAE: fused|^infer_steps=' \
    "${run_log}" | cut -c1-400 | sort -u | head -30 || true
echo "tracebacks: $(grep -a -c -E 'Traceback|OutOfMemory' "${run_log}" || true)"
for rank in 0 1; do
    step_times="$(grep -a "Rank ${rank} - .*Run Dit every step" "${run_log}" | grep -oE 'cost [0-9.]+' | awk '{print $2}' | paste -sd' ' || true)"
    echo "rank ${rank} step times: ${step_times}"
done
total="$(grep -a 'Total Cost' "${run_log}" | grep -oE 'cost [0-9.]+' | awk '{print $2}' | sort -n | tail -1 || true)"
if [[ -n "${total}" ]]; then
    printf 'Total %.2f s\n' "${total}"
fi
echo "video: ${save_result_path}   log: ${run_log}"
exit "${rc}"
