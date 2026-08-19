#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
source "${run_root}/scripts/lib/minimax_h3_prompt_presets.sh"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_path="${MODEL_PATH:-/models/minimax_h3}"
source_mode="${H3_SOURCE_MODE:-image}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}"
last_frame_path="${LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}"
overlay_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_fl2av_4step_768p_gfx1201.json}"
publish_json="${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json"

visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+){0,3}$ ]] || {
    echo "ERROR: MiniMax-H3 Day0 requires one, two, or four comma-separated HIP GPU ids, got '${visible_devices}'" >&2
    exit 1
}
IFS=',' read -r -a visible_device_ids <<<"${visible_devices}"
gpu_count="${#visible_device_ids[@]}"
case "${gpu_count}" in
    1 | 2 | 4) ;;
    *) echo "ERROR: MiniMax-H3 Day0 supports exactly 1, 2, or 4 GPUs" >&2; exit 1 ;;
esac
declare -A seen_visible_device_ids=()
for visible_device_id in "${visible_device_ids[@]}"; do
    [[ -z "${seen_visible_device_ids[${visible_device_id}]:-}" ]] || {
        echo "ERROR: MiniMax-H3 Day0 GPU ids must be unique, got '${visible_devices}'" >&2
        exit 1
    }
    seen_visible_device_ids["${visible_device_id}"]=1
done
export HIP_VISIBLE_DEVICES="${visible_devices}"
unset CUDA_VISIBLE_DEVICES

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
# The measured H3 BF16 GEMM/LoRA winner is ordinary torch.mm with PyTorch's
# default ROCm BLAS heuristic (hipBLASLt for the collected hot shapes).
# TunableOp replay regressed QKV LoRA-up and is deliberately excluded.
export PYTORCH_TUNABLEOP_ENABLED=0
unset TORCH_BLAS_PREFER_CUBLASLT TORCH_BLAS_PREFER_HIPBLASLT
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
[[ "${GPU_ARCHS}" == gfx1201 && "${CU_NUM}" == 24 ]] || {
    echo "ERROR: MiniMax-H3 Day0 requires GPU_ARCHS=gfx1201 and CU_NUM=24" >&2
    exit 1
}
case "${source_mode}" in
    image) ;;
    *) echo "ERROR: H3_SOURCE_MODE must be image; host source overlays are unsupported" >&2; exit 1 ;;
esac

aiter_jit_root="${AITER_JIT_DIR:-/workspace/.cache/aiter}"
export AITER_JIT_DIR="${aiter_jit_root%/}/gfx1201-cu24-h3"
mkdir -p "${AITER_JIT_DIR}"

[[ -f "${publish_json}" ]] || { echo "ERROR: published H3 config not found: ${publish_json}" >&2; exit 1; }
[[ -f "${overlay_json}" ]] || { echo "ERROR: H3 overlay config not found: ${overlay_json}" >&2; exit 1; }
[[ -d "${model_path}" ]] || { echo "ERROR: H3 model directory not found: ${model_path}" >&2; exit 1; }

evidence_dir="${H3_EVIDENCE_DIR:-}"
runtime_config_dir="${RUNTIME_CONFIG_DIR:-${evidence_dir:-/tmp/lightx2v-run}}"
mkdir -p "${runtime_config_dir}"
if [[ -n "${evidence_dir}" ]]; then
    runtime_config_json="${runtime_config_dir}/runtime_config.json"
    preflight_json="${evidence_dir}/preflight.json"
else
    runtime_config_json="${runtime_config_dir}/minimax-h3-$$.json"
    preflight_json="${runtime_config_dir}/minimax-h3-preflight-$$.json"
    ulysses_preflight_log="${runtime_config_dir}/minimax-h3-ulysses-preflight-$$.log"
    trap 'rm -f "${runtime_config_json}" "${preflight_json}" "${ulysses_preflight_log}"' EXIT
fi

smoke_flag=()
kernel_smoke_flag=()
target_height=768
target_width=1344
prompt_duration_seconds=5.17
if [[ "${H3_SMOKE:-0}" == 1 ]]; then
    smoke_flag+=(--smoke)
    target_height=480
    target_width=480
fi
if [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    kernel_smoke_flag+=(--kernel-smoke)
fi

python3 "${script_dir}/build_minimax_h3_runtime_config.py" \
    --publish-json "${publish_json}" \
    --overlay-json "${overlay_json}" \
    --output-json "${runtime_config_json}" \
    --seq-parallel-size "${gpu_count}" \
    "${smoke_flag[@]}"

python3 "${script_dir}/preflight_minimax_h3_day0.py" \
    --model-dir "${model_path}" \
    --turbo-dir /models/minimax_h3_turbo \
    --first-frame "${image_path}" \
    --last-frame "${last_frame_path}" \
    --runtime-config "${runtime_config_json}" \
    --expected-world-size "${gpu_count}" \
    --source-mode "${source_mode}" \
    "${kernel_smoke_flag[@]}" | tee "${preflight_json}"

if (( gpu_count > 1 )) && [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    ulysses_preflight_log="${ulysses_preflight_log:-${runtime_config_dir}/ulysses_preflight.log}"
    torchrun --standalone --nproc_per_node="${gpu_count}" \
        "${script_dir}/preflight_minimax_h3_day0.py" \
        --distributed-attention-only 2>&1 | tee "${ulysses_preflight_log}"
fi

if [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 || "${CONFIG_ONLY:-0}" == 1 ]]; then
    cat "${runtime_config_json}"
    exit 0
fi

resolve_minimax_h3_prompt \
    "${run_root}" \
    "${H3_PROMPT_PRESET:-bird-stable}" \
    "${H3_PROMPT_FILE:-}" \
    "${prompt_duration_seconds}"
prompt="${MINIMAX_H3_PROMPT}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_fl2av.mp4}"

echo "MiniMax-H3 FL2V gfx1201"
echo "GPUs: ${visible_devices} (world_size=${gpu_count})"
parallel_mode="single GPU"
if (( gpu_count > 1 )); then
    parallel_mode="Ulysses SP${gpu_count} + VAE tile parallel"
fi
echo "Parallel: ${parallel_mode}"
echo "H1 PatchEmbed API: torch F.linear (ATen operator: addmm; measured device backend: hipBLASLt)"
echo "Core Attention: aiter_fav3_sage_bf16_attn (sage_attn_v2_gfx1201)"
echo "Dense GEMM/LoRA: torch.mm -> default ROCm BLAS heuristic (TunableOp disabled)"
echo "Source mode: ${source_mode}"
echo "Model: ${model_path}"
echo "Config: ${runtime_config_json}"
echo "First/last: ${image_path} ${last_frame_path}"
echo "Prompt: ${MINIMAX_H3_PROMPT_SOURCE} (${prompt_duration_seconds}s alignment)"
echo "Output: ${save_result_path}"

source "${lightx2v_path}/scripts/base/base.sh"

cd /models
infer_args=(
    --seed "${SEED:-42}"
    --model_cls minimax_h3
    --task fl2av
    --model_path "${model_path}"
    --config_json "${runtime_config_json}"
    --prompt "${prompt}"
    --image_path "${image_path}"
    --last_frame_path "${last_frame_path}"
    --target_shape "${target_height}" "${target_width}"
    --save_result_path "${save_result_path}"
)
if (( gpu_count > 1 )); then
    torchrun --standalone --nproc_per_node="${gpu_count}" -m lightx2v.infer "${infer_args[@]}"
else
    python3 -m lightx2v.infer "${infer_args[@]}"
fi

if [[ -n "${evidence_dir}" ]]; then
    ffprobe -v error -show_streams -show_format -of json "${save_result_path}" \
        >"${evidence_dir}/ffprobe.json"
    python3 - "${evidence_dir}/ffprobe.json" "${target_width}" "${target_height}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
missing = {"video", "audio"} - kinds
if missing:
    raise SystemExit(f"MiniMax-H3 output is missing streams: {sorted(missing)}")
video = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
actual = (int(video["width"]), int(video["height"]))
expected = (int(sys.argv[2]), int(sys.argv[3]))
if actual != expected:
    raise SystemExit(f"MiniMax-H3 output size is {actual[0]}x{actual[1]}, expected {expected[0]}x{expected[1]}")
PY
fi
