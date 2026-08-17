#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_path="${MODEL_PATH:-/models/minimax_h3}"
source_mode="${H3_SOURCE_MODE:-image}"
overlay_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_fl2av_4step_768p_gfx1201.json}"
publish_json="${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json"

visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)?$ ]] || {
    echo "ERROR: MiniMax-H3 Day0 requires one GPU or two comma-separated HIP GPU ids, got '${visible_devices}'" >&2
    exit 1
}
IFS=',' read -r -a visible_device_ids <<<"${visible_devices}"
gpu_count="${#visible_device_ids[@]}"
if [[ "${gpu_count}" == 2 && "${visible_device_ids[0]}" == "${visible_device_ids[1]}" ]]; then
    echo "ERROR: MiniMax-H3 Day0 GPU ids must be unique, got '${visible_devices}'" >&2
    exit 1
fi
export HIP_VISIBLE_DEVICES="${visible_devices}"
unset CUDA_VISIBLE_DEVICES

export PLATFORM=amd_rocm
export DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
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
if [[ "${H3_SMOKE:-0}" == 1 ]]; then
    smoke_flag+=(--smoke)
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
    --first-frame /inputs/minimax_h3_first_frame \
    --last-frame /inputs/minimax_h3_last_frame \
    --runtime-config "${runtime_config_json}" \
    --expected-world-size "${gpu_count}" \
    --source-mode "${source_mode}" | tee "${preflight_json}"

if [[ "${gpu_count}" == 2 ]]; then
    ulysses_preflight_log="${ulysses_preflight_log:-${runtime_config_dir}/ulysses_preflight.log}"
    torchrun --standalone --nproc_per_node=2 \
        "${script_dir}/preflight_minimax_h3_day0.py" \
        --distributed-attention-only 2>&1 | tee "${ulysses_preflight_log}"
fi

if [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 || "${CONFIG_ONLY:-0}" == 1 ]]; then
    cat "${runtime_config_json}"
    exit 0
fi

prompt="${PROMPT:-A cinematic tracking shot between the first and last frames, natural motion, photorealistic lighting.}"
image_path="${IMAGE_PATH:-/inputs/minimax_h3_first_frame}"
last_frame_path="${LAST_FRAME_PATH:-/inputs/minimax_h3_last_frame}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_fl2av.mp4}"

echo "MiniMax-H3 FL2V gfx1201"
echo "GPUs: ${visible_devices} (world_size=${gpu_count})"
parallel_mode="single GPU"
if [[ "${gpu_count}" == 2 ]]; then
    parallel_mode="Ulysses SP2 + VAE tile parallel"
fi
echo "Parallel: ${parallel_mode}"
echo "Source mode: ${source_mode}"
echo "Model: ${model_path}"
echo "Config: ${runtime_config_json}"
echo "First/last: ${image_path} ${last_frame_path}"
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
    --save_result_path "${save_result_path}"
)
if [[ "${gpu_count}" == 2 ]]; then
    torchrun --standalone --nproc_per_node=2 -m lightx2v.infer "${infer_args[@]}"
else
    python3 -m lightx2v.infer "${infer_args[@]}"
fi

if [[ -n "${evidence_dir}" ]]; then
    ffprobe -v error -show_streams -show_format -of json "${save_result_path}" \
        >"${evidence_dir}/ffprobe.json"
    python3 - "${evidence_dir}/ffprobe.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
missing = {"video", "audio"} - kinds
if missing:
    raise SystemExit(f"MiniMax-H3 output is missing streams: {sorted(missing)}")
PY
fi
