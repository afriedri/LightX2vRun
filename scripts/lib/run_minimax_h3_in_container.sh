#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
source "${run_root}/scripts/lib/minimax_h3_prompt_presets.sh"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_path="${MODEL_PATH:-/models/minimax_h3}"
source_mode="${H3_SOURCE_MODE:-image}"
task="${H3_TASK:-fl2av}"
case "$task" in fl2av|ref2av) ;; *) echo "unsupported H3 task: $task" >&2; exit 2 ;; esac
profile="${H3_PROFILE:-}"
if [[ -z "$profile" ]]; then
    if [[ "$task" == ref2av ]]; then profile=ref2av_bf16_base; else profile=fl2av_turbo_4step; fi
fi
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png}"
last_frame_path="${LAST_FRAME_PATH:-${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png}"
case "$profile" in
    fl2av_turbo_4step)
        [[ "$task" == fl2av ]] || { echo "profile $profile requires task fl2av" >&2; exit 2; }
        overlay_json="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_fl2av_4step_768p_gfx1201.json}"
        publish_json="${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_bf16_4step_single_gpu_offload.json"
        ;;
    ref2av_bf16_base)
        [[ "$task" == ref2av ]] || { echo "profile $profile requires task ref2av" >&2; exit 2; }
        last_frame_path=
        overlay_json="${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json"
        publish_json="${lightx2v_path}/configs/minimax_h3/minimax_h3_ref2av.json"
        ;;
    ref2av_turbo_8step)
        [[ "$task" == ref2av ]] || { echo "profile $profile requires task ref2av" >&2; exit 2; }
        last_frame_path=
        overlay_json="${lightx2v_path}/configs/platforms/amd_rocm/minimax_h3_ref2av_8step_768p_gfx1201.json"
        publish_json="${lightx2v_path}/configs/minimax_h3/dmd/minimax_h3_ref2av_8step.json"
        ;;
    *) echo "unsupported H3 profile: $profile" >&2; exit 2 ;;
esac

visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+){0,15}$ ]] || {
    echo "ERROR: MiniMax-H3 gfx1201 requires one, two, four, eight, or sixteen comma-separated HIP GPU ids, got '${visible_devices}'" >&2
    exit 1
}
IFS=',' read -r -a visible_device_ids <<<"${visible_devices}"
gpu_count="${#visible_device_ids[@]}"
case "${gpu_count}" in
    1 | 2 | 4 | 8 | 16) ;;
    *) echo "ERROR: MiniMax-H3 gfx1201 supports exactly 1, 2, 4, 8, or 16 GPUs" >&2; exit 1 ;;
esac
declare -A seen_visible_device_ids=()
for visible_device_id in "${visible_device_ids[@]}"; do
    [[ -z "${seen_visible_device_ids[${visible_device_id}]:-}" ]] || {
        echo "ERROR: MiniMax-H3 gfx1201 GPU ids must be unique, got '${visible_devices}'" >&2
        exit 1
    }
    seen_visible_device_ids["${visible_device_id}"]=1
done
export HIP_VISIBLE_DEVICES="${visible_devices}"
unset CUDA_VISIBLE_DEVICES

export PLATFORM=amd_rocm
export DTYPE=BF16
# Required H3 kernels must execute; a fallback is a failed run, not a slower result.
export LIGHTX2V_NO_FALLBACK_MODELS="${LIGHTX2V_NO_FALLBACK_MODELS:+${LIGHTX2V_NO_FALLBACK_MODELS},}minimax_h3"
export TOKENIZERS_PARALLELISM=false
export GPU_ARCHS="${GPU_ARCHS:-gfx1201}"
export CU_NUM="${CU_NUM:-24}"
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
# Ordinary torch.mm is the production baseline. The collected FL2AV rank-128
# LoRA did not benefit from TunableOp replay; Ref2AV rank shapes are collected
# independently before any profile-specific promotion.
export PYTORCH_TUNABLEOP_ENABLED=0
unset TORCH_BLAS_PREFER_CUBLASLT TORCH_BLAS_PREFER_HIPBLASLT
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
[[ "${GPU_ARCHS}" == gfx1201 && "${CU_NUM}" == 24 ]] || {
    echo "ERROR: MiniMax-H3 gfx1201 requires GPU_ARCHS=gfx1201 and CU_NUM=24" >&2
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
    attention_preflight_log="${runtime_config_dir}/minimax-h3-attention-preflight-$$.log"
    trap 'rm -f "${runtime_config_json}" "${preflight_json}" "${attention_preflight_log}"' EXIT
fi

smoke_flag=()
kernel_smoke_flag=()
runcollector_schema_flag=()
target_height="${TARGET_HEIGHT:-768}"
target_width="${TARGET_WIDTH:-1344}"
target_video_length="${TARGET_VIDEO_LENGTH:-124}"
target_fps="${TARGET_FPS:-24}"
prompt_duration_seconds=5.17
output_duration=
for target_value in "${target_height}" "${target_width}" "${target_video_length}" "${target_fps}"; do
    [[ "${target_value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: MiniMax-H3 target dimensions, frames, and FPS must be positive integers" >&2
        exit 1
    }
done
if [[ "${target_fps}" != 24 || ( "$task" == fl2av && "${target_video_length}" != 124 ) ]]; then
    echo "ERROR: MiniMax-H3 production timing is frozen at 124 frames and 24 FPS" >&2
    exit 1
fi
if [[ "$task" == ref2av ]]; then
    if (( target_video_length < 124 || target_video_length > 362 || target_video_length % 17 != 5 )); then
        echo 'ERROR: Ref2AV requires 17*n+5 frames within 124..362' >&2; exit 1
    fi
    prompt_duration_seconds="$(python3 -c 'import sys; print(round(int(sys.argv[1])/24, 3))' "$target_video_length")"
fi
if [[ "${H3_SMOKE:-0}" == 1 ]]; then
    smoke_flag+=(--smoke)
    target_height=480
    target_width=480
else
    case "${target_width}x${target_height}" in
        1344x768 | 832x480) ;;
        *)
            echo "ERROR: MiniMax-H3 production timing supports 1344x768 or 832x480" >&2
            exit 1
            ;;
    esac
fi
if [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    kernel_smoke_flag+=(--kernel-smoke)
fi
if [[ "${LIGHTX2V_H3_RUN_COLLECTOR:-0}" == 1 ]]; then
    runcollector_schema_flag+=(--required-runcollector-schema 2)
fi

profile_args=(--task "$task" --profile "$profile")
if [[ "$task" == ref2av ]]; then
    profile_args+=(--target-video-length "$target_video_length")
    if [[ -n "${H3_OUTPUT_DURATION:-}" ]]; then
        [[ "$H3_OUTPUT_DURATION" =~ ^([5-9]|1[0-5])$ ]] &&
            (( target_video_length == (H3_OUTPUT_DURATION * 24 - 5 + 16) / 17 * 17 + 5 )) || {
            echo 'ERROR: H3 output duration must match native aligned frames (integer 5-15 seconds)' >&2; exit 1;
        }
        # Fail before conversion/inference if an old image cannot honor exact duration.
        python3 - "$lightx2v_path" <<'CHECK'
import pathlib, sys
source = pathlib.Path(sys.argv[1]) / "lightx2v/models/runners/minimax_h3/minimax_h3_runner.py"
if "h3_output_duration" not in source.read_text():
    raise SystemExit("Rebuild the LightX2V image: this version does not support exact H3 output duration")
CHECK
        output_duration="$H3_OUTPUT_DURATION"
        profile_args+=(--output-duration "$output_duration")
        prompt_duration_seconds="$H3_OUTPUT_DURATION"
    fi
    prepared_model="${runtime_config_dir}/ref2av_model"
    python3 "${lightx2v_path}/tools/convert/minimax_h3_comfy.py" \
        --checkpoint /models/minimax_h3_comfy/diffusion_models/minimax_h3_ref2va_bf16.safetensors \
        --cache-dir /workspace/.cache/h3_ref2av \
        --base-model-dir "$model_path" --model-output "$prepared_model"
    model_path="$prepared_model"
    if [[ -n "${H3_INFER_STEPS:-}" ]]; then profile_args+=(--infer-steps "$H3_INFER_STEPS"); fi
    if [[ -n "${H3_STEP_UPDATE:-}" ]]; then profile_args+=(--step-update "$H3_STEP_UPDATE"); fi
    if [[ -n "${H3_REFERENCE_IMAGE_RESIZE_MODE:-}" ]]; then profile_args+=(--reference-image-resize-mode "$H3_REFERENCE_IMAGE_RESIZE_MODE"); fi
fi

python3 "${script_dir}/build_minimax_h3_runtime_config.py" \
    "${profile_args[@]}" \
    --publish-json "${publish_json}" \
    --overlay-json "${overlay_json}" \
    --output-json "${runtime_config_json}" \
    --seq-parallel-size "${gpu_count}" \
    --target-height "${target_height}" \
    --target-width "${target_width}" \
    "${smoke_flag[@]}"

python3 "${script_dir}/preflight_minimax_h3_gfx1201.py" \
    --model-dir "${model_path}" \
    --turbo-dir /models/minimax_h3_turbo \
    --first-frame "${image_path}" \
    --last-frame "${last_frame_path}" \
    --runtime-config "${runtime_config_json}" \
    --expected-world-size "${gpu_count}" \
    --source-mode "${source_mode}" \
    "${runcollector_schema_flag[@]}" \
    "${kernel_smoke_flag[@]}" | tee "${preflight_json}"

if (( gpu_count > 1 )) && [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    attention_preflight_log="${attention_preflight_log:-${runtime_config_dir}/attention_preflight.log}"
    torchrun --standalone --nproc_per_node="${gpu_count}" \
        "${script_dir}/preflight_minimax_h3_gfx1201.py" \
        --distributed-attention-only 2>&1 | tee "${attention_preflight_log}"
fi

if [[ "${H3_PREFLIGHT_ONLY:-0}" == 1 || "${CONFIG_ONLY:-0}" == 1 ]]; then
    cat "${runtime_config_json}"
    exit 0
fi

case "${H3_MEGADISCOVERY:-0}" in
    0 | 1) ;;
    *) echo "ERROR: H3_MEGADISCOVERY must be 0 or 1" >&2; exit 1 ;;
esac
collective_benchmark=
megadiscovery_timeout_seconds="${H3_MEGADISCOVERY_TIMEOUT_SECONDS:-300}"
if [[ "${H3_MEGADISCOVERY:-0}" == 1 ]]; then
    case "${gpu_count}" in
        8 | 16) ;;
        *) echo "ERROR: MegaDiscovery requires exactly 8 or 16 GPUs" >&2; exit 1 ;;
    esac
    [[ -n "${evidence_dir}" ]] || {
        echo "ERROR: MegaDiscovery requires H3_EVIDENCE_DIR" >&2
        exit 1
    }
    collective_benchmark="${run_root}/scripts/collector/benchmark_minimax_h3_collectives.py"
    [[ -f "${collective_benchmark}" ]] || {
        echo "ERROR: MegaDiscovery collective benchmark not found: ${collective_benchmark}" >&2
        exit 1
    }
    command -v timeout >/dev/null 2>&1 || {
        echo "ERROR: MegaDiscovery requires the coreutils timeout command" >&2
        exit 1
    }
    [[ "${megadiscovery_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: H3_MEGADISCOVERY_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 1
    }
fi

run_megadiscovery_collectives() {
    local benchmark_status=0
    local tee_status=0
    local output="${evidence_dir}/megadiscovery_collectives.json"
    local log="${evidence_dir}/megadiscovery_collectives.log"
    echo "MegaDiscovery: production inference is complete; starting isolated exact-shape communication probes"
    echo "MegaDiscovery: expected probe time is 1-3 minutes; hard timeout is ${megadiscovery_timeout_seconds}s"
    echo "MegaDiscovery: live probe log: ${log}"
    set +e
    timeout --signal=TERM --kill-after=30s "${megadiscovery_timeout_seconds}s" \
        torchrun --standalone --nproc_per_node="${gpu_count}" \
        "${collective_benchmark}" \
        --output "${output}" 2>&1 | tee "${log}"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    benchmark_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"
    if (( tee_status != 0 )); then
        echo "WARNING: MegaDiscovery could not completely write ${log} (status=${tee_status})" >&2
    fi
    if (( benchmark_status == 0 )); then
        echo "MegaDiscovery: communication probes complete"
        return 0
    fi
    if (( benchmark_status == 124 || benchmark_status == 137 )); then
        echo "WARNING: MegaDiscovery communication probes hit the ${megadiscovery_timeout_seconds}s timeout; production evidence is preserved" >&2
    else
        echo "WARNING: MegaDiscovery communication probes failed with status ${benchmark_status}; production evidence is preserved" >&2
    fi
    python3 "${collective_benchmark}" \
        --output "${output}" \
        --world-size "${gpu_count}" \
        --record-failure "torchrun status ${benchmark_status}; see megadiscovery_collectives.log"
}

resolve_minimax_h3_prompt \
    "${run_root}" \
    "${H3_PROMPT_PRESET:-bird-stable}" \
    "${H3_PROMPT_FILE:-}" \
    "${prompt_duration_seconds}"
prompt="${MINIMAX_H3_PROMPT}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/minimax_h3_fl2av.mp4}"

echo "MiniMax-H3 ${task} gfx1201 (${profile})"
echo "Workload: ${target_width}x${target_height}, ${target_video_length} frames @ ${target_fps} FPS (${prompt_duration_seconds}s)"
echo "GPUs: ${visible_devices} (world_size=${gpu_count})"
parallel_mode="single GPU"
if (( gpu_count > 1 )); then
    if (( gpu_count == 16 )); then
        parallel_mode="CP2 x Ulysses8 SP16 + VAE tile parallel"
    else
        parallel_mode="Ulysses SP${gpu_count} + VAE tile parallel"
    fi
fi
echo "Parallel: ${parallel_mode}"
echo "H1 PatchEmbed API: torch F.linear (ATen operator: addmm; measured device backend: hipBLASLt)"
echo "H1/Refiner MHA: aiter_triton_mha"
if (( gpu_count == 16 )); then
    echo "Core Attention: aiter_fav3_sage_bf16_attn (sage_attn_v2_gfx1201_cp2_u8_2d)"
else
    echo "Core Attention: aiter_fav3_sage_bf16_attn (sage_attn_v2_gfx1201)"
fi
echo "Core Q/K RMS: minimax_h3_aiter_triton_qk"
echo "Core RoPE: minimax_h3_triton_strict_rope"
echo "H4 Video Attention: aiter_triton_mha"
if [[ "$profile" == ref2av_bf16_base ]]; then
    echo "Dense GEMM: BF16 torch.mm -> default ROCm BLAS heuristic (TunableOp disabled); LoRA disabled"
else
    echo "Dense GEMM/LoRA: torch.mm -> default ROCm BLAS heuristic (TunableOp disabled)"
fi
echo "Source mode: ${source_mode}"
echo "LightX2V revision: ${LIGHTX2V_SOURCE_REVISION:-unknown (see preflight.json)}"
echo "Model: ${model_path}"
echo "Config: ${runtime_config_json}"
echo "Input images (${task}): ${image_path} ${last_frame_path}"
echo "Prompt: ${MINIMAX_H3_PROMPT_SOURCE} (${prompt_duration_seconds}s alignment)"
echo "Output: ${save_result_path}"

source "${lightx2v_path}/scripts/base/base.sh"

cd /models
infer_args=(
    --seed "${SEED:-42}"
    --model_cls minimax_h3
    --task "$task"
    --model_path "${model_path}"
    --config_json "${runtime_config_json}"
    --prompt "${prompt}"
    --image_path "${image_path}"
    --target_shape "${target_height}" "${target_width}"
    --save_result_path "${save_result_path}"
)
if [[ "$task" == fl2av ]]; then
    infer_args+=(--last_frame_path "${last_frame_path}")
fi
if (( gpu_count > 1 )); then
    infer_command=(torchrun --standalone --nproc_per_node="${gpu_count}" -m lightx2v.infer)
else
    infer_command=(python3 -m lightx2v.infer)
fi
if [[ "$task" == ref2av && -n "$evidence_dir" ]]; then
    "${infer_command[@]}" "${infer_args[@]}" 2>&1 | tee "${evidence_dir}/run.log"
else
    "${infer_command[@]}" "${infer_args[@]}"
fi

if [[ -n "${evidence_dir}" ]]; then
    ffprobe -v error -show_streams -show_format -of json "${save_result_path}" \
        >"${evidence_dir}/ffprobe.json"
    python3 - "${evidence_dir}/ffprobe.json" "${target_width}" "${target_height}" "${output_duration}" <<'PY'
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
if sys.argv[4]:
    from fractions import Fraction
    duration = int(sys.argv[4])
    if int(video.get("nb_frames", 0)) != duration * 24 or Fraction(video["avg_frame_rate"]) != 24:
        raise SystemExit("H3 output frame count/FPS does not match the requested duration")
    for stream in payload["streams"]:
        if stream.get("codec_type") in {"video", "audio"} and abs(float(stream.get("duration", 0)) - duration) > 1/24:
            raise SystemExit(f"H3 {stream['codec_type']} duration does not match {duration}s")
if actual != expected:
    raise SystemExit(f"MiniMax-H3 output size is {actual[0]}x{actual[1]}, expected {expected[0]}x{expected[1]}")
PY
fi

if [[ "${H3_MEGADISCOVERY:-0}" == 1 ]]; then
    run_megadiscovery_collectives
fi
