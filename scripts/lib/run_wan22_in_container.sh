#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
model_path="${MODEL_PATH:-${models_root}/Wan-AI/Wan2.2-I2V-A14B}"
base_config_json="${BASE_CONFIG_JSON:-${run_root}/configs/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201_lazy.json}"

visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
[[ "${visible_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "ERROR: HIP_VISIBLE_DEVICES must be comma-separated GPU ids" >&2
    exit 1
}
IFS=',' read -r -a gpu_devices <<<"${visible_devices}"
expected_gpu_count="${EXPECTED_GPU_COUNT:-${#gpu_devices[@]}}"
case "${expected_gpu_count}" in
    2 | 4 | 8 | 16) ;;
    *) echo "ERROR: Wan2.2 timing supports 2, 4, 8, or 16 GPUs" >&2; exit 1 ;;
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
        echo "ERROR: Wan2.2 target dimensions, frames, and FPS must be positive integers" >&2
        exit 1
    }
done
(( (target_video_length - 1) % 4 == 0 )) || {
    echo "ERROR: TARGET_VIDEO_LENGTH must satisfy the Wan VAE 4n+1 contract" >&2
    exit 1
}

cpu_offload="${CPU_OFFLOAD:-1}"
offload_granularity="${OFFLOAD_GRANULARITY:-block}"
offload_ratio="${OFFLOAD_RATIO:-1.0}"
t5_cpu_offload="${T5_CPU_OFFLOAD:-1}"
vae_cpu_offload="${VAE_CPU_OFFLOAD:-0}"
use_tiling_vae="${USE_TILING_VAE:-0}"
schedule_request="${LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE:-auto}"
for flag_name in cpu_offload t5_cpu_offload vae_cpu_offload use_tiling_vae; do
    flag_value="${!flag_name}"
    [[ "${flag_value}" == 0 || "${flag_value}" == 1 ]] || {
        echo "ERROR: ${flag_name} must be 0 or 1" >&2
        exit 1
    }
done
[[ "${offload_granularity}" == block || "${offload_granularity}" == model ]] || {
    echo "ERROR: OFFLOAD_GRANULARITY must be block or model" >&2
    exit 1
}
[[ "${offload_ratio}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || {
    echo "ERROR: OFFLOAD_RATIO must be numeric" >&2
    exit 1
}
case "${schedule_request}" in
    auto | baseline | b0 | b1) ;;
    *) echo "ERROR: schedule must be auto, baseline, b0, or b1" >&2; exit 1 ;;
esac

self_attn_type="${SELF_ATTN_TYPE:-aiter_fav3_sage_bf16_attn}"
case "${self_attn_type}" in
    aiter_flydsl_bf16_flash_attn | aiter_triton_mha | aiter_fav3_sage_bf16_attn) ;;
    *) echo "ERROR: unsupported SELF_ATTN_TYPE=${self_attn_type}" >&2; exit 1 ;;
esac
if [[ "${expected_gpu_count}" == 16 ]]; then
    seq_p_attn_type=ring
else
    seq_p_attn_type=ulysses
fi
rope_type="${ROPE_TYPE:-triton_fused_qk_rope}"
case "${rope_type}" in
    torch_real_rope | triton_fused_qk_rope) ;;
    *) echo "ERROR: unsupported ROPE_TYPE=${rope_type}" >&2; exit 1 ;;
esac

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
[[ -f "${run_root}/python_compat/wan22_lazy/sitecustomize.py" ]] || {
    echo "ERROR: Wan2.2 lazy compatibility shim is missing" >&2
    exit 1
}

runtime_config_dir="${RUNTIME_CONFIG_DIR:-/tmp/lightx2v-run}"
mkdir -p "${runtime_config_dir}"
runtime_config_json="${runtime_config_dir}/wan22-${expected_gpu_count}gpu-$$.json"
trap 'rm -f "${runtime_config_json}"' EXIT

export CPU_OFFLOAD="${cpu_offload}"
export OFFLOAD_GRANULARITY="${offload_granularity}"
export OFFLOAD_RATIO="${offload_ratio}"
export T5_CPU_OFFLOAD="${t5_cpu_offload}"
export VAE_CPU_OFFLOAD="${vae_cpu_offload}"
export USE_TILING_VAE="${use_tiling_vae}"
export LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE="${schedule_request}"

python3 - "${base_config_json}" "${runtime_config_json}" "${expected_gpu_count}" \
    "${seq_p_attn_type}" "${self_attn_type}" "${rope_type}" "${target_height}" \
    "${target_width}" "${target_video_length}" "${target_fps}" <<'PY'
import json
import os
import sys

(
    source_path,
    output_path,
    seq_p_size_text,
    seq_p_attn_type,
    self_attn_type,
    rope_type,
    target_height_text,
    target_width_text,
    target_video_length_text,
    target_fps_text,
) = sys.argv[1:]
with open(source_path, encoding="utf-8") as source_file:
    config = json.load(source_file)

seq_p_size = int(seq_p_size_text)
offload_ratio = float(os.environ["OFFLOAD_RATIO"])
if not 0.0 <= offload_ratio <= 1.0:
    raise SystemExit(f"OFFLOAD_RATIO must be in [0, 1], got {offload_ratio}")

config.update(
    {
        "infer_steps": 4,
        "target_height": int(target_height_text),
        "target_width": int(target_width_text),
        "target_video_length": int(target_video_length_text),
        "fps": int(target_fps_text),
        "enable_cfg": False,
        "self_attn_1_type": self_attn_type,
        "rope_type": rope_type,
        "rope_chunk": False,
        "cpu_offload": os.environ["CPU_OFFLOAD"] == "1",
        "offload_granularity": os.environ["OFFLOAD_GRANULARITY"],
        "offload_ratio": offload_ratio,
        "t5_cpu_offload": os.environ["T5_CPU_OFFLOAD"] == "1",
        "vae_cpu_offload": os.environ["VAE_CPU_OFFLOAD"] == "1",
        "use_tiling_vae": os.environ["USE_TILING_VAE"] == "1",
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

request = os.environ["LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE"]
full_lazy = (
    config["cpu_offload"]
    and config["offload_granularity"] == "block"
    and offload_ratio == 1.0
    and config.get("lazy_load", False)
    and config.get("feature_caching", "NoCaching") == "NoCaching"
)
if request == "auto":
    schedule = "b1" if full_lazy and seq_p_size in {8, 16} else "baseline"
else:
    schedule = request
if schedule in {"b0", "b1"} and not full_lazy:
    raise SystemExit(
        f"Wan {schedule} requires full lazy block offload with NoCaching"
    )
config["wan_block_offload_schedule_request"] = request
config["wan_block_offload_schedule"] = schedule

with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(config, output_file, indent=4, ensure_ascii=False)
    output_file.write("\n")
PY

resolved_schedule="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["wan_block_offload_schedule"])' "${runtime_config_json}")"
prompt="${PROMPT:-Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard, with crystal-clear water and distant green hills in the background.}"
negative_prompt="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，最差质量，低质量，畸形，杂乱的背景}"
image_path="${IMAGE_PATH:-${lightx2v_path}/assets/inputs/imgs/img_0.jpg}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/wan22_${target_height}p_${target_video_length}f_${target_fps}fps_${expected_gpu_count}gpu.mp4}"

echo "Wan2.2 I2V A14B: ${target_height}x${target_width}, ${target_video_length} frames @ ${target_fps} FPS, 4 steps"
echo "GPUs: ${visible_devices} (${expected_gpu_count} ranks)"
echo "Parallel: SP=${expected_gpu_count}, attention=${seq_p_attn_type}"
echo "Offload: ${offload_granularity}/${offload_ratio}; T5=${t5_cpu_offload}"
echo "Wan block-offload schedule: ${resolved_schedule} (request=${schedule_request})"
echo "VAE: CK Conv3d v1, resident=$((1 - vae_cpu_offload)), tiling=${use_tiling_vae}, parallel=1"
echo "Config: ${runtime_config_json}"
echo "Output: ${save_result_path}"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
    cat "${runtime_config_json}"
    exit 0
fi
python3 -c 'import importlib; import aiter; assert callable(aiter.ck_conv3d_v1); assert aiter.is_ck_conv3d_v1_prebuilt(); module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'

export PYTHONPATH="${run_root}/python_compat/wan22_lazy${PYTHONPATH:+:${PYTHONPATH}}"
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
if [[ "${LIGHTX2V_WARMUP:-0}" == 1 ]]; then
    infer_args+=(--warmup)
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
if [[ "${LIGHTX2V_DIT_PROGRESS:-1}" == 1 ]]; then
    torchrun "${torchrun_args[@]}" 2>&1 | python3 "${script_dir}/../report_dit_progress.py" --ranks "${expected_gpu_count}"
else
    torchrun "${torchrun_args[@]}"
fi
