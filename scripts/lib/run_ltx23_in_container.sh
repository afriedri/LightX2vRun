#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
model_path="${MODEL_PATH:-/models/ltx23}"
checkpoint="${LTX23_CHECKPOINT_CONTAINER:-/models/ltx23/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors}"
upscaler="${LTX23_UPSCALER_CONTAINER:-/models/ltx23/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
gemma_dir="${LTX23_GEMMA_CONTAINER:-/models/ltx23/gemma}"
evidence_dir="${LTX23_EVIDENCE_DIR:-${lightx2v_path}/save_results}"
readonly fps=24
readonly frames=121
preflight_only="${LTX23_PREFLIGHT_ONLY:-0}"
collect="${LTX23_COLLECT:-1}"
quick_gemm_probe="${LTX23_QUICK_GEMM_PROBE:-0}"
weight_probe_only="${LTX23_WEIGHT_PROBE_ONLY:-0}"
weight_probe_sha256="${LTX23_WEIGHT_PROBE_SHA256:-0}"

case "${preflight_only}" in 0 | 1) ;; *) echo "ERROR: LTX23_PREFLIGHT_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "${collect}" in 0 | 1) ;; *) echo "ERROR: LTX23_COLLECT must be 0 or 1" >&2; exit 2 ;; esac
case "${quick_gemm_probe}" in 0 | 1) ;; *) echo "ERROR: LTX23_QUICK_GEMM_PROBE must be 0 or 1" >&2; exit 2 ;; esac
case "${weight_probe_only}" in 0 | 1) ;; *) echo "ERROR: LTX23_WEIGHT_PROBE_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "${weight_probe_sha256}" in 0 | 1) ;; *) echo "ERROR: LTX23_WEIGHT_PROBE_SHA256 must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${quick_gemm_probe}" == 1 && "${preflight_only}" != 1 ]]; then
    echo "ERROR: LTX23_QUICK_GEMM_PROBE=1 is diagnostic-only and requires preflight-only mode" >&2
    exit 2
fi

visible_devices="${HIP_VISIBLE_DEVICES:-}"
[[ "${visible_devices}" =~ ^[0-9]+,[0-9]+$ ]] || {
    echo "ERROR: LTX-2.3 B0 requires exactly two comma-separated HIP GPU ids" >&2
    exit 2
}
IFS=',' read -r gpu0 gpu1 <<<"${visible_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: LTX-2.3 B0 GPU ids must be unique" >&2; exit 2; }

config_json="${lightx2v_path}/configs/ltx2/ltx2_3_distill_fp8_gfx1201_b0_704x1280_24fps.json"
preflight="${run_root}/scripts/lib/preflight_ltx23_fp8_gfx1201.py"
weight_probe="${run_root}/scripts/collector/collect_ltx23_weight_probe.py"
video_path="${evidence_dir}/ltx23_b0_704x1280_${fps}fps.mp4"
mkdir -p "${evidence_dir}"
for required_file in "${checkpoint}" "${upscaler}" "${config_json}" "${preflight}" "${weight_probe}"; do
    [[ -f "${required_file}" ]] || { echo "ERROR: required file not found: ${required_file}" >&2; exit 1; }
done
[[ -d "${gemma_dir}" ]] || { echo "ERROR: Gemma directory not found: ${gemma_dir}" >&2; exit 1; }

export PLATFORM=amd_rocm
export DTYPE=BF16
export GPU_ARCHS=gfx1201
export CU_NUM=24
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ROCM_ARCH=gfx1201
export TOKENIZERS_PARALLELISM=false
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-/workspace/.cache/aiter}/gfx1201-cu24-ltx23"
mkdir -p "${AITER_JIT_DIR}"

if [[ "${collect}" == 1 ]]; then
    export LIGHTX2V_WAN_RUNTUNE=1
    export LIGHTX2V_WAN_RUNTUNE_BACKEND=ltx23_fp8_b0_gfx1201
    export LIGHTX2V_WAN_RUNTUNE_OUTPUT="${evidence_dir}/ltx23_collector.rank{rank}.json"
else
    export LIGHTX2V_WAN_RUNTUNE=0
    unset LIGHTX2V_WAN_RUNTUNE_OUTPUT
fi

echo "LTX-2.3 FP8 B0 gfx1201"
echo "Workload: 704x1280 final, 352x640 Stage 1, ${frames} frames @ ${fps} FPS"
echo "Schedule: Stage 1=8 steps, spatial upsample x2, Stage 2=3 steps"
echo "Parallel: Torch Ulysses SP2; Torch all-to-all/pre-post"
echo "Residency: phase-serialized Gemma -> DiT -> VAE (32 GiB/rank)"
echo "DiT stream: monolithic checkpoint -> BF16/FP8 pinned host x2 -> GPU x2"
echo "L1: Gemma 3"
echo "L2: torch._scaled_mm FP8, torch.mm BF16, Torch SDPA/norm/RoPE/modulate + Torch upsampler"
echo "L3: tiled Torch Video VAE + Torch Audio VAE"
echo "Collector: ${collect}"

if [[ "${weight_probe_only}" == 1 ]]; then
    probe_args=(
        --checkpoint "${checkpoint}"
        --output "${evidence_dir}/ltx23_weight_probe.json"
        --expect-key "model.diffusion_model.transformer_blocks.4.attn1.to_gate_logits.weight"
    )
    if [[ "${weight_probe_sha256}" == 1 ]]; then
        probe_args+=(--sha256)
    fi
    echo "Action: in-container header-only weight probe"
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${weight_probe}" "${probe_args[@]}"
    echo "Weight report: ${evidence_dir}/ltx23_weight_probe.json"
    exit 0
fi

python3 "${preflight}" \
    --mode runtime \
    --checkpoint "${checkpoint}" \
    --upscaler "${upscaler}" \
    --gemma-dir "${gemma_dir}" \
    --runtime-config "${config_json}" \
    --output "${evidence_dir}/preflight_runtime.json"

probe_args=()
if [[ "${quick_gemm_probe}" == 1 ]]; then
    probe_args+=(--quick-gemm-probe)
fi
torchrun --standalone --nproc_per_node=2 \
    "${preflight}" \
    --mode distributed \
    --checkpoint "${checkpoint}" \
    --upscaler "${upscaler}" \
    --gemma-dir "${gemma_dir}" \
    --runtime-config "${config_json}" \
    --output "${evidence_dir}/preflight_distributed.json" \
    "${probe_args[@]}"

if [[ "${preflight_only}" == 1 ]]; then
    echo "LTX-2.3 B0 runtime/distributed preflight complete"
    exit 0
fi

source "${lightx2v_path}/scripts/base/base.sh"
cd /models
infer_args=(
    --seed "${SEED:-42}"
    --model_cls ltx2
    --task t2av
    --model_path "${model_path}"
    --config_json "${config_json}"
    --prompt "${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
    --negative_prompt "${NEGATIVE_PROMPT:-}"
    --save_result_path "${video_path}"
)
# Deliberately omit --target_shape. For an upsampled LTX run, LightX2V derives
# Stage 1 as 352x640 from the config's final 704x1280 target, then doubles once.
torchrun --standalone --nproc_per_node=2 -m lightx2v.infer "${infer_args[@]}"

ffprobe -v error -count_frames -show_streams -show_format -of json "${video_path}" \
    >"${evidence_dir}/ffprobe.json"
python3 - "${evidence_dir}/ffprobe.json" "${fps}" "${frames}" <<'PY'
import json
import math
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
fps = int(sys.argv[2])
frames = int(sys.argv[3])
streams = payload.get("streams", [])
video = next((item for item in streams if item.get("codec_type") == "video"), None)
audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
if video is None or audio is None:
    raise SystemExit("LTX-2.3 output must contain both video and audio streams")
if (int(video.get("width", 0)), int(video.get("height", 0))) != (1280, 704):
    raise SystemExit(f"unexpected output size: {video.get('width')}x{video.get('height')}")
rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
num, den = (int(value) for value in rate.split("/", 1))
actual_fps = num / den if den else 0.0
if not math.isclose(actual_fps, fps, rel_tol=0.0, abs_tol=0.01):
    raise SystemExit(f"unexpected video FPS: {actual_fps}, expected {fps}")
observed_frames = video.get("nb_read_frames") or video.get("nb_frames")
if observed_frames not in (None, "N/A") and int(observed_frames) != frames:
    raise SystemExit(f"unexpected frame count: {observed_frames}, expected {frames}")
PY

echo "LTX-2.3 FP8 B0 complete: ${video_path}"
