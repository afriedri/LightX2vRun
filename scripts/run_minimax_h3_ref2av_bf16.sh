#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
image=
model_dir="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
comfy_dir="${H3_COMFY_DIR:-/disk/ssd1/difumodels/MiniMax-H3-ComfyUI}"
result_dir="${H3_RESULT_DIR:-}"
gpus="${GPU_DEVICES:-0,1}"
references=()
prompt="${PROMPT:-}"
prompt_file=
smoke=0
preflight=0
collect="${LIGHTX2V_H3_RUN_COLLECTOR:-0}"
usage() {
    cat <<'EOF'
Usage: bash scripts/run_minimax_h3_ref2av_bf16.sh \
  --image IMAGE --reference-image FILE --prompt-file FILE [options]

Runs in the image using the current mounted Run checkout. The image must include
LightX2V's tools/convert/minimax_h3_comfy.py. First use prepares/caches native BF16
weights (about 66.3 GB additional disk); subsequent runs reuse the cache.

  --model-dir DIR       Existing complete native H3 directory for Qwen/tokenizer/VAEs
  --comfy-dir DIR       ComfyUI root with diffusion_models/
  --reference-image FILE  Repeat for 1-9 images, in prompt reference order
  --prompt-file FILE    Prompt with reference descriptions; PROMPT overrides file
  --result-dir DIR      Reports/video output (default: timestamped)
  --gpus IDS            GPU ids (default: 0,1; supports 1/2/4/8/16 GPUs)
  --infer-steps N      Base schedule grid points (default: 30, N-1 evaluations)
  --smoke              480x480 instead of 1344x768, 124 frames at 24 FPS
  --preflight-only     Validate assets and run kernel/communication probes
  --collect            Enable existing H3 runtime collector
  --help, -h           Show help

CACHE_DIR selects the persistent host cache. DRY_RUN=1 prints Docker arguments.
The original native Qwen/VAEs are reused; ComfyUI Qwen/VAEs are not selected here.
EOF
}
while (( $# )); do
    case "$1" in
        --image|--model-dir|--comfy-dir|--reference-image|--prompt-file|--result-dir|--gpus|--infer-steps)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || { echo "missing value for $1" >&2; exit 2; }
            case "$1" in
                --image) image="$2" ;;
                --model-dir) model_dir="$2" ;;
                --comfy-dir) comfy_dir="$2" ;;
                --reference-image) references+=("$2") ;;
                --prompt-file) prompt_file="$2" ;;
                --result-dir) result_dir="$2" ;;
                --gpus) gpus="$2" ;;
                --infer-steps) export H3_INFER_STEPS="$2" ;;
            esac
            shift 2 ;;
        --smoke) smoke=1; shift ;;
        --preflight-only) preflight=1; shift ;;
        --collect) collect=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$image" ]] || { echo '--image is required' >&2; exit 2; }
(( ${#references[@]} >= 1 && ${#references[@]} <= 9 )) || { echo 'provide 1-9 --reference-image files' >&2; exit 2; }
for i in "${!references[@]}"; do
    [[ "${references[i]}" != *$'\n'* && "${references[i]}" != *,* ]] || { echo 'reference path cannot contain newline or comma' >&2; exit 2; }
    [[ "${references[i]}" == /* ]] || references[i]="${PWD}/${references[i]}"
done
if [[ -z "$prompt" && -n "$prompt_file" ]]; then
    [[ -f "$prompt_file" ]] || { echo "prompt file not found: $prompt_file" >&2; exit 2; }
    prompt="$(<"$prompt_file")"
fi
[[ -n "$prompt" ]] || { echo 'provide --prompt-file or PROMPT for your references' >&2; exit 2; }
[[ "$model_dir" == /* ]] || model_dir="${PWD}/${model_dir}"
[[ "$comfy_dir" == /* ]] || comfy_dir="${PWD}/${comfy_dir}"
result_dir="${result_dir:-${OUTPUT_DIR:-/disk/ssd1/lightx2v-runtime/outputs}/minimax_h3_ref2av_$(date +%Y%m%d_%H%M%S)}"
[[ "$result_dir" == /* ]] || result_dir="${PWD}/${result_dir}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/disk/ssd1/lightx2v-runtime}"
export IMAGE="$image" LIGHTX2V_H3_EXPLICIT_IMAGE=1 H3_TASK=ref2av
export H3_MODEL_DIR="$model_dir" H3_COMFY_DIR="$comfy_dir" GPU_DEVICES="$gpus"
export H3_REFERENCE_IMAGES="$(printf '%s\n' "${references[@]}")"
export PROMPT="$prompt" H3_SMOKE="$smoke" H3_PREFLIGHT_ONLY="$preflight"
export LIGHTX2V_H3_RUN_COLLECTOR="$collect"
export OUTPUT_DIR="$result_dir" H3_EVIDENCE_DIR=/workspace/LightX2V/save_results
export SAVE_RESULT_PATH=/workspace/LightX2V/save_results/minimax_h3_ref2av.mp4
export LIGHTX2V_H3_RUN_COLLECTOR_OUTPUT="${H3_EVIDENCE_DIR}/h3_runcollector.rank{rank}.json"
exec bash "${MODEL_RUN:-${run_root}/lightx2vModelrun}" --run-minimax-h3-gfx1201
