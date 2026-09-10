#!/usr/bin/env bash
# MiniMax-H3 Ref2AV Turbo LoRA: 8 model evaluations at 1344x768 / 15 seconds.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_image="${IMAGE:-lightx2v-rocm:hvat-gfx1201-R9600}"
export H3_MODEL_DIR="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
export H3_TURBO_DIR="${H3_TURBO_DIR:-/disk/ssd1/difumodels/Minimax-h3-Turbo}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/disk/ssd1/lightx2v-runtime}"
export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export H3_PROFILE=ref2av_turbo_8step
export H3_STEP_UPDATE=training_euler
export H3_INFER_STEPS=9
export LIGHTX2V_H3_RUN_COLLECTOR="${LIGHTX2V_H3_RUN_COLLECTOR:-1}"
asset_dir="${script_dir}/../assets/minimax_h3/rooftop_15s"
reference_images=("${asset_dir}/Image Gen-1.png" "${asset_dir}/RYAN-2.png")
if [[ -n "${H3_REFERENCE_IMAGE:-}" ]]; then reference_images=("$H3_REFERENCE_IMAGE"); fi
prompt_file="${H3_PROMPT_FILE:-${asset_dir}/prompt.txt}"

export TARGET_HEIGHT=768 TARGET_WIDTH=1344 TARGET_VIDEO_LENGTH=362 TARGET_FPS=24
reference_args=()
custom_references=0
for argument in "$@"; do
    case "$argument" in
        --reference-image) custom_references=1 ;;
        --help|-h)
            cat <<HELP
MiniMax-H3 Ref2AV Turbo 8-step LoRA E2E

Base partition:   ${H3_MODEL_DIR}/transformer_ref (native Diffusers layout)
Turbo LoRA:       ${H3_TURBO_DIR}/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
Shared Qwen/VAEs: ${H3_MODEL_DIR}
GPUs:             ${GPU_DEVICES}
Sampling:         training_euler, 9 grid points / 8 model evaluations, shift 12/3
Output:           1344x768, 15 seconds at 24 FPS
Collector:        enabled by default

CLI options are provided by run_minimax_h3_ref2av_bf16.sh. The Turbo profile
rejects sampling overrides that change the released 8-step contract.
HELP
            exec bash "${script_dir}/run_minimax_h3_ref2av_bf16.sh" --help
            ;;
    esac
done
if [[ "$custom_references" == 0 ]]; then
    for reference in "${reference_images[@]}"; do
        reference_args+=(--reference-image "$reference")
    done
fi

exec bash "${script_dir}/run_minimax_h3_ref2av_bf16.sh" \
    --image "$run_image" \
    --turbo-dir "$H3_TURBO_DIR" \
    --prompt-file "$prompt_file" --duration 15 \
    "${reference_args[@]}" "$@"
