#!/usr/bin/env bash
# H3 original BF16 reference-to-video + audio. No Turbo LoRA.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Edit these defaults for your machine/materials, or override with CLI options.
# Use a rebuilt image containing the current LightX2V H3 BF16 conversion tool.
run_image="${IMAGE:-lightx2v-rocm:hvat-gfx1201-R9600}"
export H3_MODEL_DIR="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
export H3_COMFY_DIR="${H3_COMFY_DIR:-/disk/ssd1/difumodels/MiniMax-H3-ComfyUI}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/disk/ssd1/lightx2v-runtime}"
export GPU_DEVICES="${GPU_DEVICES:-0,1}"
export H3_INFER_STEPS="${H3_INFER_STEPS:-30}"
reference_images=(
    "${H3_REFERENCE_IMAGE:-${RUNTIME_ROOT}/inputs/h3-ref2v/reference.png}"
    # Add more image paths here in prompt reference order (maximum 9).
)
prompt_file="${H3_PROMPT_FILE:-${RUNTIME_ROOT}/inputs/h3-ref2v/prompt.txt}"

# Keep this profile independent of stale geometry variables from other models.
# --smoke is handled by the shared entry and selects 480x480.
export TARGET_HEIGHT=768 TARGET_WIDTH=1344 TARGET_VIDEO_LENGTH=124 TARGET_FPS=24

reference_args=()
custom_references=0
for argument in "$@"; do
    case "$argument" in
        --reference-image) custom_references=1 ;;
        --help|-h)
            cat <<HELP
H3 original BF16 Ref2V dedicated profile (no LoRA)

Default image:     ${run_image}
Shared Qwen/VAEs:  ${H3_MODEL_DIR}
ComfyUI BF16 root: ${H3_COMFY_DIR}
Reference image:  ${reference_images[0]}
Prompt file:      ${prompt_file}
GPUs:             ${GPU_DEVICES} (default two 32 GiB cards)
Sampling:         ${H3_INFER_STEPS} grid points, video/audio shift 12/3
Output geometry:  1344x768, 124 frames at 24 FPS

Edit the defaults at the top of this script or use the options below.
Explicit --reference-image options replace the entire default image list.
Two-card full-resolution VRAM fit still requires testing; use --smoke first.
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

# The shared launcher mounts current Run and inputs, then converts/infers inside
# the image. CLI values occur last so --image / --prompt-file override defaults.
exec bash "${script_dir}/run_minimax_h3_ref2av_bf16.sh" \
    --image "$run_image" \
    --prompt-file "$prompt_file" \
    "${reference_args[@]}" "$@"
