#!/usr/bin/env bash
# Run the current checkout's weight inventory inside an existing image.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
image="${IMAGE:-compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4}"
model_root="/disk/ssd1/difumodels/MiniMax-H3-ComfyUI"
output_dir="${PWD}/h3-ref2va-weight-inventory"
asset=both

usage() {
    cat <<'HELP'
Usage: bash scripts/collector/run_minimax_h3_ref2v_weights.sh [options]
  --image IMAGE       Python runtime image (defaults to the H3 Ref2V Turbo8 V4 image)
  --model-root DIR    ComfyUI model root (default: /disk/ssd1/difumodels/MiniMax-H3-ComfyUI)
  --output-dir DIR    Host report directory (default: ./h3-ref2va-weight-inventory)
  --asset ASSET       both (default), bf16, or int8-convrot
  --help, -h          Show this help

Mounts this Run checkout and weights read-only; writes JSON/Markdown reports
through the output mount. Python runs inside the image; no host Python, Torch,
GPU or rebuilt image is needed for this inventory. IMAGE overrides the default.
DRY_RUN=1 prints the Docker command without requiring Docker or model files.
HELP
}

while (( $# )); do
    case "$1" in
        --image|--model-root|--output-dir|--asset)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                echo "missing value for $1" >&2; exit 2
            fi
            case "$1" in
                --image) image="$2" ;;
                --model-root) model_root="$2" ;;
                --output-dir) output_dir="$2" ;;
                --asset) asset="$2" ;;
            esac
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$asset" in both|bf16|int8-convrot) ;; *) echo "invalid --asset: $asset" >&2; exit 2 ;; esac

[[ "$model_root" == /* ]] || model_root="${PWD}/${model_root}"
[[ "$output_dir" == /* ]] || output_dir="${PWD}/${output_dir}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -d "$model_root" ]] || { echo "model root not found: $model_root" >&2; exit 2; }
    command -v docker >/dev/null || { echo 'docker is required' >&2; exit 2; }
    mkdir -p -- "$output_dir"
    model_root="$(cd -- "$model_root" && pwd -P)"
    output_dir="$(cd -- "$output_dir" && pwd -P)"
fi

command=(docker run --rm --network none
    --user "$(id -u):$(id -g)"
    --volume "${run_root}:/workspace/LightX2VRun:ro"
    --volume "${model_root}:/models/minimax_h3_comfyui:ro"
    --volume "${output_dir}:/output:rw"
    --workdir /workspace/LightX2VRun
    --entrypoint python3
    "$image" -B -E -s
    /workspace/LightX2VRun/scripts/collector/collect_minimax_h3_ref2v_weights.py
    --model-root /models/minimax_h3_comfyui --output-dir /output --asset "$asset")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
printf 'H3 Ref2VA inventory image: %s\nHost reports: %s\n' "$image" "$output_dir"
exec "${command[@]}"
