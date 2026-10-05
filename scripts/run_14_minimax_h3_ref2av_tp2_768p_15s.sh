#!/usr/bin/env bash
# MiniMax-H3 full BF16 Ref2AV, TP2 on two AMD Radeon R9600D (gfx1201) GPUs,
# 1344x768, 362 frames at 24 FPS (15 s), 20 steps, seed 42, in the
# rocm/vllm-dev:gfx1201-h3-ho-sk7 image. See docs/minimax-h3-ref2av-tp2-gfx1201.md.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"

image="${IMAGE:-rocm/vllm-dev:gfx1201-h3-ho-sk7}"
gpus="${GPU_DEVICES:-0,1}"
model_dir="${H3_MODEL_DIR:-/disk/ssd1/difumodels/MiniMax-h3}"
material="${H3_MATERIAL:-rooftop}"
steps="${H3_INFER_STEPS:-20}"
sage_preset="${H3_SAGE_PRESET:-fast}"
numa_node="${NUMA_NODE:-}"
docker_network="${DOCKER_NETWORK:-bridge}"
dri_all="${DRI_ALL:-1}"
container_name="${CONTAINER_NAME:-minimax-h3-ref2av-tp2-$$}"
result_dir="${H3_RESULT_DIR:-}"
prompt_file=
references=()

usage() {
    cat <<EOF
Usage: bash scripts/run_14_minimax_h3_ref2av_tp2_768p_15s.sh [options]

Runs MiniMax-H3 Ref2AV (BF16, no LoRA) with tensor parallelism over two gfx1201
GPUs inside an image built from the LightX2V/aiter h3/gfx1201-ho-sk7 branches.
The workload is fixed at 1344x768, 362 frames, 24 FPS and seed 42.

  --image IMAGE          Docker image (IMAGE; default: ${image})
  --gpus A,B             Two physical HIP GPU ids (GPU_DEVICES; default: ${gpus})
  --model-dir DIR        MiniMax-H3 model directory, HF layout (H3_MODEL_DIR;
                         default: ${model_dir})
  --material NAME        rooftop (2 refs, default) or storeroom (9 refs), from
                         configs/minimax_h3/ (H3_MATERIAL)
  --reference-image FILE Repeat 1-9 times, in prompt order; replaces --material
  --prompt-file FILE     Prompt for --reference-image (H3_PROMPT_FILE)
  --result-dir DIR       Host directory for mp4 + log (H3_RESULT_DIR;
                         default: \${OUTPUT_DIR:-\$PWD/outputs}/minimax_h3_ref2av_tp2_<time>)
  --infer-steps N        Steps (H3_INFER_STEPS; default 20; 1/3/20 have prebuilt AdaLN caches)
  --sage-preset NAME     fast | exact | t2 | esp321w (H3_SAGE_PRESET; default: fast)
  --numa-node N          Pin CPUs and memory to NUMA node N (NUMA_NODE; default: none)
  --help, -h             Show this help

Other environment: DOCKER_NETWORK (default bridge; do not use host),
DRI_ALL (1: all of /dev/dri + HIP_VISIBLE_DEVICES, default; 0: only H3_DRI_NODES),
H3_DRI_NODES (e.g. "card1 renderD128 card2 renderD129", used with DRI_ALL=0),
CONTAINER_NAME, DOCKER_EXTRA_ARGS, DRY_RUN=1. Explicit AITER_GFX1201_SAGE_ASM /
AITER_GFX1201_SAGE_ASM_SCHED override the preset. Pass-through switches:
FUSED_TAIL, TE_PAGEABLE, VAE_ENC_PAR, H3_TEXT_ENCODER_TP, AITER_GFX1201_DOWN_BLAS,
LIGHTX2V_H3_*, SEED, NCCL_DEBUG.
EOF
}

while (( $# )); do
    case "$1" in
        --image|--gpus|--model-dir|--material|--reference-image|--prompt-file|--result-dir|--infer-steps|--sage-preset|--numa-node)
            (( $# >= 2 )) && [[ -n "$2" ]] || { echo "missing value for $1" >&2; exit 2; }
            case "$1" in
                --image) image="$2" ;;
                --gpus) gpus="$2" ;;
                --model-dir) model_dir="$2" ;;
                --material) material="$2" ;;
                --reference-image) references+=("$2") ;;
                --prompt-file) prompt_file="$2" ;;
                --result-dir) result_dir="$2" ;;
                --infer-steps) steps="$2" ;;
                --sage-preset) sage_preset="$2" ;;
                --numa-node) numa_node="$2" ;;
            esac
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "${gpus}" =~ ^[0-9]+,[0-9]+$ ]] || die "TP2 needs exactly two comma-separated GPU ids, got '${gpus}'"
IFS=',' read -r -a gpu_ids <<<"${gpus}"
[[ "${gpu_ids[0]}" != "${gpu_ids[1]}" ]] || die "GPU ids must differ"
[[ "${steps}" =~ ^[1-9][0-9]*$ ]] || die "--infer-steps must be a positive integer"
[[ -z "${numa_node}" || "${numa_node}" =~ ^[0-9]+$ ]] || die "--numa-node must be a NUMA node index"
[[ "${docker_network}" != host ]] || \
    echo "WARNING: DOCKER_NETWORK=host can make RCCL init fail (hipIpcGetMemHandle invalid argument) when another ROCm container runs; see the doc" >&2

# Attention core presets (aiter AITER_GFX1201_SAGE_ASM[_SCHED]).
case "${sage_preset}" in
    fast) preset_asm=esp321wsk7; preset_sched=esp321wsk7:250,esp321wsk7t2 ;;
    exact) preset_asm=esp321wsk7; preset_sched= ;;
    t2) preset_asm=esp321wsk7t2; preset_sched= ;;
    esp321w) preset_asm=esp321w; preset_sched= ;;
    *) die "unknown --sage-preset '${sage_preset}' (fast, exact, t2, esp321w)" ;;
esac
export AITER_GFX1201_SAGE_ASM="${AITER_GFX1201_SAGE_ASM:-${preset_asm}}"
export AITER_GFX1201_SAGE_ASM_SCHED="${AITER_GFX1201_SAGE_ASM_SCHED-${preset_sched}}"

# Materials: explicit references, or one of the repository's Ref2AV materials.
if (( ${#references[@]} == 0 )); then
    case "${material}" in
        rooftop) material_config="${run_root}/configs/minimax_h3/ref2av_rooftop_2ref_15s.sh" ;;
        storeroom) material_config="${run_root}/configs/minimax_h3/ref2av_storeroom_courtyard_9ref_15s.sh" ;;
        *) die "unknown --material '${material}' (rooftop, storeroom)" ;;
    esac
    explicit_prompt_file="${prompt_file:-${H3_PROMPT_FILE:-}}"
    # shellcheck source=/dev/null
    source "${material_config}"
    prompt_file="${explicit_prompt_file:-${H3_PROMPT_FILE}}"
    mapfile -t references <<<"${H3_REFERENCE_IMAGES}"
else
    prompt_file="${prompt_file:-${H3_PROMPT_FILE:-}}"
    [[ -n "${prompt_file}" ]] || die "--reference-image needs --prompt-file"
fi
(( ${#references[@]} >= 1 && ${#references[@]} <= 9 )) || die "provide 1-9 reference images"
for i in "${!references[@]}"; do
    [[ "${references[i]}" != *,* && "${references[i]}" != *$'\n'* ]] || die "reference path cannot contain a comma or newline"
    [[ "${references[i]}" == /* ]] || references[i]="${PWD}/${references[i]}"
done
[[ "${prompt_file}" == /* ]] || prompt_file="${PWD}/${prompt_file}"
[[ "${prompt_file}" != *,* ]] || die "prompt path cannot contain a comma"
[[ "${model_dir}" == /* ]] || model_dir="${PWD}/${model_dir}"
result_dir="${result_dir:-${OUTPUT_DIR:-${PWD}/outputs}/minimax_h3_ref2av_tp2_$(date +%Y%m%d_%H%M%S)}"
[[ "${result_dir}" == /* ]] || result_dir="${PWD}/${result_dir}"

container_refs=()
docker_args=(
    run --rm --init
    --name "${container_name}"
    "--network=${docker_network}"
    --shm-size=64G
    --cap-add=SYS_PTRACE --security-opt=seccomp=unconfined
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864
    --device=/dev/kfd
    --mount "type=bind,source=${run_root},target=/workspace/LightX2VRun,readonly"
    --mount "type=bind,source=${model_dir},target=/models/minimax_h3,readonly"
    --mount "type=bind,source=${result_dir},target=/workspace/save_results"
    --mount "type=bind,source=${prompt_file},target=/inputs/h3_prompt.txt,readonly"
    --env MODEL_PATH=/models/minimax_h3
    --env H3_PROMPT_FILE=/inputs/h3_prompt.txt
    --env "H3_INFER_STEPS=${steps}"
    --env "SAVE_RESULT_PATH=/workspace/save_results/minimax_h3_ref2av_tp2_${steps}.mp4"
)
for i in "${!references[@]}"; do
    reference_name="$(basename -- "${references[i]}")"
    target="/inputs/h3_reference_${i}.${reference_name##*.}"
    [[ "${reference_name}" == *.* ]] || target="/inputs/h3_reference_${i}"
    container_refs+=("${target}")
    docker_args+=(--mount "type=bind,source=${references[i]},target=${target},readonly")
done
docker_args+=(--env "H3_REFERENCE_IMAGES=$(printf '%s\n' "${container_refs[@]}")")

if [[ "${dri_all}" == 1 ]]; then
    # All DRM nodes; HIP selects the two physical GPUs. Used for the measured runs.
    docker_args+=(--ipc=host --device=/dev/dri --env "HIP_VISIBLE_DEVICES=${gpus}")
else
    [[ -n "${H3_DRI_NODES:-}" ]] || die "DRI_ALL=0 requires H3_DRI_NODES, e.g. 'card1 renderD128 card2 renderD129'"
    docker_args+=(--ipc=private --env HIP_VISIBLE_DEVICES=0,1)
    for node in ${H3_DRI_NODES}; do docker_args+=("--device=/dev/dri/${node}"); done
fi
if [[ -n "${numa_node}" ]]; then
    cpulist_file="/sys/devices/system/node/node${numa_node}/cpulist"
    [[ -r "${cpulist_file}" ]] || die "NUMA node ${numa_node} not found"
    docker_args+=(--cpuset-cpus "$(<"${cpulist_file}")" --cpuset-mems "${numa_node}")
fi
for host_group in video render; do
    if group_entry="$(getent group "${host_group}" 2>/dev/null)"; then
        docker_args+=(--group-add "$(cut -d: -f3 <<<"${group_entry}")")
    fi
done
for passthrough_env in \
    AITER_GFX1201_SAGE_ASM AITER_GFX1201_SAGE_ASM_SCHED AITER_GFX1201_DOWN_BLAS \
    FUSED_TAIL TE_PAGEABLE VAE_ENC_PAR H3_TEXT_ENCODER_TP H3_BASE_RUNTIME_CONFIG \
    LIGHTX2V_H3_VAE_DEC_BATCH LIGHTX2V_H3_VAE_TRITON_ATTN LIGHTX2V_H3_TE_PREFETCH_LATE \
    LIGHTX2V_H3_TE_ANON_HOST LIGHTX2V_H3_GC_FREEZE LIGHTX2V_H3_ASYNC_VAE_LOAD \
    LIGHTX2V_H3_VAE_ENC_PREWARM LIGHTX2V_H3_DIT_PIN_SLAB LIGHTX2V_H3_AUDIO_POLY_UPSAMPLE \
    LIGHTX2V_H3_VAE_STREAM_SAVE LIGHTX2V_H3_VAE_STREAM_SHIFT \
    SEED NCCL_DEBUG NCCL_DEBUG_SUBSYS; do
    if [[ -n "${!passthrough_env+x}" ]]; then
        docker_args+=(--env "${passthrough_env}=${!passthrough_env}")
    fi
done
if [[ -n "${DOCKER_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_docker_args <<<"${DOCKER_EXTRA_ARGS}"
    docker_args+=("${extra_docker_args[@]}")
fi
docker_args+=(
    --entrypoint /bin/bash "${image}"
    /workspace/LightX2VRun/scripts/lib/run_minimax_h3_ref2av_tp2_in_container.sh
)

echo "Image: ${image}"
echo "GPUs: ${gpus}  NUMA: ${numa_node:-none}  network: ${docker_network}"
echo "Sage: preset=${sage_preset} AITER_GFX1201_SAGE_ASM=${AITER_GFX1201_SAGE_ASM} AITER_GFX1201_SAGE_ASM_SCHED=${AITER_GFX1201_SAGE_ASM_SCHED}"
echo "Result dir: ${result_dir}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'docker'; printf ' %q' "${docker_args[@]}"; printf '\n'
    exit 0
fi
command -v docker >/dev/null 2>&1 || die "docker is not installed or not in PATH"
[[ -c /dev/kfd ]] || die "ROCm device /dev/kfd is unavailable"
[[ -d "${model_dir}" ]] || die "model directory not found: ${model_dir}"
[[ -f "${prompt_file}" ]] || die "prompt file not found: ${prompt_file}"
for reference in "${references[@]}"; do [[ -f "${reference}" ]] || die "reference image not found: ${reference}"; done
mkdir -p "${result_dir}"
exec docker "${docker_args[@]}"
