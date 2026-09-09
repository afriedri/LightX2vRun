#!/usr/bin/env bash
# One host entry for image-based LTX-2.5 SP2/SP4/SP8/SP16 E2E evidence.
set -Eeuo pipefail

G='\033[1;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
info() { echo -e "[$(date '+%H:%M:%S')] ${C}INFO${N} $*"; }
ok()   { echo -e "[$(date '+%H:%M:%S')] ${G}OK${N}   $*"; }
warn() { echo -e "[$(date '+%H:%M:%S')] ${Y}WARN${N} $*"; }
err()  { echo -e "[$(date '+%H:%M:%S')] ${R}ERR${N}  $*" >&2; }
die()  { err "$*"; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
scale_entry="${run_root}/scripts/run_10_ltx25_c1_704x1280_121f_24fps_scale.sh"

image="${IMAGE:-}"
result_dir="${RESULT_DIR:-}"
dry_run="${DRY_RUN:-0}"
source_overlay="${SOURCE_OVERLAY:-0}"

lightx2v_src="${LIGHTX2V_SRC:-}"
lightx2v_git_url="${LTX25_LIGHTX2V_GIT_URL:-${LIGHTX2V_GIT_URL:-git@gitcode.com:hvat-ai/LightX2V.git}}"
lightx2v_git_ref="${LTX25_LIGHTX2V_REF:-${LIGHTX2V_GIT_REF:-hvat-gfx1201-R9600}}"
lightx2v_git_dir="${LTX25_LIGHTX2V_GIT_DIR:-${LIGHTX2V_GIT_DIR:-/disk/ssd/lightx2v-runtime/source-cache/ltx25-scale/LightX2V-hvat-gfx1201-R9600}}"
allow_dirty_lightx2v="${ALLOW_DIRTY_LIGHTX2V:-0}"

aiter_src="${AITER_SRC:-}"
aiter_git_url="${AITER_GIT_URL:-git@gitcode.com:hvat-ai/aiter.git}"
aiter_git_ref="${AITER_GIT_REF:-LightX2V/hvat-gfx1201-R9600}"
aiter_git_dir="${AITER_GIT_DIR:-/disk/ssd/lightx2v-runtime/source-cache/ltx25-scale/aiter-hvat-gfx1201-R9600}"
allow_dirty_aiter="${ALLOW_DIRTY_AITER:-0}"

run_args=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/LTX25-ScaleCollector.sh --image IMAGE [OPTIONS]

Collects LTX-2.5 SP2/SP4/SP8/SP16 E2E scale evidence from IMAGE and delegates
scale grouping, memory observation, video output, and timing summaries to the
existing numbered task 10 launcher. The default path uses the image-installed
LightX2V and Aiter wheel, including its AOT CK Conv3d module.

Options:
  --image IMAGE
  --counts 2,4,8,16     Forwarded to numbered task 10
  --gpu-count N         Run one of 2, 4, 8, or 16
  --output-dir DIR      Exact host timing/evidence directory

Source controls:
  SOURCE_OVERLAY=0|1    Default: 0. Set to 1 only for explicit source debugging
  LIGHTX2V_SRC=/path    Use this local Git checkout instead of fetching
  LIGHTX2V_GIT_URL=URL
  LIGHTX2V_GIT_REF=REF  Default: hvat-gfx1201-R9600
  LIGHTX2V_GIT_DIR=DIR  Managed checkout cache
  LTX25_LIGHTX2V_GIT_URL / LTX25_LIGHTX2V_REF / LTX25_LIGHTX2V_GIT_DIR
                         CollectorUltra-compatible aliases
  AITER_SRC=/path       Use this local Git checkout instead of fetching
  AITER_GIT_URL=URL
  AITER_GIT_REF=REF     Default: LightX2V/hvat-gfx1201-R9600
  AITER_GIT_DIR=DIR     Managed checkout cache, including CK submodule
  ALLOW_DIRTY_LIGHTX2V=0|1
  ALLOW_DIRTY_AITER=0|1

Other controls:
  LTX25_MODEL_ROOT=/disk/ssd/difumodels-Dis/LTX-2.5
  OUTPUT_DIR=/disk/ssd/lightx2v-runtime/outputs
  CACHE_DIR=/disk/ssd/lightx2v-runtime/cache
  DRY_RUN=1             Use existing/provided checkouts and print all commands

Source controls are ignored unless SOURCE_OVERLAY=1. This collector does not
select a new Hot kernel and does not modify either managed checkout. Hot
selection remains in aiter-tune-gfx1201.
EOF
}

validate_lightx2v_source() {
    lightx2v_src="$(realpath -- "${lightx2v_src}")"
    [[ -f "${lightx2v_src}/lightx2v/__init__.py" ]] ||
        die "LIGHTX2V_SRC is not a LightX2V source tree: ${lightx2v_src}"
    [[ -f "${lightx2v_src}/lightx2v/models/video_encoders/hf/ltx2/video_vae/distributed_decoder.py" ]] ||
        die "LightX2V source does not contain the distributed LTX-2.5 Video VAE implementation"
    git -C "${lightx2v_src}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
        die "LIGHTX2V_SRC must be a Git checkout"
    if [[ "${allow_dirty_lightx2v}" == 0 && -n "$(git -C "${lightx2v_src}" status --porcelain)" ]]; then
        die "LIGHTX2V_SRC is dirty; use the managed checkout or set ALLOW_DIRTY_LIGHTX2V=1"
    fi
    lightx2v_revision="$(git -C "${lightx2v_src}" rev-parse HEAD)"
    [[ "${lightx2v_revision}" =~ ^[0-9a-fA-F]{40}$ ]] || die "invalid LightX2V revision"
}

validate_aiter_source() {
    aiter_src="$(realpath -- "${aiter_src}")"
    [[ -f "${aiter_src}/aiter/__init__.py" ]] || die "AITER_SRC is not an Aiter source tree: ${aiter_src}"
    git -C "${aiter_src}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "AITER_SRC must be a Git checkout"
    if [[ "${allow_dirty_aiter}" == 0 && -n "$(git -C "${aiter_src}" status --porcelain)" ]]; then
        die "AITER_SRC is dirty; use the managed checkout or set ALLOW_DIRTY_AITER=1"
    fi
    aiter_revision="$(git -C "${aiter_src}" rev-parse HEAD)"
    [[ "${aiter_revision}" =~ ^[0-9a-fA-F]{40}$ ]] || die "invalid Aiter revision"
    if [[ -f "${aiter_src}/.gitmodules" ]]; then
        [[ -d "${aiter_src}/3rdparty/composable_kernel" ]] || die "Aiter CK submodule checkout is missing"
        aiter_ck_revision="$(git -C "${aiter_src}/3rdparty/composable_kernel" rev-parse HEAD 2>/dev/null || true)"
        [[ "${aiter_ck_revision}" =~ ^[0-9a-fA-F]{40}$ ]] || die "invalid Aiter CK submodule revision"
    else
        aiter_ck_revision=not-recorded
    fi
}

prepare_managed_source() {
    local label="$1" url="$2" ref="$3" directory="$4"
    directory="$(realpath -m -- "${directory}")"
    mkdir -p -- "$(dirname -- "${directory}")"
    if [[ -e "${directory}" && ! -d "${directory}/.git" ]]; then
        die "managed ${label} path is not a Git checkout: ${directory}"
    fi
    if [[ ! -d "${directory}/.git" ]]; then
        info "Creating managed ${label} checkout: ${directory}"
        git clone --no-checkout "${url}" "${directory}"
    else
        local actual_url
        actual_url="$(git -C "${directory}" remote get-url origin)"
        [[ "${actual_url}" == "${url}" ]] ||
            die "managed ${label} origin mismatch: expected ${url}, got ${actual_url}"
        [[ -z "$(git -C "${directory}" status --porcelain)" ]] ||
            die "managed ${label} checkout is dirty: ${directory}"
    fi
    info "Fetching ${label} ref: ${ref}"
    git -C "${directory}" fetch --no-tags origin "${ref}"
    local resolved_ref
    resolved_ref="$(git -C "${directory}" rev-parse --verify 'FETCH_HEAD^{commit}')"
    git -C "${directory}" checkout -q --detach "${resolved_ref}"
    prepared_source="${directory}"
}

run_source_overlay_container() {
    [[ "${1:-}" == --run-ltx25-c1-gfx1201 && $# == 1 ]] ||
        die "internal collector MODEL_RUN only supports --run-ltx25-c1-gfx1201"

    local runtime_image="${IMAGE:-}"
    local gpu_devices="${GPU_DEVICES:-}"
    local model_root="${LTX25_MODEL_ROOT:-/disk/ssd/difumodels-Dis/LTX-2.5}"
    local runtime_root="${RUNTIME_ROOT:-/disk/ssd/lightx2v-runtime}"
    local input_dir="${INPUT_DIR:-${runtime_root}/inputs}"
    local output_dir="${OUTPUT_DIR:-${runtime_root}/outputs}"
    local cache_dir="${CACHE_DIR:-${runtime_root}/cache}"
    local container_name="${CONTAINER_NAME:-lightx2v-ltx25-scale-collector}"
    local use_source_overlay="${LTX25_SCALE_SOURCE_OVERLAY:-0}"
    local mounted_lightx2v="${LTX25_SCALE_LIGHTX2V_SOURCE:-}"
    local mounted_aiter="${LTX25_SCALE_AITER_SOURCE:-}"
    local mounted_lightx2v_revision="${LTX25_SCALE_LIGHTX2V_REVISION:-image}"
    local mounted_aiter_revision="${LTX25_SCALE_AITER_REVISION:-image}"
    local tunableop_root=/workspace/LightX2V/configs/ltx25/gfx1201
    local inner=/workspace/LightX2VRun/scripts/lib/run_ltx25_c1_in_container.sh
    local -a gpu_list docker_args command

    [[ -n "${runtime_image}" ]] || die "internal collector launch received no IMAGE"
    [[ "${gpu_devices}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "invalid GPU_DEVICES: ${gpu_devices}"
    IFS=',' read -r -a gpu_list <<<"${gpu_devices}"
    case "${#gpu_list[@]}" in 2|4|8|16) ;; *) die "ScaleCollector supports SP2, SP4, SP8, or SP16" ;; esac

    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        command -v docker >/dev/null 2>&1 || die "docker is required"
        [[ -d "${model_root}" ]] || die "LTX-2.5 model root not found: ${model_root}"
        [[ -f "${run_root}/scripts/lib/run_ltx25_c1_in_container.sh" ]] || die "LightX2VRun inner entry is missing"
        if [[ "${use_source_overlay}" == 1 ]]; then
            for cache_rank in 0 1; do
                [[ -f "${mounted_lightx2v}/configs/ltx25/gfx1201/tunableop_results_full${cache_rank}.csv" ]] ||
                    die "LightX2V source is missing TunableOp rank cache ${cache_rank}"
            done
        fi
        mkdir -p -- "${input_dir}" "${output_dir}" \
            "${cache_dir}/aiter" "${cache_dir}/huggingface" \
            "${cache_dir}/torch_extensions" "${cache_dir}/triton"
        docker image inspect "${runtime_image}" >/dev/null 2>&1 || die "Docker image unavailable: ${runtime_image}"
    fi

    docker_args=(
        run --rm
        --name "${container_name}"
        --hostname "${container_name}"
        --init --ipc=host
        --device=/dev/kfd --device=/dev/dri
        --ulimit=memlock=-1:-1
        --ulimit=stack=67108864:67108864
        --workdir=/workspace/LightX2V
        --env "HIP_VISIBLE_DEVICES=${gpu_devices}"
        --env PLATFORM=amd_rocm --env DTYPE=BF16
        --env TOKENIZERS_PARALLELISM=false
        --env "MODEL_PATH=${model_root}"
        --env LIGHTX2V_PATH=/workspace/LightX2V
        --env TRITON_CACHE_DIR=/workspace/.cache/triton
        --env TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions
        --env HF_HOME=/root/.cache/huggingface
        --env PYTORCH_ROCM_ARCH=gfx1201 --env AITER_ROCM_ARCH=gfx1201
        --env GPU_ARCHS=gfx1201 --env CU_NUM=24
        --env AITER_LOG_TUNED_CONFIG=1
        --env HSA_ENABLE_IPC_MODE_LEGACY=1 --env HIP_FORCE_DEV_KERNARG=1
        --env SAFETENSORS_FAST_GPU=1
        --env LIGHTX2V_LTX25_C1_IN_CONTAINER=1
        --env LTX25_C1_RESULT_DIR=/workspace/LightX2V/save_results
        --env "LTX25_MODEL_ROOT=${model_root}"
        --env "LTX25_WORLD_SIZE=${#gpu_list[@]}"
        --env PYTORCH_TUNABLEOP_ENABLED=1
        --env PYTORCH_TUNABLEOP_TUNING=0
        --env PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
        --env "PYTORCH_TUNABLEOP_FILENAME=${tunableop_root}/tunableop_results_full%d.csv"
        --env "LTX25_TUNABLEOP_CACHE_RANK0=${tunableop_root}/tunableop_results_full0.csv"
        --env "LTX25_TUNABLEOP_CACHE_RANK1=${tunableop_root}/tunableop_results_full1.csv"
        --env LIGHTX2V_AMD_RMSNORM_BACKEND=ck
        --env LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
        --env LIGHTX2V_CK_CONV3D_PROFILE_DIR=/workspace/LightX2V/save_results/ck_conv3d_routes
        --mount "type=bind,source=${run_root},target=/workspace/LightX2VRun,readonly"
        --mount "type=bind,source=${model_root},target=${model_root},readonly"
        --mount "type=bind,source=${input_dir},target=/inputs"
        --mount "type=bind,source=${output_dir},target=/workspace/LightX2V/save_results"
        --mount "type=bind,source=${cache_dir}/aiter,target=/workspace/.cache/aiter"
        --mount "type=bind,source=${cache_dir}/huggingface,target=/root/.cache/huggingface"
        --mount "type=bind,source=${cache_dir}/torch_extensions,target=/workspace/.cache/torch_extensions"
        --mount "type=bind,source=${cache_dir}/triton,target=/workspace/.cache/triton"
    )

    if [[ "${use_source_overlay}" == 1 ]]; then
        [[ -n "${mounted_lightx2v}" && -n "${mounted_aiter}" ]] ||
            die "source overlay requested without prepared LightX2V/Aiter sources"
        docker_args+=(
            --env PYTHONPATH=/workspace/aiter-source
            --env "AITER_JIT_DIR=/workspace/.cache/aiter/ltx25-source-${mounted_aiter_revision:0:12}"
            --env "LIGHTX2V_SOURCE_REVISION=${mounted_lightx2v_revision}"
            --env "AITER_SOURCE_REVISION=${mounted_aiter_revision}"
            --env "AITER_COMMIT=${mounted_aiter_revision}"
            --mount "type=bind,source=${mounted_lightx2v},target=/workspace/LightX2V,readonly"
            --mount "type=bind,source=${mounted_aiter},target=/workspace/aiter-source,readonly"
        )
    fi

    if [[ -n "${CK_CONV3D_TUNED_CSV:-}" ]]; then
        local conv3d_csv
        conv3d_csv="$(realpath -m -- "${CK_CONV3D_TUNED_CSV}")"
        [[ "${DRY_RUN:-0}" == 1 || -f "${conv3d_csv}" ]] || die "Conv3D CSV not found: ${conv3d_csv}"
        docker_args+=(
            --mount "type=bind,source=${conv3d_csv},target=/workspace/ck-conv3d-config/conv3d_v1_tuned.csv,readonly"
            --env AITER_CONFIG_CONV3D_V1=/workspace/ck-conv3d-config/conv3d_v1_tuned.csv
        )
    fi

    for passthrough in \
        PROMPT NEGATIVE_PROMPT SAVE_RESULT_PATH SEED MASTER_ADDR MASTER_PORT \
        TARGET_VIDEO_LENGTH TARGET_HEIGHT TARGET_WIDTH TARGET_FPS INFER_STEPS \
        BASE_CONFIG_JSON LIGHTX2V_EXTRA_ARGS CPU_OFFLOAD OFFLOAD_GRANULARITY OFFLOAD_RATIO \
        PROFILE_WORKLOAD NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE; do
        if [[ -n "${!passthrough:-}" ]]; then
            docker_args+=(--env "${passthrough}=${!passthrough}")
        fi
    done
    if [[ -e /etc/localtime ]]; then
        docker_args+=(--mount type=bind,source=/etc/localtime,target=/etc/localtime,readonly)
    fi

    command=(docker "${docker_args[@]}" "${runtime_image}" /bin/bash "${inner}")
    if [[ "${use_source_overlay}" == 1 ]]; then
        printf 'LTX-2.5 ScaleCollector source overlay: LightX2V=%s Aiter=%s\n' \
            "${mounted_lightx2v_revision}" "${mounted_aiter_revision}"
    else
        printf 'LTX-2.5 ScaleCollector source: image-installed LightX2V + Aiter AOT wheel\n'
    fi
    printf 'LTX-2.5 ScaleCollector container: SP%s GPUs=%s\n' "${#gpu_list[@]}" "${gpu_devices}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf 'DRY RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return
    fi
    exec "${command[@]}"
}

if [[ "${LTX25_SCALE_COLLECTOR_INNER:-0}" == 1 ]]; then
    run_source_overlay_container "$@"
    exit
fi

while (( $# > 0 )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --image) (( $# >= 2 )) || die "--image requires a value"; image="$2"; shift 2 ;;
        --image=*) image="${1#--image=}"; shift ;;
        --output-dir) (( $# >= 2 )) || die "--output-dir requires a value"; result_dir="$2"; shift 2 ;;
        --output-dir=*) result_dir="${1#--output-dir=}"; shift ;;
        --counts|--gpu-count)
            (( $# >= 2 )) || die "$1 requires a value"
            run_args+=("$1" "$2")
            shift 2
            ;;
        --counts=*|--gpu-count=*) run_args+=("$1"); shift ;;
        2|4|8|16) run_args+=("$1"); shift ;;
        *) die "unsupported argument: $1" ;;
    esac
done

for value in "${dry_run}" "${source_overlay}" "${allow_dirty_lightx2v}" "${allow_dirty_aiter}"; do
    [[ "${value}" == 0 || "${value}" == 1 ]] || die "boolean controls must be 0 or 1"
done
[[ -n "${image}" ]] || die "--image or IMAGE is required"
[[ "${image}" != *[[:space:]]* ]] || die "image reference may not contain whitespace"
[[ -f "${scale_entry}" ]] || die "numbered LTX-2.5 scale entry not found: ${scale_entry}"
if [[ "${source_overlay}" == 1 ]]; then
    command -v git >/dev/null 2>&1 || die "git is required for SOURCE_OVERLAY=1"
    if [[ -n "${lightx2v_src}" ]]; then
        validate_lightx2v_source
    elif [[ "${dry_run}" == 1 ]]; then
        [[ -d "${lightx2v_git_dir}/.git" ]] || die "DRY_RUN source overlay requires LIGHTX2V_SRC or an existing managed checkout"
        lightx2v_src="${lightx2v_git_dir}"
        validate_lightx2v_source
        warn "DRY_RUN uses the existing LightX2V checkout without fetching"
    else
        prepare_managed_source LightX2V "${lightx2v_git_url}" "${lightx2v_git_ref}" "${lightx2v_git_dir}"
        lightx2v_src="${prepared_source}"
        validate_lightx2v_source
    fi

    if [[ -n "${aiter_src}" ]]; then
        validate_aiter_source
    elif [[ "${dry_run}" == 1 ]]; then
        [[ -d "${aiter_git_dir}/.git" ]] || die "DRY_RUN source overlay requires AITER_SRC or an existing managed checkout"
        aiter_src="${aiter_git_dir}"
        validate_aiter_source
        warn "DRY_RUN uses the existing Aiter checkout without fetching"
    else
        prepare_managed_source Aiter "${aiter_git_url}" "${aiter_git_ref}" "${aiter_git_dir}"
        aiter_src="${prepared_source}"
        git -C "${aiter_src}" submodule sync --recursive
        git -C "${aiter_src}" submodule update --init --recursive
        validate_aiter_source
    fi

    ok "LightX2V source ready: ${lightx2v_revision}"
    ok "Aiter source ready: ${aiter_revision} (CK ${aiter_ck_revision})"
else
    lightx2v_src=""
    aiter_src=""
    lightx2v_revision=image
    aiter_revision=image
    info "Source overlay: disabled; using LightX2V and Aiter AOT wheel from IMAGE"
fi
info "Runtime image : ${image}"
info "Scale entry   : ${scale_entry}"
info "Collection    : task-10 video/log/memory/timing; no Hot selection"

export LTX25_SCALE_LIGHTX2V_SOURCE="${lightx2v_src}"
export LTX25_SCALE_AITER_SOURCE="${aiter_src}"
export LTX25_SCALE_LIGHTX2V_REVISION="${lightx2v_revision}"
export LTX25_SCALE_AITER_REVISION="${aiter_revision}"
export LTX25_SCALE_SOURCE_OVERLAY="${source_overlay}"
export LTX25_SCALE_COLLECTOR_INNER=1
export MODEL_RUN="${script_dir}/LTX25-ScaleCollector.sh"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-ltx25-scale-collector}"
export MEMORY_MONITOR_CONTAINER="${MEMORY_MONITOR_CONTAINER:-${CONTAINER_NAME}}"
export DRY_RUN="${dry_run}"
if [[ -n "${result_dir}" ]]; then
    export TIMING_RESULT_DIR="${result_dir}"
fi

exec bash "${scale_entry}" --image "${image}" "${run_args[@]}"
