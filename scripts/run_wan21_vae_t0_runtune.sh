#!/usr/bin/env bash
# Four-card Wan2.1 VAE T0: legacy -> MIOpen -> untiled -> GPU resident.
set -Eeuo pipefail

G='\033[1;32m'; C='\033[0;36m'; Y='\033[0;33m'; R='\033[0;31m'; N='\033[0m'
info() { echo -e "[$(date '+%H:%M:%S')] ${C}INFO${N} $*"; }
ok()   { echo -e "[$(date '+%H:%M:%S')] ${G}OK${N}   $*"; }
warn() { echo -e "[$(date '+%H:%M:%S')] ${Y}WARN${N} $*"; }
err()  { echo -e "[$(date '+%H:%M:%S')] ${R}ERR${N}  $*" >&2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RUN_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUNNER="${SCRIPT_DIR}/benchmark_wan21_vae_t0.py"
IMAGE="${IMAGE:-ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260804-SageUtimateWithMM-T1}"
GPU_IDS="${GPU_IDS:-${GPU_DEVICES:-0,1,2,3}}"
MODEL_STORE="${MODEL_STORE:-/disk/ssd/difumodels-Dis}"
WAN21_DISTILL_DIR="${WAN21_DISTILL_DIR:-${MODEL_STORE}/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v/distill_fp8}"
VAE_PATH="${VAE_PATH:-${WAN21_DISTILL_DIR}/Wan2.1_VAE.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}}"
T0_ROUNDS="${T0_ROUNDS:-1}"
T0_ROTATION="${T0_ROTATION:-0}"
T0_TIMEOUT_SECONDS="${T0_TIMEOUT_SECONDS:-900}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${RUNNER}" ]] || { err "runner not found: ${RUNNER}"; exit 1; }
[[ "${T0_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { err "T0_ROUNDS must be a positive integer"; exit 1; }
[[ "${T0_ROTATION}" =~ ^[0-3]$ ]] || { err "T0_ROTATION must be 0, 1, 2, or 3"; exit 1; }
[[ "${T0_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { err "T0_TIMEOUT_SECONDS must be positive"; exit 1; }

IFS=',' read -r -a RAW_GPU_LIST <<<"${GPU_IDS}"
GPU_LIST=()
declare -A GPU_SEEN=()
for raw_gpu in "${RAW_GPU_LIST[@]}"; do
    gpu="${raw_gpu//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { err "GPU_IDS must contain numeric ROCm device IDs"; exit 1; }
    [[ -z "${GPU_SEEN[${gpu}]:-}" ]] || { err "duplicate GPU ID: ${gpu}"; exit 1; }
    GPU_SEEN["${gpu}"]=1
    GPU_LIST+=("${gpu}")
done
[[ "${#GPU_LIST[@]}" -eq 4 ]] || { err "Wan2.1 VAE T0 requires exactly four GPU IDs"; exit 1; }

ARM_NAMES=(legacy miopen miopen_untiled optimized)
ARM_MIOPEN=(0 1 1 1)
ARM_TILING=(1 1 0 0)
ARM_OFFLOAD=(1 1 1 0)

NOW="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/wan21_vae_t0_${NOW}}"
if [[ -d "${RESULT_DIR}" && -n "$(find "${RESULT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    err "result directory is not empty: ${RESULT_DIR}"
    exit 1
fi
mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(realpath "${RESULT_DIR}")"
exec > >(tee -a "${RESULT_DIR}/launcher.log") 2>&1

print_mapping() {
    local ordinal arm_index
    for ordinal in "${!GPU_LIST[@]}"; do
        arm_index=$(((ordinal + T0_ROTATION) % 4))
        info "GPU ${GPU_LIST[${ordinal}]} -> ${ARM_NAMES[${arm_index}]} "\
"(MIOpen=${ARM_MIOPEN[${arm_index}]}, tiling=${ARM_TILING[${arm_index}]}, VAE_CPU_OFFLOAD=${ARM_OFFLOAD[${arm_index}]})"
    done
}

info "Image      : ${IMAGE}"
info "VAE        : ${VAE_PATH}"
info "Workload   : Wan2.1 720p, 81 frames, BF16 encode + decode"
info "Rounds     : ${T0_ROUNDS}"
info "Rotation   : ${T0_ROTATION}"
info "Results    : ${RESULT_DIR}"
print_mapping

if [[ "${DRY_RUN}" == 1 ]]; then
    ok "Dry run complete; no containers started"
    exit 0
fi

command -v docker >/dev/null || { err "docker is required"; exit 1; }
docker image inspect "${IMAGE}" >/dev/null
[[ -f "${VAE_PATH}" ]] || { err "Wan2.1 VAE checkpoint not found: ${VAE_PATH}"; exit 1; }

SYNC_DIR="$(mktemp -d /tmp/wan21-vae-t0-sync.XXXXXX)"
REFERENCE_DIR="$(mktemp -d /tmp/wan21-vae-t0-reference.XXXXXX)"
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    rm -rf -- "${SYNC_DIR}" "${REFERENCE_DIR}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

{
    date --iso-8601=seconds
    uname -a
    git -C "${REPO_ROOT}" rev-parse HEAD
    git -C "${REPO_ROOT}" status --short --branch
    docker image inspect --format '{{.Id}} {{.RepoDigests}}' "${IMAGE}"
    docker image inspect --format 'LightX2V={{index .Config.Labels "org.opencontainers.image.revision"}} Aiter={{index .Config.Labels "org.opencontainers.image.aiter.revision"}}' "${IMAGE}"
    echo "GPU_IDS=${GPU_IDS}"
    echo "T0_ROUNDS=${T0_ROUNDS}"
    echo "T0_ROTATION=${T0_ROTATION}"
    if command -v rocm-smi >/dev/null; then
        rocm-smi --showproductname --showuniqueid --showserial --showmeminfo vram || true
    fi
} >"${RESULT_DIR}/host_environment.txt" 2>&1

run_gpu() {
    local gpu=$1
    local arm_index=$2
    local arm=${ARM_NAMES[${arm_index}]}
    local gpu_result="${RESULT_DIR}/gpu${gpu}"
    local container_tmp="${gpu_result}/container_tmp"
    mkdir -p "${gpu_result}" "${container_tmp}"
    chmod 1777 "${container_tmp}"

    docker run --rm \
        --network=host \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt=seccomp=unconfined \
        --device=/dev/kfd \
        --device=/dev/dri \
        --ulimit=memlock=-1:-1 \
        --ulimit=stack=67108864:67108864 \
        --workdir=/workspace/LightX2V \
        --env "HIP_VISIBLE_DEVICES=${gpu}" \
        --env PLATFORM=amd_rocm \
        --env DTYPE=BF16 \
        --env "LIGHTX2V_AMD_MIOPEN=${ARM_MIOPEN[${arm_index}]}" \
        --env USE_CHANNELS_LAST_3D=0 \
        --env TMPDIR=/tmp \
        --env PYTHONUNBUFFERED=1 \
        --mount "type=bind,source=${RUNNER},target=/workspace/benchmark_wan21_vae_t0.py,readonly" \
        --mount "type=bind,source=${VAE_PATH},target=/models/Wan2.1_VAE.pth,readonly" \
        --mount "type=bind,source=${gpu_result},target=/results" \
        --mount "type=bind,source=${container_tmp},target=/tmp" \
        --mount "type=bind,source=${SYNC_DIR},target=/t0-sync" \
        --mount "type=bind,source=${REFERENCE_DIR},target=/t0-reference" \
        --entrypoint=python3 \
        "${IMAGE}" \
        -u /workspace/benchmark_wan21_vae_t0.py \
        --output-dir /results \
        --reference-dir /t0-reference \
        --sync-dir /t0-sync \
        --participants 4 \
        --arm "${arm}" \
        --gpu-id "${gpu}" \
        --miopen "${ARM_MIOPEN[${arm_index}]}" \
        --tiling "${ARM_TILING[${arm_index}]}" \
        --vae-cpu-offload "${ARM_OFFLOAD[${arm_index}]}" \
        --vae-path /models/Wan2.1_VAE.pth \
        --rounds "${T0_ROUNDS}" \
        --timeout-seconds "${T0_TIMEOUT_SECONDS}"
}

run_and_report() {
    local gpu=$1
    local arm_index=$2
    local arm=${ARM_NAMES[${arm_index}]}
    set -o pipefail
    if run_gpu "${gpu}" "${arm_index}" 2>&1 | tee "${RESULT_DIR}/gpu${gpu}/run.log" | awk -v prefix="[gpu${gpu}:${arm}] " '{ print prefix $0; fflush() }'; then
        return 0
    fi
    touch "${SYNC_DIR}/abort"
    return 1
}

PIDS=()
ARM_INDEXES=()
for ordinal in "${!GPU_LIST[@]}"; do
    gpu=${GPU_LIST[${ordinal}]}
    arm_index=$(((ordinal + T0_ROTATION) % 4))
    run_and_report "${gpu}" "${arm_index}" &
    PIDS+=("$!")
    ARM_INDEXES+=("${arm_index}")
done

FAILED=0
for ordinal in "${!PIDS[@]}"; do
    gpu=${GPU_LIST[${ordinal}]}
    arm=${ARM_NAMES[${ARM_INDEXES[${ordinal}]}]}
    if wait "${PIDS[${ordinal}]}"; then
        ok "GPU ${gpu} ${arm} finished"
    else
        err "GPU ${gpu} ${arm} failed; see ${RESULT_DIR}/gpu${gpu}/run.log"
        FAILED=1
    fi
done
[[ "${FAILED}" == 0 ]] || exit 1

python3 -B "${RUNNER}" --aggregate-root "${RESULT_DIR}" | tee "${RESULT_DIR}/aggregate.log"
ARCHIVE="${RESULT_DIR}.tar.gz"
tar -C "$(dirname "${RESULT_DIR}")" -czf "${ARCHIVE}" "$(basename "${RESULT_DIR}")"
ok "T0 summary : ${RESULT_DIR}/summary.json"
ok "T0 archive : ${ARCHIVE}"
warn "The four arms use different cards. Re-run with T0_ROTATION=1 to check card bias before production gating."
