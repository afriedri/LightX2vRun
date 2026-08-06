#!/usr/bin/env bash
# Two-card Wan2.1 VAE A/B: original PyTorch offload versus CK v1 resident.
set -Eeuo pipefail

G='\033[1;32m'; C='\033[0;36m'; Y='\033[0;33m'; R='\033[0;31m'; N='\033[0m'
info() { echo -e "[$(date '+%H:%M:%S')] ${C}INFO${N} $*"; }
ok()   { echo -e "[$(date '+%H:%M:%S')] ${G}OK${N}   $*"; }
warn() { echo -e "[$(date '+%H:%M:%S')] ${Y}WARN${N} $*"; }
err()  { echo -e "[$(date '+%H:%M:%S')] ${R}ERR${N}  $*" >&2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_REPO_ROOT="${RUN_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUNNER="${SCRIPT_DIR}/runtune_ck_conv3d_v1.py"
CONTAINER_REPO_ROOT="/workspace/LightX2V"
CONTAINER_RUNNER="/workspace/runtune_ck_conv3d_v1.py"
LEGACY_IMAGE="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260804-SageUtimateWithMM-T1"
CK_AOT_IMAGE="ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260805-CKConv3dV1-AOT"
IMAGE="${IMAGE:-${CK_AOT_IMAGE}}"
if [[ "${IMAGE}" == "${LEGACY_IMAGE}" ]]; then
    IMAGE="${CK_AOT_IMAGE}"
fi
AITER_CK_REV="af7118e342580ecd3f71edce7b1d0ba465012ecf"
EXPECTED_AITER_REV="${EXPECTED_AITER_REV:-1abd99a06c66af00fdd22f711b1222fad44a64fc}"
EXPECTED_LIGHTX2V_REV="${EXPECTED_LIGHTX2V_REV:-2c2f4204cdf2d442fe76a76173604f763c166d63}"
GPU_IDS="${GPU_IDS:-0,1}"
MODEL_STORE="${MODEL_STORE:-/disk/ssd/difumodels-Dis}"
WAN21_DISTILL_DIR="${WAN21_DISTILL_DIR:-${MODEL_STORE}/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v/distill_fp8}"
VAE_PATH="${VAE_PATH:-${WAN21_DISTILL_DIR}/Wan2.1_VAE.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_REPO_ROOT}}"
ROUNDS="${ROUNDS:-1}"
ROTATION="${ROTATION:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
SEED="${SEED:-20260805}"
PROXY_URL="${PROXY_URL-http://192.168.2.3:10808}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "${RUNNER}" ]] || { err "runner not found: ${RUNNER}"; exit 1; }
[[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { err "ROUNDS must be positive"; exit 1; }
[[ "${ROTATION}" =~ ^[01]$ ]] || { err "ROTATION must be 0 or 1"; exit 1; }
[[ "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || { err "TIMEOUT_SECONDS must be positive"; exit 1; }
[[ "${SEED}" =~ ^[0-9]+$ ]] || { err "SEED must be a non-negative integer"; exit 1; }

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
[[ "${#GPU_LIST[@]}" -eq 2 ]] || { err "CK Conv3d v1 runtune requires exactly two GPU IDs"; exit 1; }

ARM_NAMES=(pytorch_offload ck_v1_resident)
ARM_BACKENDS=(pytorch Aiter_CK_Conv3d_v1)
ARM_OFFLOAD=(1 0)
ARM_TILING=(1 0)

NOW="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_DIR}/ck_conv3d_v1_runtune_${NOW}}"
if [[ -d "${RESULT_DIR}" && -n "$(find "${RESULT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    err "result directory is not empty: ${RESULT_DIR}"
    exit 1
fi
mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(realpath "${RESULT_DIR}")"

exec 3>&1 4>&2
exec > >(tee -a "${RESULT_DIR}/launcher.log" >&3) 2>&1
LOGGER_PID=$!

print_mapping() {
    local ordinal arm_index
    for ordinal in "${!GPU_LIST[@]}"; do
        arm_index=$(((ordinal + ROTATION) % 2))
        info "GPU ${GPU_LIST[${ordinal}]} -> ${ARM_NAMES[${arm_index}]} "\
"(backend=${ARM_BACKENDS[${arm_index}]}, MIOpen=0, tiling=${ARM_TILING[${arm_index}]}, VAE_CPU_OFFLOAD=${ARM_OFFLOAD[${arm_index}]}, channels_last_3d=0)"
    done
}

info "Image      : ${IMAGE}"
info "VAE        : ${VAE_PATH}"
info "Workload   : Wan2.1 720p, 81 frames, BF16 encode + decode"
info "GPUs       : ${GPU_IDS}"
info "Rounds     : ${ROUNDS}"
info "Rotation   : ${ROTATION}"
info "Results    : ${RESULT_DIR}"
info "LightX2V   : image-bundled ${CONTAINER_REPO_ROOT}"
info "Run runner : ${RUNNER}"
if [[ -n "${PROXY_URL}" ]]; then
    info "Proxy      : ${PROXY_URL}"
else
    info "Proxy      : disabled"
fi
print_mapping

if [[ "${DRY_RUN}" == 1 ]]; then
    ok "Dry run complete; no containers started"
    exit 0
fi

command -v docker >/dev/null || { err "docker is required"; exit 1; }
docker image inspect "${IMAGE}" >/dev/null
IMAGE_AITER_REV="$(docker image inspect --format \
    '{{with .Config.Labels}}{{index . "org.opencontainers.image.aiter.revision"}}{{end}}' \
    "${IMAGE}")"
IMAGE_LIGHTX2V_REV="$(docker image inspect --format \
    '{{with .Config.Labels}}{{index . "org.opencontainers.image.revision"}}{{end}}' \
    "${IMAGE}")"
[[ -n "${IMAGE_AITER_REV}" && "${IMAGE_AITER_REV}" != unknown ]] || {
    err "image has no concrete Aiter revision OCI label"
    exit 1
}
[[ -n "${IMAGE_LIGHTX2V_REV}" && "${IMAGE_LIGHTX2V_REV}" != unknown ]] || {
    err "image has no concrete LightX2V revision OCI label"
    exit 1
}
if [[ -n "${EXPECTED_AITER_REV}" && "${IMAGE_AITER_REV}" != "${EXPECTED_AITER_REV}" ]]; then
    err "image Aiter revision mismatch: expected ${EXPECTED_AITER_REV}, got ${IMAGE_AITER_REV}"
    exit 1
fi
if [[ -n "${EXPECTED_LIGHTX2V_REV}" && "${IMAGE_LIGHTX2V_REV}" != "${EXPECTED_LIGHTX2V_REV}" ]]; then
    err "image LightX2V revision mismatch: expected ${EXPECTED_LIGHTX2V_REV}, got ${IMAGE_LIGHTX2V_REV}"
    exit 1
fi
[[ -c /dev/kfd ]] || { err "/dev/kfd is unavailable"; exit 1; }
[[ -d /dev/dri ]] || { err "/dev/dri is unavailable"; exit 1; }
[[ -f "${VAE_PATH}" ]] || { err "Wan2.1 VAE checkpoint not found: ${VAE_PATH}"; exit 1; }

SYNC_DIR="$(mktemp -d /tmp/ck-conv3d-v1-sync.XXXXXX)"
REFERENCE_DIR="$(mktemp -d /tmp/ck-conv3d-v1-reference.XXXXXX)"
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
    git -C "${RUN_REPO_ROOT}" rev-parse HEAD
    git -C "${RUN_REPO_ROOT}" status --short --branch
    docker image inspect --format '{{.Id}} {{.RepoDigests}}' "${IMAGE}"
    docker image inspect --format 'LightX2V={{index .Config.Labels "org.opencontainers.image.revision"}} Aiter={{index .Config.Labels "org.opencontainers.image.aiter.revision"}}' "${IMAGE}"
    echo "GPU_IDS=${GPU_IDS}"
    echo "ROUNDS=${ROUNDS}"
    echo "ROTATION=${ROTATION}"
    echo "SEED=${SEED}"
    echo "IMAGE_AITER_REV=${IMAGE_AITER_REV}"
    echo "IMAGE_LIGHTX2V_REV=${IMAGE_LIGHTX2V_REV}"
    echo "LIGHTX2V_SOURCE_MODE=image"
    echo "RUNTUNE_RUNNER_SOURCE=${RUNNER}"
    echo "AITER_CK_GITLINK_REV=${AITER_CK_REV}"
    echo "NETWORK=host"
    echo "AUTOMATIC_FREQUENCY=1"
    echo "VAE_ARMS=pytorch_offload:backend=pytorch,tiling=1,cpu_offload=1;ck_v1_resident:backend=Aiter_CK_Conv3d_v1,tiling=0,cpu_offload=0"
    echo "FROZEN=MIOpen:0,channels_last_3d:0,BF16,720p,81f"
} >"${RESULT_DIR}/host_environment.txt" 2>&1

docker_common=(
    --network=host
    --ipc=host
    --cap-add=SYS_PTRACE
    --security-opt=seccomp=unconfined
    --device=/dev/kfd
    --device=/dev/dri
    --ulimit=memlock=-1:-1
    --ulimit=stack=67108864:67108864
    --workdir="${CONTAINER_REPO_ROOT}"
    --env PLATFORM=amd_rocm
    --env DTYPE=BF16
    --env LIGHTX2V_AMD_MIOPEN=0
    --env USE_CHANNELS_LAST_3D=0
    --env GPU_ARCHS=gfx1201
    --env CU_NUM=24
    --env "AITER_COMMIT=${IMAGE_AITER_REV}"
    --env PYTHONUNBUFFERED=1
)
if [[ -n "${PROXY_URL}" ]]; then
    docker_common+=(
        --env "HTTP_PROXY=${PROXY_URL}"
        --env "HTTPS_PROXY=${PROXY_URL}"
        --env "ALL_PROXY=${PROXY_URL}"
        --env "http_proxy=${PROXY_URL}"
        --env "https_proxy=${PROXY_URL}"
        --env "all_proxy=${PROXY_URL}"
        --env "NO_PROXY=127.0.0.1,localhost"
        --env "no_proxy=127.0.0.1,localhost"
    )
fi

info "Checking image-built CK Conv3d v1 extension (runtime compilation is disabled)"
set +e
docker run --rm \
    "${docker_common[@]}" \
    --env "EXPECTED_CK_REV=${AITER_CK_REV}" \
    --entrypoint=python3 \
    "${IMAGE}" \
    -c 'import importlib, os; import aiter; assert callable(aiter.ck_conv3d_v1); assert aiter.is_ck_conv3d_v1_prebuilt(); assert aiter.CK_CONV3D_V1_CK_REVISION == os.environ["EXPECTED_CK_REV"]; module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(module.__file__)' \
    2>&1 | tee "${RESULT_DIR}/preflight.log"
preflight_status=("${PIPESTATUS[@]}")
set -e
PREFLIGHT_FAILED=0
if [[ "${preflight_status[0]}" != 0 || "${preflight_status[1]}" != 0 ]]; then
    err "prebuilt CK v1 module check failed; rebuild the image before runtune"
    PREFLIGHT_FAILED=1
fi

run_gpu() {
    local gpu=$1
    local arm_index=$2
    local arm=${ARM_NAMES[${arm_index}]}
    local gpu_result="${RESULT_DIR}/gpu${gpu}"
    local container_tmp="${gpu_result}/container_tmp"
    mkdir -p "${gpu_result}" "${container_tmp}"
    chmod 1777 "${container_tmp}"

    docker run --rm \
        "${docker_common[@]}" \
        --env "HIP_VISIBLE_DEVICES=${gpu}" \
        --env "HOST_GPU_ID=${gpu}" \
        --env "LIGHTX2V_VAE_CONV3D_BACKEND=${ARM_BACKENDS[${arm_index}]}" \
        --env TMPDIR=/tmp \
        --mount "type=bind,source=${RUNNER},target=${CONTAINER_RUNNER},readonly" \
        --mount "type=bind,source=${VAE_PATH},target=/models/Wan2.1_VAE.pth,readonly" \
        --mount "type=bind,source=${gpu_result},target=/task" \
        --mount "type=bind,source=${container_tmp},target=/tmp" \
        --mount "type=bind,source=${SYNC_DIR},target=/runtune-sync" \
        --mount "type=bind,source=${REFERENCE_DIR},target=/runtune-reference" \
        --entrypoint=python3 \
        "${IMAGE}" \
        -u "${CONTAINER_RUNNER}" \
        --output-dir /task \
        --reference-dir /runtune-reference \
        --sync-dir /runtune-sync \
        --participants 2 \
        --arm "${arm}" \
        --gpu-id "${gpu}" \
        --vae-path /models/Wan2.1_VAE.pth \
        --rounds "${ROUNDS}" \
        --seed "${SEED}" \
        --timeout-seconds "${TIMEOUT_SECONDS}"
}

run_and_report() {
    local gpu=$1
    local arm_index=$2
    local arm=${ARM_NAMES[${arm_index}]}
    local gpu_result="${RESULT_DIR}/gpu${gpu}"
    local result_file="${gpu_result}/result.json"
    local status_file="${gpu_result}/pipeline_status.txt"
    mkdir -p "${gpu_result}"

    set +e
    run_gpu "${gpu}" "${arm_index}" 2>&1 \
        | tee "${gpu_result}/run.log" \
        | awk -v prefix="[gpu${gpu}:${arm}] " '{ print prefix $0; fflush() }'
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    local docker_status=${pipeline_status[0]}
    local tee_status=${pipeline_status[1]}
    local awk_status=${pipeline_status[2]}
    printf 'docker=%s\ntee=%s\nawk=%s\n' \
        "${docker_status}" "${tee_status}" "${awk_status}" >"${status_file}"

    if [[ "${docker_status}" == 0 && "${tee_status}" == 0 && "${awk_status}" == 0 ]]; then
        return 0
    fi
    if [[ -s "${result_file}" ]] && python3 - "${result_file}" "${arm}" "${ROUNDS}" <<'PY'
import json
import sys

result_path, expected_arm, expected_rounds_text = sys.argv[1:]
with open(result_path, encoding="utf-8") as result_file:
    result = json.load(result_file)
if result.get("experiment") != "wan21_vae_ck_conv3d_v1":
    raise SystemExit("unexpected experiment")
if result.get("arm") != expected_arm:
    raise SystemExit("unexpected arm")
if len(result.get("rounds", [])) != int(expected_rounds_text):
    raise SystemExit("incomplete rounds")
PY
    then
        printf 'accepted_result=1\n' >>"${status_file}"
        warn "GPU ${gpu} ${arm} returned a complete result with a failed gate; deferring the decision to aggregate"
        return 0
    fi

    printf 'accepted_result=0\n' >>"${status_file}"
    touch "${SYNC_DIR}/abort"
    return 1
}

FAILED=${PREFLIGHT_FAILED}
if [[ "${PREFLIGHT_FAILED}" == 0 ]]; then
    PIDS=()
    ARM_INDEXES=()
    for ordinal in "${!GPU_LIST[@]}"; do
        gpu=${GPU_LIST[${ordinal}]}
        arm_index=$(((ordinal + ROTATION) % 2))
        run_and_report "${gpu}" "${arm_index}" &
        PIDS+=("$!")
        ARM_INDEXES+=("${arm_index}")
    done

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
fi

if [[ "${FAILED}" == 0 ]]; then
    set +e
    python3 -B "${RUNNER}" --aggregate-root "${RESULT_DIR}" \
        2>&1 | tee "${RESULT_DIR}/aggregate.log"
    aggregate_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${aggregate_status[0]}" != 0 || "${aggregate_status[1]}" != 0 ]]; then
        FAILED=1
    fi
else
    warn "Skipping aggregate because one or both VAE arms did not finish"
fi

ARCHIVE="${RESULT_DIR}.tar.gz"
info "Closing launcher.log before packaging results"
exec 1>&3 2>&4
exec 3>&- 4>&-
LOGGER_STATUS=0
wait "${LOGGER_PID}" || LOGGER_STATUS=$?

ARCHIVE_OK=1
if ! tar -C "$(dirname "${RESULT_DIR}")" -czf "${ARCHIVE}" "$(basename "${RESULT_DIR}")"; then
    ARCHIVE_OK=0
    FAILED=1
fi
{
    if [[ "${FAILED}" == 0 ]]; then
        ok "Summary: ${RESULT_DIR}/summary.json"
    else
        err "Runtune did not pass; inspect evidence in ${RESULT_DIR}"
    fi
    if [[ "${LOGGER_STATUS}" != 0 ]]; then
        warn "launcher.log tee exited with status ${LOGGER_STATUS}; per-GPU logs remain authoritative"
    fi
    if [[ "${ARCHIVE_OK}" == 1 ]]; then
        ok "Archive: ${ARCHIVE}"
    else
        warn "Archive creation failed; uncompressed results remain at ${RESULT_DIR}"
    fi
    warn "The two VAE arms use different cards. Re-run with ROTATION=1 to check card bias."
} 2>&1 | tee -a "${RESULT_DIR}/launcher.log"
exit "${FAILED}"
