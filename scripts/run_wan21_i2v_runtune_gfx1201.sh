#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_repo_dir="$(cd -- "${script_dir}/.." && pwd)"
modelrun="${run_repo_dir}/lightx2vModelrun"

[[ -x "${modelrun}" ]] || {
    echo "ERROR: lightx2vModelrun is not executable: ${modelrun}" >&2
    exit 1
}
[[ "${GPU_DEVICES:-0}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Wan2.1 runtune requires exactly one GPU_DEVICES id" >&2
    exit 1
}

runtime_root="${RUNTIME_ROOT:-/disk/ssd/lightx2v-runtime}"
output_dir="${OUTPUT_DIR:-${runtime_root}/outputs}"
runtune_name="${RUNTUNE_NAME:-wan21_runtune_$(date +%Y%m%d_%H%M%S)}"
[[ "${runtune_name}" != */* && "${runtune_name}" != "." && "${runtune_name}" != ".." ]] || {
    echo "ERROR: RUNTUNE_NAME must be one directory name, got '${runtune_name}'" >&2
    exit 1
}
host_results="${output_dir}/${runtune_name}"
container_results="/workspace/LightX2V/save_results/${runtune_name}"
mkdir -p "${host_results}"

backend_list="${RUNTUNE_BACKENDS:-flydsl,sage}"
IFS=',' read -r -a backends <<<"${backend_list}"
if [[ "${#backends[@]}" -ne 2 ]]; then
    echo "ERROR: RUNTUNE_BACKENDS must contain flydsl,sage in either order" >&2
    exit 1
fi
declare -A seen=()
for backend in "${backends[@]}"; do
    case "${backend}" in
        flydsl | sage) ;;
        *)
            echo "ERROR: unsupported runtune backend '${backend}'" >&2
            exit 1
            ;;
    esac
    [[ -z "${seen[${backend}]:-}" ]] || {
        echo "ERROR: duplicate runtune backend ${backend}" >&2
        exit 1
    }
    seen["${backend}"]=1
done
[[ -n "${seen[flydsl]:-}" && -n "${seen[sage]:-}" ]] || {
    echo "ERROR: RUNTUNE_BACKENDS must include both flydsl and sage" >&2
    exit 1
}

warmup="${LIGHTX2V_WARMUP:-0}"
rope_chunk_size="${ROPE_CHUNK_SIZE:-8192}"
offload_ratio="${OFFLOAD_RATIO:-1.0}"

echo "Wan2.1 RunTune"
echo "Results : ${host_results}"
echo "Backends: ${backends[*]}"
echo "Workload: 720p, 81 frames, 4 steps, H40/S74592"
echo "Common  : warmup=${warmup}, rope_chunk=${rope_chunk_size}, offload_ratio=${offload_ratio}, seed=${SEED:-1234}"

run_backend() {
    local backend=$1
    local mode
    case "${backend}" in
        flydsl) mode=--benchmark-wan21-4090 ;;
        sage) mode=--benchmark-wan21-4090-sage ;;
    esac

    echo "[$(date +%H:%M:%S)] START ${backend}"
    GPU_DEVICES="${GPU_DEVICES:-0}" \
    OUTPUT_DIR="${output_dir}" \
    CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-gfx1201}-runtune-${backend}" \
    LIGHTX2V_WARMUP="${warmup}" \
    LIGHTX2V_DIT_PROGRESS="${LIGHTX2V_DIT_PROGRESS:-1}" \
    ROPE_CHUNK_SIZE="${rope_chunk_size}" \
    OFFLOAD_RATIO="${offload_ratio}" \
    LIGHTX2V_WAN_RUNTUNE=1 \
    LIGHTX2V_WAN_RUNTUNE_BACKEND="${backend}" \
    LIGHTX2V_WAN_RUNTUNE_OUTPUT="${container_results}/${backend}.json" \
    SAVE_RESULT_PATH="${container_results}/${backend}.mp4" \
    "${modelrun}" "${mode}" 2>&1 | tee "${host_results}/${backend}.log"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[$(date +%H:%M:%S)] DRY   ${backend}"
        return
    fi
    [[ -s "${host_results}/${backend}.json" ]] || {
        echo "ERROR: ${backend} runtune report was not created" >&2
        exit 1
    }
    echo "[$(date +%H:%M:%S)] DONE  ${backend}"
}

for backend in "${backends[@]}"; do
    run_backend "${backend}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "RunTune dry-run complete: both backend commands validated"
    exit 0
fi

python3 "${script_dir}/summarize_wan21_runtune.py" \
    --flydsl "${host_results}/flydsl.json" \
    --sage "${host_results}/sage.json" \
    --output-dir "${host_results}"

echo "RunTune complete: ${host_results}"
