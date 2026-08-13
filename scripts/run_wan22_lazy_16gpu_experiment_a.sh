#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
memdiag_launcher="${script_dir}/run_wan22_ckconv3d_v1_720p_5s_lazy_16gpu_memdiag.sh"
summarizer="${script_dir}/summarize_wan22_lazy_experiment_a.py"

[[ -n "${IMAGE:-}" ]] || {
    echo "ERROR: IMAGE must name an image built from the Experiment-A LightX2V instrumentation" >&2
    exit 2
}
cache_state="${WAN22_CACHE_STATE:-}"
case "${cache_state}" in
    cold | warm) ;;
    *)
        echo "ERROR: WAN22_CACHE_STATE must explicitly be cold or warm (the launcher labels but does not alter page cache)" >&2
        exit 2
        ;;
esac
[[ -x "${memdiag_launcher}" ]] || { echo "ERROR: missing launcher: ${memdiag_launcher}" >&2; exit 1; }
[[ -f "${summarizer}" ]] || { echo "ERROR: missing summarizer: ${summarizer}" >&2; exit 1; }

export SWEEP_TAG="${SWEEP_TAG:-wan22_lazy16_experiment_a_$(date +%Y%m%d_%H%M%S)}"
host_output_root="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
host_result_dir="${SWEEP_RESULT_DIR:-${host_output_root}/${SWEEP_TAG}}"
container_result_dir="/workspace/LightX2V/save_results/${SWEEP_TAG}"
if [[ -n "${SWEEP_RESULT_DIR:-}" && "${SWEEP_RESULT_DIR}" != "${host_output_root}/${SWEEP_TAG}" ]]; then
    echo "ERROR: Experiment A requires SWEEP_RESULT_DIR=${host_output_root}/${SWEEP_TAG} so per-rank reports use the mounted output tree" >&2
    exit 2
fi

export LIGHTX2V_WAN_RUNTUNE=1
export LIGHTX2V_WAN_RUNTUNE_BACKEND="wan22-lazy-full-offload-baseline-${cache_state}-cache"
export LIGHTX2V_WAN_RUNTUNE_OUTPUT="${container_result_dir}/runtune/rank{rank:03d}.json"

echo "Experiment A instrumentation: enabled (schema 2, one report per rank)"
echo "Experiment A cache state: ${cache_state} (label only; page cache is not modified)"
echo "Experiment A result directory: ${host_result_dir}"
echo "Experiment A rank reports: ${host_result_dir}/runtune/rankNNN.json"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed" >&2; exit 1; }
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
        echo "ERROR: Experiment-A image is unavailable locally: ${IMAGE}" >&2
        exit 1
    }
    docker run --rm --network=none --entrypoint /bin/bash "${IMAGE}" -c \
        'grep -q "^WAN_RUNTUNE_SCHEMA_VERSION = 2$" /workspace/LightX2V/lightx2v/utils/wan_runtune.py' || {
        echo "ERROR: image does not contain the Experiment-A schema-2 instrumentation: ${IMAGE}" >&2
        exit 1
    }
fi

bash "${memdiag_launcher}" "$@"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Experiment A dry-run complete; no per-rank reports were expected"
    exit 0
fi

python3 "${summarizer}" \
    --input-dir "${host_result_dir}/runtune" \
    --log "${host_result_dir}/720p121f_16gpu.log" \
    --memory-summary "${host_result_dir}/720p121f_16gpu_memory.jsonl.summary.json" \
    --output "${host_result_dir}/experiment_a_summary.json" \
    --expected-ranks 16
