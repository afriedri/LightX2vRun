#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
memdiag_launcher="${script_dir}/run_wan22_ckconv3d_v1_720p_5s_lazy_16gpu_memdiag.sh"
summarizer="${script_dir}/summarize_wan22_lazy_experiment_a.py"
uploader="${script_dir}/upload_wan22_memory_bundle.py"

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
offload_schedule="${LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE:-baseline}"
case "${offload_schedule}" in
    baseline | b0 | b1) ;;
    *)
        echo "ERROR: LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE must be baseline, b0, or b1" >&2
        exit 2
        ;;
esac
gpu_count="${WAN22_GPU_COUNT:-16}"
case "${gpu_count}" in
    2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: WAN22_GPU_COUNT must be 2, 4, 8, or 16, got ${gpu_count}" >&2
        exit 2
        ;;
esac
[[ -x "${memdiag_launcher}" ]] || { echo "ERROR: missing launcher: ${memdiag_launcher}" >&2; exit 1; }
[[ -f "${summarizer}" ]] || { echo "ERROR: missing summarizer: ${summarizer}" >&2; exit 1; }
upload_enabled="${WAN22_UPLOAD:-1}"
case "${upload_enabled}" in
    0 | 1) ;;
    *)
        echo "ERROR: WAN22_UPLOAD must be 0 or 1" >&2
        exit 2
        ;;
esac
upload_prefix="${WAN22_UPLOAD_PREFIX:-${S3_PREFIX:-wan22/experiment-a}}"
if [[ "${upload_enabled}" == "1" ]]; then
    [[ -f "${uploader}" ]] || { echo "ERROR: missing uploader: ${uploader}" >&2; exit 1; }
fi

export SWEEP_TAG="${SWEEP_TAG:-wan22_lazy${gpu_count}_experiment_a_$(date +%Y%m%d_%H%M%S)}"
export WAN22_GPU_COUNT="${gpu_count}"
host_output_root="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
host_result_dir="${SWEEP_RESULT_DIR:-${host_output_root}/${SWEEP_TAG}}"
container_result_dir="/workspace/LightX2V/save_results/${SWEEP_TAG}"
if [[ -n "${SWEEP_RESULT_DIR:-}" && "${SWEEP_RESULT_DIR}" != "${host_output_root}/${SWEEP_TAG}" ]]; then
    echo "ERROR: Experiment A requires SWEEP_RESULT_DIR=${host_output_root}/${SWEEP_TAG} so per-rank reports use the mounted output tree" >&2
    exit 2
fi

export LIGHTX2V_WAN_RUNTUNE=1
export LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE="${offload_schedule}"
export LIGHTX2V_WAN_RUNTUNE_BACKEND="wan22-lazy-full-offload-${offload_schedule}-${cache_state}-cache"
export LIGHTX2V_WAN_RUNTUNE_OUTPUT="${container_result_dir}/runtune/rank{rank:03d}.json"

echo "Experiment A instrumentation: enabled (schema 2, one report per rank)"
echo "Experiment A GPU count: ${gpu_count}"
echo "Wan block-offload schedule: ${offload_schedule}"
echo "Experiment A cache state: ${cache_state} (label only; page cache is not modified)"
echo "Experiment A result directory: ${host_result_dir}"
echo "Experiment A rank reports: ${host_result_dir}/runtune/rankNNN.json"
if [[ "${upload_enabled}" == "1" ]]; then
    echo "Experiment A upload: enabled (prefix=${upload_prefix})"
else
    echo "Experiment A upload: disabled (WAN22_UPLOAD=0)"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed" >&2; exit 1; }
    if [[ "${upload_enabled}" == "1" ]]; then
        python3 -c 'import boto3, botocore' >/dev/null 2>&1 || {
            echo "ERROR: automatic upload requires boto3; install it with: python3 -m pip install boto3" >&2
            exit 1
        }
    fi
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
        echo "ERROR: Experiment-A image is unavailable locally: ${IMAGE}" >&2
        exit 1
    }
    docker run --rm --network=none --entrypoint /bin/bash "${IMAGE}" -c \
        'grep -q "^WAN_RUNTUNE_SCHEMA_VERSION = 2$" /workspace/LightX2V/lightx2v/utils/wan_runtune.py &&
         grep -q '\''runtuner.begin_scope("request_end_to_end")'\'' /workspace/LightX2V/lightx2v/models/runners/wan/wan_runner.py' || {
        echo "ERROR: image does not contain the complete Experiment-A Wan request instrumentation: ${IMAGE}" >&2
        exit 1
    }
    if [[ "${offload_schedule}" != "baseline" ]]; then
        docker run --rm --network=none --entrypoint /bin/bash "${IMAGE}" -c \
            'grep -q '\''wan_block_offload_schedule'\'' /workspace/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py &&
             grep -q '\''def _finish_block_compute'\'' /workspace/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py &&
             grep -q '\''activated = activate()'\'' /workspace/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py &&
             grep -q '\''B1 computes block 0 outside the steady-state loop'\'' /workspace/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py &&
             grep -q '\''def activate_cuda_buffers'\'' /workspace/LightX2V/lightx2v/common/modules/weight_module.py' || {
            echo "ERROR: image does not contain the fixed Wan B0/B1 schedules with CUDA-buffer activation and B1 block-index initialization: ${IMAGE}" >&2
            exit 1
        }
    fi
fi

set +e
bash "${memdiag_launcher}" "$@"
run_status=$?
set -e

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Experiment A dry-run complete; no per-rank reports were expected"
    exit 0
fi

summarizer_log="${host_result_dir}/experiment_a_summarizer.log"
set +e
python3 "${summarizer}" \
    --input-dir "${host_result_dir}/runtune" \
    --log "${host_result_dir}/720p121f_${gpu_count}gpu.log" \
    --memory-summary "${host_result_dir}/720p121f_${gpu_count}gpu_memory.jsonl.summary.json" \
    --output "${host_result_dir}/experiment_a_summary.json" \
    --expected-ranks "${gpu_count}" 2>&1 | tee "${summarizer_log}"
summarizer_status=${PIPESTATUS[0]}
set -e

upload_status=0
if [[ "${upload_enabled}" == "1" ]]; then
    python3 "${uploader}" \
        "${host_result_dir}" \
        --experiment-a \
        --expected-ranks "${gpu_count}" \
        --prefix "${upload_prefix}" || upload_status=$?
fi

if (( summarizer_status != 0 )); then
    echo "ERROR: Experiment-A summarizer failed with status ${summarizer_status}; diagnostic bundle upload was still attempted" >&2
fi
if (( upload_status != 0 )); then
    echo "ERROR: Experiment-A automatic upload failed with status ${upload_status}" >&2
fi
if (( run_status != 0 )); then
    echo "ERROR: Experiment-A workload failed with status ${run_status}; summarization and diagnostic bundle upload were still attempted" >&2
    exit "${run_status}"
fi
if (( summarizer_status != 0 )); then
    exit "${summarizer_status}"
fi
if (( upload_status != 0 )); then
    exit "${upload_status}"
fi
