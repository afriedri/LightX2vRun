#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  WAN22_CACHE_STATE=warm IMAGE=<B0B1-Fix1-or-newer-image> \
    bash scripts/run_wan22_lazy_b1_scale.sh

Runs the existing Wan2.2 lazy B1 schedule sequentially at 2, 4, and 8 GPUs
for the same 720p/121-frame workload. Every arm retains rank instrumentation,
host/physical-GPU memory monitoring, deterministic packaging, and automatic
upload. The final acceptance report treats physical VRAM and allocator-reserved
VRAM as hard gates.

Overrides:
  B1_SCALE_TAG                       Result tag
  OUTPUT_DIR                         Output root
  WAN22_B1_COUNTS                    Unique subset of 2,4,8 (default: 2,4,8)
  WAN22_B1_GPU_GROUPS                Groups for 2;4;8 GPUs
  WAN22_B1_GPU_BUDGET_GIB            Per-card budget (default: 32)
  WAN22_B1_MIN_GPU_HEADROOM_GIB      Required per-card headroom (default: 1)
  WAN22_UPLOAD=0                     Disable arm and combined uploads
  WAN22_B1_SCALE_UPLOAD_PREFIX       Combined prefix
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    echo "ERROR: unsupported arguments: $*" >&2
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
arm_launcher="${script_dir}/run_wan22_lazy_16gpu_experiment_a.sh"
summarizer="${script_dir}/summarize_wan22_lazy_b1_scale.py"
uploader="${script_dir}/upload_wan22_memory_bundle.py"
for required in "${arm_launcher}" "${summarizer}" "${uploader}"; do
    [[ -f "${required}" ]] || { echo "ERROR: missing required script: ${required}" >&2; exit 1; }
done

[[ -n "${IMAGE:-}" ]] || {
    echo "ERROR: IMAGE must name an image containing the fixed B1 schedule" >&2
    exit 2
}
case "${WAN22_CACHE_STATE:-}" in
    cold | warm) ;;
    *)
        echo "ERROR: WAN22_CACHE_STATE must explicitly be cold or warm" >&2
        exit 2
        ;;
esac
upload_enabled="${WAN22_UPLOAD:-1}"
case "${upload_enabled}" in
    0 | 1) ;;
    *) echo "ERROR: WAN22_UPLOAD must be 0 or 1" >&2; exit 2 ;;
esac

requested_counts="${WAN22_B1_COUNTS:-2,4,8}"
IFS=',' read -r -a counts <<<"${requested_counts}"
declare -A seen_counts=()
for count in "${counts[@]}"; do
    case "${count}" in
        2 | 4 | 8) ;;
        *) echo "ERROR: WAN22_B1_COUNTS may contain only 2, 4, and 8" >&2; exit 2 ;;
    esac
    [[ -z "${seen_counts[${count}]:-}" ]] || {
        echo "ERROR: WAN22_B1_COUNTS contains duplicate count ${count}" >&2
        exit 2
    }
    seen_counts["${count}"]=1
done
(( ${#counts[@]} > 0 )) || { echo "ERROR: WAN22_B1_COUNTS must not be empty" >&2; exit 2; }

gpu_groups_value="${WAN22_B1_GPU_GROUPS:-0,1;0,1,2,3;0,1,2,3,4,5,6,7}"
IFS=';' read -r -a gpu_groups <<<"${gpu_groups_value}"
[[ "${#gpu_groups[@]}" -eq 3 ]] || {
    echo "ERROR: WAN22_B1_GPU_GROUPS must contain groups for 2;4;8 GPUs" >&2
    exit 2
}
declare -A gpu_group_by_count=(
    [2]="${gpu_groups[0]}"
    [4]="${gpu_groups[1]}"
    [8]="${gpu_groups[2]}"
)
for expected_count in 2 4 8; do
    group="${gpu_group_by_count[${expected_count}]}"
    [[ "${group}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        echo "ERROR: invalid ${expected_count}-GPU group: ${group}" >&2
        exit 2
    }
    IFS=',' read -r -a group_gpus <<<"${group}"
    [[ "${#group_gpus[@]}" -eq "${expected_count}" ]] || {
        echo "ERROR: ${expected_count}-GPU group contains ${#group_gpus[@]} ids: ${group}" >&2
        exit 2
    }
    declare -A seen_group_gpus=()
    for gpu in "${group_gpus[@]}"; do
        [[ -z "${seen_group_gpus[${gpu}]:-}" ]] || {
            echo "ERROR: ${expected_count}-GPU group contains duplicate id ${gpu}" >&2
            exit 2
        }
        seen_group_gpus["${gpu}"]=1
    done
    unset seen_group_gpus
done

gpu_budget_gib="${WAN22_B1_GPU_BUDGET_GIB:-32}"
minimum_headroom_gib="${WAN22_B1_MIN_GPU_HEADROOM_GIB:-1}"
[[ "${gpu_budget_gib}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "ERROR: WAN22_B1_GPU_BUDGET_GIB must be a non-negative number" >&2
    exit 2
}
[[ "${minimum_headroom_gib}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    echo "ERROR: WAN22_B1_MIN_GPU_HEADROOM_GIB must be a non-negative number" >&2
    exit 2
}

scale_tag="${B1_SCALE_TAG:-wan22_lazy_b1_scale_$(date +%Y%m%d_%H%M%S)}"
[[ "${scale_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: B1_SCALE_TAG may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
}
host_output_root="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
scale_dir="${host_output_root}/${scale_tag}"
if [[ -d "${scale_dir}" && -n "$(find "${scale_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: B1 scale result directory is not empty: ${scale_dir}" >&2
    exit 1
fi
mkdir -p "${scale_dir}"

combined_prefix="${WAN22_B1_SCALE_UPLOAD_PREFIX:-wan22/experiment-b1-scale}"
arm_prefix="${combined_prefix%/}/${scale_tag}/arms"
counts_csv="$(IFS=,; echo "${counts[*]}")"

echo "Wan2.2 lazy B1 GPU-count validation"
echo "Image: ${IMAGE}"
echo "Counts: ${counts_csv} (sequential)"
echo "Cache state: ${WAN22_CACHE_STATE} (label only; page cache is not modified)"
echo "GPU memory gate: budget=${gpu_budget_gib} GiB/card, minimum headroom=${minimum_headroom_gib} GiB/card"
echo "Result directory: ${scale_dir}"
if [[ "${upload_enabled}" == "1" ]]; then
    echo "Upload: per-count recovery bundles + combined scale bundle (${combined_prefix})"
else
    echo "Upload: disabled (WAN22_UPLOAD=0)"
fi

failed_count=""
run_status=0
for count in "${counts[@]}"; do
    selected="${gpu_group_by_count[${count}]}"
    echo "[$(date +%H:%M:%S)] Starting B1 ${count}-GPU arm (${selected})"
    set +e
    env \
        OUTPUT_DIR="${scale_dir}" \
        SWEEP_TAG="${count}gpu" \
        GPU_DEVICES="${selected}" \
        WAN22_GPU_COUNT="${count}" \
        LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE=b1 \
        MEMORY_MONITOR_GPU_LIMIT_GIB="${gpu_budget_gib}" \
        WAN22_UPLOAD="${upload_enabled}" \
        WAN22_UPLOAD_PREFIX="${arm_prefix}" \
        bash "${arm_launcher}"
    run_status=$?
    set -e
    if (( run_status != 0 )); then
        failed_count="${count}"
        break
    fi
    echo "[$(date +%H:%M:%S)] Completed B1 ${count}-GPU arm"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "B1 scale dry-run complete; summary and upload were skipped"
    exit 0
fi

if (( run_status != 0 )); then
    failed_dir="${scale_dir}/${failed_count}gpu"
    if [[ "${upload_enabled}" == "1" && -d "${failed_dir}" ]]; then
        echo "B1 ${failed_count}-GPU arm failed; attempting a memory/log-only recovery upload" >&2
        python3 "${uploader}" \
            "${failed_dir}" \
            --prefix "${combined_prefix%/}/${scale_tag}/failed-diagnostics" || true
    fi
    echo "ERROR: B1 ${failed_count}-GPU arm failed with status ${run_status}; later counts were not run" >&2
    exit "${run_status}"
fi

summary_json="${scale_dir}/b1_scale_summary.json"
summary_log="${scale_dir}/b1_scale_summary.log"
set +e
python3 "${summarizer}" \
    --result-dir "${scale_dir}" \
    --output "${summary_json}" \
    --counts "${counts_csv}" \
    --gpu-budget-gib "${gpu_budget_gib}" \
    --minimum-gpu-headroom-gib "${minimum_headroom_gib}" 2>&1 | tee "${summary_log}"
summary_status=${PIPESTATUS[0]}
set -e

upload_status=0
if [[ "${upload_enabled}" == "1" && -f "${summary_json}" ]]; then
    python3 "${uploader}" \
        "${scale_dir}" \
        --b1-scale \
        --counts "${counts_csv}" \
        --prefix "${combined_prefix}" || upload_status=$?
fi

if (( summary_status != 0 )); then
    echo "ERROR: B1 scale acceptance failed with status ${summary_status}; combined upload was still attempted" >&2
    exit "${summary_status}"
fi
if (( upload_status != 0 )); then
    echo "ERROR: B1 scale automatic upload failed with status ${upload_status}" >&2
    exit "${upload_status}"
fi

echo "Wan2.2 lazy B1 GPU-count validation complete: ${scale_tag}"
echo "Summary: ${summary_json}"
