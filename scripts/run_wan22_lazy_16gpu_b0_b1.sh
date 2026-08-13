#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  WAN22_CACHE_STATE=warm IMAGE=<b0-b1-image> \
    bash scripts/run_wan22_lazy_16gpu_b0_b1.sh

Runs the fixed Wan2.2 720p/121f workload twice on the same 16 GPUs:
  1. B0: constructor-buffer reuse plus terminal-wrap elimination
  2. B1: the same transfer count with three-stage host-fill/H2D/compute overlap

Both arms retain Experiment-A rank instrumentation and memory monitoring. Each
arm is summarized and uploaded for failure recovery; after both complete, the
script validates transfer counts and output identity, writes a B0/B1 comparison,
then creates and uploads one combined deterministic archive.

Overrides:
  B0_B1_TAG                 Paired result tag
  OUTPUT_DIR                Output root (default: /disk/ssd/lightx2v-runtime/outputs)
  WAN22_UPLOAD=0            Disable arm and combined uploads
  WAN22_B0_B1_UPLOAD_PREFIX Combined S3 prefix (default: wan22/experiment-b0-b1)
  WAN22_CACHE_STATE         Required: cold or warm; label only
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
comparator="${script_dir}/compare_wan22_lazy_b0_b1.py"
uploader="${script_dir}/upload_wan22_memory_bundle.py"

[[ -n "${IMAGE:-}" ]] || {
    echo "ERROR: IMAGE must name an image containing the Wan B0/B1 schedules" >&2
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
    *)
        echo "ERROR: WAN22_UPLOAD must be 0 or 1" >&2
        exit 2
        ;;
esac
for required in "${arm_launcher}" "${comparator}" "${uploader}"; do
    [[ -f "${required}" ]] || { echo "ERROR: missing required script: ${required}" >&2; exit 1; }
done

pair_tag="${B0_B1_TAG:-wan22_lazy16_b0_b1_$(date +%Y%m%d_%H%M%S)}"
[[ "${pair_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: B0_B1_TAG may contain only letters, digits, dot, underscore, and dash" >&2
    exit 2
}
host_output_root="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}"
pair_dir="${host_output_root}/${pair_tag}"
if [[ -d "${pair_dir}" && -n "$(find "${pair_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: paired result directory is not empty: ${pair_dir}" >&2
    exit 1
fi
mkdir -p "${pair_dir}"

combined_prefix="${WAN22_B0_B1_UPLOAD_PREFIX:-wan22/experiment-b0-b1}"
arm_prefix="${combined_prefix%/}/${pair_tag}/arms"

echo "Wan2.2 lazy B0/B1 paired experiment"
echo "Image: ${IMAGE}"
echo "Cache state: ${WAN22_CACHE_STATE} (label only; page cache is not modified)"
echo "Result directory: ${pair_dir}"
echo "Run order: B0, then B1"
if [[ "${upload_enabled}" == "1" ]]; then
    echo "Upload: arm recovery bundles + combined bundle (${combined_prefix})"
else
    echo "Upload: disabled (WAN22_UPLOAD=0)"
fi

run_arm() {
    local arm="$1"
    echo "[$(date +%H:%M:%S)] Starting ${arm^^} 16-GPU arm"
    env \
        OUTPUT_DIR="${pair_dir}" \
        SWEEP_TAG="${arm}" \
        LIGHTX2V_WAN_BLOCK_OFFLOAD_SCHEDULE="${arm}" \
        WAN22_UPLOAD="${upload_enabled}" \
        WAN22_UPLOAD_PREFIX="${arm_prefix}" \
        bash "${arm_launcher}"
    echo "[$(date +%H:%M:%S)] Completed ${arm^^} 16-GPU arm"
}

run_arm b0
run_arm b1

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "B0/B1 dry-run complete; comparison and upload were skipped"
    exit 0
fi

b0_dir="${pair_dir}/b0"
b1_dir="${pair_dir}/b1"
b0_video="${pair_dir}/b0_720p121f_16gpu.mp4"
b1_video="${pair_dir}/b1_720p121f_16gpu.mp4"
comparison_json="${pair_dir}/b0_b1_comparison.json"
comparison_log="${pair_dir}/b0_b1_comparison.log"

set +e
python3 "${comparator}" \
    --b0-dir "${b0_dir}" \
    --b1-dir "${b1_dir}" \
    --b0-video "${b0_video}" \
    --b1-video "${b1_video}" \
    --output "${comparison_json}" \
    --expected-ranks 16 2>&1 | tee "${comparison_log}"
comparison_status=${PIPESTATUS[0]}
set -e
if (( comparison_status != 0 )); then
    echo "ERROR: B0/B1 comparison failed; individual arm bundles were already uploaded" >&2
    exit "${comparison_status}"
fi

if [[ "${upload_enabled}" == "1" ]]; then
    python3 "${uploader}" \
        "${pair_dir}" \
        --b0-b1 \
        --expected-ranks 16 \
        --prefix "${combined_prefix}"
fi

echo "Wan2.2 lazy B0/B1 paired experiment complete: ${pair_tag}"
echo "B0: ${b0_dir}"
echo "B1: ${b1_dir}"
echo "Comparison: ${comparison_json}"
