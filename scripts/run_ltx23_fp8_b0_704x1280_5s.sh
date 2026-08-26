#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/.." && pwd)"
model_run="${MODEL_RUN:-${run_root}/lightx2vModelrun}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_ltx23_fp8_b0_704x1280_5s.sh --image IMAGE [OPTIONS]

Runs the LTX-2.3 22B Distilled FP8 B0 on exactly two gfx1201 GPUs. The final
video is 704x1280; the official two-stage path denoises 352x640 for eight steps,
upsamples by two, then refines 704x1280 for three steps. The workload is fixed
at 24 FPS / 121 frames. Collector is on by default and writes rank-local
schema-2 timing plus host/memory evidence.

Options:
  --image IMAGE        Runtime image containing the current LightX2V and Aiter
  --gpus IDS           Exactly two unique physical GPU ids (default: 0,1)
  --checkpoint FILE    Host LTX-2.3 Distilled FP8 monolith
  --upscaler FILE      Host LTX-2.3 spatial upscaler x2 checkpoint
  --gemma-dir DIR      Host Gemma 3 12B QAT unquantized asset directory
  --result-dir DIR     Exact empty output directory (default: timestamped)
  --prompt TEXT        T2AV prompt; PROMPT is also accepted
  --mock-memory-only   Header-only memory gate; needs no image or GPU
  --preflight-only     Run SHA/header/image/RCCL/exact-shape FP8 probes, then exit
  --collect            Enable collector (default)
  --no-collect         Disable timing/memory observers; safety preflight remains

Environment:
  DRY_RUN=1            Validate launcher arguments and print the Docker command
  MEMORY_MONITOR=0|1   Host memory/VRAM observer (defaults to collector setting)
  LTX23_QUICK_GEMM_PROBE=1
                       Use small matrices for an explicit diagnostic only
  LTX23_WEIGHT_PROBE_ONLY=1
                       Collector wrapper only: inspect weights in the B0 container
  LTX23_DYNAMIC_MEMORY_BUDGET_GIB=16
                       No-GPU gate reserve for activations and kernel workspace
  LTX23_HANDOFF_VRAM_LIMIT_MIB=1024
                       Require each card below this physical VRAM use before DiT
  LTX23_HANDOFF_TIMEOUT_SECONDS=60
                       Maximum wait for the process/VRAM handoff gate
EOF
}

image=
readonly fps=24
readonly frames=121
gpu_devices="${GPU_DEVICES:-0,1}"
checkpoint="${LTX23_CHECKPOINT:-/disk/ssd/difumodels-Dis/LTX-2.3-fp8/ltx-2.3-22b-distilled-fp8.safetensors}"
upscaler="${LTX23_UPSCALER:-/disk/ssd/difumodels-Dis/LTX-2.3-fp8/ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"
gemma_dir="${LTX23_GEMMA_DIR:-/disk/ssd/difumodels-Dis/LTX-2.3-fp8/gemma-3-12b-it-qat-q4_0-unquantized}"
result_dir="${LTX23_RESULT_DIR:-}"
prompt="${PROMPT:-A cinematic wide shot of a red fox walking through fresh snow at sunrise, with natural synchronized forest ambience.}"
preflight_only=0
collect="${LTX23_COLLECT:-1}"
quick_gemm_probe="${LTX23_QUICK_GEMM_PROBE:-0}"
weight_probe_only="${LTX23_WEIGHT_PROBE_ONLY:-0}"
weight_probe_sha256="${LTX23_WEIGHT_PROBE_SHA256:-0}"
mock_memory_only=0
dynamic_memory_budget_gib="${LTX23_DYNAMIC_MEMORY_BUDGET_GIB:-16}"
handoff_vram_limit_mib="${LTX23_HANDOFF_VRAM_LIMIT_MIB:-1024}"
handoff_timeout_seconds="${LTX23_HANDOFF_TIMEOUT_SECONDS:-60}"

while (( $# > 0 )); do
    case "$1" in
        --help | -h) usage; exit 0 ;;
        --image) (( $# >= 2 )) || { echo "ERROR: --image requires a value" >&2; exit 2; }; image="$2"; shift 2 ;;
        --image=*) image="${1#--image=}"; shift ;;
        --gpus | --gpu) (( $# >= 2 )) || { echo "ERROR: $1 requires a value" >&2; exit 2; }; gpu_devices="$2"; shift 2 ;;
        --checkpoint) (( $# >= 2 )) || { echo "ERROR: --checkpoint requires a value" >&2; exit 2; }; checkpoint="$2"; shift 2 ;;
        --upscaler) (( $# >= 2 )) || { echo "ERROR: --upscaler requires a value" >&2; exit 2; }; upscaler="$2"; shift 2 ;;
        --gemma-dir) (( $# >= 2 )) || { echo "ERROR: --gemma-dir requires a value" >&2; exit 2; }; gemma_dir="$2"; shift 2 ;;
        --result-dir) (( $# >= 2 )) || { echo "ERROR: --result-dir requires a value" >&2; exit 2; }; result_dir="$2"; shift 2 ;;
        --prompt) (( $# >= 2 )) || { echo "ERROR: --prompt requires a value" >&2; exit 2; }; prompt="$2"; shift 2 ;;
        --mock-memory-only) mock_memory_only=1; shift ;;
        --preflight-only) preflight_only=1; shift ;;
        --collect) collect=1; shift ;;
        --no-collect) collect=0; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "${mock_memory_only}" != 1 ]]; then
    [[ -n "${image}" ]] || { echo "ERROR: --image is required; B0 must use an image built from the current hvat-gfx1201 source" >&2; exit 2; }
fi
case "${collect}" in 0 | 1) ;; *) echo "ERROR: LTX23_COLLECT must be 0 or 1" >&2; exit 2 ;; esac
case "${quick_gemm_probe}" in 0 | 1) ;; *) echo "ERROR: LTX23_QUICK_GEMM_PROBE must be 0 or 1" >&2; exit 2 ;; esac
case "${weight_probe_only}" in 0 | 1) ;; *) echo "ERROR: LTX23_WEIGHT_PROBE_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "${weight_probe_sha256}" in 0 | 1) ;; *) echo "ERROR: LTX23_WEIGHT_PROBE_SHA256 must be 0 or 1" >&2; exit 2 ;; esac
[[ "${handoff_vram_limit_mib}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ERROR: LTX23_HANDOFF_VRAM_LIMIT_MIB must be numeric" >&2; exit 2; }
[[ "${handoff_timeout_seconds}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ERROR: LTX23_HANDOFF_TIMEOUT_SECONDS must be numeric" >&2; exit 2; }
if [[ "${quick_gemm_probe}" == 1 && "${preflight_only}" != 1 ]]; then
    echo "ERROR: LTX23_QUICK_GEMM_PROBE=1 is diagnostic-only and requires --preflight-only" >&2
    exit 2
fi
if [[ "${weight_probe_only}" == 1 && "${preflight_only}" == 1 ]]; then
    echo "ERROR: weight-probe-only and preflight-only are mutually exclusive" >&2
    exit 2
fi
if [[ "${mock_memory_only}" == 1 && ( "${weight_probe_only}" == 1 || "${preflight_only}" == 1 ) ]]; then
    echo "ERROR: mock-memory-only cannot be combined with another probe-only mode" >&2
    exit 2
fi
[[ "${gpu_devices}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "ERROR: --gpus requires exactly two comma-separated ids" >&2; exit 2; }
IFS=',' read -r gpu0 gpu1 <<<"${gpu_devices}"
[[ "${gpu0}" != "${gpu1}" ]] || { echo "ERROR: --gpus ids must be unique" >&2; exit 2; }

run_kind=ltx23_fp8_b0_704x1280
if [[ "${mock_memory_only}" == 1 ]]; then
    run_kind=ltx23_memory_mock
fi
if [[ "${weight_probe_only}" == 1 ]]; then
    run_kind=ltx23_weight_probe
fi
run_tag="${run_kind}_${fps}fps_$(date +%Y%m%d_%H%M%S)"
if [[ -z "${result_dir}" ]]; then
    result_dir="${OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/outputs}/${run_tag}"
fi
if [[ -d "${result_dir}" && -n "$(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: result directory is not empty: ${result_dir}" >&2
    exit 2
fi
mkdir -p "${result_dir}"

memory_mock="${run_root}/scripts/lib/mock_ltx23_b0_memory.py"
[[ -f "${memory_mock}" ]] || { echo "ERROR: memory mock not found: ${memory_mock}" >&2; exit 1; }
if [[ "${mock_memory_only}" == 1 ]]; then
    python3 "${memory_mock}" \
        --checkpoint "${checkpoint}" \
        --output "${result_dir}/mock_memory.json" \
        --dynamic-budget-gib "${dynamic_memory_budget_gib}"
    echo "LTX-2.3 no-GPU memory report: ${result_dir}/mock_memory.json"
    exit 0
fi

preflight="${run_root}/scripts/lib/preflight_ltx23_fp8_gfx1201.py"
host_collector="${run_root}/scripts/collector/collect_ltx23_host_evidence.py"
summary_script="${run_root}/scripts/lib/summarize_ltx23_b0.py"
weight_probe="${run_root}/scripts/collector/collect_ltx23_weight_probe.py"
handoff_probe="${run_root}/scripts/lib/check_ltx23_phase_handoff.py"
for script in "${model_run}" "${preflight}" "${memory_mock}" "${host_collector}" "${summary_script}" "${weight_probe}" "${handoff_probe}"; do
    [[ -f "${script}" ]] || { echo "ERROR: required launcher component not found: ${script}" >&2; exit 1; }
done

python3 - "${result_dir}/run_manifest.json" "${image}" "${fps}" "${frames}" \
    "${gpu_devices}" "${checkpoint}" "${upscaler}" "${gemma_dir}" \
    "${prompt}" "${preflight_only}" "${collect}" "${weight_probe_only}" \
    "${weight_probe_sha256}" "${dynamic_memory_budget_gib}" \
    "${handoff_vram_limit_mib}" "${handoff_timeout_seconds}" <<'PY'
import json
import pathlib
import sys

(output, image, fps, frames, gpus, checkpoint, upscaler, gemma, prompt,
 preflight_only, collect, weight_probe_only, weight_probe_sha256,
 dynamic_memory_budget_gib, handoff_vram_limit_mib,
 handoff_timeout_seconds) = sys.argv[1:]
payload = {
    "schema_name": "lightx2v.ltx23_fp8.b0.manifest",
    "schema_version": 1,
    "kind": "ltx23_fp8_gfx1201_b0",
    "image": image,
    "task": "t2av",
    "gpus": [int(item) for item in gpus.split(",")],
    "checkpoint": {
        "path": checkpoint,
        "expected_size_bytes": 29531884062,
        "expected_sha256": "d9646b6f2d5c42d337b23671634c43bfeece6989644f51b4a3aa088465ccd3b2",
    },
    "upscaler": upscaler,
    "gemma_dir": gemma,
    "prompt": prompt,
    "preflight_only": preflight_only == "1",
    "weight_probe_only": weight_probe_only == "1",
    "weight_probe_sha256": weight_probe_sha256 == "1",
    "collector": {
        "enabled": collect == "1",
        "default_enabled": True,
        "runtime_schema_version": 2,
        "source_process_output": "ltx23_collector.gemma.json",
        "output_pattern": "ltx23_collector.rank{rank}.json",
        "observer_only": True,
    },
    "workload": {
        "final_height": 704,
        "final_width": 1280,
        "stage1_height": 352,
        "stage1_width": 640,
        "frames": int(frames),
        "fps": int(fps),
        "stage1_steps": 8,
        "stage2_steps": 3,
    },
    "backends": {
        "fp8_gemm": "torch._scaled_mm",
        "bf16_gemm": "torch.mm",
        "attention": "torch_sdpa",
        "norm_rope_modulate": "torch",
        "sequence_parallel": "ulysses_sp2_torch_a2a",
    },
    "residency": {
        "strategy": "process_isolated_gemma_then_disk_streamed_mixed_bf16_fp8",
        "prompt_handoff": {
            "producer": "single_source_process_accelerate_auto",
            "artifact": "ltx23_prompt_embeddings.safetensors",
            "consumer": "fresh_sp2_dit_processes",
            "vram_release_gate": "phase_handoff.json",
            "vram_release_limit_mib": float(handoff_vram_limit_mib),
            "timeout_seconds": float(handoff_timeout_seconds)
        },
        "gpu_slots": {"bf16": 2, "fp8": 2},
        "pinned_host_slots": {"bf16": 2, "fp8": 2},
        "mock_memory_report": "mock_memory.json",
        "dynamic_activation_workspace_budget_gib": float(dynamic_memory_budget_gib),
    },
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

snapshot_written=0
monitor_pid=
collect_host() {
    local phase="$1"
    [[ "${collect}" == 1 && "${DRY_RUN:-0}" != 1 ]] || return 0
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${host_collector}" \
        --output "${result_dir}/host_${phase}.json" \
        --phase "${phase}" \
        --image "${image}" \
        --run-root "${run_root}" \
        --checkpoint "${checkpoint}" \
        --upscaler "${upscaler}" \
        --gemma-dir "${gemma_dir}" \
        --gpu-indices "${gpu_devices}"
}
stop_monitor() {
    if [[ -n "${monitor_pid:-}" ]]; then
        if kill -0 "${monitor_pid}" 2>/dev/null; then kill "${monitor_pid}" 2>/dev/null || true; fi
        wait "${monitor_pid}" 2>/dev/null || true
        monitor_pid=
    fi
}
wait_monitor() {
    local status=0
    if [[ -n "${monitor_pid:-}" ]]; then
        set +e
        wait "${monitor_pid}"
        status=$?
        set -e
        monitor_pid=
    fi
    if (( status != 0 )); then
        echo "ERROR: memory monitor failed with status ${status}" >&2
        return "${status}"
    fi
}
cleanup() {
    local status=$?
    trap - EXIT
    stop_monitor
    if [[ "${snapshot_written}" == 0 && "${collect}" == 1 && "${DRY_RUN:-0}" != 1 ]]; then
        set +e
        collect_host failure
        local collector_status=$?
        set -e
        snapshot_written=1
        if (( status == 0 && collector_status != 0 )); then status=${collector_status}; fi
    fi
    exit "${status}"
}
trap cleanup EXIT

if [[ "${DRY_RUN:-0}" != 1 && "${weight_probe_only}" != 1 ]]; then
    python3 "${memory_mock}" \
        --checkpoint "${checkpoint}" \
        --output "${result_dir}/mock_memory.json" \
        --dynamic-budget-gib "${dynamic_memory_budget_gib}"
    python3 "${preflight}" \
        --mode host \
        --checkpoint "${checkpoint}" \
        --upscaler "${upscaler}" \
        --gemma-dir "${gemma_dir}" \
        --output "${result_dir}/preflight_host.json"
    collect_host before
fi

export IMAGE="${image}"
export LIGHTX2V_LTX23_EXPLICIT_IMAGE=1
export GPU_DEVICES="${gpu_devices}"
export LTX23_CHECKPOINT="${checkpoint}"
export LTX23_UPSCALER="${upscaler}"
export LTX23_GEMMA_DIR="${gemma_dir}"
export LTX23_PREFLIGHT_ONLY="${preflight_only}"
export LTX23_COLLECT="${collect}"
export LTX23_WEIGHT_PROBE_ONLY="${weight_probe_only}"
export LTX23_WEIGHT_PROBE_SHA256="${weight_probe_sha256}"
export LTX23_HANDOFF_VRAM_LIMIT_MIB="${handoff_vram_limit_mib}"
export LTX23_HANDOFF_TIMEOUT_SECONDS="${handoff_timeout_seconds}"
export LTX23_EVIDENCE_DIR=/workspace/LightX2V/save_results
export PROMPT="${prompt}"
export OUTPUT_DIR="${result_dir}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-ltx23-b0}"

default_memory_monitor="${collect}"
if [[ "${weight_probe_only}" == 1 ]]; then
    default_memory_monitor=0
fi
memory_monitor="${MEMORY_MONITOR:-${default_memory_monitor}}"
case "${memory_monitor}" in 0 | 1) ;; *) echo "ERROR: MEMORY_MONITOR must be 0 or 1" >&2; exit 2 ;; esac
if [[ "${memory_monitor}" == 1 && "${DRY_RUN:-0}" != 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 -B "${run_root}/scripts/lib/monitor_run_memory.py" \
        --container "${CONTAINER_NAME}" \
        --log "${result_dir}/run.log" \
        --output "${result_dir}/memory.jsonl" \
        --interval "${MEMORY_MONITOR_INTERVAL:-2.0}" \
        --print-interval "${MEMORY_MONITOR_PRINT_INTERVAL:-30.0}" \
        --start-timeout "${MEMORY_MONITOR_START_TIMEOUT:-900}" \
        --expected-ranks 2 \
        --gpu-indices "${gpu_devices}" \
        --host-limit-gib "${MEMORY_MONITOR_HOST_LIMIT_GIB:-512}" \
        --gpu-limit-gib "${MEMORY_MONITOR_GPU_LIMIT_GIB:-32}" \
        --phase-schema ltx2 &
    monitor_pid=$!
fi

echo "LTX-2.3 FP8 B0 results: ${result_dir}"
echo "Workload: 704x1280 final / 352x640 Stage 1, ${frames}f @ ${fps} FPS, 8+3 steps"
echo "GPUs: ${gpu_devices}"
echo "Collector: ${collect} (default on)"
set +e
"${model_run}" --run-ltx23-gfx1201 2>&1 | tee "${result_dir}/run.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if (( pipeline_status[0] != 0 )); then
    echo "ERROR: LTX-2.3 B0 failed; evidence remains in ${result_dir}" >&2
    exit "${pipeline_status[0]}"
fi
if (( pipeline_status[1] != 0 )); then
    echo "ERROR: could not write ${result_dir}/run.log" >&2
    exit "${pipeline_status[1]}"
fi
wait_monitor

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    snapshot_written=1
    echo "Dry run complete: ${result_dir}"
    exit 0
fi

if [[ "${weight_probe_only}" == 1 ]]; then
    snapshot_written=1
    probe_report="${result_dir}/ltx23_weight_probe.json"
    [[ -s "${probe_report}" ]] || { echo "ERROR: container did not write ${probe_report}" >&2; exit 1; }
    echo "LTX-2.3 in-container weight probe complete: ${probe_report}"
    exit 0
fi

collect_host after
snapshot_written=1
if [[ "${preflight_only}" == 1 ]]; then
    echo "LTX-2.3 B0 preflight complete: ${result_dir}"
    exit 0
fi

summary_args=(--result-dir "${result_dir}" --output-json "${result_dir}/summary.json")
if [[ "${collect}" == 1 ]]; then summary_args+=(--expect-collector); fi
PYTHONDONTWRITEBYTECODE=1 python3 -B "${summary_script}" "${summary_args[@]}"
echo "LTX-2.3 FP8 B0 complete: ${result_dir}"
echo "Video: ${result_dir}/ltx23_b0_704x1280_${fps}fps.mp4"
echo "Summary: ${result_dir}/summary.json"
