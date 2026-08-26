#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/collector/run_ltx23_weight_probe.sh --image IMAGE [B0 OPTIONS]

Runs the LTX-2.3 weight probe inside the exact B0 container path. It reuses the
production launcher, image, two-GPU visibility, model mounts, cache mounts and
result mount, but exits after writing ltx23_weight_probe.json without loading
tensor payloads or starting inference.

Accepted B0 options include --gpus, --checkpoint, --upscaler, --gemma-dir and
--result-dir. Add --sha256 to stream and verify the full checkpoint in-container.
EOF
}

args=()
while (( $# > 0 )); do
    case "$1" in
        --help | -h) usage; exit 0 ;;
        --sha256) export LTX23_WEIGHT_PROBE_SHA256=1; shift ;;
        *) args+=("$1"); shift ;;
    esac
done

export LTX23_WEIGHT_PROBE_ONLY=1
exec bash "${run_root}/scripts/run_ltx23_fp8_b0_704x1280_5s.sh" "${args[@]}"
