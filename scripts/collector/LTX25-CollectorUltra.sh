#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$(cd -- "${script_dir}/../.." && pwd)"
git_url="${LTX25_LIGHTX2V_GIT_URL:-git@gitcode.com:hvat-ai/LightX2V.git}"
git_ref="${LTX25_LIGHTX2V_REF:-hvat-gfx1201}"
git_dir="${LTX25_LIGHTX2V_GIT_DIR:-/disk/ssd/lightx2v-runtime/source-cache/LightX2V-ltx25-ultra}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage:
  bash scripts/collector/LTX25-CollectorUltra.sh [OPTIONS]

One host entry for the LTX-2.5 Ultra contract collection. It reuses the
standard two-GPU ROCm container run and additionally requires projected L1
Q/K/V, RoPE cos/sin tensors, GroupNorm parameters, and Ulysses collective
payload contracts. Official DiffVAE remains an explicitly unexecuted
capability while the runtime route is conv_diagnostic.

All options are forwarded to run_ltx25_contract_collector.sh. Common options:
  --image IMAGE
  --output-dir DIR
  --model-root DIR
  --gpus 0,1
  --run-id ID

Environment:
  DRY_RUN=1  Print the Docker command without starting it.
  LTX25_LIGHTX2V_GIT_URL  LightX2V repository URL
  LTX25_LIGHTX2V_REF      Branch/tag/SHA to fetch (default: hvat-gfx1201)
  LTX25_LIGHTX2V_GIT_DIR  Managed source cache directory

The launcher fetches the requested LightX2V revision into its managed cache,
checks out the exact FETCH_HEAD commit, verifies a clean source tree, and runs
that source directly in the existing image. No image rebuild or source-path
argument is required.
EOF
    exit 0
fi

export LIGHTX2V_LTX25_CONTRACT_ULTRA=1
export LTX25_CONTRACT_OUTPUT_DIR="${LTX25_CONTRACT_OUTPUT_DIR:-/disk/ssd/lightx2v-runtime/ltx25-collector-ultra-$(date +%Y%m%d_%H%M%S)}"
export LIGHTX2V_LTX25_CONTRACT_RUN_ID="${LIGHTX2V_LTX25_CONTRACT_RUN_ID:-ltx25-ultra-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
export CONTAINER_NAME="${CONTAINER_NAME:-lightx2v-ltx25-collector-ultra}"

[[ "${git_dir}" == /* ]] || { echo "ERROR: LTX25_LIGHTX2V_GIT_DIR must be absolute" >&2; exit 2; }
[[ -n "${git_url}" && -n "${git_ref}" ]] || { echo "ERROR: LightX2V Git URL/ref cannot be empty" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "CollectorUltra source plan: fetch ${git_ref} from ${git_url} into ${git_dir}"
    export LTX25_LIGHTX2V_SOURCE="${git_dir}"
    export LIGHTX2V_SOURCE_REVISION="DRY_RUN_FETCH_HEAD"
else
    command -v git >/dev/null 2>&1 || { echo "ERROR: git is required" >&2; exit 1; }
    mkdir -p -- "$(dirname -- "${git_dir}")"
    if [[ -e "${git_dir}" && ! -d "${git_dir}/.git" ]]; then
        echo "ERROR: managed LightX2V path is not a Git checkout: ${git_dir}" >&2
        exit 1
    fi
    if [[ ! -d "${git_dir}/.git" ]]; then
        git clone --no-checkout "${git_url}" "${git_dir}"
    else
        actual_url="$(git -C "${git_dir}" remote get-url origin)"
        [[ "${actual_url}" == "${git_url}" ]] || {
            echo "ERROR: managed LightX2V origin mismatch: ${actual_url}" >&2
            exit 1
        }
        [[ -z "$(git -C "${git_dir}" status --porcelain --ignore-submodules=all)" ]] || {
            echo "ERROR: managed LightX2V checkout is dirty: ${git_dir}" >&2
            exit 1
        }
    fi
    git -C "${git_dir}" fetch --no-tags origin "${git_ref}"
    resolved_ref="$(git -C "${git_dir}" rev-parse --verify 'FETCH_HEAD^{commit}')"
    git -C "${git_dir}" checkout --detach "${resolved_ref}"
    [[ -f "${git_dir}/lightx2v/utils/ltx2_contract_collector.py" ]] || {
        echo "ERROR: fetched revision is not a CollectorUltra LightX2V tree" >&2
        exit 1
    }
    export LTX25_LIGHTX2V_SOURCE="$(cd -- "${git_dir}" && pwd -P)"
    export LIGHTX2V_SOURCE_REVISION="${resolved_ref}"
    echo "CollectorUltra source ready: ${resolved_ref}"
fi

exec "${script_dir}/run_ltx25_contract_collector.sh" "$@"
