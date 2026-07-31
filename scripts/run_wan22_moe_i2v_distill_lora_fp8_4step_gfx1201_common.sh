#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
models_root="${MODELS_ROOT:-/models}"
base_model_dir="${MODEL_PATH:-${models_root}/Wan-AI/Wan2.2-I2V-A14B}"
lora_dir="${DISTILL_LORAS_DIR:-${models_root}/lightx2v/Wan2.2-Distill-Loras}"
output_dir="${LORA_FP8_OUTPUT_DIR:-${models_root}/lightx2v/Wan2.2-Distill-Models/lora-fp8-strength1}"
lora_strength="${LORA_STRENGTH:-1.0}"
convert_device="${LORA_CONVERT_DEVICE:-cpu}"
force_convert="${FORCE_LORA_CONVERT:-0}"

high_source="${HIGH_NOISE_SOURCE:-${base_model_dir}/high_noise_model}"
low_source="${LOW_NOISE_SOURCE:-${base_model_dir}/low_noise_model}"
high_output_name=wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step
low_output_name=wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step
high_output="${output_dir}/${high_output_name}.safetensors"
low_output="${output_dir}/${low_output_name}.safetensors"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_source() {
    local source_path=$1
    local label=$2
    if [[ -d "${source_path}" ]]; then
        compgen -G "${source_path}/*.safetensors" >/dev/null || \
            die "${label} source has no safetensors shards: ${source_path}"
    elif [[ ! -f "${source_path}" || "${source_path}" != *.safetensors ]]; then
        die "${label} source must be a safetensors file or a directory containing safetensors: ${source_path}"
    fi
}

resolve_lora() {
    local noise=$1
    local explicit_path=$2
    local explicit_variable=$3
    local -a matches=()
    if [[ -n "${explicit_path}" ]]; then
        [[ -f "${explicit_path}" ]] || die "${noise} LoRA not found: ${explicit_path}"
        printf '%s\n' "${explicit_path}"
        return
    fi

    shopt -s nullglob
    matches=("${lora_dir}"/wan2.2_i2v_A14b_${noise}_lora_rank64_lightx2v_4step*.safetensors)
    shopt -u nullglob
    if (( ${#matches[@]} == 0 )); then
        die "No ${noise} I2V distill LoRA found under ${lora_dir}"
    fi
    if (( ${#matches[@]} > 1 )); then
        printf 'ERROR: multiple %s LoRAs found; set %s explicitly:\n' \
            "${noise}" "${explicit_variable}" >&2
        printf '  %s\n' "${matches[@]}" >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

validate_checkpoint() {
    local checkpoint=$1
    python3 - "${checkpoint}" <<'PY'
import sys
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as handle:
    keys = set(handle.keys())
    fp8_keys = [
        key
        for key in keys
        if str(handle.get_slice(key).get_dtype()) in {"F8_E4M3", "F8_E4M3FN"}
    ]
    missing_scales = [key for key in fp8_keys if f"{key}_scale" not in keys]

if not fp8_keys:
    raise SystemExit(f"{path}: no E4M3 tensors found")
if missing_scales:
    preview = ", ".join(missing_scales[:5])
    raise SystemExit(f"{path}: {len(missing_scales)} FP8 tensors have no scale, e.g. {preview}")
print(f"Validated {path}: {len(fp8_keys)} E4M3 tensors with scales")
PY
}

metadata_matches() {
    local metadata_path=$1
    local source_path=$2
    local lora_path=$3
    python3 - "${metadata_path}" "${source_path}" "${lora_path}" "${lora_strength}" <<'PY'
import json
import os
import sys

metadata_path, source_path, lora_path, strength = sys.argv[1:]
try:
    with open(metadata_path, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
except (OSError, ValueError):
    raise SystemExit(1)

expected = {
    "source": os.path.realpath(source_path),
    "lora": os.path.realpath(lora_path),
    "strength": strength,
}
raise SystemExit(0 if all(metadata.get(key) == value for key, value in expected.items()) else 1)
PY
}

write_metadata() {
    local metadata_path=$1
    local noise=$2
    local source_path=$3
    local lora_path=$4
    python3 - "${metadata_path}" "${noise}" "${source_path}" "${lora_path}" "${lora_strength}" <<'PY'
import json
import os
import sys

metadata_path, noise, source_path, lora_path, strength = sys.argv[1:]
metadata = {
    "noise": noise,
    "source": os.path.realpath(source_path),
    "lora": os.path.realpath(lora_path),
    "strength": strength,
    "format": "lightx2v-scaled-fp8-e4m3",
}
with open(metadata_path, "w", encoding="utf-8") as metadata_file:
    json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    metadata_file.write("\n")
PY
}

temporary_dirs=()
cleanup() {
    local temporary_dir
    for temporary_dir in "${temporary_dirs[@]}"; do
        if [[ -n "${temporary_dir}" && "${temporary_dir}" == "${output_dir}"/.lora-convert-* ]]; then
            rm -rf -- "${temporary_dir}"
        fi
    done
}
trap cleanup EXIT

convert_one() {
    local noise=$1
    local source_path=$2
    local lora_path=$3
    local output_name=$4
    local final_path=$5
    local metadata_path="${final_path}.json"

    if [[ -f "${final_path}" && "${force_convert}" != 1 ]]; then
        metadata_matches "${metadata_path}" "${source_path}" "${lora_path}" || \
            die "Existing ${noise} output metadata does not match this source/LoRA/strength; set FORCE_LORA_CONVERT=1 or use another LORA_FP8_OUTPUT_DIR"
        validate_checkpoint "${final_path}"
        echo "Reusing ${noise} LoRA-FP8 checkpoint: ${final_path}"
        return
    fi

    local temporary_dir
    temporary_dir="$(mktemp -d "${output_dir}/.lora-convert-${noise}.XXXXXX")"
    temporary_dirs+=("${temporary_dir}")
    local log_path="${temporary_dir}/convert.log"
    local candidate_path="${temporary_dir}/${output_name}.safetensors"

    echo "Converting ${noise}: merge LoRA -> scaled FP8"
    python3 "${lightx2v_path}/tools/convert/converter.py" \
        --source "${source_path}" \
        --output "${temporary_dir}" \
        --output_ext .safetensors \
        --output_name "${output_name}" \
        --model_type wan_dit \
        --lora_path "${lora_path}" \
        --lora_strength "${lora_strength}" \
        --quantized \
        --linear_type fp8 \
        --non_linear_dtype torch.bfloat16 \
        --device "${convert_device}" \
        --single_file 2>&1 | tee "${log_path}"

    local summary
    summary="$(grep -E 'Applied [0-9]+ LoRA weight adjustments out of [0-9]+ possible' "${log_path}" | tail -n 1 || true)"
    if [[ ! "${summary}" =~ Applied\ ([0-9]+)\ LoRA\ weight\ adjustments\ out\ of\ ([0-9]+)\ possible ]]; then
        die "${noise} conversion did not report a LoRA application summary"
    fi
    local applied_count="${BASH_REMATCH[1]}"
    local possible_count="${BASH_REMATCH[2]}"
    if (( applied_count == 0 || applied_count != possible_count )); then
        die "${noise} LoRA application was incomplete: ${applied_count}/${possible_count}"
    fi
    [[ -f "${candidate_path}" ]] || die "${noise} converter output not found: ${candidate_path}"
    validate_checkpoint "${candidate_path}"

    mv -f -- "${candidate_path}" "${final_path}"
    write_metadata "${metadata_path}" "${noise}" "${source_path}" "${lora_path}"
    echo "Created ${noise} LoRA-FP8 checkpoint: ${final_path}"
}

case "${force_convert}" in
    0 | 1) ;;
    *) die "FORCE_LORA_CONVERT must be 0 or 1" ;;
esac

[[ -d "${lora_dir}" ]] || die "Distill LoRA directory not found: ${lora_dir}"
mkdir -p "${output_dir}" || die "Cannot create LoRA-FP8 output directory: ${output_dir}"

high_lora="$(resolve_lora high_noise "${HIGH_NOISE_LORA_PATH:-}" HIGH_NOISE_LORA_PATH)"
low_lora="$(resolve_lora low_noise "${LOW_NOISE_LORA_PATH:-}" LOW_NOISE_LORA_PATH)"

if [[ "${CONFIG_ONLY:-0}" != 1 && "${SKIP_LORA_CONVERT:-0}" != 1 ]]; then
    require_source "${high_source}" "high_noise"
    require_source "${low_source}" "low_noise"
    convert_one high_noise "${high_source}" "${high_lora}" "${high_output_name}" "${high_output}"
    convert_one low_noise "${low_source}" "${low_lora}" "${low_output_name}" "${low_output}"
fi

export HIGH_NOISE_QUANTIZED_CKPT="${high_output}"
export LOW_NOISE_QUANTIZED_CKPT="${low_output}"

if [[ "${LORA_CONVERT_ONLY:-0}" == 1 ]]; then
    echo "LoRA merge and FP8 quantization completed."
    exit 0
fi

exec bash "${script_dir}/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_common.sh" "$@"
