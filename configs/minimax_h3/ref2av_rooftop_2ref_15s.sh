#!/usr/bin/env bash
# Original two-reference materials used when no --prompt-profile is given.
h3_material_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
h3_material_repo_root="$(cd -- "${h3_material_config_dir}/../.." && pwd)"
h3_material_asset_dir="${h3_material_repo_root}/assets/minimax_h3/rooftop_15s"

export H3_REF2AV_MATERIAL_NAME=rooftop_2ref_15s
export H3_PROMPT_FILE="${h3_material_asset_dir}/prompt.txt"
printf -v H3_REFERENCE_IMAGES '%s\n' \
    "${h3_material_asset_dir}/Image Gen-1.png" \
    "${h3_material_asset_dir}/RYAN-2.png"
H3_REFERENCE_IMAGES="${H3_REFERENCE_IMAGES%$'\n'}"
export H3_REFERENCE_IMAGES

unset h3_material_config_dir h3_material_repo_root h3_material_asset_dir
