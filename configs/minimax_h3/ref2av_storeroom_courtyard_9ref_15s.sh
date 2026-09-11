#!/usr/bin/env bash
# Ordered materials shared by the BF16-base and Turbo-8-step Ref2AV scales.
h3_material_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
h3_material_repo_root="$(cd -- "${h3_material_config_dir}/../.." && pwd)"
h3_material_asset_dir="${h3_material_repo_root}/assets/minimax_h3/storeroom_courtyard_9ref_15s"

export H3_REF2AV_MATERIAL_NAME=storeroom_courtyard_9ref_15s
export H3_PROMPT_FILE="${h3_material_asset_dir}/prompt.txt"

printf -v H3_REFERENCE_IMAGES '%s\n' \
    "${h3_material_asset_dir}/wang_hao_front.png" \
    "${h3_material_asset_dir}/wang_hao_full.png" \
    "${h3_material_asset_dir}/chen_fan_front.png" \
    "${h3_material_asset_dir}/chen_fan_full.png" \
    "${h3_material_asset_dir}/discipline_disciple.png" \
    "${h3_material_asset_dir}/su_xue_front.png" \
    "${h3_material_asset_dir}/su_xue_full.png" \
    "${h3_material_asset_dir}/character_layout.png" \
    "${h3_material_asset_dir}/zhao_xiaoman.png"
H3_REFERENCE_IMAGES="${H3_REFERENCE_IMAGES%$'\n'}"
export H3_REFERENCE_IMAGES

unset h3_material_config_dir h3_material_repo_root h3_material_asset_dir
