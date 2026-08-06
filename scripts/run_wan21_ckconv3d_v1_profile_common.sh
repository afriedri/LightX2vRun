#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
expected_gpu_count="${EXPECTED_GPU_COUNT:?EXPECTED_GPU_COUNT must be set by a GPU-count wrapper}"
case "${expected_gpu_count}" in
    1 | 2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: CK Conv3d profile supports 1, 2, 4, 8, or 16 GPUs" >&2
        exit 1
        ;;
esac

profile_tag="${PROFILE_TAG:-wan21_ckconv3d_v1_${expected_gpu_count}gpu_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${profile_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: PROFILE_TAG may contain only letters, digits, dot, underscore, and dash" >&2
    exit 1
fi

export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export LIGHTX2V_AMD_MIOPEN=0
export USE_TILING_VAE=0
export VAE_CPU_OFFLOAD=0
export T5_CPU_OFFLOAD="${T5_CPU_OFFLOAD:-1}"
export CLIP_CPU_OFFLOAD="${CLIP_CPU_OFFLOAD:-1}"
export USE_CHANNELS_LAST_3D=0
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_profiles/${profile_tag}}"
export SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${LIGHTX2V_CK_CONV3D_PROFILE_DIR}/output.mp4}"

if [[ "${expected_gpu_count}" == 16 ]]; then
    export SEQ_P_ATTN_TYPE="${SEQ_P_ATTN_TYPE:-ring}"
    export SELF_ATTN_TYPE="${SELF_ATTN_TYPE:-aiter_triton_bf16_flash_attn}"
else
    export SEQ_P_ATTN_TYPE="${SEQ_P_ATTN_TYPE:-ulysses}"
    export SELF_ATTN_TYPE="${SELF_ATTN_TYPE:-aiter_flydsl_bf16_flash_attn}"
fi

echo "Wan2.1 CK Conv3d v1 profile: ${expected_gpu_count} GPUs"
echo "VAE: resident, spatial_parallel=$((expected_gpu_count > 1 ? 1 : 0)), tiling=0, MIOpen=0"
echo "Attention: ${SEQ_P_ATTN_TYPE}/${SELF_ATTN_TYPE}"
echo "Profile: ${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
echo "Tuned CSV: ${AITER_CONFIG_CONV3D_V1:-Aiter packaged default}"

if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
    mkdir -p "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
    python3 -c 'import importlib; import aiter; assert callable(aiter.ck_conv3d_v1); assert callable(aiter.get_ck_conv3d_v1_config); assert aiter.is_ck_conv3d_v1_prebuilt(); module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'
fi

bash "${script_dir}/run_wan21_i2v_distill_fp8_4step_gfx1201_1gpu_benchmark_4090.sh" "$@"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

encode_count="$(find "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" -maxdepth 1 -type f -name 'rank*_encode_*.json' | wc -l)"
decode_count="$(find "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" -maxdepth 1 -type f -name 'rank*_decode_*.json' | wc -l)"
if (( encode_count < expected_gpu_count || decode_count < expected_gpu_count )); then
    echo "ERROR: incomplete CK Conv3d evidence: encode=${encode_count}, decode=${decode_count}, expected at least ${expected_gpu_count} each" >&2
    exit 1
fi

python3 - "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" "${expected_gpu_count}" "${SEQ_P_ATTN_TYPE}" "${SELF_ATTN_TYPE}" <<'PY'
import json
import os
import pathlib
import sys

profile_dir = pathlib.Path(sys.argv[1])
snapshots = sorted(profile_dir.glob("rank*_*.json"))
ck_calls = 0
csv_hit_calls = 0
non_csv_ck = []
for snapshot_path in snapshots:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for route in snapshot.get("routes", []):
        if route.get("route") != "ck":
            continue
        calls = int(route.get("calls", 0))
        ck_calls += calls
        if route.get("selection") == "csv_lookup":
            csv_hit_calls += calls
        else:
            non_csv_ck.append(
                {
                    "snapshot": snapshot_path.name,
                    "selection": route.get("selection"),
                    "instance_id": route.get("instance_id"),
                    "calls": calls,
                    "input_shape": route.get("input_shape"),
                    "weight_shape": route.get("weight_shape"),
                    "output_shape": route.get("output_shape"),
                }
            )

replay = {
    "tuned_csv": os.getenv("AITER_CONFIG_CONV3D_V1"),
    "ck_calls": ck_calls,
    "csv_hit_calls": csv_hit_calls,
    "all_ck_calls_hit_csv": ck_calls > 0 and ck_calls == csv_hit_calls,
    "non_csv_ck_routes": non_csv_ck,
}
(profile_dir / "csv_replay_validation.json").write_text(
    json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
payload = {
    "kind": "wan21_ck_conv3d_v1_profile_manifest",
    "world_size": int(sys.argv[2]),
    "seq_p_attn_type": sys.argv[3],
    "self_attn_type": sys.argv[4],
    "vae_parallel": int(sys.argv[2]) > 1,
    "vae_cpu_offload": False,
    "use_tiling_vae": False,
    "miopen": False,
    "tuned_csv": os.getenv("AITER_CONFIG_CONV3D_V1"),
    "snapshots": [path.name for path in snapshots],
    "csv_replay": replay,
}
(profile_dir / "manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if replay["tuned_csv"] and not replay["all_ck_calls_hit_csv"]:
    raise SystemExit(
        "generated tuned CSV did not cover every successful CK call: "
        f"hits={csv_hit_calls}, ck_calls={ck_calls}; inspect csv_replay_validation.json"
    )
PY

echo "CK Conv3d profile complete: ${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
