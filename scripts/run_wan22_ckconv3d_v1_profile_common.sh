#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-/workspace/LightX2V}"
expected_gpu_count="${EXPECTED_GPU_COUNT:?EXPECTED_GPU_COUNT must be set by a GPU-count wrapper}"

case "${expected_gpu_count}" in
    1 | 2 | 4 | 8 | 16) ;;
    *)
        echo "ERROR: Wan2.2 CK Conv3d profiling supports 1, 2, 4, 8, or 16 GPUs" >&2
        exit 1
        ;;
esac

profile_tag="${PROFILE_TAG:-wan22_ckconv3d_v1_${expected_gpu_count}gpu_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${profile_tag}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: PROFILE_TAG may contain only letters, digits, dot, underscore, and dash" >&2
    exit 1
fi

export TARGET_VIDEO_LENGTH="${TARGET_VIDEO_LENGTH:-81}"
export TARGET_HEIGHT="${TARGET_HEIGHT:-720}"
export TARGET_WIDTH="${TARGET_WIDTH:-1280}"
export TARGET_FPS="${TARGET_FPS:-16}"
for target_name in TARGET_VIDEO_LENGTH TARGET_HEIGHT TARGET_WIDTH TARGET_FPS; do
    target_value="${!target_name}"
    if [[ ! "${target_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${target_name} must be a positive integer, got ${target_value}" >&2
        exit 1
    fi
done
if (( (TARGET_VIDEO_LENGTH - 1) % 4 != 0 )); then
    echo "ERROR: TARGET_VIDEO_LENGTH must satisfy the Wan VAE 4n+1 contract, got ${TARGET_VIDEO_LENGTH}" >&2
    exit 1
fi
if [[ -n "${PROFILE_WORKLOAD:-}" ]]; then
    export PROFILE_WORKLOAD
elif [[ "${TARGET_HEIGHT}x${TARGET_WIDTH}x${TARGET_VIDEO_LENGTH}x${TARGET_FPS}" == "720x1280x81x16" ]]; then
    export PROFILE_WORKLOAD=720p
elif [[ "${TARGET_HEIGHT}x${TARGET_WIDTH}x${TARGET_VIDEO_LENGTH}x${TARGET_FPS}" == "480x832x81x16" ]]; then
    export PROFILE_WORKLOAD=480p
elif [[ "${TARGET_HEIGHT}x${TARGET_WIDTH}x${TARGET_VIDEO_LENGTH}x${TARGET_FPS}" == "480x832x121x24" ]]; then
    export PROFILE_WORKLOAD=480p121f
else
    export PROFILE_WORKLOAD="${TARGET_HEIGHT}x${TARGET_WIDTH}_${TARGET_VIDEO_LENGTH}f_${TARGET_FPS}fps"
fi
if [[ ! "${PROFILE_WORKLOAD}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: PROFILE_WORKLOAD may contain only letters, digits, dot, underscore, and dash" >&2
    exit 1
fi

# Match the Wan2.1 CK release's VAE policy. This makes the captured shapes
# independent of spatial tiling and keeps the VAE resident for the whole run.
export LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1
export CPU_OFFLOAD=1
export OFFLOAD_GRANULARITY=block
export OFFLOAD_RATIO="${OFFLOAD_RATIO:-1.0}"
export T5_CPU_OFFLOAD="${T5_CPU_OFFLOAD:-1}"
export VAE_CPU_OFFLOAD=0
export USE_TILING_VAE=0
export LIGHTX2V_AMD_MIOPEN=0
export USE_CHANNELS_LAST_3D=0
export LIGHTX2V_WARMUP=0
export SELF_ATTN_TYPE=aiter_fav3_sage_bf16_attn
export BASE_CONFIG_JSON="${BASE_CONFIG_JSON:-${lightx2v_path}/configs/platforms/amd_rocm/wan22_moe_i2v_distill_fp8_4step_sage_gfx1201.json}"
export LIGHTX2V_CK_CONV3D_PROFILE_DIR="${LIGHTX2V_CK_CONV3D_PROFILE_DIR:-/workspace/LightX2V/save_results/ck_conv3d_profiles/${profile_tag}}"
export SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${LIGHTX2V_CK_CONV3D_PROFILE_DIR}/output.mp4}"

echo "Wan2.2 CK Conv3d v1 profile: ${expected_gpu_count} GPU(s)"
echo "Workload: ${PROFILE_WORKLOAD}, ${TARGET_HEIGHT}x${TARGET_WIDTH}, ${TARGET_VIDEO_LENGTH} frames @ ${TARGET_FPS} fps"
echo "Offload: DiT=block/${OFFLOAD_RATIO}, T5=${T5_CPU_OFFLOAD}; VAE resident, tiling=0"
echo "VAE: MIOpen=0, channels_last_3d=0, spatial_parallel=$((expected_gpu_count > 1 ? 1 : 0))"
echo "Profile: ${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
echo "Tuned CSV: ${AITER_CONFIG_CONV3D_V1:-Aiter packaged default}"

if [[ "${CONFIG_ONLY:-0}" != 1 ]]; then
    if [[ -d "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" ]] && \
        [[ -n "$(find "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: CK Conv3d profile directory is not empty: ${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" >&2
        exit 1
    fi
    mkdir -p "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
    python3 -c 'import importlib; import aiter; assert callable(aiter.ck_conv3d_v1); assert callable(aiter.get_ck_conv3d_v1_config); assert aiter.is_ck_conv3d_v1_prebuilt(); module = importlib.import_module("aiter.jit.module_ck_vae_conv3d"); assert module.__file__.endswith(".so"), module.__file__; print(f"CK Conv3d v1 AOT: {module.__file__}")'
fi

bash "${script_dir}/run_wan22_moe_i2v_distill_fp8_4step_gfx1201_common.sh" "$@"

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
    exit 0
fi

python3 - "${LIGHTX2V_CK_CONV3D_PROFILE_DIR}" "${expected_gpu_count}" <<'PY'
import json
import os
from pathlib import Path
import sys

profile_dir = Path(sys.argv[1])
world_size = int(sys.argv[2])
expected = {
    (rank, phase)
    for rank in range(world_size)
    for phase in ("encode", "decode")
}
snapshots = {}
routes = []
contracts = set()
required_route_fields = {
    "route",
    "calls",
    "input_shape",
    "weight_shape",
    "output_shape",
    "conv_stride",
    "padding",
    "dilation",
    "groups",
}
for path in sorted(profile_dir.glob("rank*_*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "lightx2v_ck_conv3d_profile":
        raise SystemExit(f"unexpected CK Conv3d snapshot kind: {path}")
    rank = int(payload.get("rank", -1))
    phase = str(payload.get("phase", ""))
    snapshot_world_size = int(payload.get("world_size", -1))
    key = (rank, phase)
    if key not in expected:
        raise SystemExit(f"unexpected rank/phase snapshot {key}: {path}")
    if snapshot_world_size != world_size:
        raise SystemExit(
            f"snapshot world_size mismatch: expected {world_size}, "
            f"got {snapshot_world_size}: {path}"
        )
    if key in snapshots:
        raise SystemExit(f"duplicate rank/phase snapshot {key}: {path}")
    expected_name = f"rank{rank:03d}_{phase}_000.json"
    if path.name != expected_name:
        raise SystemExit(
            f"unexpected snapshot sequence for {key}: expected {expected_name}, got {path.name}"
        )
    if payload.get("backend") != "Aiter_CK_Conv3d_v1":
        raise SystemExit(f"unexpected Conv3d backend in {path}: {payload.get('backend')!r}")
    if payload.get("gfx") != "gfx1201" or int(payload.get("wgp", 0)) != 24:
        raise SystemExit(
            f"unexpected gfx/WGP in {path}: {payload.get('gfx')}/{payload.get('wgp')}"
        )
    snapshot_routes = payload.get("routes")
    if not isinstance(snapshot_routes, list) or not snapshot_routes:
        raise SystemExit(f"snapshot contains no Conv3d routes: {path}")
    for route in snapshot_routes:
        missing = sorted(required_route_fields - set(route))
        if missing:
            raise SystemExit(f"route in {path} is missing fields: {', '.join(missing)}")
        if int(route["calls"]) <= 0:
            raise SystemExit(f"route in {path} has non-positive calls: {route['calls']}")
        for field, dimensions in (
            ("input_shape", 5),
            ("weight_shape", 5),
            ("output_shape", 5),
            ("conv_stride", 3),
            ("padding", 3),
            ("dilation", 3),
        ):
            values = tuple(int(value) for value in route[field])
            if len(values) != dimensions:
                raise SystemExit(
                    f"route in {path} has invalid {field}: {route[field]!r}"
                )
        contracts.add(
            (
                tuple(int(value) for value in route["input_shape"]),
                tuple(int(value) for value in route["weight_shape"]),
                tuple(int(value) for value in route["output_shape"]),
                tuple(int(value) for value in route["conv_stride"]),
                tuple(int(value) for value in route["padding"]),
                tuple(int(value) for value in route["dilation"]),
                int(route["groups"]),
            )
        )
    snapshots[key] = path
    routes.extend(snapshot_routes)

missing_snapshots = sorted(expected - set(snapshots))
if missing_snapshots:
    raise SystemExit(f"missing rank/phase snapshots: {missing_snapshots}")

manifest = {
    "kind": "wan22_ck_conv3d_v1_profile_manifest",
    "model": "wan2.2_moe_distill",
    "workload": os.environ["PROFILE_WORKLOAD"],
    "target_video_length": int(os.environ["TARGET_VIDEO_LENGTH"]),
    "target_height": int(os.environ["TARGET_HEIGHT"]),
    "target_width": int(os.environ["TARGET_WIDTH"]),
    "fps": int(os.environ["TARGET_FPS"]),
    "duration_seconds": (int(os.environ["TARGET_VIDEO_LENGTH"]) - 1) / int(os.environ["TARGET_FPS"]),
    "world_size": world_size,
    "vae_parallel": world_size > 1,
    "cpu_offload": True,
    "offload_granularity": "block",
    "offload_ratio": float(os.environ["OFFLOAD_RATIO"]),
    "t5_cpu_offload": os.environ["T5_CPU_OFFLOAD"] == "1",
    "vae_cpu_offload": False,
    "use_tiling_vae": False,
    "miopen": False,
    "channels_last_3d": False,
    "self_attn_type": os.environ["SELF_ATTN_TYPE"],
    "tuned_csv": os.getenv("AITER_CONFIG_CONV3D_V1"),
    "snapshots": [snapshots[key].name for key in sorted(snapshots)],
    "rank_phase_snapshot_count": len(snapshots),
    "route_count": len(routes),
    "shape_contract_count": len(contracts),
    "total_calls": sum(int(route["calls"]) for route in routes),
    "ck_calls": sum(int(route.get("calls", 0)) for route in routes if route.get("route") == "ck"),
    "fallback_calls": sum(
        int(route.get("calls", 0)) for route in routes if route.get("route") != "ck"
    ),
}
(profile_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

echo "Wan2.2 CK Conv3d profile complete: ${LIGHTX2V_CK_CONV3D_PROFILE_DIR}"
