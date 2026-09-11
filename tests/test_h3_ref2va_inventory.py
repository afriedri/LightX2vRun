import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/collector"))
from collect_minimax_h3_ref2v_weights import EXPECTED_CORE_QUANT_SHAPES, FILES, audit_int8_convrot, collect


def write(path, tensors, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}; payload = bytearray()
    if metadata is not None:
        header["__metadata__"] = {
            key: value if isinstance(value, str) else json.dumps(value)
            for key, value in metadata.items()
        }
    for name, dtype, shape, data in tensors:
        start = len(payload); payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_collects_both_formats_and_actual_convrot_scalar(tmp_path):
    write(tmp_path / FILES["bf16"], [("blocks.0.weight", "BF16", [2, 2], bytes(8))])
    write(tmp_path / FILES["int8_convrot"], [("blocks.0.weight", "I8", [2, 2], bytes(4)),
          ("blocks.0.weight_scale", "F32", [1], struct.pack("<f", 1)),
          ("blocks.0.convrot_groupsize", "I64", [], struct.pack("<q", 64))])
    report = collect(tmp_path)
    assert report["status"] == "ok"
    assert report["comparison"]["dtype_transitions"] == {"BF16->I8": 1}
    int8 = report["assets"]["int8_convrot"]
    assert int8["convrot_groupsize_counts"] == {"64": 1}
    assert int8["probe_contract"]["scalar_metadata_bytes_read"] == 8
    assert not int8["probe_contract"]["weight_matrix_payload_read"]
    assert not report["runtime_validated"]


def test_missing_or_truncated_weight_keeps_both_reports(tmp_path):
    path = tmp_path / FILES["bf16"]
    write(path, [("w", "BF16", [2, 2], bytes(8))])
    path.write_bytes(path.read_bytes()[:-1])
    report = collect(tmp_path)
    assert report["status"] == "error"
    assert set(report["assets"]) == set(FILES)
    assert all(asset["status"] == "error" for asset in report["assets"].values())


def test_quant_audit_reports_sidecar_contract_errors(tmp_path):
    quant = json.dumps({"format": "int8_tensorwise", "convrot": False, "convrot_groupsize": 32}).encode()
    write(tmp_path / FILES["int8_convrot"], [
        ("blocks.0.attn.qkv_proj.weight", "I8", [12, 8], bytes(96)),
        ("blocks.0.attn.qkv_proj.weight_scale", "BF16", [1], bytes(2)),
        ("blocks.0.attn.qkv_proj.comfy_quant", "U8", [len(quant)], quant),
    ])
    readiness = collect(tmp_path, ("int8_convrot",))["assets"]["int8_convrot"]["conversion_readiness"]
    codes = {item["code"] for item in readiness["issues"]}
    assert {
        "convrot_not_enabled", "unsupported_convrot_groupsize", "invalid_weight_scale_dtype",
        "main_block_set", "refiner_block_set", "missing_h3_anchors",
    } <= codes
    assert readiness["layer_kind_counts"] == {"fused_qkv_split": 1}


def test_collects_only_available_int8_convrot_without_bf16(tmp_path):
    quant = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
    write(tmp_path / FILES["int8_convrot"], [
        ("blocks.0.weight", "I8", [2, 2], bytes(4)),
        ("blocks.0.weight_scale", "F32", [2], bytes(8)),
        ("blocks.0.comfy_quant", "U8", [len(quant)], quant),
    ])
    report = collect(tmp_path, ("int8_convrot",))
    assert report["status"] == "ok"
    assert report["selected_assets"] == ["int8_convrot"]
    assert set(report["assets"]) == {"int8_convrot"}
    assert report["comparison"] is None
    int8 = report["assets"]["int8_convrot"]
    assert int8["convrot_groupsize_counts"] == {"256": 1}
    assert int8["quantization_format_counts"] == {"int8_tensorwise": 1}
    assert int8["comfy_quant_status_counts"] == {"ok": 1}
    assert int8["comfy_quant_layers"]["blocks.0.comfy_quant"]["config"]["convrot"] is True
    readiness = int8["conversion_readiness"]
    assert readiness["quantized_layer_count"] == 1
    assert readiness["expected_core_quantized_layer_count"] == 250
    assert readiness["layer_kind_counts"] == {"main_block_direct_rename": 1}
    assert readiness["layers"][0]["weight_scale"]["shape"] == [2]
    assert readiness["status"] == "blocked"  # tiny fixture is not a complete 50+2 block H3 checkpoint
    assert len(readiness["missing_core_quant_layers"]) == 250


def test_normalizes_legacy_safetensors_quantization_metadata(tmp_path):
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
    write(
        tmp_path / FILES["int8_convrot"],
        [
            ("blocks.0.weight", "I8", [2, 64], bytes(128)),
            ("blocks.0.weight_scale", "F32", [2], bytes(8)),
        ],
        metadata={"_quantization_metadata": {"layers": {"blocks.0": config}}},
    )
    int8 = collect(tmp_path, ("int8_convrot",))["assets"]["int8_convrot"]
    assert int8["comfy_quant_layers"]["blocks.0.comfy_quant"] == {
        "status": "ok", "config": config, "source": "safetensors_metadata",
    }
    assert int8["quant_metadata_source_counts"] == {"safetensors_metadata": 1}
    assert int8["quantization_format_counts"] == {"int8_tensorwise": 1}
    assert int8["convrot_groupsize_counts"] == {"64": 1}
    assert int8["probe_contract"]["header_only"] is True
    assert int8["probe_contract"]["bounded_metadata_bytes_read"] == 0


def test_legacy_metadata_takes_comfyui_precedence_over_duplicate_tensor(tmp_path):
    metadata_config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
    tensor_config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    quant = json.dumps(tensor_config).encode()
    write(
        tmp_path / FILES["int8_convrot"],
        [
            ("blocks.0.weight", "I8", [2, 64], bytes(128)),
            ("blocks.0.weight_scale", "F32", [2, 1], bytes(8)),
            ("blocks.0.comfy_quant", "U8", [len(quant)], quant),
        ],
        metadata={"_quantization_metadata": {"layers": {"blocks.0": metadata_config}}},
    )
    int8 = collect(tmp_path, ("int8_convrot",))["assets"]["int8_convrot"]
    parsed = int8["comfy_quant_layers"]["blocks.0.comfy_quant"]
    assert parsed == {
        "status": "ok",
        "config": metadata_config,
        "source": "tensor+safetensors_metadata",
        "effective_source": "safetensors_metadata",
        "representations_match": False,
        "shadowed_tensor_config": tensor_config,
    }
    assert int8["convrot_groupsize_counts"] == {"64": 1}
    readiness = int8["conversion_readiness"]
    assert readiness["layers"][0]["effective_metadata_source"] == "safetensors_metadata"
    assert readiness["shadowed_tensor_metadata_count"] == 1


def test_complete_core_quant_coverage_is_conversion_ready(tmp_path):
    tensors = []
    quant_layers = {}
    for block in range(50):
        for suffix, shape in EXPECTED_CORE_QUANT_SHAPES.items():
            base = f"blocks.{block}.{suffix}"
            tensors.extend([
                {"name": f"{base}.weight", "dtype": "I8", "shape": shape},
                {"name": f"{base}.weight_scale", "dtype": "F32", "shape": [shape[0]]},
                {"name": f"{base}.comfy_quant", "dtype": "U8", "shape": [1]},
            ])
            groupsize = 64 if suffix == "adaln_proj.linear" else 256
            quant_layers[f"{base}.comfy_quant"] = {
                "status": "ok",
                "config": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize},
            }
    for anchor in (
        "video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight",
        "final_layer.video_out.weight", "final_layer.audio_out.weight",
    ):
        tensors.append({"name": anchor, "dtype": "BF16", "shape": [1, 1]})
    for block in range(2):
        tensors.append({"name": f"token_refiner.blocks.{block}.norm1.weight", "dtype": "BF16", "shape": [1]})
    readiness = audit_int8_convrot({
        "tensor_inventory": tensors,
        "comfy_quant_layers": quant_layers,
        "metadata": {"config": {"transformer": {"num_layers": 50}}},
    })
    assert readiness["status"] == "ready"
    assert readiness["quantized_layer_count"] == readiness["valid_layer_count"] == 250
    assert readiness["expected_target_quantized_linear_count"] == 350
    assert readiness["full_precision_matrix_mult_enabled_count"] == 0
    assert readiness["shadowed_tensor_metadata_count"] == 0
    assert readiness["layer_kind_counts"] == {
        "fused_qkv_split": 50,
        "fused_fc1_half_swap": 50,
        "main_block_direct_rename": 150,
    }
    assert not readiness["missing_core_quant_layers"]
    assert not readiness["unsupported_extra_quant_layers"]
