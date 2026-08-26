import importlib.util
import json
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/collector/collect_ltx23_weight_probe.py"
SPEC = importlib.util.spec_from_file_location("collect_ltx23_weight_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def write_checkpoint(
    path: Path,
    *,
    prefix: str = "",
    missing_gate: str | None = None,
    missing_scale: bool = False,
    quantized_gate_block: int | None = None,
) -> None:
    entries = {}
    payload_size = 0

    def add(name: str, dtype: str, shape: list[int], nbytes: int) -> None:
        nonlocal payload_size
        entries[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [payload_size, payload_size + nbytes],
        }
        payload_size += nbytes

    namespace = f"{prefix}." if prefix else ""
    for block in range(2):
        for attention in PROBE.LTX2_ATTENTION_PREFIXES:
            for parameter, shape, nbytes in (
                ("weight", [2, 4], 16),
                ("bias", [2], 4),
            ):
                name = (
                    f"{namespace}transformer_blocks.{block}.{attention}."
                    f"to_gate_logits.{parameter}"
                )
                if name != missing_gate:
                    quantized = parameter == "weight" and block == quantized_gate_block
                    add(name, "F8_E4M3" if quantized else "BF16", shape, nbytes // 2 if quantized else nbytes)
                    if quantized:
                        add(name.removesuffix(".weight") + ".input_scale", "F32", [1], 4)
                        add(name.removesuffix(".weight") + ".weight_scale", "F32", [1], 4)
        layer = f"{namespace}transformer_blocks.{block}.attn1.to_q"
        dtype = "BF16" if block == 0 else "F8_E4M3"
        add(f"{layer}.weight", dtype, [4, 4], 32 if dtype == "BF16" else 16)
        if block == 1:
            add(f"{layer}.input_scale", "F32", [1], 4)
            if not missing_scale:
                add(f"{layer}.weight_scale", "F32", [1], 4)

    header = {
        "__metadata__": {
            "model_version": "2.3.0",
            "config": json.dumps({"transformer": {"num_layers": 2}}),
        },
        **entries,
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload_size))


def test_collects_alternate_namespace_and_complete_inventory(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    write_checkpoint(checkpoint)
    expected = "model.diffusion_model.transformer_blocks.1.attn1.to_gate_logits.weight"

    report = PROBE.collect_report(checkpoint, expected_keys=[expected])

    gate = report["contracts"]["gate_logits"]
    assert gate["status_counts"] == {"alternate_prefix": 24}
    assert not gate["missing_expected_names"]
    assert report["namespaces"]["transformer_block_prefix_counts"] == {"": 28}
    assert report["contracts"]["explicit_runtime_keys"][0]["status"] == "alternate_prefix"
    assert len(report["tensor_inventory"]) == 28
    assert report["probe_contract"]["payload_read"] is False


def test_reports_missing_gate_and_fp8_scale_without_aborting(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    missing = "model.diffusion_model.transformer_blocks.1.attn1.to_gate_logits.weight"
    write_checkpoint(
        checkpoint,
        prefix="model.diffusion_model",
        missing_gate=missing,
        missing_scale=True,
    )

    report = PROBE.collect_report(checkpoint)

    assert report["contracts"]["gate_logits"]["status_counts"] == {
        "exact": 23,
        "missing": 1,
    }
    assert report["contracts"]["gate_logits"]["missing_expected_names"] == [missing]
    assert report["contracts"]["fp8"]["invalid_or_missing_scale_count"] == 1
    assert {issue["code"] for issue in report["issues"]} >= {
        "missing_gate_logits",
        "invalid_fp8_scale_contract",
    }


def test_cli_writes_report_and_strict_preserves_evidence(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    output = tmp_path / "probe.json"
    write_checkpoint(checkpoint)

    status = PROBE.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--strict",
        ]
    )

    assert status == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "lightx2v.ltx23.weight_probe"
    assert payload["issues"]


def test_official_mixed_fp8_gate_weights_are_not_reported_as_errors(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    write_checkpoint(
        checkpoint,
        prefix="model.diffusion_model",
        quantized_gate_block=1,
    )

    report = PROBE.collect_report(checkpoint)

    gate = report["contracts"]["gate_logits"]
    assert gate["status_counts"] == {"exact": 24}
    assert len(gate["quantized_to_gate_logits"]) == 6
    assert gate["quantized_gate_logits_are_valid_fp8_linears"] is True
    assert "quantized_gate_logits" not in {
        issue["code"] for issue in report["issues"]
    }
    assert report["contracts"]["fp8"]["invalid_or_missing_scale_count"] == 0
