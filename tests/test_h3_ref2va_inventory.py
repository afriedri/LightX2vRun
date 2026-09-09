import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/collector"))
from collect_minimax_h3_ref2v_weights import FILES, collect


def write(path, tensors):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}; payload = bytearray()
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
