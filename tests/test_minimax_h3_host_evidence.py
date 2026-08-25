from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/collector/collect_minimax_h3_host_evidence.py"


def test_host_evidence_cli_writes_sp8_read_only_contract(tmp_path):
    output = tmp_path / "host-before.json"
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output",
            str(output),
            "--phase",
            "before",
            "--gpu-indices",
            "0,1,2,3,4,5,6,7",
            "--image",
            "example.invalid/minimax-h3:production",
            "--run-root",
            str(REPO_ROOT),
            "--path",
            f"result={tmp_path}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "lightx2v.minimax_h3.sp8_supercollect.host"
    assert payload["phase"] == "before"
    assert payload["read_only"] is True
    assert payload["workload_contract"] == {
        "cpu_block_buffers": 2,
        "dit_dtype": "BF16",
        "dit_quant_scheme": "Default",
        "fps": 24,
        "frames": 124,
        "gpu_block_buffers": 2,
        "height": 768,
        "lazy_load": True,
        "model": "minimax_h3",
        "parallel": "ulysses_sp8",
        "width": 1344,
    }
    assert payload["gpus"]["requested_rocm_indices"] == list(range(8))
    assert payload["paths"]["result"]["exists"] is True


def test_host_evidence_cli_writes_cp2_ulysses8_sp16_rank_mapping(tmp_path):
    output = tmp_path / "host-before.json"
    completed = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output",
            str(output),
            "--phase",
            "before",
            "--profile",
            "megadiscovery",
            "--gpu-indices",
            ",".join(str(index) for index in range(16)),
            "--image",
            "example.invalid/minimax-h3:production",
            "--run-root",
            str(REPO_ROOT),
            "--path",
            f"result={tmp_path}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "lightx2v.minimax_h3.megadiscovery.host"
    assert payload["profile"] == "megadiscovery"
    assert payload["workload_contract"]["parallel"] == "cp2_ulysses8_sp16"
    mapping = payload["gpus"]["local_rank_mapping"]
    assert [item["local_rank"] for item in mapping] == list(range(16))
    assert [item["rocm_index"] for item in mapping] == list(range(16))
