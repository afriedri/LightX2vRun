from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = REPO_ROOT / "scripts/lib/collect_minimax_h3_sage_shapes.py"
SPEC = importlib.util.spec_from_file_location("collect_minimax_h3_sage_shapes", COLLECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def sage_log(world_size: int, sequence: int = 42196) -> str:
    local_heads = 56 // world_size
    lines = []
    for rank in range(world_size):
        lines.append(
            "[Runtime] kind=dispatch model=minimax_h3 stage=H3-Core "
            "component=dit.core.attention status=active "
            "backend=aiter_fav3_sage_bf16_attn "
            f"rank={rank}/{world_size} causal=false kernel=sage_attention2 "
            "layout=BSHD numerics=int8_qk_fp8_pv "
            f"route=sage_attn_v2_gfx1201 shape=[1,{sequence},{local_heads},128]"
        )
    return "\n".join(lines) + "\n"


def test_collector_builds_exact_sp4_sp8_summary(tmp_path):
    sp4 = tmp_path / "sp4.log"
    sp8 = tmp_path / "sp8.log"
    sp4.write_text(sage_log(4), encoding="utf-8")
    sp8.write_text(sage_log(8), encoding="utf-8")

    payload = COLLECTOR.build_summary([(4, sp4), (8, sp8)])

    assert payload["schema_name"] == "lightx2v.minimax_h3.sage_shape_sweep"
    assert payload["sp_sizes"] == [4, 8]
    assert payload["sequence_length"] == 42196
    assert payload["source"].endswith("schema-2 runcollector disabled")
    assert [run["shape_bshd"] for run in payload["runs"]] == [
        [1, 42196, 14, 128],
        [1, 42196, 7, 128],
    ]
    assert [run["rank_count"] for run in payload["runs"]] == [4, 8]


def test_collector_rejects_missing_rank(tmp_path):
    path = tmp_path / "sp4.log"
    path.write_text("\n".join(sage_log(4).splitlines()[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing Sage dispatches for ranks \[3\]"):
        COLLECTOR.collect_log(path, 4)


def test_collector_rejects_cross_topology_sequence_disagreement(tmp_path):
    sp4 = tmp_path / "sp4.log"
    sp8 = tmp_path / "sp8.log"
    sp4.write_text(sage_log(4), encoding="utf-8")
    sp8.write_text(sage_log(8, sequence=12345), encoding="utf-8")

    with pytest.raises(ValueError, match="do not share one global sequence length"):
        COLLECTOR.build_summary([(4, sp4), (8, sp8)])


def test_collector_cli_writes_json(tmp_path):
    sp4 = tmp_path / "sp4.log"
    output = tmp_path / "sage_shapes.json"
    sp4.write_text(sage_log(4), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR_PATH),
            "--run",
            f"4={sp4}",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SP4: [1, 42196, 14, 128] (4 ranks)" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["runs"][0]["local_heads"] == 14
