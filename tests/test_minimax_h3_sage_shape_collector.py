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


def ring_sage_log(local_query_sequence: int = 4951) -> str:
    lines = []
    for rank in range(16):
        lines.append(
            "[Runtime] kind=dispatch model=minimax_h3 stage=H3-Core "
            "component=dit.core.attention status=active "
            "backend=aiter_fav3_sage_bf16_attn "
            f"rank={rank}/16 causal=false kernel=sage_attention2 "
            "layout=BSHD numerics=int8_qk_fp8_pv return_lse=true "
            "route=sage_attn_v2_gfx1201_ring_hybrid "
            f"shape=[1,{local_query_sequence},56,128]"
        )
    return "\n".join(lines) + "\n"


def test_collector_builds_exact_sp4_sp8_ring16_summary(tmp_path):
    sp4 = tmp_path / "sp4.log"
    sp8 = tmp_path / "sp8.log"
    ring16 = tmp_path / "ring16.log"
    sp4.write_text(sage_log(4), encoding="utf-8")
    sp8.write_text(sage_log(8), encoding="utf-8")
    ring16.write_text(ring_sage_log(), encoding="utf-8")

    payload = COLLECTOR.build_summary([(4, sp4), (8, sp8), (16, ring16)])

    assert payload["schema_name"] == "lightx2v.minimax_h3.sage_shape_sweep"
    assert payload["schema_version"] == 2
    assert payload["sp_sizes"] == [4, 8, 16]
    assert payload["sequence_length"] == 42196
    assert payload["source"].endswith("schema-2 runcollector disabled")
    assert [run["shape_bshd"] for run in payload["runs"]] == [
        [1, 42196, 14, 128],
        [1, 42196, 7, 128],
        [1, 4951, 56, 128],
    ]
    assert [run["rank_count"] for run in payload["runs"]] == [4, 8, 16]
    ring = payload["runs"][2]
    assert ring["topology"] == "ring"
    assert ring["dispatch_sequence_scope"] == "local_query_with_replicated_aux"
    assert ring["aux_sequence_length"] == 2468
    assert ring["main_shard_sequence_length"] == 2483
    assert ring["kernel_contracts"] == [
        {
            "name": "ring_nonfinal_aux_rectangular",
            "q_shape_bshd": [1, 2468, 56, 128],
            "kv_shape_bshd": [1, 2483, 56, 128],
            "calls_per_attention": 15,
            "backend_config": None,
        },
        {
            "name": "ring_nonfinal_main_square",
            "q_shape_bshd": [1, 2483, 56, 128],
            "kv_shape_bshd": [1, 2483, 56, 128],
            "calls_per_attention": 15,
            "backend_config": {"backend": "sage_attn_v2_gfx1201"},
        },
        {
            "name": "ring_final_combined_square",
            "q_shape_bshd": [1, 4951, 56, 128],
            "kv_shape_bshd": [1, 4951, 56, 128],
            "calls_per_attention": 1,
            "backend_config": {"backend": "sage_attn_v2_gfx1201"},
        },
    ]


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


def test_collector_rejects_ring_without_global_sequence_reference(tmp_path):
    ring16 = tmp_path / "ring16.log"
    ring16.write_text(ring_sage_log(), encoding="utf-8")

    with pytest.raises(ValueError, match="requires at least one dense/Ulysses reference"):
        COLLECTOR.build_summary([(16, ring16)])


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
    assert "SP4: dispatch=[1, 42196, 14, 128] (4 ranks)" in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["runs"][0]["local_heads"] == 14
