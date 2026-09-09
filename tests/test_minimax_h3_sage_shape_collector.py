from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = REPO_ROOT / "scripts/collector/collect_minimax_h3_sage_shapes.py"
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


def hybrid_sage_log(local_query_sequence: int = 21098) -> str:
    lines = []
    for rank in range(16):
        lines.append(
            "[Runtime] kind=dispatch model=minimax_h3 stage=H3-Core "
            "component=dit.core.attention status=active "
            "backend=aiter_fav3_sage_bf16_attn "
            f"rank={rank}/16 causal=false kernel=sage_attention2 "
            "layout=BSHD numerics=int8_qk_fp8_pv return_lse=true "
            "route=sage_attn_v2_gfx1201_blockwise_aux_split "
            f"shape=[1,{local_query_sequence},7,128]"
        )
    return "\n".join(lines) + "\n"


def test_collector_builds_exact_sp4_sp8_cp2_ulysses8_summary(tmp_path):
    sp4 = tmp_path / "sp4.log"
    sp8 = tmp_path / "sp8.log"
    hybrid16 = tmp_path / "hybrid16.log"
    sp4.write_text(sage_log(4), encoding="utf-8")
    sp8.write_text(sage_log(8), encoding="utf-8")
    hybrid16.write_text(hybrid_sage_log(), encoding="utf-8")

    payload = COLLECTOR.build_summary([(4, sp4), (8, sp8), (16, hybrid16)])

    assert payload["schema_name"] == "lightx2v.minimax_h3.sage_shape_sweep"
    assert payload["schema_version"] == 2
    assert payload["sp_sizes"] == [4, 8, 16]
    assert payload["sequence_length"] == 42196
    assert payload["source"].startswith("normalized runtime dispatch")
    assert [run["shape_bshd"] for run in payload["runs"]] == [
        [1, 42196, 14, 128],
        [1, 42196, 7, 128],
        [1, 21098, 7, 128],
    ]
    assert [run["rank_count"] for run in payload["runs"]] == [4, 8, 16]
    hybrid = payload["runs"][2]
    assert hybrid["topology"] == "ulysses_ring_2d"
    assert hybrid["dispatch_sequence_scope"] == "context_local_query"
    assert hybrid["aux_sequence_length"] == 4
    assert hybrid["local_aux_query_sequence_length"] == 2
    assert hybrid["main_shard_sequence_length"] == 2637
    assert hybrid["ulysses_main_sequence_length"] == 21096
    assert hybrid["kernel_contracts"] == [
        {
            "name": "cp2_nonfinal_aux_rectangular",
            "q_shape_bshd": [1, 2, 7, 128],
            "kv_shape_bshd": [1, 21096, 7, 128],
            "calls_per_attention": 1,
            "backend_config": None,
        },
        {
            "name": "cp2_nonfinal_main_square",
            "q_shape_bshd": [1, 21096, 7, 128],
            "kv_shape_bshd": [1, 21096, 7, 128],
            "calls_per_attention": 1,
            "backend_config": {"backend": "sage_attn_v2_gfx1201"},
        },
        {
            "name": "cp2_final_combined_square",
            "q_shape_bshd": [1, 21100, 7, 128],
            "kv_shape_bshd": [1, 21100, 7, 128],
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


def test_collector_rejects_hybrid_without_global_sequence_reference(tmp_path):
    hybrid16 = tmp_path / "hybrid16.log"
    hybrid16.write_text(hybrid_sage_log(), encoding="utf-8")

    with pytest.raises(ValueError, match="requires at least one dense/Ulysses reference"):
        COLLECTOR.build_summary([(16, hybrid16)])


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


@pytest.mark.parametrize("remainder", [0, 1, 3, 4, 15])
def test_ref2v_cp2_shapes_follow_odd_aux_and_minimum_prefix(tmp_path, remainder):
    total = 112000 + remainder
    aux = remainder if remainder >= 2 else remainder + 16
    main = (total - aux) // 2
    lines = []
    for rank in range(16):
        local_aux = aux // 2 + int(rank < 8 and aux % 2 != 0)
        line = hybrid_sage_log(main + local_aux).splitlines()[rank]
        lines.append(line)
    lines.append(f"[Runtime] kind=state model=minimax_h3 stage=H3-Prepare "
                 f"component=dit.sequence_partition status=active backend=ulysses_ring "
                 f"rank=0/16 total_rows={total} aux_rows={aux}")
    path = tmp_path / "ref16.log"
    path.write_text("\n".join(lines))
    result = COLLECTOR.build_summary([(16, path)])
    assert result["global_sequence_length"] == total
    run = result["runs"][0]
    assert run["aux_sequence_length"] == aux
    assert run["ulysses_main_sequence_length"] == main
    contracts = run["kernel_contracts"]
    if aux % 2:
        assert contracts[0]["q_shape_bshd"][1] == (aux + 1) // 2
        assert contracts[1]["q_shape_bshd"][1] == aux // 2
        assert contracts[1]["ranks"] == list(range(8, 16))
    assert contracts[-1]["q_shape_bshd"][1] == main + aux
