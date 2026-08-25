from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTIVE_SCRIPT = REPO_ROOT / "scripts/collector/benchmark_minimax_h3_collectives.py"
ANALYZER_SCRIPT = REPO_ROOT / "scripts/collector/analyze_minimax_h3_megadiscovery.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECTIVES = load_module("benchmark_minimax_h3_collectives", COLLECTIVE_SCRIPT)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def benchmark(name: str, mean_ms: float, gbps: float = 20.0) -> dict:
    return {
        "name": name,
        "critical_path": {"mean_ms": mean_ms},
        "effective_logical_GBps": gbps,
    }


def test_exact_shape_traffic_contracts_match_production_calls():
    sp8 = COLLECTIVES.build_traffic_contract(8)
    assert sp8["local_main_tokens"] == 4_966
    assert sp8["one_a2a_payload_bytes"] == 71_192_576
    assert sp8["collective_calls_per_block"] == 5
    assert sp8["collective_calls_per_run"] == 1_000
    assert sp8["estimated_remote_bytes_per_run_per_rank"] == 56_026_521_600

    hybrid16 = COLLECTIVES.build_traffic_contract(16)
    assert hybrid16["topology"] == "cp2_ulysses8_sp16"
    assert hybrid16["local_main_tokens"] == 2_637
    assert hybrid16["ulysses_main_tokens"] == 21_096
    assert hybrid16["aux_tokens"] == 4
    assert hybrid16["one_ulysses_a2a_payload_bytes"] == 37_804_032
    assert hybrid16["one_context_k_or_v_payload_bytes"] == 37_804_032
    assert hybrid16["context_ring_hops_per_block"] == 1
    assert hybrid16["context_ring_rounds_per_run"] == 200
    assert hybrid16["p2p_send_recv_ops_per_block"] == 4
    assert hybrid16["p2p_send_recv_ops_per_run"] == 800
    assert hybrid16["estimated_remote_bytes_per_run_per_rank"] == 41_595_187_200


def test_ring_edges_are_split_into_two_disjoint_all_rank_matchings():
    matchings = COLLECTIVES.build_ring_edge_matchings(16)
    assert len(matchings) == 2
    assert all(len(matching) == 8 for matching in matchings)

    for matching in matchings:
        participating_ranks = [rank for edge in matching for rank in edge]
        assert sorted(participating_ranks) == list(range(16))

    assert {edge for matching in matchings for edge in matching} == {
        (source, (source + 1) % 16) for source in range(16)
    }


def test_ring_edge_matching_requires_an_even_world_size():
    with pytest.raises(ValueError, match="even world size"):
        COLLECTIVES.build_ring_edge_matchings(15)


def test_failure_record_preserves_completed_checkpoints(tmp_path):
    output = tmp_path / "megadiscovery_collectives.json"
    completed = benchmark("hybrid16_ulysses_one_a2a", 10.0)
    write_json(
        output,
        {
            "schema_name": COLLECTIVES.SCHEMA_NAME,
            "schema_version": COLLECTIVES.SCHEMA_VERSION,
            "probe_started_at": "2026-08-25T02:11:04+00:00",
            "status": "running",
            "active_benchmark": "hybrid16_block_fused_transport",
            "environment": {"world_size": 16, "torch_version": "test"},
            "traffic_contract": COLLECTIVES.build_traffic_contract(16),
            "benchmarks": [completed],
        },
    )

    COLLECTIVES.write_failure(output, 16, "torchrun status 124")

    failed = json.loads(output.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["active_benchmark"] == "hybrid16_block_fused_transport"
    assert failed["completed_benchmark_count"] == 1
    assert failed["benchmarks"] == [completed]
    assert failed["environment"]["torch_version"] == "test"


def test_analyzer_builds_cp2_ulysses8_sp16_diagnosis(tmp_path):
    write_json(tmp_path / "run_manifest.json", {"gpu_count": 16})
    write_json(
        tmp_path / "h3_t1_summary.json",
        {
            "world_size": 16,
            "critical_path_totals_ms": {
                "h3_step_gpu_ms": 100_000.0,
                "sp_layout_and_communication_gpu_ms": 35_000.0,
                "exposed_h2d_gpu_ms": 1_000.0,
                "dense_attention_gpu_ms": 20_000.0,
            },
            "per_rank_totals_ms": {
                str(rank): {"h3_step_gpu_ms": 100_000.0 + rank * 10.0}
                for rank in range(16)
            },
        },
    )
    benches = [
        benchmark("hybrid16_block_current_transport", 100.0),
        benchmark("hybrid16_block_fused_transport", 80.0),
        benchmark("hybrid16_context_kv_separate", 10.0),
        benchmark("hybrid16_context_kv_fused", 8.0),
    ]
    write_json(
        tmp_path / "megadiscovery_collectives.json",
        {
            "environment": {"world_size": 16},
            "traffic_contract": COLLECTIVES.build_traffic_contract(16),
            "benchmarks": benches,
        },
    )

    devices = []
    mapping = []
    for rank in range(16):
        upstream = "0000:00:01.0" if rank < 8 else "0000:80:01.0"
        pci_address = f"0000:{rank + 1:02x}:00.0"
        root_bus = "00" if rank < 8 else "80"
        sysfs_path = f"/sys/devices/pci0000:{root_bus}/{upstream}/{pci_address}"
        devices.append(
            {
                "rocm_index": rank,
                "total_bytes": 32 * 2**30,
                "pci": {"sysfs_path": sysfs_path},
            }
        )
        mapping.append(
            {
                "local_rank": rank,
                "rocm_index": rank,
                "pci_address": pci_address,
                "pci_upstream": upstream,
                "numa_node": 0 if rank < 8 else 1,
                "iommu_group": rank,
            }
        )
    host_base = {
        "host": {
            "memory": {"total_bytes": 512 * 2**30},
            "numa_balancing": 1,
            "kfd_topology": {"available": True, "nodes": []},
            "vmstat": {"pgmajfault": 4, "pswpin": 0},
        },
        "gpus": {"devices": devices, "local_rank_mapping": mapping},
        "paths": {"model": {"mount": {"filesystems": []}}},
    }
    host_after = json.loads(json.dumps(host_base))
    host_after["host"]["vmstat"] = {"pgmajfault": 7, "pswpin": 0}
    write_json(tmp_path / "megadiscovery_host_before.json", host_base)
    write_json(tmp_path / "megadiscovery_host_after.json", host_after)
    write_json(
        tmp_path / "memory.jsonl.summary.json",
        {
            "phase_maxima": {
                "h3": {
                    "cgroup_current_bytes": 180 * 2**30,
                    "max_gpu_used_bytes": 12 * 2**30,
                }
            },
            "host_counter_window": {"deltas": {"vmstat": {"pgmajfault": 3}}},
            "rank_lifecycle": {"0": {"process_counter_deltas": {"proc_stat": {"majflt": 1}}}},
        },
    )
    write_json(tmp_path / "summary.json", {"total_cost_seconds": 196.0})
    (tmp_path / "rccl.node.123.log").write_text(
        "node:123:123 [0] NCCL INFO ncclCommInitRankConfig_impl rank 0 nranks 16 "
        "cudaDev 0 - Init COMPLETE\n"
        "node:123:123 [0] NCCL INFO Channel 00/02 : 0 1 2 3 4 5 6 7 12 13 14 15 8 9 10 11\n"
        "node:123:123 [0] NCCL INFO P2P Channel 00/0 : 0[0] -> 1[1] via P2P/IPC comm 0x1\n"
        "node:123:123 [0] NCCL INFO Send: opCount 7 count 17798144\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["python3", str(ANALYZER_SCRIPT), "--result-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "megadiscovery.json").read_text(encoding="utf-8"))
    assert report["diagnosis"]["communication_dominant_under_current_lazy_policy"] is True
    assert report["fusion_discovery"]["synthetic_speedup"] == pytest.approx(1.25)
    assert report["fusion_discovery"]["context_synthetic_speedup"] == pytest.approx(1.25)
    assert report["profile"] == "cp2_ulysses8_sp16"
    assert report["ring_edge_discovery"] is None
    assert report["opportunities"][1]["decision"] == "inspect_subgroup_placement"
    assert report["system"]["pcie_root_complex_groups"] == {
        "pci0000:00": list(range(8)),
        "pci0000:80": list(range(8, 16)),
    }
    assert report["system"]["vmstat_delta"]["pgmajfault"] == 3
    assert report["system"]["rccl"]["transport_evidence_present"] is True
    assert report["system"]["rccl"]["init_complete_count"] == 1
    assert report["system"]["rccl"]["connection_transports"] == {"P2P/IPC": 1}
    assert report["system"]["rccl"]["net_fallback_connection_count"] == 0
    assert report["system"]["rccl"]["channel_orders"] == [
        [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11]
    ]
    assert report["system"]["rccl"]["max_p2p_op_count_by_rank"] == {"0": 7}
    markdown = (tmp_path / "megadiscovery.md").read_text(encoding="utf-8")
    assert "Topology: `CP2 x Ulysses8`" in markdown
    assert "Production configuration changed: `false`" in markdown

    failure = subprocess.run(
        [
            "python3",
            str(COLLECTIVE_SCRIPT),
            "--output",
            str(tmp_path / "megadiscovery_collectives.json"),
            "--world-size",
            "16",
            "--record-failure",
            "torchrun status 124; see megadiscovery_collectives.log",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failure.returncode == 0, failure.stdout + failure.stderr
    partial_run = subprocess.run(
        ["python3", str(ANALYZER_SCRIPT), "--result-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert partial_run.returncode == 0, partial_run.stdout + partial_run.stderr
    partial = json.loads((tmp_path / "megadiscovery.json").read_text(encoding="utf-8"))
    assert partial["status"] == "partial"
    assert partial["collective_probe_status"] == "failed"
    assert len(partial["collective_benchmarks"]) == len(benches)
    assert partial["opportunities"][0]["decision"] == "candidate"
    partial_markdown = (tmp_path / "megadiscovery.md").read_text(encoding="utf-8")
    assert "Report status: `partial`" in partial_markdown
    assert "Production and schema-2 evidence remain available" in partial_markdown


def test_sp16_missing_dense_attention_keeps_communication_attribution_unknown(
    tmp_path,
):
    write_json(tmp_path / "run_manifest.json", {"gpu_count": 16})
    write_json(
        tmp_path / "h3_t1_summary.json",
        {
            "world_size": 16,
            "critical_path_totals_ms": {
                "h3_step_gpu_ms": 120_000.0,
                "sp_layout_and_communication_gpu_ms": 100_000.0,
                "exposed_h2d_gpu_ms": 0.0,
                "dense_attention_gpu_ms": 0.0,
            },
            "per_rank_totals_ms": {
                str(rank): {"h3_step_gpu_ms": 120_000.0 + rank}
                for rank in range(16)
            },
        },
    )
    write_json(
        tmp_path / "megadiscovery_collectives.json",
        {
            "status": "failed",
            "active_benchmark": "hybrid16_ulysses_one_a2a",
            "error": "torchrun status 124",
            "environment": {"world_size": 16},
            "traffic_contract": COLLECTIVES.build_traffic_contract(16),
            "benchmarks": [],
        },
    )
    host = {
        "host": {"memory": {"total_bytes": 1}, "vmstat": {}},
        "gpus": {"devices": [], "local_rank_mapping": []},
    }
    write_json(tmp_path / "megadiscovery_host_before.json", host)
    write_json(tmp_path / "megadiscovery_host_after.json", host)

    completed = subprocess.run(
        ["python3", str(ANALYZER_SCRIPT), "--result-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "megadiscovery.json").read_text(encoding="utf-8"))
    diagnosis = report["diagnosis"]
    assert diagnosis["sequence_parallel_envelope_dominant_under_current_lazy_policy"] is True
    assert diagnosis["ring_dense_attention_timing_available"] is False
    assert diagnosis["communication_dominant_under_current_lazy_policy"] is None
    assert (
        diagnosis["communication_attribution_status"]
        == "unresolved_2d_attention_nested_in_envelope"
    )
    markdown = (tmp_path / "megadiscovery.md").read_text(encoding="utf-8")
    assert "Communication dominant under the current lazy policy: `unknown`" in markdown
