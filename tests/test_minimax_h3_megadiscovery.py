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

    ring16 = COLLECTIVES.build_traffic_contract(16)
    assert ring16["local_main_tokens"] == 2_483
    assert ring16["one_k_or_v_payload_bytes"] == 35_596_288
    assert ring16["aux_broadcast_payload_bytes"] == 35_381_248
    assert ring16["ring_hops_per_block"] == 15
    assert ring16["ring_rounds_per_run"] == 3_000
    assert ring16["p2p_send_recv_ops_per_block"] == 60
    assert ring16["p2p_send_recv_ops_per_run"] == 12_000
    assert ring16["logical_send_bytes_per_run_per_rank"] == 213_577_728_000


def test_analyzer_builds_ring16_diagnosis_and_finds_slow_neighbor(tmp_path):
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
        benchmark("ring16_full_rotation_separate_kv", 100.0),
        benchmark("ring16_full_rotation_fused_kv", 80.0),
        benchmark("ring16_aux_broadcast", 2.0),
    ]
    benches.extend(
        benchmark(
            f"ring16_edge_{source:02d}_to_{(source + 1) % 16:02d}",
            20.0 if source == 7 else 10.0,
        )
        for source in range(16)
    )
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
        sysfs_path = f"/sys/devices/pci0000:00/{upstream}/{pci_address}"
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
        "node:123:123 [0] NCCL INFO P2P Channel 00/0 : 0[0] -> 1[1] via IPC\n",
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
    edges = report["ring_edge_discovery"]
    assert edges["measured_edge_count"] == 16
    assert edges["slowest_edge"]["source_local_rank"] == 7
    assert edges["slowest_over_median"] == pytest.approx(2.0)
    assert report["opportunities"][1]["decision"] == "candidate"
    assert report["system"]["vmstat_delta"]["pgmajfault"] == 3
    assert report["system"]["rccl"]["transport_evidence_present"] is True
    markdown = (tmp_path / "megadiscovery.md").read_text(encoding="utf-8")
    assert "Current-order neighbor edges measured: `16`" in markdown
    assert "Production configuration changed: `false`" in markdown
