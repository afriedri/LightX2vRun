#!/usr/bin/env python3
"""Build a unified MiniMax-H3 MegaDiscovery diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_NAME = "lightx2v.minimax_h3.megadiscovery"
SCHEMA_VERSION = 1


def load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise ValueError(f"required MegaDiscovery artifact is missing: {path.name}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"MegaDiscovery artifact must contain a JSON object: {path.name}")
    return payload


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def benchmark_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in payload.get("benchmarks", [])}


def critical_mean(benchmark: dict[str, Any] | None) -> float | None:
    if not benchmark:
        return None
    value = (benchmark.get("critical_path") or {}).get("mean_ms")
    return float(value) if value is not None else None


def parse_rccl_logs(paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "p2p_or_ipc": re.compile(r"\b(?:P2P|IPC)\b", re.IGNORECASE),
        "shared_memory": re.compile(r"\bSHM\b|shared memory", re.IGNORECASE),
        "network_transport": re.compile(r"\bNET/|network", re.IGNORECASE),
        "graph_or_topology": re.compile(r"graph|topolog", re.IGNORECASE),
        "channel": re.compile(r"channel", re.IGNORECASE),
        "warning": re.compile(r"\bWARN(?:ING)?\b|\bERROR\b|failed|failure", re.IGNORECASE),
    }
    counts = {name: 0 for name in patterns}
    excerpts = []
    connection_transports: dict[str, int] = {}
    channel_orders: set[tuple[int, ...]] = set()
    initialized_ranks: set[int] = set()
    max_p2p_op_count_by_rank: dict[str, int] = {}
    numa_balancing_warning_count = 0
    for path in paths:
        file_rank: int | None = None
        file_max_op_count: int | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            init_match = re.search(r"\brank ([0-9]+) nranks [0-9]+ .* Init COMPLETE", line)
            if init_match:
                file_rank = int(init_match.group(1))
                initialized_ranks.add(file_rank)
            op_count_match = re.search(r"\b(?:Send|Recv): opCount ([0-9]+)\b", line)
            if op_count_match:
                op_count = int(op_count_match.group(1))
                file_max_op_count = max(file_max_op_count or 0, op_count)
            connection_match = re.search(r"\bvia ([^ ]+) comm\b", line)
            if connection_match:
                transport = connection_match.group(1)
                connection_transports[transport] = connection_transports.get(transport, 0) + 1
            channel_match = re.search(
                r"\bChannel [0-9]+/[0-9]+\s*:\s*((?:[0-9]+\s+)+[0-9]+)\s*$",
                line,
            )
            if channel_match:
                channel_orders.add(tuple(int(item) for item in channel_match.group(1).split()))
            if "NUMA auto balancing enabled" in line:
                numa_balancing_warning_count += 1
            matched = False
            for name, pattern in patterns.items():
                if pattern.search(line):
                    counts[name] += 1
                    matched = True
            if matched and len(excerpts) < 80:
                excerpts.append({"file": path.name, "line": line[:1000]})
        if file_rank is not None and file_max_op_count is not None:
            max_p2p_op_count_by_rank[str(file_rank)] = file_max_op_count
    return {
        "files": [path.name for path in paths],
        "file_count": len(paths),
        "keyword_counts": counts,
        "excerpts": excerpts,
        "initialized_ranks": sorted(initialized_ranks),
        "init_complete_count": len(initialized_ranks),
        "connection_transports": dict(sorted(connection_transports.items())),
        "net_fallback_connection_count": sum(
            count
            for transport, count in connection_transports.items()
            if transport.upper().startswith("NET")
        ),
        "channel_orders": [list(order) for order in sorted(channel_orders)],
        "max_p2p_op_count_by_rank": max_p2p_op_count_by_rank,
        "numa_balancing_warning_count": numa_balancing_warning_count,
        "transport_evidence_present": bool(
            connection_transports
            or counts["p2p_or_ipc"]
            or counts["shared_memory"]
            or counts["network_transport"]
        ),
    }


def pci_groups(host: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for gpu in (host.get("gpus") or {}).get("devices", []):
        sysfs_path = ((gpu.get("pci") or {}).get("sysfs_path"))
        if not sysfs_path:
            continue
        path = Path(str(sysfs_path))
        upstream = path.parent.name
        groups.setdefault(upstream, []).append(int(gpu["rocm_index"]))
    return {key: sorted(value) for key, value in sorted(groups.items())}


def pcie_root_complex(sysfs_path: str | None) -> str | None:
    if not sysfs_path:
        return None
    for component in Path(sysfs_path).parts:
        if re.fullmatch(r"pci[0-9a-fA-F]{4}:[0-9a-fA-F]{2}", component):
            return component
    return None


def pcie_root_complex_groups(host: dict[str, Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for gpu in (host.get("gpus") or {}).get("devices", []):
        root_complex = pcie_root_complex((gpu.get("pci") or {}).get("sysfs_path"))
        if root_complex is None:
            continue
        groups.setdefault(root_complex, []).append(int(gpu["rocm_index"]))
    return {key: sorted(value) for key, value in sorted(groups.items())}


def ring_edge_discovery(
    benches: dict[str, dict[str, Any]], host: dict[str, Any]
) -> dict[str, Any]:
    pattern = re.compile(r"ring16_edge_([0-9]{2})_to_([0-9]{2})")
    devices = {
        int(item["rocm_index"]): item
        for item in (host.get("gpus") or {}).get("devices", [])
    }
    mapping = {
        int(item["local_rank"]): {
            **item,
            "pcie_root_complex": pcie_root_complex(
                ((devices.get(int(item["rocm_index"])) or {}).get("pci") or {}).get(
                    "sysfs_path"
                )
            ),
        }
        for item in (host.get("gpus") or {}).get("local_rank_mapping", [])
    }
    edges = []
    for name, benchmark in benches.items():
        match = pattern.fullmatch(name)
        if not match:
            continue
        source = int(match.group(1))
        destination = int(match.group(2))
        source_gpu = mapping.get(source) or {}
        destination_gpu = mapping.get(destination) or {}
        latency_ms = critical_mean(benchmark)
        edges.append(
            {
                "source_local_rank": source,
                "destination_local_rank": destination,
                "source_rocm_index": source_gpu.get("rocm_index"),
                "destination_rocm_index": destination_gpu.get("rocm_index"),
                "source_pci_address": source_gpu.get("pci_address"),
                "destination_pci_address": destination_gpu.get("pci_address"),
                "source_pcie_root_complex": source_gpu.get("pcie_root_complex"),
                "destination_pcie_root_complex": destination_gpu.get("pcie_root_complex"),
                "same_pcie_root_complex": (
                    source_gpu.get("pcie_root_complex")
                    == destination_gpu.get("pcie_root_complex")
                    if source_gpu.get("pcie_root_complex") is not None
                    and destination_gpu.get("pcie_root_complex") is not None
                    else None
                ),
                "same_pci_upstream": (
                    source_gpu.get("pci_upstream") == destination_gpu.get("pci_upstream")
                    if source_gpu.get("pci_upstream") is not None
                    and destination_gpu.get("pci_upstream") is not None
                    else None
                ),
                "same_numa_node": (
                    source_gpu.get("numa_node") == destination_gpu.get("numa_node")
                    if source_gpu.get("numa_node") is not None
                    and destination_gpu.get("numa_node") is not None
                    else None
                ),
                "critical_mean_ms": latency_ms,
                "effective_logical_GBps": benchmark.get("effective_logical_GBps"),
            }
        )
    edges.sort(key=lambda item: item["source_local_rank"])
    measured = [item for item in edges if item["critical_mean_ms"] is not None]
    latencies = [float(item["critical_mean_ms"]) for item in measured]
    median_ms = statistics.median(latencies) if latencies else None
    slowest = max(measured, key=lambda item: float(item["critical_mean_ms"])) if measured else None
    fastest = min(measured, key=lambda item: float(item["critical_mean_ms"])) if measured else None
    return {
        "rank_mapping": [mapping[index] for index in sorted(mapping)],
        "edges": edges,
        "measured_edge_count": len(measured),
        "median_edge_ms": median_ms,
        "slowest_edge": slowest,
        "fastest_edge": fastest,
        "slowest_over_median": safe_ratio(
            float((slowest or {}).get("critical_mean_ms") or 0.0),
            float(median_ms or 0.0),
        ),
    }


def build_report(result_dir: Path) -> dict[str, Any]:
    manifest = load_json(result_dir / "run_manifest.json")
    t1 = load_json(result_dir / "h3_t1_summary.json")
    collective = load_json(result_dir / "megadiscovery_collectives.json")
    host_before = load_json(result_dir / "megadiscovery_host_before.json")
    host_after = load_json(result_dir / "megadiscovery_host_after.json")
    memory = load_json(result_dir / "memory.jsonl.summary.json", required=False)
    production = load_json(result_dir / "summary.json", required=False)
    assert manifest is not None and t1 is not None and collective is not None
    assert host_before is not None and host_after is not None

    world_size = int(t1["world_size"])
    if world_size not in (8, 16):
        raise ValueError(f"MegaDiscovery requires world size 8 or 16, got {world_size}")
    if int(manifest.get("gpu_count") or 0) != world_size:
        raise ValueError("run manifest and T1 summary world sizes disagree")
    if int(((collective.get("environment") or {}).get("world_size")) or 0) != world_size:
        raise ValueError("collective benchmark and T1 summary world sizes disagree")
    topology = "cp2_ulysses8_sp16" if world_size == 16 else "ulysses_sp8"
    totals = t1.get("critical_path_totals_ms") or {}
    h3_ms = float(totals.get("h3_step_gpu_ms") or 0.0)
    sp_ms = float(totals.get("sp_layout_and_communication_gpu_ms") or 0.0)
    exposed_h2d_ms = float(totals.get("exposed_h2d_gpu_ms") or 0.0)
    dense_attention_ms = float(totals.get("dense_attention_gpu_ms") or 0.0)
    sp_share = safe_ratio(sp_ms, h3_ms)
    exposed_share = safe_ratio(exposed_h2d_ms, h3_ms)

    rank_totals = {
        int(rank): float((metrics or {}).get("h3_step_gpu_ms") or 0.0)
        for rank, metrics in (t1.get("per_rank_totals_ms") or {}).items()
    }
    rank_values = list(rank_totals.values())
    rank_median = statistics.median(rank_values) if rank_values else 0.0
    rank_spread = max(rank_values) - min(rank_values) if rank_values else 0.0

    benches = benchmark_by_name(collective)
    collective_status = str(collective.get("status") or "complete")
    collective_active_benchmark = collective.get("active_benchmark")
    fusion = None
    projected = {}
    edge_discovery: dict[str, Any] | None = None
    if world_size == 16:
        current_ms = critical_mean(benches.get("hybrid16_block_current_transport"))
        fused_ms = critical_mean(benches.get("hybrid16_block_fused_transport"))
        context_separate_ms = critical_mean(benches.get("hybrid16_context_kv_separate"))
        context_fused_ms = critical_mean(benches.get("hybrid16_context_kv_fused"))
        fusion = {
            "current_2d_mean_ms": current_ms,
            "fused_2d_mean_ms": fused_ms,
            "synthetic_speedup": safe_ratio(current_ms or 0.0, fused_ms or 0.0),
            "context_separate_kv_mean_ms": context_separate_ms,
            "context_fused_kv_mean_ms": context_fused_ms,
            "context_synthetic_speedup": safe_ratio(
                context_separate_ms or 0.0,
                context_fused_ms or 0.0,
            ),
            "bytes_changed": False,
            "context_hops_changed": False,
            "context_p2p_ops_per_hop": {"current": 4, "fused": 2},
        }
        current_ceiling_ms = current_ms * 200 if current_ms is not None else None
        fused_ceiling_ms = fused_ms * 200 if fused_ms is not None else None
        projected = {
            "current_2d_transport_ceiling_ms_for_200_blocks": current_ceiling_ms,
            "fused_2d_transport_ceiling_ms_for_200_blocks": fused_ceiling_ms,
            "synthetic_2d_fusion_ceiling_delta_ms_for_200_blocks": (
                current_ceiling_ms - fused_ceiling_ms
                if current_ceiling_ms is not None and fused_ceiling_ms is not None
                else None
            ),
            "current_isolated_share_of_observed_envelope": safe_ratio(
                float(current_ceiling_ms or 0.0), sp_ms
            ),
            "non_additive": True,
            "guardrail": "The 2D transport is interleaved with layout and attention; isolated projections are ceilings, not production savings.",
        }
    else:
        current_ms = critical_mean(benches.get("ulysses_block_current_transport"))
        fused_ms = critical_mean(benches.get("ulysses_block_qkv_fused_transport"))
        fusion = {
            "current_mean_ms": current_ms,
            "qkv_fused_mean_ms": fused_ms,
            "synthetic_speedup": safe_ratio(current_ms or 0.0, fused_ms or 0.0),
            "bytes_changed": False,
        }
        current_projection = current_ms * 200 if current_ms is not None else None
        fused_projection = fused_ms * 200 if fused_ms is not None else None
        projected = {
            "current_transport_isolated_ms_for_200_blocks": current_projection,
            "qkv_fused_transport_isolated_ms_for_200_blocks": fused_projection,
            "current_isolated_share_of_observed_envelope": safe_ratio(
                float(current_projection or 0.0), sp_ms
            ),
            "guardrail": "Ulysses projections exclude layout kernels and model-time fabric contention.",
        }

    host_memory_total = float(((host_before.get("host") or {}).get("memory") or {}).get("total_bytes") or 0.0)
    h3_memory = ((memory or {}).get("phase_maxima") or {}).get("h3") or {}
    h3_cgroup = float(h3_memory.get("cgroup_current_bytes") or 0.0)
    h3_gpu = float(h3_memory.get("max_gpu_used_bytes") or 0.0)
    gpu_totals = [float(item.get("total_bytes") or 0.0) for item in (host_before.get("gpus") or {}).get("devices", [])]
    gpu_total = max(gpu_totals, default=0.0)

    before_vmstat = ((host_before.get("host") or {}).get("vmstat") or {})
    after_vmstat = ((host_after.get("host") or {}).get("vmstat") or {})
    vmstat_delta = {
        name: int(after_vmstat.get(name, 0)) - int(before_vmstat.get(name, 0))
        for name in sorted(set(before_vmstat) | set(after_vmstat))
    }
    model_mount = ((host_before.get("paths") or {}).get("model") or {}).get("mount")
    rccl = parse_rccl_logs(sorted(result_dir.glob("rccl.*.log")))
    groups = pci_groups(host_before)
    root_complex_groups = pcie_root_complex_groups(host_before)
    memory_host_deltas = ((memory or {}).get("host_counter_window") or {}).get("deltas") or {}
    rank_lifecycle = (memory or {}).get("rank_lifecycle") or {}

    sequence_parallel_envelope_dominant = bool(
        sp_share is not None
        and exposed_share is not None
        and sp_share >= 0.25
        and exposed_share <= 0.03
    )
    distributed_dense_attention_timing_available = not (
        world_size == 16 and dense_attention_ms <= 0.0
    )
    if distributed_dense_attention_timing_available:
        communication_dominant: bool | None = sequence_parallel_envelope_dominant
        communication_attribution_status = (
            "supported_by_collector_decomposition"
            if communication_dominant
            else "not_supported_by_collector_decomposition"
        )
    else:
        communication_dominant = None
        communication_attribution_status = "unresolved_2d_attention_nested_in_envelope"
    opportunities = []
    if world_size == 16:
        fusion_speedup = (fusion or {}).get("synthetic_speedup")
        opportunities.append(
            {
                "priority": 1,
                "name": "2d_qkv_and_kv_payload_fusion",
                "evidence": "isolated production-shape microbenchmark",
                "measured_synthetic_speedup": fusion_speedup,
                "production_change_applied": False,
                "decision": (
                    "probe_unavailable"
                    if fusion_speedup is None
                    else "candidate"
                    if fusion_speedup > 1.03
                    else "not_supported_by_microbenchmark"
                ),
            }
        )
        opportunities.append(
            {
                "priority": 2,
                "name": "2d_subgroup_rank_order_from_pcie_topology",
                "evidence": {
                    "pci_upstream_groups": groups,
                    "pcie_root_complex_groups": root_complex_groups,
                    "rccl_transport_evidence_present": rccl["transport_evidence_present"],
                    "ulysses_rows": [list(range(8)), list(range(8, 16))],
                    "context_pairs": [[rank, rank + 8] for rank in range(8)],
                },
                "production_change_applied": False,
                "decision": "inspect_subgroup_placement",
            }
        )
        opportunities.append(
            {
                "priority": 3,
                "name": "2d_online_merge_and_layout_fusion",
                "evidence": projected.get("current_2d_transport_ceiling_ms_for_200_blocks"),
                "production_change_applied": False,
                "decision": "measure_against_2d_envelope_before_code_change",
            }
        )

    input_failures = []
    if collective_status != "complete":
        input_failures.append(
            {
                "artifact": "megadiscovery_collectives.json",
                "status": collective_status,
                "error": collective.get("error"),
                "active_benchmark": collective_active_benchmark,
                "completed_benchmark_count": len(benches),
            }
        )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial" if input_failures else "complete",
        "input_failures": input_failures,
        "profile": topology,
        "world_size": world_size,
        "observer_only": True,
        "production_configuration_changed": False,
        "artifacts": sorted(path.name for path in result_dir.iterdir() if path.is_file()),
        "production_summary": production,
        "diagnosis": {
            "communication_dominant_under_current_lazy_policy": communication_dominant,
            "communication_attribution_status": communication_attribution_status,
            "h3_collector_gpu_ms": h3_ms,
            "sequence_parallel_envelope_gpu_ms": sp_ms,
            "sequence_parallel_envelope_share": sp_share,
            "sequence_parallel_envelope_dominant_under_current_lazy_policy": sequence_parallel_envelope_dominant,
            "dense_attention_gpu_ms": dense_attention_ms,
            "ring_dense_attention_timing_available": distributed_dense_attention_timing_available,
            "distributed_dense_attention_timing_available": distributed_dense_attention_timing_available,
            "exposed_h2d_gpu_ms": exposed_h2d_ms,
            "exposed_h2d_share": exposed_share,
            "rank_h3_total_ms": rank_totals,
            "rank_spread_ms": rank_spread,
            "rank_spread_over_median": safe_ratio(rank_spread, rank_median),
        },
        "traffic_contract": collective.get("traffic_contract"),
        "collective_benchmarks": collective.get("benchmarks"),
        "collective_probe_status": collective_status,
        "collective_probe_active_benchmark": collective_active_benchmark,
        "collective_probe_completed_benchmark_count": len(benches),
        "fusion_discovery": fusion,
        "ring_edge_discovery": edge_discovery,
        "isolated_projection": projected,
        "system": {
            "numa_balancing": (host_before.get("host") or {}).get("numa_balancing"),
            "pci_upstream_groups": groups,
            "pcie_root_complex_groups": root_complex_groups,
            "kfd_topology": (host_before.get("host") or {}).get("kfd_topology"),
            "rccl": rccl,
            "vmstat_delta": vmstat_delta,
            "memory_monitor_host_counter_deltas": memory_host_deltas,
            "rank_lifecycle": rank_lifecycle,
            "model_mount": model_mount,
            "h3_cgroup_bytes": h3_cgroup,
            "h3_cgroup_share_of_host": safe_ratio(h3_cgroup, host_memory_total),
            "h3_max_gpu_used_bytes": h3_gpu,
            "h3_gpu_used_share": safe_ratio(h3_gpu, gpu_total),
        },
        "opportunities": opportunities,
        "guardrails": [
            "Collector latency is not the production latency baseline.",
            "The sequence-parallel envelope includes layout, communication, online merge, and synchronization; it is not pure wire time.",
            "A zero distributed dense-attention timer means attention remains nested in the sequence-parallel envelope, so pure communication attribution is unknown.",
            "2D communication is interleaved with layout and attention, so isolated transport projections are non-additive.",
            "MegaDiscovery does not change BF16, lazy loading, CPU/GPU buffer counts, LoRA, TunableOp, attention kernels, or SP topology.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    diagnosis = report["diagnosis"]
    traffic = report.get("traffic_contract") or {}
    fusion = report.get("fusion_discovery") or {}
    edge_discovery = report.get("ring_edge_discovery") or {}
    system = report.get("system") or {}
    rccl = system.get("rccl") or {}
    communication_dominant = diagnosis["communication_dominant_under_current_lazy_policy"]
    communication_dominant_text = (
        "unknown" if communication_dominant is None else str(communication_dominant).lower()
    )
    lines = [
        "# MiniMax-H3 MegaDiscovery",
        "",
        f"- Profile: `{report['profile']}`",
        f"- World size: `{report['world_size']}`",
        f"- Report status: `{report['status']}`",
        f"- Production configuration changed: `{str(report['production_configuration_changed']).lower()}`",
        "",
        "## Diagnosis",
        "",
        f"- Communication dominant under the current lazy policy: `{communication_dominant_text}`",
        f"- Communication attribution: `{diagnosis['communication_attribution_status']}`",
        f"- Sequence-parallel envelope dominant: `{str(diagnosis['sequence_parallel_envelope_dominant_under_current_lazy_policy']).lower()}`",
        f"- Distributed dense-attention timing available: `{str(diagnosis['distributed_dense_attention_timing_available']).lower()}`",
        f"- Sequence-parallel envelope: `{diagnosis['sequence_parallel_envelope_gpu_ms'] / 1000.0:.3f}s` ({(diagnosis['sequence_parallel_envelope_share'] or 0.0) * 100.0:.2f}%)",
        f"- Exposed H2D: `{diagnosis['exposed_h2d_gpu_ms'] / 1000.0:.3f}s` ({(diagnosis['exposed_h2d_share'] or 0.0) * 100.0:.2f}%)",
        f"- Cross-rank H3 spread: `{diagnosis['rank_spread_ms'] / 1000.0:.3f}s`",
        "",
        "## Traffic",
        "",
    ]
    if report.get("input_failures"):
        failure = report["input_failures"][0]
        active = failure.get("active_benchmark") or "unknown"
        lines[6:6] = [
            f"- Collective probe: `{failure.get('status')}` ({failure.get('error')})",
            f"- Probe stopped in: `{active}`; completed checkpoints: `{failure.get('completed_benchmark_count', 0)}`",
            "- Production and schema-2 evidence remain available; completed transport checkpoints are retained.",
        ]
    if report["world_size"] == 16:
        lines.extend(
            [
                f"- Topology: `CP{traffic.get('context_size')} x Ulysses{traffic.get('ulysses_size')}`",
                f"- Ulysses A2As per block: `{traffic.get('ulysses_a2a_calls_per_block')}`",
                f"- Context Ring hops per block: `{traffic.get('context_ring_hops_per_block')}`",
                f"- P2P send/recv operations per run and rank: `{traffic.get('p2p_send_recv_ops_per_run')}`",
                f"- Estimated remote traffic per run and rank: `{(traffic.get('estimated_remote_bytes_per_run_per_rank') or 0) / 1e9:.3f} GB`",
                f"- Isolated full 2D fusion speedup: `{(fusion.get('synthetic_speedup') or 0.0):.3f}x`",
                f"- Isolated CP2 K/V fusion speedup: `{(fusion.get('context_synthetic_speedup') or 0.0):.3f}x`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Collective calls per run: `{traffic.get('collective_calls_per_run')}`",
                f"- Estimated remote traffic per run and rank: `{(traffic.get('estimated_remote_bytes_per_run_per_rank') or 0) / 1e9:.3f} GB`",
                f"- Isolated QKV-fused speedup: `{(fusion.get('synthetic_speedup') or 0.0):.3f}x`",
            ]
        )
    lines.extend(
        [
            "",
            "## System Evidence",
            "",
            f"- RCCL log files: `{rccl.get('file_count', 0)}`",
            f"- RCCL ranks with Init COMPLETE: `{rccl.get('init_complete_count', 0)}`",
            f"- RCCL active connection transports: `{rccl.get('connection_transports') or {}}`",
            f"- RCCL NET fallback connections: `{rccl.get('net_fallback_connection_count', 0)}`",
            f"- RCCL warning/error matches: `{((rccl.get('keyword_counts') or {}).get('warning', 0))}`",
            f"- RCCL NUMA-balancing warnings: `{rccl.get('numa_balancing_warning_count', 0)}`",
            f"- PCIe root-complex groups: `{len(system.get('pcie_root_complex_groups') or {})}`",
            f"- NUMA balancing: `{system.get('numa_balancing')}`",
            f"- H3 host/cgroup use: `{(system.get('h3_cgroup_bytes') or 0) / 2**30:.2f} GiB`",
            f"- H3 peak VRAM per card: `{(system.get('h3_max_gpu_used_bytes') or 0) / 2**30:.2f} GiB`",
        ]
    )
    lines.extend(["", "## Opportunities", ""])
    if report["opportunities"]:
        for item in report["opportunities"]:
            lines.append(f"{item['priority']}. `{item['name']}`: {item['decision']}")
    else:
        lines.append("No topology-specific opportunity was emitted.")
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in report["guardrails"])
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.result_dir)
    except ValueError as error:
        parser.error(str(error))
    output_json = args.output_json or args.result_dir / "megadiscovery.json"
    output_markdown = args.output_markdown or args.result_dir / "megadiscovery.md"
    write_atomic(output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_atomic(output_markdown, render_markdown(report))
    print(f"MiniMax-H3 MegaDiscovery JSON: {output_json}")
    print(f"MiniMax-H3 MegaDiscovery report: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
