#!/usr/bin/env python3
"""Benchmark the exact MiniMax-H3 SP8 and CP2 x Ulysses8 SP16 payloads."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


GLOBAL_MAIN_TOKENS = 39_728
AUX_TOKENS = 2_468
SP16_GLOBAL_MAIN_TOKENS = 42_192
SP16_AUX_TOKENS = 4
HEADS = 56
HEAD_DIM = 128
DTYPE_BYTES = 2
BLOCKS = 50
MODEL_EVALUATIONS = 4

SCHEMA_NAME = "lightx2v.minimax_h3.megadiscovery.collectives"
SCHEMA_VERSION = 1


def build_ring_edge_matchings(world_size: int) -> list[list[tuple[int, int]]]:
    """Split a directed ring into two disjoint all-rank P2P matchings."""
    if world_size < 2 or world_size % 2:
        raise ValueError("ring edge matching requires an even world size")
    return [
        [(source, (source + 1) % world_size) for source in range(parity, world_size, 2)]
        for parity in (0, 1)
    ]


def build_traffic_contract(world_size: int) -> dict[str, Any]:
    if world_size not in (8, 16):
        raise ValueError("MiniMax-H3 collective discovery supports world size 8 or 16")
    if GLOBAL_MAIN_TOKENS % world_size:
        raise ValueError("global main token count must divide evenly across ranks")
    local_main = GLOBAL_MAIN_TOKENS // world_size
    one_main_payload = local_main * HEADS * HEAD_DIM * DTYPE_BYTES
    aux_full_payload = AUX_TOKENS * HEADS * HEAD_DIM * DTYPE_BYTES
    calls = BLOCKS * MODEL_EVALUATIONS
    if world_size == 8:
        local_aux_payload = aux_full_payload // world_size
        remote_per_block = (
            4 * one_main_payload * (world_size - 1) // world_size
            + aux_full_payload
            - local_aux_payload
        )
        return {
            "topology": "ulysses_sp8",
            "world_size": world_size,
            "global_main_tokens": GLOBAL_MAIN_TOKENS,
            "local_main_tokens": local_main,
            "aux_tokens": AUX_TOKENS,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "torch.bfloat16",
            "one_a2a_payload_bytes": one_main_payload,
            "aux_local_payload_bytes": local_aux_payload,
            "aux_full_payload_bytes": aux_full_payload,
            "collective_calls_per_block": 5,
            "collective_calls_per_run": calls * 5,
            "estimated_remote_bytes_per_block_per_rank": remote_per_block,
            "estimated_remote_bytes_per_run_per_rank": remote_per_block * calls,
            "guardrail": "Remote bytes are an end-to-end traffic estimate; RCCL protocol traffic may differ.",
        }

    context_size = 2
    ulysses_size = 8
    global_main = SP16_GLOBAL_MAIN_TOKENS
    aux_tokens = SP16_AUX_TOKENS
    local_main = global_main // world_size
    shard_heads = HEADS // ulysses_size
    ulysses_main = local_main * ulysses_size
    one_ulysses_payload = local_main * HEADS * HEAD_DIM * DTYPE_BYTES
    one_context_payload = ulysses_main * shard_heads * HEAD_DIM * DTYPE_BYTES
    context_aux_shard_payload = (aux_tokens // context_size) * shard_heads * HEAD_DIM * DTYPE_BYTES
    ulysses_aux_payload = aux_tokens * shard_heads * HEAD_DIM * DTYPE_BYTES
    one_ulysses_remote = one_ulysses_payload * (ulysses_size - 1) // ulysses_size
    send_bytes_per_block = (
        4 * one_ulysses_remote
        + 2 * one_context_payload
        + context_aux_shard_payload
        + ulysses_aux_payload * (ulysses_size - 1)
    )
    return {
        "topology": "cp2_ulysses8_sp16",
        "world_size": world_size,
        "context_size": context_size,
        "ulysses_size": ulysses_size,
        "global_main_tokens": global_main,
        "local_main_tokens": local_main,
        "ulysses_main_tokens": ulysses_main,
        "aux_tokens": aux_tokens,
        "heads": HEADS,
        "shard_heads": shard_heads,
        "head_dim": HEAD_DIM,
        "dtype": "torch.bfloat16",
        "one_ulysses_a2a_payload_bytes": one_ulysses_payload,
        "one_context_k_or_v_payload_bytes": one_context_payload,
        "fused_context_kv_payload_bytes": 2 * one_context_payload,
        "context_aux_shard_payload_bytes": context_aux_shard_payload,
        "ulysses_aux_payload_bytes": ulysses_aux_payload,
        "ulysses_a2a_calls_per_block": 4,
        "context_ring_hops_per_block": 1,
        "context_ring_rounds_per_run": calls,
        "p2p_send_recv_ops_per_block": 4,
        "p2p_send_recv_ops_per_run": 4 * calls,
        "collective_phases_per_block": 7,
        "collective_phases_per_run": 7 * calls,
        "estimated_remote_bytes_per_block_per_rank": send_bytes_per_block,
        "estimated_remote_bytes_per_run_per_rank": send_bytes_per_block * calls,
        "guardrail": "2D traffic excludes layout, attention overlap, online merge, and RCCL protocol amplification.",
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "max_ms": max(values),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_failure(output: Path, world_size: int, reason: str) -> None:
    existing: dict[str, Any] = {}
    try:
        loaded = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("checkpoint is not a JSON object")
        existing_world_size = int(((loaded.get("environment") or {}).get("world_size")) or 0)
        if (
            loaded.get("schema_name") in (None, SCHEMA_NAME)
            and existing_world_size == world_size
        ):
            existing = loaded
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        pass

    benchmarks = existing.get("benchmarks") or []
    if not isinstance(benchmarks, list):
        benchmarks = []
    environment = existing.get("environment") or {}
    if not isinstance(environment, dict):
        environment = {}
    environment["world_size"] = world_size
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_started_at": existing.get("probe_started_at"),
        "status": "failed",
        "active_benchmark": existing.get("active_benchmark"),
        "completed_benchmark_count": len(benchmarks),
        "profiling_only": True,
        "production_configuration_changed": False,
        "environment": environment,
        "traffic_contract": existing.get("traffic_contract") or build_traffic_contract(world_size),
        "benchmarks": benchmarks,
        "error": reason,
        "guardrails": [
            "The isolated probe failed or timed out after production inference completed.",
            "Completed benchmark checkpoints are preserved in this artifact.",
            "Production artifacts remain valid; the unified diagnosis will be marked partial.",
        ],
    }
    _write_atomic(output, payload)
    print(f"[MegaDiscovery][collectives] PARTIAL {reason}", flush=True)


def run_distributed(output: Path) -> None:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print("[MegaDiscovery][collectives] INIT process-group", flush=True)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=20),
        device_id=device,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    contract = build_traffic_contract(world_size)
    dtype = torch.bfloat16
    environment = {
        "world_size": world_size,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(local_rank),
    }
    guardrails = [
        "The benchmark runs in a separate torchrun process after inference and retains no model or allocator state.",
        "2D measurements intentionally remove attention overlap, so they are transport ceilings rather than projected production latency.",
        "Synthetic fused payloads are discovery evidence and do not enable tensor fusion in the production run.",
    ]
    probe_started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    def checkpoint(
        status: str,
        *,
        active_benchmark: str | None,
        benchmarks: list[dict[str, Any]] | None = None,
    ) -> None:
        if rank != 0:
            return
        completed = results if benchmarks is None else benchmarks
        _write_atomic(
            output,
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "probe_started_at": probe_started_at,
                "status": status,
                "active_benchmark": active_benchmark,
                "completed_benchmark_count": len(completed),
                "profiling_only": True,
                "production_configuration_changed": False,
                "environment": environment,
                "traffic_contract": contract,
                "benchmarks": completed,
                "guardrails": guardrails,
            },
        )

    if rank == 0:
        print(
            f"[MegaDiscovery][collectives] READY world_size={world_size} topology={contract['topology']}",
            flush=True,
        )
    checkpoint("running", active_benchmark=None)

    def measure(
        name: str,
        operation: Callable[[], None],
        *,
        warmups: int,
        iterations: int,
        logical_bytes: int,
        logical_bytes_scope: str = "per_rank",
        interpretation: str,
    ) -> dict[str, Any]:
        if rank == 0:
            print(
                f"[MegaDiscovery][collectives] START {name} warmups={warmups} iterations={iterations}",
                flush=True,
            )
        checkpoint("running", active_benchmark=name)
        dist.barrier()
        for _ in range(warmups):
            operation()
        torch.cuda.synchronize(device)
        dist.barrier()
        local_samples = []
        for _ in range(iterations):
            dist.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            operation()
            torch.cuda.synchronize(device)
            local_samples.append((time.perf_counter() - started) * 1000.0)
        local_tensor = torch.tensor(local_samples, dtype=torch.float64, device=device)
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)
        per_rank = [item.cpu().tolist() for item in gathered]
        critical = [max(samples[index] for samples in per_rank) for index in range(iterations)]
        critical_mean_ms = statistics.fmean(critical)
        effective_gbps = logical_bytes / critical_mean_ms / 1_000_000.0
        result = {
            "name": name,
            "warmups": warmups,
            "iterations": iterations,
            "logical_bytes_per_rank_per_iteration": logical_bytes,
            "logical_bytes_scope": logical_bytes_scope,
            "critical_path": _stats(critical),
            "effective_logical_GBps": effective_gbps,
            "per_rank": {str(index): _stats(samples) for index, samples in enumerate(per_rank)},
            "interpretation": interpretation,
        }
        if rank == 0:
            print(
                f"[MegaDiscovery][collectives] DONE  {name} mean={critical_mean_ms:.3f}ms logical={effective_gbps:.3f}GB/s",
                flush=True,
            )
        checkpoint("running", active_benchmark=None, benchmarks=[*results, result])
        return result

    local_main = int(contract["local_main_tokens"])
    aux_tokens = int(contract["aux_tokens"])

    if world_size == 8:
        shard_heads = HEADS // world_size
        a2a_input = torch.empty((world_size, local_main, shard_heads, HEAD_DIM), dtype=dtype, device=device)
        a2a_output = torch.empty_like(a2a_input)
        aux_input = torch.empty((aux_tokens, shard_heads, HEAD_DIM), dtype=dtype, device=device)
        aux_output = [torch.empty_like(aux_input) for _ in range(world_size)]
        fused_input = torch.empty((world_size, local_main, 3, shard_heads, HEAD_DIM), dtype=dtype, device=device)
        fused_output = torch.empty_like(fused_input)

        def one_a2a() -> None:
            dist.all_to_all_single(a2a_output, a2a_input)

        def current_block() -> None:
            for _ in range(4):
                dist.all_to_all_single(a2a_output, a2a_input)
            dist.all_gather(aux_output, aux_input)

        def fused_block() -> None:
            dist.all_to_all_single(fused_output, fused_input)
            dist.all_to_all_single(a2a_output, a2a_input)
            dist.all_gather(aux_output, aux_input)

        one_bytes = int(contract["one_a2a_payload_bytes"] * (world_size - 1) // world_size)
        results.append(
            measure(
                "ulysses_one_a2a",
                one_a2a,
                warmups=2,
                iterations=5,
                logical_bytes=one_bytes,
                interpretation="One production-shape A2A, excluding layout kernels.",
            )
        )
        results.append(
            measure(
                "ulysses_block_current_transport",
                current_block,
                warmups=1,
                iterations=5,
                logical_bytes=int(contract["estimated_remote_bytes_per_block_per_rank"]),
                interpretation="Four serial A2As plus aux all-gather; no dense attention or layout copies.",
            )
        )
        results.append(
            measure(
                "ulysses_block_qkv_fused_transport",
                fused_block,
                warmups=1,
                iterations=5,
                logical_bytes=int(contract["estimated_remote_bytes_per_block_per_rank"]),
                interpretation="Synthetic QKV-fused comparison only; production configuration remains unchanged.",
            )
        )
    else:
        context_size = int(contract["context_size"])
        ulysses_size = int(contract["ulysses_size"])
        shard_heads = int(contract["shard_heads"])
        ulysses_main = int(contract["ulysses_main_tokens"])

        ulysses_groups = []
        for context_rank in range(context_size):
            ranks = list(
                range(
                    context_rank * ulysses_size,
                    (context_rank + 1) * ulysses_size,
                )
            )
            ulysses_groups.append(dist.new_group(ranks=ranks))
        context_groups = []
        for ulysses_rank in range(ulysses_size):
            ranks = [
                context_rank * ulysses_size + ulysses_rank
                for context_rank in range(context_size)
            ]
            context_groups.append(dist.new_group(ranks=ranks))
        ulysses_group = ulysses_groups[rank // ulysses_size]
        context_group = context_groups[rank % ulysses_size]
        context_peer = rank + ulysses_size if rank < ulysses_size else rank - ulysses_size

        a2a_input = torch.empty(
            (ulysses_size, local_main, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        a2a_output = torch.empty_like(a2a_input)
        fused_qkv_input = torch.empty(
            (ulysses_size, local_main, 3, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        fused_qkv_output = torch.empty_like(fused_qkv_input)
        context_k_send = torch.empty(
            (ulysses_main, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        context_k_receive = torch.empty_like(context_k_send)
        context_v_send = torch.empty_like(context_k_send)
        context_v_receive = torch.empty_like(context_k_send)
        context_kv_send = torch.empty(
            (2 * ulysses_main, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        context_kv_receive = torch.empty_like(context_kv_send)
        context_aux_input = torch.empty(
            (aux_tokens // context_size, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        context_aux_output = [
            torch.empty_like(context_aux_input) for _ in range(context_size)
        ]
        ulysses_aux_input = torch.empty(
            (aux_tokens, shard_heads, HEAD_DIM),
            dtype=dtype,
            device=device,
        )
        ulysses_aux_output = [
            torch.empty_like(ulysses_aux_input) for _ in range(ulysses_size)
        ]

        def one_ulysses_a2a() -> None:
            dist.all_to_all_single(
                a2a_output,
                a2a_input,
                group=ulysses_group,
            )

        def context_kv_separate() -> None:
            operations = [
                dist.P2POp(
                    dist.isend,
                    context_k_send,
                    context_peer,
                    group=context_group,
                ),
                dist.P2POp(
                    dist.irecv,
                    context_k_receive,
                    context_peer,
                    group=context_group,
                ),
                dist.P2POp(
                    dist.isend,
                    context_v_send,
                    context_peer,
                    group=context_group,
                ),
                dist.P2POp(
                    dist.irecv,
                    context_v_receive,
                    context_peer,
                    group=context_group,
                ),
            ]
            for request in dist.batch_isend_irecv(operations):
                request.wait()

        def context_kv_fused() -> None:
            operations = [
                dist.P2POp(
                    dist.isend,
                    context_kv_send,
                    context_peer,
                    group=context_group,
                ),
                dist.P2POp(
                    dist.irecv,
                    context_kv_receive,
                    context_peer,
                    group=context_group,
                ),
            ]
            for request in dist.batch_isend_irecv(operations):
                request.wait()

        def current_2d_block() -> None:
            for _ in range(3):
                one_ulysses_a2a()
            context_kv_separate()
            dist.all_gather(
                context_aux_output,
                context_aux_input,
                group=context_group,
            )
            one_ulysses_a2a()
            dist.all_gather(
                ulysses_aux_output,
                ulysses_aux_input,
                group=ulysses_group,
            )

        def fused_2d_block() -> None:
            dist.all_to_all_single(
                fused_qkv_output,
                fused_qkv_input,
                group=ulysses_group,
            )
            context_kv_fused()
            dist.all_gather(
                context_aux_output,
                context_aux_input,
                group=context_group,
            )
            one_ulysses_a2a()
            dist.all_gather(
                ulysses_aux_output,
                ulysses_aux_input,
                group=ulysses_group,
            )

        one_ulysses_remote = int(
            contract["one_ulysses_a2a_payload_bytes"]
            * (ulysses_size - 1)
            // ulysses_size
        )
        context_kv_bytes = 2 * int(contract["one_context_k_or_v_payload_bytes"])
        results.append(
            measure(
                "hybrid16_ulysses_one_a2a",
                one_ulysses_a2a,
                warmups=2,
                iterations=5,
                logical_bytes=one_ulysses_remote,
                interpretation="One production-shape A2A inside each disjoint U8 row.",
            )
        )
        results.append(
            measure(
                "hybrid16_context_kv_separate",
                context_kv_separate,
                warmups=2,
                iterations=5,
                logical_bytes=context_kv_bytes,
                interpretation="The production CP2 single-hop K/V exchange for eight disjoint rank pairs.",
            )
        )
        results.append(
            measure(
                "hybrid16_context_kv_fused",
                context_kv_fused,
                warmups=2,
                iterations=5,
                logical_bytes=context_kv_bytes,
                interpretation="Synthetic fused-KV comparison; production tensor fusion remains disabled.",
            )
        )
        results.append(
            measure(
                "hybrid16_block_current_transport",
                current_2d_block,
                warmups=1,
                iterations=3,
                logical_bytes=int(contract["estimated_remote_bytes_per_block_per_rank"]),
                interpretation="Four U8 A2As, one CP2 K/V exchange, and both auxiliary gathers.",
            )
        )
        results.append(
            measure(
                "hybrid16_block_fused_transport",
                fused_2d_block,
                warmups=1,
                iterations=3,
                logical_bytes=int(contract["estimated_remote_bytes_per_block_per_rank"]),
                interpretation="Synthetic QKV/KV-fused 2D comparison; production configuration remains unchanged.",
            )
        )

    if rank == 0:
        checkpoint("complete", active_benchmark=None)
        print(f"[MegaDiscovery][collectives] COMPLETE output={output}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(8, 16))
    parser.add_argument("--record-failure")
    args = parser.parse_args()
    if args.record_failure is not None:
        if args.world_size is None:
            parser.error("--record-failure requires --world-size")
        write_failure(args.output, args.world_size, args.record_failure)
        return 0
    run_distributed(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
