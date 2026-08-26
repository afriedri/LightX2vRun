#!/usr/bin/env python3
"""Sample LightX2V rank memory, container memory, and physical GPU VRAM.

This is a host-side, read-only probe.  It deliberately does not call rocm-smi,
change clocks, or modify the running container.  Run it on the Docker host while
``lightx2v-gfx1201`` is running (it may be started before the container appears).

The JSONL samples keep the memory scopes separate:

* per-rank RSS/PSS/private/shared values from ``smaps_rollup``;
* simultaneous sums across ranks (not sums of per-rank historical peaks);
* the container cgroup's actual anon/file/shmem charge and peak;
* host MemTotal/MemAvailable/Cached/AnonPages/Shmem;
* physical VRAM used/total for the selected ROCm-indexed AMD DRM cards.

The external probe cannot see PyTorch's per-process allocator counters.  Those
are distinct from physical VRAM used and must be logged in-process with
``torch.cuda.memory_allocated()``, ``memory_reserved()``, and their max variants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KIB = 1024
GIB = 1024**3
RANK_CMD_TOKEN = "lightx2v.infer"
SMAPS_FIELDS = (
    "Rss",
    "Pss",
    "Pss_Anon",
    "Pss_File",
    "Pss_Shmem",
    "Shared_Clean",
    "Shared_Dirty",
    "Private_Clean",
    "Private_Dirty",
    "Anonymous",
    "AnonHugePages",
    "Locked",
    "Swap",
    "SwapPss",
)


# Executed as root in the existing container so /proc/<rank>/smaps_rollup is
# readable even when the host user cannot inspect container-owned processes.
CONTAINER_PROBE = r'''
import json
import os
import re

FIELDS = {
    "Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem",
    "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty",
    "Anonymous", "AnonHugePages", "Locked", "Swap", "SwapPss",
}
RUNTIME_ENV_KEYS = {
    "AITER_JIT_DIR", "AITER_ROCM_ARCH", "CUDA_VISIBLE_DEVICES",
    "GPU_ARCHS", "HIP_VISIBLE_DEVICES", "HSA_FORCE_FINE_GRAIN_PCIE",
    "LIGHTX2V_H3_RUN_COLLECTOR", "LOCAL_RANK", "MASTER_ADDR",
    "LIGHTX2V_WAN_RUNTUNE", "LIGHTX2V_WAN_RUNTUNE_BACKEND",
    "LIGHTX2V_WAN_RUNTUNE_OUTPUT", "LTX23_COLLECT",
    "LTX23_PREFLIGHT_ONLY", "LTX23_QUICK_GEMM_PROBE",
    "MASTER_PORT", "NCCL_P2P_LEVEL", "OMP_NUM_THREADS", "RANK",
    "ROCR_VISIBLE_DEVICES", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT", "WORLD_SIZE", "PYTORCH_TUNABLEOP_ENABLED",
}

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

def read_bytes(path):
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None

def parse_kib_file(path):
    text = read_text(path)
    if text is None:
        return None
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key not in FIELDS:
            continue
        match = re.search(r"([0-9]+)", value)
        if match:
            result[key + "_bytes"] = int(match.group(1)) * 1024
    return result

def parse_environ(pid):
    raw = read_bytes("/proc/{}/environ".format(pid))
    if raw is None:
        return {}
    result = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result

def parse_status(pid):
    text = read_text("/proc/{}/status".format(pid)) or ""
    result = {}
    for line in text.splitlines():
        if line.startswith("NSpid:"):
            ids = [int(item) for item in line.split()[1:]]
            result["namespace_pids"] = ids
        elif line.startswith("Threads:"):
            result["threads"] = int(line.split()[1])
        elif line.startswith("Cpus_allowed_list:"):
            result["cpus_allowed_list"] = line.split(":", 1)[1].strip()
        elif line.startswith("Mems_allowed_list:"):
            result["mems_allowed_list"] = line.split(":", 1)[1].strip()
        elif line.startswith("voluntary_ctxt_switches:"):
            result["voluntary_ctxt_switches"] = int(line.split(":", 1)[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            result["nonvoluntary_ctxt_switches"] = int(line.split(":", 1)[1])
    return result

def parse_proc_stat(pid):
    text = read_text("/proc/{}/stat".format(pid))
    if not text or ")" not in text:
        return {}
    fields = text[text.rfind(")") + 2:].split()
    if len(fields) < 40:
        return {}
    return {
        "state": fields[0],
        "minflt": int(fields[7]),
        "majflt": int(fields[9]),
        "utime_ticks": int(fields[11]),
        "stime_ticks": int(fields[12]),
        "num_threads_stat": int(fields[17]),
        "processor": int(fields[36]),
        "delayacct_blkio_ticks": int(fields[39]),
    }

def parse_proc_io(pid):
    text = read_text("/proc/{}/io".format(pid)) or ""
    result = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip().isdigit():
            result[key] = int(value.strip())
    return result

def parse_schedstat(pid):
    text = read_text("/proc/{}/schedstat".format(pid)) or ""
    fields = text.split()
    if len(fields) < 3 or not all(value.isdigit() for value in fields[:3]):
        return {}
    return {
        "cpu_runtime_ns": int(fields[0]),
        "runqueue_wait_ns": int(fields[1]),
        "timeslices": int(fields[2]),
    }

def parse_numa_maps(pid):
    text = read_text("/proc/{}/numa_maps".format(pid)) or ""
    pages = {}
    for line in text.splitlines():
        for node, count in re.findall(r"\bN([0-9]+)=([0-9]+)\b", line):
            key = "N{}".format(node)
            pages[key] = pages.get(key, 0) + int(count)
    return pages

def find_ranks():
    ranks = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        raw_cmd = read_bytes("/proc/{}/cmdline".format(name))
        if not raw_cmd:
            continue
        argv = [item.decode(errors="replace") for item in raw_cmd.split(b"\0") if item]
        is_inference = "lightx2v.infer" in argv
        is_collective_probe = any(
            item.rsplit("/", 1)[-1] == "benchmark_minimax_h3_collectives.py"
            for item in argv
        )
        if not is_inference and not is_collective_probe:
            continue
        env = parse_environ(pid)
        local_rank = env.get("LOCAL_RANK")
        rank = env.get("RANK")
        inferred_single_process = False
        if local_rank is None and rank is None:
            is_distributed_launcher = any(
                item.rsplit("/", 1)[-1] == "torchrun" or item == "torch.distributed.run"
                for item in argv
            )
            if is_collective_probe or is_distributed_launcher:
                continue
            # Single-GPU launchers use ``python -m lightx2v.infer`` directly,
            # without torchrun's rank environment. It is still rank 0 for
            # process-memory accounting.
            local_rank = "0"
            rank = "0"
            inferred_single_process = True
        item = {
            "container_pid": pid,
            "rank": int(rank) if rank is not None and rank.lstrip("-").isdigit() else rank,
            "local_rank": int(local_rank) if local_rank is not None and local_rank.lstrip("-").isdigit() else local_rank,
            "inferred_single_process": inferred_single_process,
            "workload": "collective_probe" if is_collective_probe else "inference",
            "cmdline_argv": argv,
            "cmdline": " ".join(argv),
            "runtime_environment": {
                key: env[key] for key in sorted(RUNTIME_ENV_KEYS) if key in env
            },
        }
        item.update(parse_status(pid))
        item["proc_stat"] = parse_proc_stat(pid)
        item["proc_io"] = parse_proc_io(pid)
        item["schedstat"] = parse_schedstat(pid)
        item["numa_pages_by_node"] = parse_numa_maps(pid)
        smaps = parse_kib_file("/proc/{}/smaps_rollup".format(pid))
        if smaps is None:
            item["smaps_error"] = "unreadable"
        else:
            item.update(smaps)
        ranks.append(item)
    ranks.sort(key=lambda item: (item.get("local_rank") is None, item.get("local_rank", 0)))
    return ranks

def read_int(path):
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None

def read_cgroup():
    # Docker's private cgroup namespace normally exposes the container cgroup at
    # this root.  Keep a cgroup-v1 fallback for older hosts.
    current = read_int("/sys/fs/cgroup/memory.current")
    peak = read_int("/sys/fs/cgroup/memory.peak")
    stat_text = read_text("/sys/fs/cgroup/memory.stat")
    version = 2
    if current is None:
        current = read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        peak = read_int("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        stat_text = read_text("/sys/fs/cgroup/memory/memory.stat")
        version = 1
    stat = {}
    if stat_text:
        for line in stat_text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                stat[parts[0]] = int(parts[1])
    io_stat_text = read_text("/sys/fs/cgroup/io.stat") if version == 2 else None
    io_stat = {}
    if io_stat_text:
        for line in io_stat_text.splitlines():
            fields = line.split()
            if not fields:
                continue
            io_stat[fields[0]] = {
                key: int(value)
                for field in fields[1:]
                for key, separator, value in (field.partition("="),)
                if separator and value.isdigit()
            }
    return {
        "version": version,
        "current_bytes": current,
        "peak_bytes": peak,
        "stat_bytes": stat,
        "io_stat": io_stat,
    }

print(json.dumps({"ranks": find_ranks(), "cgroup": read_cgroup()}, sort_keys=True))
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_command(argv: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)


def container_state(container: str) -> str:
    result = run_command(["docker", "inspect", "--format", "{{.State.Running}}", container])
    if result.returncode != 0:
        return "missing"
    return "running" if result.stdout.strip().lower() == "true" else "stopped"


def sample_container(container: str) -> dict[str, Any]:
    result = run_command(["docker", "exec", container, "python3", "-c", CONTAINER_PROBE], timeout=30.0)
    if result.returncode != 0:
        return {"probe_error": result.stderr.strip() or result.stdout.strip() or f"docker exec exited {result.returncode}"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"probe_error": f"invalid probe JSON: {exc}", "probe_stdout": result.stdout[-2000:]}


def add_host_pids(container: str, container_sample: dict[str, Any]) -> None:
    """Map container PIDs to host PIDs using NSpid from Docker's host PID list."""
    result = run_command(["docker", "top", container, "-eo", "pid"])
    if result.returncode != 0:
        container_sample["host_pid_map_error"] = result.stderr.strip() or "docker top failed"
        return
    container_to_host: dict[int, int] = {}
    for line in result.stdout.splitlines()[1:]:
        value = line.strip()
        if not value.isdigit():
            continue
        host_pid = int(value)
        try:
            status = Path(f"/proc/{host_pid}/status").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for status_line in status.splitlines():
            if status_line.startswith("NSpid:"):
                namespace_pids = [int(item) for item in status_line.split()[1:]]
                if namespace_pids:
                    container_to_host[namespace_pids[-1]] = host_pid
                break
    for rank in container_sample.get("ranks", []):
        rank["host_pid"] = container_to_host.get(rank.get("container_pid"))


def read_key_value_kib(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            match = re.search(r"([0-9]+)", value)
            if match:
                result[key] = int(match.group(1)) * KIB
    return result


def sample_host_memory() -> dict[str, int | None]:
    values = read_key_value_kib(Path("/proc/meminfo"))
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_excluding_available_bytes": used,
        "cached_bytes": values.get("Cached"),
        "buffers_bytes": values.get("Buffers"),
        "anon_pages_bytes": values.get("AnonPages"),
        "shmem_bytes": values.get("Shmem"),
        "slab_bytes": values.get("Slab"),
    }


VMSTAT_COUNTERS = (
    "pgfault",
    "pgmajfault",
    "pgpgin",
    "pgpgout",
    "pswpin",
    "pswpout",
    "workingset_refault_anon",
    "workingset_refault_file",
    "numa_hit",
    "numa_miss",
    "numa_foreign",
    "numa_interleave",
    "numa_local",
    "numa_other",
)


def sample_vmstat() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/vmstat").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, OSError):
        return values
    selected = set(VMSTAT_COUNTERS)
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in selected and fields[1].isdigit():
            values[fields[0]] = int(fields[1])
    return values


def sample_pressure() -> dict[str, str]:
    result = {}
    for resource in ("cpu", "memory", "io"):
        value = read_optional_text(Path("/proc/pressure") / resource)
        if value is not None:
            result[resource] = value
    return result


def parse_diskstats(text: str) -> dict[str, dict[str, int]]:
    result = {}
    names = (
        "reads_completed",
        "reads_merged",
        "sectors_read",
        "read_ms",
        "writes_completed",
        "writes_merged",
        "sectors_written",
        "write_ms",
        "io_in_progress",
        "io_ms",
        "weighted_io_ms",
    )
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        values = fields[3:14]
        if not all(value.isdigit() for value in values):
            continue
        result[fields[2]] = {
            "major": int(fields[0]),
            "minor": int(fields[1]),
            **{name: int(value) for name, value in zip(names, values)},
        }
    return result


def sample_diskstats() -> dict[str, dict[str, int]]:
    try:
        text = Path("/proc/diskstats").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return {}
    return parse_diskstats(text)


def sample_host() -> dict[str, Any]:
    result: dict[str, Any] = sample_host_memory()
    result["vmstat"] = sample_vmstat()
    result["pressure"] = sample_pressure()
    result["diskstats"] = sample_diskstats()
    return result


def counter_delta(first: Any, last: Any) -> Any:
    if isinstance(first, int) and isinstance(last, int):
        return last - first
    if isinstance(first, dict) and isinstance(last, dict):
        return {
            key: delta
            for key in sorted(first.keys() & last.keys())
            if (delta := counter_delta(first[key], last[key])) is not None
        }
    return None


def read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def drm_card_index(name: str) -> int | None:
    """Return a DRM card index while rejecting connector entries."""
    match = re.fullmatch(r"card([0-9]+)", name)
    return int(match.group(1)) if match else None


def sample_gpus(selected_indices: set[int] | None = None) -> list[dict[str, Any]]:
    cards = []
    indexed_card_paths = [
        (index, path)
        for path in Path("/sys/class/drm").glob("card*")
        if (index := drm_card_index(path.name)) is not None
    ]
    for _, card_path in sorted(indexed_card_paths):
        device_path = card_path / "device"
        vendor = read_optional_int(device_path / "vendor")
        if vendor != 0x1002:
            continue
        total = read_optional_int(device_path / "mem_info_vram_total")
        used = read_optional_int(device_path / "mem_info_vram_used")
        if total is None and used is None:
            continue
        cards.append(
            {
                "rocm_index": len(cards),
                "card": card_path.name,
                "pci_address": Path(os.path.realpath(device_path)).name,
                "device_id": read_optional_text(device_path / "device"),
                "unique_id": read_optional_text(device_path / "unique_id"),
                "numa_node": read_optional_int(device_path / "numa_node"),
                "total_bytes": total,
                "used_bytes": used,
            }
        )
    if selected_indices is None:
        return cards
    return [card for card in cards if card["rocm_index"] in selected_indices]


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers")
    indices = tuple(int(item) for item in value.split(","))
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return indices


class PhaseTracker:
    """Best-effort phase labels from the existing profile log."""

    _ORDER = {
        "loading": 0,
        "vae_encode": 1,
        "post_vae_encode": 2,
        "transformer_shell": 3,
        "dit_high": 4,
        "dit_low": 5,
        "vae_decode": 6,
        "transformer_released": 7,
        "complete": 8,
    }

    def __init__(self, log_path: Path | None, expected_ranks: int):
        self.log_path = log_path
        self.expected_ranks = expected_ranks
        self.offset = 0
        self.phase = "loading"
        self.load_done_ranks: set[int] = set()
        self.vae_encode_done_ranks: set[int] = set()
        self.memory_label_ranks: dict[str, set[int]] = {}

    @staticmethod
    def _record_ranks(text: str, marker: str, destination: set[int]) -> None:
        pattern = re.compile(rf"(?:Rank (?P<rank>\d+)|Single GPU).*{re.escape(marker)} cost", re.IGNORECASE)
        for match in pattern.finditer(text):
            destination.add(int(match.group("rank") or 0))

    def _advance(self, candidate: str) -> None:
        if self._ORDER[candidate] > self._ORDER[self.phase]:
            self.phase = candidate

    def _record_memory_labels(self, text: str) -> None:
        pattern = re.compile(
            r"\[WanRunTune\] device_memory label=(?P<label>[a-z0-9_]+) "
            r"rank=(?P<rank>\d+)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            self.memory_label_ranks.setdefault(match.group("label"), set()).add(
                int(match.group("rank"))
            )

    def _label_complete(self, label: str) -> bool:
        return len(self.memory_label_ranks.get(label, ())) >= self.expected_ranks

    def update(self) -> str:
        if self.log_path is None:
            return "unknown"
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self.offset)
                text = stream.read()
                self.offset = stream.tell()
        except (FileNotFoundError, PermissionError, OSError):
            return self.phase

        self._record_ranks(text, "Load models", self.load_done_ranks)
        self._record_ranks(text, "Run VAE Encoder", self.vae_encode_done_ranks)
        self._record_memory_labels(text)

        # Only advance: rank-interleaved logs may contain delayed lines.
        if re.search(r"RUN pipeline cost|Total Cost", text, re.IGNORECASE):
            self._advance("complete")
        elif self._label_complete("transformer_released"):
            self._advance("transformer_released")
        elif self._label_complete("vae_decode_entry") or self._label_complete("vae_decode_stream_entry"):
            self._advance("vae_decode")
        elif re.search(r"DiT denoising[^\n]*100%|Run VAE Decoder cost", text, re.IGNORECASE):
            self._advance("vae_decode")
        elif self._label_complete("transformer_low_ready"):
            self._advance("dit_low")
        elif self._label_complete("transformer_high_ready"):
            self._advance("dit_high")
        elif self._label_complete("transformer_shell_ready"):
            self._advance("transformer_shell")
        elif re.search(r"==>\s*step_index:", text):
            self._advance("dit_high")
        elif len(self.vae_encode_done_ranks) >= self.expected_ranks:
            self._advance("post_vae_encode")
        elif len(self.load_done_ranks) >= self.expected_ranks:
            self._advance("vae_encode")
        return self.phase


class MiniMaxH3PhaseTracker:
    """Best-effort H0-H6 labels from MiniMax-H3 stage logs. Wan regex is unchanged."""

    _ORDER = {
        "loading": 0,
        "h0": 1,
        "h1": 2,
        "h2": 3,
        "h3": 4,
        "h4": 5,
        "h5": 6,
        "h6": 7,
        "complete": 8,
    }

    def __init__(self, log_path: Path | None, expected_ranks: int):
        self.log_path = log_path
        self.expected_ranks = expected_ranks
        self.offset = 0
        self.phase = "loading"

    def _advance(self, candidate: str) -> None:
        if self._ORDER[candidate] > self._ORDER[self.phase]:
            self.phase = candidate

    def update(self) -> str:
        if self.log_path is None:
            return "unknown"
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self.offset)
                text = stream.read()
                self.offset = stream.tell()
        except (FileNotFoundError, PermissionError, OSError):
            return self.phase
        if re.search(r"RUN pipeline cost|Total Cost", text, re.IGNORECASE):
            self._advance("complete")
        for stage, name in (
            ("H6", "h6"),
            ("H5", "h5"),
            ("H4", "h4"),
            ("H3", "h3"),
            ("H2", "h2"),
            ("H1", "h1"),
            ("H0", "h0"),
        ):
            runtime_stage = re.search(
                rf"\[Runtime\]\s+kind=stage\s+model=minimax_h3\s+"
                rf"stage={stage}(?:-[A-Za-z0-9_]+)?\b",
                text,
            )
            legacy_stage = re.search(rf"\[H3Stage\]\s+{stage}\b", text)
            if runtime_stage or legacy_stage:
                self._advance(name)
                break
        return self.phase


class LTX2PhaseTracker:
    """Track the normalized L1/L2/L3 lifecycle for LTX-2.3 B0."""

    _ORDER = {
        "loading": 0,
        "l1": 1,
        "l2_s1": 2,
        "l2_upsample": 3,
        "l2_s2": 4,
        "l3": 5,
        "complete": 6,
    }

    def __init__(self, log_path: Path | None, expected_ranks: int):
        self.log_path = log_path
        self.expected_ranks = expected_ranks
        self.offset = 0
        self.phase = "loading"

    def _advance(self, candidate: str) -> None:
        if self._ORDER[candidate] > self._ORDER[self.phase]:
            self.phase = candidate

    def update(self) -> str:
        if self.log_path is None:
            return "unknown"
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(self.offset)
                text = stream.read()
                self.offset = stream.tell()
        except (FileNotFoundError, PermissionError, OSError):
            return self.phase
        if re.search(r"RUN pipeline cost|LTX-2\.3 FP8 B0 complete", text, re.IGNORECASE):
            self._advance("complete")
        for stage, name in (
            ("L3", "l3"),
            ("L2-S2", "l2_s2"),
            ("L2-Upsample", "l2_upsample"),
            ("L2-S1", "l2_s1"),
            ("L1", "l1"),
        ):
            if re.search(
                rf"\[Runtime\]\s+kind=stage\s+model=ltx2\s+stage={re.escape(stage)}\b",
                text,
            ):
                self._advance(name)
                break
        return self.phase


def add_rank_totals(sample: dict[str, Any]) -> None:
    ranks = sample.get("container", {}).get("ranks", [])
    totals: dict[str, int] = {field + "_bytes": 0 for field in SMAPS_FIELDS}
    for rank in ranks:
        for key in totals:
            value = rank.get(key)
            if isinstance(value, int):
                totals[key] += value
    private = totals["Private_Clean_bytes"] + totals["Private_Dirty_bytes"]
    shared = totals["Shared_Clean_bytes"] + totals["Shared_Dirty_bytes"]
    totals["Private_bytes"] = private
    totals["Shared_bytes"] = shared
    totals["rank_count"] = len(ranks)
    sample["rank_sums"] = totals


def gib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f} GiB"


def should_emit_status(
    *,
    phase: str,
    previous_phase: str | None,
    monotonic_seconds: float,
    last_status_time: float | None,
    print_interval: float,
) -> bool:
    """Emit phase transitions immediately and rate-limit unchanged heartbeats."""
    if phase != previous_phase:
        return True
    return bool(
        print_interval > 0
        and (
            last_status_time is None
            or monotonic_seconds - last_status_time >= print_interval
        )
    )


class Summary:
    def __init__(self) -> None:
        self.samples = 0
        self.max_rank_sums: dict[str, int] = {}
        self.rank_peak_rss: dict[str, int] = {}
        self.max_cgroup_current = 0
        self.reported_cgroup_peak = 0
        self.max_host_used = 0
        self.gpu_peaks: dict[str, int] = {}
        self.gpu_totals: dict[str, int] = {}
        self.phase_maxima: dict[str, dict[str, int]] = {}
        self.rank_lifecycle: dict[str, dict[str, Any]] = {}
        self.active_rank_keys: set[str] = set()
        self.first_host_counters: dict[str, Any] | None = None
        self.last_host_counters: dict[str, Any] | None = None

    @staticmethod
    def _rank_key(rank: dict[str, Any]) -> str:
        rank_id = rank.get("rank", rank.get("local_rank", rank.get("container_pid")))
        workload = rank.get("workload")
        if workload and workload != "inference":
            return f"{workload}:{rank_id}"
        return str(rank_id)

    def _observe_rank_lifecycle(self, sample: dict[str, Any]) -> None:
        container = sample.get("container", {})
        if container.get("probe_error"):
            return
        timestamp = sample.get("timestamp_utc")
        monotonic_seconds = sample.get("monotonic_seconds")
        current_rank_keys: set[str] = set()
        for rank in container.get("ranks", []):
            rank_key = self._rank_key(rank)
            current_rank_keys.add(rank_key)
            lifecycle = self.rank_lifecycle.setdefault(
                rank_key,
                {
                    "rank": rank.get("rank"),
                    "local_rank": rank.get("local_rank"),
                    "container_pid": rank.get("container_pid"),
                    "host_pid": rank.get("host_pid"),
                    "workload": rank.get("workload", "inference"),
                    "first_seen_utc": timestamp,
                    "first_seen_monotonic_seconds": monotonic_seconds,
                },
            )
            lifecycle["last_seen_utc"] = timestamp
            lifecycle["last_seen_monotonic_seconds"] = monotonic_seconds
            lifecycle["cpus_allowed_list"] = rank.get("cpus_allowed_list")
            lifecycle["mems_allowed_list"] = rank.get("mems_allowed_list")
            lifecycle["last_numa_pages_by_node"] = rank.get("numa_pages_by_node", {})
            counters = {
                "proc_stat": {
                    key: value
                    for key, value in rank.get("proc_stat", {}).items()
                    if key in {
                        "minflt",
                        "majflt",
                        "utime_ticks",
                        "stime_ticks",
                        "delayacct_blkio_ticks",
                    }
                },
                "proc_io": rank.get("proc_io", {}),
                "schedstat": rank.get("schedstat", {}),
                "voluntary_ctxt_switches": rank.get("voluntary_ctxt_switches"),
                "nonvoluntary_ctxt_switches": rank.get("nonvoluntary_ctxt_switches"),
            }
            lifecycle.setdefault("first_process_counters", counters)
            lifecycle["last_process_counters"] = counters

        for rank_key in self.active_rank_keys - current_rank_keys:
            lifecycle = self.rank_lifecycle[rank_key]
            lifecycle.setdefault("exit_observed_utc", timestamp)
            lifecycle.setdefault("exit_observed_monotonic_seconds", monotonic_seconds)
            lifecycle.setdefault("exit_observation", "rank_missing_from_probe")
        self.active_rank_keys = current_rank_keys

    def close_active_ranks(
        self,
        timestamp: str,
        monotonic_seconds: float,
        observation: str,
    ) -> None:
        for rank_key in self.active_rank_keys:
            lifecycle = self.rank_lifecycle[rank_key]
            lifecycle.setdefault("exit_observed_utc", timestamp)
            lifecycle.setdefault("exit_observed_monotonic_seconds", monotonic_seconds)
            lifecycle.setdefault("exit_observation", observation)
        self.active_rank_keys.clear()

    def add(self, sample: dict[str, Any]) -> None:
        self.samples += 1
        self._observe_rank_lifecycle(sample)
        host = sample.get("host", {})
        host_counters = {
            "vmstat": host.get("vmstat", {}),
            "diskstats": {
                device: {
                    key: value
                    for key, value in counters.items()
                    if key not in {"major", "minor", "io_in_progress"}
                }
                for device, counters in host.get("diskstats", {}).items()
            },
        }
        if self.first_host_counters is None:
            self.first_host_counters = host_counters
        self.last_host_counters = host_counters
        for key, value in sample.get("rank_sums", {}).items():
            if isinstance(value, int):
                self.max_rank_sums[key] = max(self.max_rank_sums.get(key, 0), value)
        for rank in sample.get("container", {}).get("ranks", []):
            rank_key = self._rank_key(rank)
            self.rank_peak_rss[rank_key] = max(self.rank_peak_rss.get(rank_key, 0), rank.get("Rss_bytes", 0))
        cgroup = sample.get("container", {}).get("cgroup", {})
        self.max_cgroup_current = max(self.max_cgroup_current, cgroup.get("current_bytes") or 0)
        self.reported_cgroup_peak = max(self.reported_cgroup_peak, cgroup.get("peak_bytes") or 0)
        self.max_host_used = max(self.max_host_used, sample.get("host", {}).get("used_excluding_available_bytes") or 0)
        for gpu in sample.get("gpus", []):
            key = f"{gpu.get('card')}@{gpu.get('pci_address')}"
            self.gpu_peaks[key] = max(self.gpu_peaks.get(key, 0), gpu.get("used_bytes") or 0)
            total_bytes = gpu.get("total_bytes") or 0
            if total_bytes:
                previous_total = self.gpu_totals.get(key)
                if previous_total is not None and previous_total != total_bytes:
                    raise ValueError(
                        f"physical GPU total changed for {key}: "
                        f"{previous_total} -> {total_bytes}"
                    )
                self.gpu_totals[key] = total_bytes

        phase = sample.get("phase", "unknown")
        phase_max = self.phase_maxima.setdefault(phase, {})
        candidates = {
            "rank_rss_sum_bytes": sample.get("rank_sums", {}).get("Rss_bytes", 0),
            "rank_pss_sum_bytes": sample.get("rank_sums", {}).get("Pss_bytes", 0),
            "rank_private_sum_bytes": sample.get("rank_sums", {}).get("Private_bytes", 0),
            "cgroup_current_bytes": cgroup.get("current_bytes") or 0,
            "host_used_bytes": sample.get("host", {}).get("used_excluding_available_bytes") or 0,
            "max_gpu_used_bytes": max((gpu.get("used_bytes") or 0 for gpu in sample.get("gpus", [])), default=0),
        }
        for key, value in candidates.items():
            phase_max[key] = max(phase_max.get(key, 0), value)

    def as_dict(self) -> dict[str, Any]:
        rank_lifecycle = {}
        for rank, lifecycle in self.rank_lifecycle.items():
            entry = dict(lifecycle)
            entry["process_counter_deltas"] = counter_delta(
                entry.get("first_process_counters", {}),
                entry.get("last_process_counters", {}),
            )
            rank_lifecycle[rank] = entry
        return {
            "samples": self.samples,
            "max_simultaneous_rank_sums_bytes": self.max_rank_sums,
            "sum_of_individual_rank_peak_rss_bytes_do_not_use_as_host_peak": sum(self.rank_peak_rss.values()),
            "individual_rank_peak_rss_bytes": self.rank_peak_rss,
            "max_observed_cgroup_current_bytes": self.max_cgroup_current,
            "kernel_reported_cgroup_peak_bytes": self.reported_cgroup_peak,
            "max_observed_host_used_bytes": self.max_host_used,
            "physical_gpu_peak_used_bytes": self.gpu_peaks,
            "physical_gpu_total_bytes": self.gpu_totals,
            "phase_maxima": self.phase_maxima,
            "rank_lifecycle": rank_lifecycle,
            "host_counter_window": {
                "first": self.first_host_counters or {},
                "last": self.last_host_counters or {},
                "deltas": counter_delta(
                    self.first_host_counters or {},
                    self.last_host_counters or {},
                ),
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--container", default="lightx2v-gfx1201", help="Docker container name")
    parser.add_argument("--log", type=Path, help="Host-side inference log used for best-effort phase labels")
    parser.add_argument("--output", type=Path, help="JSONL output (default: lightx2v-memory-<UTC>.jsonl)")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=30.0,
        help=(
            "Periodic terminal status interval in seconds; phase changes and the "
            "final peak summary are always printed (default: 30, 0 disables heartbeats)"
        ),
    )
    parser.add_argument("--start-timeout", type=float, default=900.0, help="Seconds to wait for the container")
    parser.add_argument("--expected-ranks", type=int, default=16, help="Warn unless this many rank processes are found")
    parser.add_argument("--host-limit-gib", type=float, default=512.0, help="Diagnostic threshold for cgroup memory")
    parser.add_argument("--gpu-limit-gib", type=float, default=32.0, help="Diagnostic threshold for per-card VRAM")
    parser.add_argument(
        "--gpu-indices",
        type=parse_gpu_indices,
        help="comma-separated host ROCm indices to sample (default: all AMD cards)",
    )
    parser.add_argument("--once", action="store_true", help="Take one sample and exit")
    parser.add_argument(
        "--phase-schema",
        choices=("wan", "minimax_h3", "ltx2"),
        default="wan",
        help="Log phase labels: wan keeps legacy regex; minimax_h3 and ltx2 use normalized [Runtime] stage events",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    if args.print_interval < 0:
        raise SystemExit("--print-interval must be non-negative")
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = Path(f"lightx2v-memory-{stamp}.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")

    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    deadline = time.monotonic() + args.start_timeout
    while not stop and container_state(args.container) != "running":
        if time.monotonic() >= deadline:
            print(f"ERROR: container {args.container!r} did not start within {args.start_timeout:.0f}s", file=sys.stderr)
            return 2
        time.sleep(min(args.interval, 2.0))

    tracker_cls = {
        "wan": PhaseTracker,
        "minimax_h3": MiniMaxH3PhaseTracker,
        "ltx2": LTX2PhaseTracker,
    }[args.phase_schema]
    tracker = tracker_cls(args.log, args.expected_ranks)
    summary = Summary()
    previous_phase = None
    last_status_time: float | None = None
    host_limit_warned = False
    gpu_limit_warned = False
    print(
        f"Memory monitor: samples={args.interval:g}s, "
        f"terminal={'phase-only' if args.print_interval == 0 else f'{args.print_interval:g}s + phase changes'}, "
        f"output={args.output}",
        file=sys.stderr,
    )

    final_state = "unknown"
    with args.output.open("a", encoding="utf-8") as output:
        while not stop:
            state = container_state(args.container)
            if state != "running":
                final_state = state
                break
            sample: dict[str, Any] = {
                "schema_version": 2,
                "timestamp_utc": utc_now(),
                "monotonic_seconds": time.monotonic(),
                "phase": tracker.update(),
                "container_name": args.container,
                "container": sample_container(args.container),
                "host": sample_host(),
                "gpus": sample_gpus(
                    set(args.gpu_indices) if args.gpu_indices is not None else None
                ),
            }
            if any(
                rank.get("workload") == "collective_probe"
                for rank in sample["container"].get("ranks", [])
            ):
                sample["phase"] = "collective_probe"
            add_host_pids(args.container, sample["container"])
            add_rank_totals(sample)
            summary.add(sample)
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            output.flush()

            rank_count = sample["rank_sums"]["rank_count"]
            rss = sample["rank_sums"].get("Rss_bytes")
            pss = sample["rank_sums"].get("Pss_bytes")
            private = sample["rank_sums"].get("Private_bytes")
            cgroup_current = sample.get("container", {}).get("cgroup", {}).get("current_bytes")
            max_gpu = max((gpu.get("used_bytes") or 0 for gpu in sample["gpus"]), default=None)
            changed = sample["phase"] != previous_phase
            emit_status = should_emit_status(
                phase=sample["phase"],
                previous_phase=previous_phase,
                monotonic_seconds=sample["monotonic_seconds"],
                last_status_time=last_status_time,
                print_interval=args.print_interval,
            )
            previous_phase = sample["phase"]
            if emit_status:
                prefix = "MEM-PHASE" if changed else "MEM"
                print(
                    f"{prefix}={sample['phase']} ranks={rank_count}/{args.expected_ranks} "
                    f"sumRSS={gib(rss)} sumPSS={gib(pss)} sumPrivate={gib(private)} "
                    f"cgroup={gib(cgroup_current)} maxGPU={gib(max_gpu)}",
                    file=sys.stderr,
                )
                last_status_time = sample["monotonic_seconds"]
            if (
                not host_limit_warned
                and cgroup_current is not None
                and cgroup_current >= args.host_limit_gib * GIB
            ):
                print(f"WARNING: cgroup memory reached diagnostic limit {args.host_limit_gib:.2f} GiB", file=sys.stderr)
                host_limit_warned = True
            if (
                not gpu_limit_warned
                and max_gpu is not None
                and max_gpu >= args.gpu_limit_gib * GIB
            ):
                print(f"WARNING: a GPU reached diagnostic limit {args.gpu_limit_gib:.2f} GiB", file=sys.stderr)
                gpu_limit_warned = True
            if args.once:
                break
            time.sleep(args.interval)

    if summary.active_rank_keys:
        observation = "monitor_stopped" if stop else f"container_{final_state}"
        summary.close_active_ranks(utc_now(), time.monotonic(), observation)

    summary_data = summary.as_dict()
    summary_path.write_text(json.dumps(summary_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}", file=sys.stderr)
    kernel_peak = summary_data["kernel_reported_cgroup_peak_bytes"]
    if kernel_peak > 0:
        cgroup_peak_text = f"kernel cgroup peak={gib(kernel_peak)}"
    else:
        cgroup_peak_text = (
            "kernel cgroup peak=unavailable; "
            "sampled max cgroup current (lower bound)="
            f"{gib(summary_data['max_observed_cgroup_current_bytes'])}"
        )
    max_gpu_peak = max(
        summary_data["physical_gpu_peak_used_bytes"].values(), default=0
    )
    print(
        "Peak scopes: "
        f"simultaneous sumRSS={gib(summary_data['max_simultaneous_rank_sums_bytes'].get('Rss_bytes'))}, "
        f"simultaneous sumPSS={gib(summary_data['max_simultaneous_rank_sums_bytes'].get('Pss_bytes'))}, "
        f"{cgroup_peak_text}, "
        f"max physical GPU VRAM={gib(max_gpu_peak)}, "
        f"sum(individual rank peak RSS; invalid as host peak)={gib(summary_data['sum_of_individual_rank_peak_rss_bytes_do_not_use_as_host_peak'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
