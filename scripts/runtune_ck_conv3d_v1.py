#!/usr/bin/env python3
"""Two-arm Wan2.1 VAE runtune for PyTorch offload versus CK v1 resident."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path


MIB = 1024 * 1024
ARM_ORDER = ("pytorch_offload", "ck_v1_resident")
ARM_CONFIG = {
    "pytorch_offload": {
        "backend": "pytorch",
        "vae_cpu_offload": True,
    },
    "ck_v1_resident": {
        "backend": "Aiter_CK_Conv3d_v1",
        "vae_cpu_offload": False,
    },
}


def log(arm: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [ck-v1:{arm}] {message}", flush=True)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_route_histogram_csv(path: Path, rounds: list[dict[str, object]]) -> None:
    fields = (
        "round",
        "phase",
        "route",
        "selection",
        "instance_id",
        "calls",
        "input_shape",
        "weight_shape",
        "output_shape",
        "input_stride",
        "weight_stride",
        "conv_stride",
        "padding",
        "dilation",
        "groups",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_entry in rounds:
            for phase in ("encode", "decode"):
                for route in round_entry[phase].get("route_histogram", []):
                    row = {
                        "round": round_entry["round"],
                        "phase": phase,
                        **route,
                    }
                    for name in (
                        "input_shape",
                        "weight_shape",
                        "output_shape",
                        "input_stride",
                        "weight_stride",
                        "conv_stride",
                        "padding",
                        "dilation",
                    ):
                        row[name] = json.dumps(row[name], separators=(",", ":"))
                    writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--participants", type=int, default=2)
    parser.add_argument("--arm", choices=ARM_ORDER)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--min-close-percent", type=float, default=99.9)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()

    if args.aggregate_root is not None:
        return args

    required = ("output_dir", "reference_dir", "sync_dir", "arm", "vae_path")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing run arguments: " + ", ".join(missing))
    if args.participants != 2:
        parser.error("this runtune requires exactly two participants")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must satisfy (frames - 1) % 4 == 0")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def configure_environment(arm: str) -> dict[str, object]:
    config = ARM_CONFIG[arm]
    os.environ["PLATFORM"] = "amd_rocm"
    os.environ["DTYPE"] = "BF16"
    os.environ["LIGHTX2V_AMD_MIOPEN"] = "0"
    os.environ["USE_CHANNELS_LAST_3D"] = "0"
    os.environ["LIGHTX2V_VAE_CONV3D_BACKEND"] = str(config["backend"])
    return config


def wait_for(predicate, *, description: str, sync_dir: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if (sync_dir / "abort").exists():
            raise RuntimeError(f"the other runtune arm failed while waiting for {description}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.2)


def barrier(args: argparse.Namespace, name: str) -> None:
    marker = args.sync_dir / f"{name}.{args.arm}.ready"
    marker.touch()
    wait_for(
        lambda: len(list(args.sync_dir.glob(f"{name}.*.ready"))) >= args.participants,
        description=f"{name} barrier",
        sync_dir=args.sync_dir,
        timeout=args.timeout_seconds,
    )
    log(args.arm, f"barrier passed: {name}")


def atomic_torch_save(torch, tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def tensor_digest(torch, tensor, chunk_elements: int = 4_194_304) -> str:
    digest = hashlib.sha256()
    flattened = tensor.contiguous().view(-1).view(torch.uint16)
    for start in range(0, flattened.numel(), chunk_elements):
        digest.update(flattened[start : start + chunk_elements].numpy().tobytes())
    return digest.hexdigest()


def tensor_summary(torch, tensor) -> dict[str, object]:
    flattened = tensor.contiguous().view(-1)
    total = 0.0
    squared = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    chunk_elements = 2_097_152
    for start in range(0, flattened.numel(), chunk_elements):
        chunk = flattened[start : start + chunk_elements].float()
        total += float(chunk.sum().item())
        squared += float((chunk * chunk).sum().item())
        minimum = min(minimum, float(chunk.min().item()))
        maximum = max(maximum, float(chunk.max().item()))
    mean = total / flattened.numel()
    variance = max(0.0, squared / flattened.numel() - mean * mean)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
        "sha256": tensor_digest(torch, tensor),
    }


def compare_tensors(
    torch,
    candidate,
    reference,
    *,
    atol: float,
    rtol: float,
    chunk_elements: int = 2_097_152,
) -> dict[str, object]:
    if tuple(candidate.shape) != tuple(reference.shape):
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "passed": False,
        }

    candidate = candidate.contiguous().view(-1)
    reference = reference.contiguous().view(-1)
    count = candidate.numel()
    close_count = 0
    exact_count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    dot = 0.0
    candidate_squared = 0.0
    reference_squared = 0.0
    for start in range(0, count, chunk_elements):
        lhs_raw = candidate[start : start + chunk_elements]
        rhs_raw = reference[start : start + chunk_elements]
        exact_count += int((lhs_raw == rhs_raw).sum().item())
        lhs = lhs_raw.float()
        rhs = rhs_raw.float()
        difference = lhs - rhs
        absolute = difference.abs()
        close_count += int(torch.isclose(lhs, rhs, atol=atol, rtol=rtol).sum().item())
        absolute_sum += float(absolute.sum().item())
        squared_sum += float((difference * difference).sum().item())
        max_abs = max(max_abs, float(absolute.max().item()))
        dot += float((lhs * rhs).sum().item())
        candidate_squared += float((lhs * lhs).sum().item())
        reference_squared += float((rhs * rhs).sum().item())

    mse = squared_sum / count
    denominator = math.sqrt(candidate_squared * reference_squared)
    return {
        "shape_match": True,
        "elements": count,
        "exact_percent": exact_count * 100.0 / count,
        "close_percent": close_count * 100.0 / count,
        "mean_abs": absolute_sum / count,
        "max_abs": max_abs,
        "rmse": math.sqrt(mse),
        "cosine": dot / denominator if denominator else 1.0,
        "psnr_db_peak2": None if mse == 0 else 10.0 * math.log10(4.0 / mse),
    }


def reference_metrics(
    torch,
    output,
    *,
    phase: str,
    round_index: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    reference_path = args.reference_dir / f"round{round_index}_{phase}.pt"
    ready_path = reference_path.with_suffix(".ready")
    if args.arm == "pytorch_offload":
        log(args.arm, f"publishing round {round_index} {phase} reference")
        atomic_torch_save(torch, output, reference_path)
        ready_path.touch()
        reference = output
    else:
        log(args.arm, f"waiting for round {round_index} PyTorch {phase} reference")
        wait_for(
            ready_path.exists,
            description=f"PyTorch {phase} reference for round {round_index}",
            sync_dir=args.sync_dir,
            timeout=args.timeout_seconds,
        )
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)

    metrics = compare_tensors(
        torch,
        output,
        reference,
        atol=args.atol,
        rtol=args.rtol,
    )
    metrics["passed"] = bool(
        metrics.get("shape_match", False)
        and metrics.get("close_percent", 0.0) >= args.min_close_percent
        and metrics.get("cosine", 0.0) >= args.min_cosine
    )
    if reference is not output:
        del reference
    return metrics


def memory_snapshot(torch) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "free_mib": free_bytes / MIB,
        "total_mib": total_bytes / MIB,
    }


def measure_cuda(torch, function) -> tuple[object, dict[str, object]]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = memory_snapshot(torch)
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    output = function()
    end.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    return output, {
        "wall_seconds": wall_seconds,
        "gpu_seconds": float(start.elapsed_time(end)) / 1000.0,
        "memory_before": before,
        "memory_after": memory_snapshot(torch),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "host_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_video_input(torch, args: argparse.Namespace, round_index: int):
    generator = torch.Generator(device="cpu").manual_seed(args.seed + round_index)
    first_frame = torch.randn(
        (1, 3, args.height, args.width), generator=generator, dtype=torch.float32
    ).clamp_(-3.0, 3.0).div_(3.0)
    video = torch.zeros(
        (1, 3, args.frames, args.height, args.width),
        device="cuda",
        dtype=torch.bfloat16,
    )
    video[:, :, 0].copy_(first_frame.to(device="cuda", dtype=torch.bfloat16))
    return video


def build_latent_input(torch, args: argparse.Namespace, round_index: int):
    latent_frames = 1 + (args.frames - 1) // 4
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + round_index)
    latent = torch.randn(
        (
            args.latent_channels,
            latent_frames,
            args.height // 8,
            args.width // 8,
        ),
        generator=generator,
        dtype=torch.float32,
    )
    return latent.to(device="cuda", dtype=torch.bfloat16)


def device_metadata(torch, args: argparse.Namespace) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "host_gpu_id": args.gpu_id,
        "visible_device_index": 0,
        "device_name": properties.name,
        "gcn_arch_name": getattr(properties, "gcnArchName", "unknown"),
        "total_memory_mib": properties.total_memory / MIB,
        "automatic_frequency": True,
    }


def run_arm(args: argparse.Namespace) -> dict[str, object]:
    config = configure_environment(args.arm)

    import torch

    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
    from lightx2v_platform.ops.conv3d import (
        get_ck_conv3d_provenance,
        get_ck_conv3d_route_counts,
        get_ck_conv3d_route_report,
        reset_ck_conv3d_state,
    )

    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        raise RuntimeError("this runtune requires PyTorch ROCm and a visible GPU")
    torch.backends.cudnn.enabled = False
    if torch.backends.cudnn.enabled:
        raise RuntimeError("MIOpen must remain disabled in both VAE arms")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    args.sync_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    reset_ck_conv3d_state()

    cpu_offload = bool(config["vae_cpu_offload"])
    log(
        args.arm,
        f"backend={config['backend']}, MIOpen=0, tiling=0, "
        f"VAE_CPU_OFFLOAD={int(cpu_offload)}, channels_last_3d=0",
    )
    model, load_measurement = measure_cuda(
        torch,
        lambda: WanVAE(
            z_dim=args.latent_channels,
            vae_path=str(args.vae_path),
            dtype=torch.bfloat16,
            device="cpu" if cpu_offload else "cuda",
            parallel=False,
            use_tiling=False,
            cpu_offload=cpu_offload,
            load_from_rank0=False,
        ),
    )
    log(
        args.arm,
        f"model loaded in {load_measurement['wall_seconds']:.3f}s; "
        f"peak allocated={load_measurement['peak_allocated_mib']:.1f} MiB",
    )
    barrier(args, "model_loaded")

    rounds = []
    gate_reasons = []
    with torch.inference_mode():
        for round_index in range(args.rounds):
            video = build_video_input(torch, args, round_index)
            get_ck_conv3d_route_counts(reset=True)
            get_ck_conv3d_route_report(reset=True)
            barrier(args, f"round{round_index}_encode_start")
            log(args.arm, f"round {round_index}: VAE encode start")
            encoded, encode_measurement = measure_cuda(torch, lambda: model.encode(video))
            encode_routes = get_ck_conv3d_route_counts(reset=True)
            encode_route_histogram = get_ck_conv3d_route_report(reset=True)
            encoded_cpu = encoded.detach().cpu().contiguous()
            encode_summary = tensor_summary(torch, encoded_cpu)
            encode_comparison = reference_metrics(
                torch,
                encoded_cpu,
                phase="encode",
                round_index=round_index,
                args=args,
            )
            if not encode_comparison["passed"]:
                gate_reasons.append(f"round {round_index} encode correctness failed")
            if args.arm == "ck_v1_resident" and encode_routes.get("ck", 0) == 0:
                gate_reasons.append(f"round {round_index} encode had zero CK calls")
            log(
                args.arm,
                f"round {round_index}: encode {encode_measurement['wall_seconds']:.3f}s, "
                f"CK calls={encode_routes.get('ck', 0)}",
            )
            del video, encoded, encoded_cpu
            gc.collect()
            torch.cuda.empty_cache()

            latent = build_latent_input(torch, args, round_index)
            get_ck_conv3d_route_counts(reset=True)
            get_ck_conv3d_route_report(reset=True)
            barrier(args, f"round{round_index}_decode_start")
            log(args.arm, f"round {round_index}: VAE decode start")
            decoded, decode_measurement = measure_cuda(torch, lambda: model.decode(latent))
            decode_routes = get_ck_conv3d_route_counts(reset=True)
            decode_route_histogram = get_ck_conv3d_route_report(reset=True)
            decoded_cpu = decoded.detach().cpu().contiguous()
            decode_summary = tensor_summary(torch, decoded_cpu)
            decode_comparison = reference_metrics(
                torch,
                decoded_cpu,
                phase="decode",
                round_index=round_index,
                args=args,
            )
            if not decode_comparison["passed"]:
                gate_reasons.append(f"round {round_index} decode correctness failed")
            if args.arm == "ck_v1_resident" and decode_routes.get("ck", 0) == 0:
                gate_reasons.append(f"round {round_index} decode had zero CK calls")
            log(
                args.arm,
                f"round {round_index}: decode {decode_measurement['wall_seconds']:.3f}s, "
                f"CK calls={decode_routes.get('ck', 0)}",
            )
            del latent, decoded, decoded_cpu
            gc.collect()
            torch.cuda.empty_cache()

            rounds.append(
                {
                    "round": round_index,
                    "encode": {
                        "measurement": encode_measurement,
                        "output": encode_summary,
                        "vs_pytorch_offload": encode_comparison,
                        "route_counts": encode_routes,
                        "route_histogram": encode_route_histogram,
                    },
                    "decode": {
                        "measurement": decode_measurement,
                        "output": decode_summary,
                        "vs_pytorch_offload": decode_comparison,
                        "route_counts": decode_routes,
                        "route_histogram": decode_route_histogram,
                    },
                }
            )

    result = {
        "schema_version": 1,
        "experiment": "wan21_vae_ck_conv3d_v1",
        "kind": "arm_result",
        "arm": args.arm,
        "gpu_id": args.gpu_id,
        "configuration": {
            "backend": config["backend"],
            "miopen": False,
            "tiling": False,
            "vae_cpu_offload": cpu_offload,
            "channels_last_3d": False,
            "dtype": "BF16",
            "automatic_frequency": True,
        },
        "workload": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "latent_channels": args.latent_channels,
            "latent_frames": 1 + (args.frames - 1) // 4,
            "seed": args.seed,
            "rounds": args.rounds,
        },
        "correctness_thresholds": {
            "atol": args.atol,
            "rtol": args.rtol,
            "min_close_percent": args.min_close_percent,
            "min_cosine": args.min_cosine,
        },
        "model_load": load_measurement,
        "rounds": rounds,
        "gate": {"passed": not gate_reasons, "reasons": sorted(set(gate_reasons))},
        "provenance": get_ck_conv3d_provenance(),
        "device": device_metadata(torch, args),
    }
    atomic_json(args.output_dir / "result.json", result)
    write_route_histogram_csv(args.output_dir / "route_histogram.csv", rounds)
    log(args.arm, "result.json written")
    return result


def median(values) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return 0.0
    return statistics.median(finite)


def min_metric(rounds, phase: str, field: str) -> float | None:
    values = [
        entry[phase]["vs_pytorch_offload"].get(field)
        for entry in rounds
    ]
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(finite) if finite else None


def max_metric(rounds, phase: str, field: str) -> float | None:
    values = [
        entry[phase]["vs_pytorch_offload"].get(field)
        for entry in rounds
    ]
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def flatten_result(result: dict[str, object]) -> dict[str, object]:
    rounds = result["rounds"]
    encode_wall = [entry["encode"]["measurement"]["wall_seconds"] for entry in rounds]
    decode_wall = [entry["decode"]["measurement"]["wall_seconds"] for entry in rounds]
    encode_gpu = [entry["encode"]["measurement"]["gpu_seconds"] for entry in rounds]
    decode_gpu = [entry["decode"]["measurement"]["gpu_seconds"] for entry in rounds]
    encode_peak = [entry["encode"]["measurement"]["peak_allocated_mib"] for entry in rounds]
    decode_peak = [entry["decode"]["measurement"]["peak_allocated_mib"] for entry in rounds]
    encode_reserved = [entry["encode"]["measurement"]["peak_reserved_mib"] for entry in rounds]
    decode_reserved = [entry["decode"]["measurement"]["peak_reserved_mib"] for entry in rounds]
    return {
        "arm": result["arm"],
        "gpu_id": result["gpu_id"],
        **result["configuration"],
        "model_load_wall_s": result["model_load"]["wall_seconds"],
        "encode_wall_s_median": median(encode_wall),
        "decode_wall_s_median": median(decode_wall),
        "vae_wall_s_median": median(encode_wall) + median(decode_wall),
        "encode_gpu_s_median": median(encode_gpu),
        "decode_gpu_s_median": median(decode_gpu),
        "vae_gpu_s_median": median(encode_gpu) + median(decode_gpu),
        "peak_allocated_mib": max(encode_peak + decode_peak),
        "peak_reserved_mib": max(encode_reserved + decode_reserved),
        "encode_cosine_min": min_metric(rounds, "encode", "cosine"),
        "encode_max_abs_max": max_metric(rounds, "encode", "max_abs"),
        "decode_cosine_min": min_metric(rounds, "decode", "cosine"),
        "decode_max_abs_max": max_metric(rounds, "decode", "max_abs"),
        "ck_calls": sum(
            entry[phase]["route_counts"].get("ck", 0)
            for entry in rounds
            for phase in ("encode", "decode")
        ),
        "gate_passed": result["gate"]["passed"],
    }


def speedup(baseline: float, candidate: float) -> float:
    return baseline / candidate if candidate > 0 else 0.0


def format_metric(value: float | None, precision: str) -> str:
    return "n/a" if value is None else format(value, precision)


def aggregate(root: Path) -> dict[str, object]:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("gpu*/result.json"))
    ]
    found = {result.get("arm") for result in results}
    if found != set(ARM_ORDER) or len(results) != len(ARM_ORDER):
        raise ValueError(
            f"expected exactly arms {ARM_ORDER}, found {sorted(map(str, found))}"
        )

    workload = results[0]["workload"]
    thresholds = results[0]["correctness_thresholds"]
    expected_workload = {
        "height": 720,
        "width": 1280,
        "frames": 81,
        "latent_channels": 16,
        "latent_frames": 21,
    }
    if any(workload.get(key) != value for key, value in expected_workload.items()):
        raise ValueError("the runtune workload is not Wan2.1 720p/81f")
    for result in results[1:]:
        if result["workload"] != workload:
            raise ValueError("the two arms did not run the same workload")
        if result["correctness_thresholds"] != thresholds:
            raise ValueError("the two arms used different correctness thresholds")
    for result in results:
        config = result["configuration"]
        expected_arm = ARM_CONFIG[result["arm"]]
        if (
            config.get("backend") != expected_arm["backend"]
            or config.get("vae_cpu_offload") != expected_arm["vae_cpu_offload"]
        ):
            raise ValueError(f"arm {result['arm']} changed its VAE contract")
        if (
            config.get("miopen")
            or config.get("tiling")
            or config.get("channels_last_3d")
            or config.get("dtype") != "BF16"
        ):
            raise ValueError(f"arm {result['arm']} changed a frozen non-VAE setting")

    rows = sorted(
        (flatten_result(result) for result in results),
        key=lambda row: ARM_ORDER.index(row["arm"]),
    )
    rows_by_arm = {row["arm"]: row for row in rows}
    baseline = rows_by_arm["pytorch_offload"]
    candidate = rows_by_arm["ck_v1_resident"]
    comparison = {
        "baseline": "pytorch_offload",
        "candidate": "ck_v1_resident",
        "encode_wall_speedup": speedup(
            baseline["encode_wall_s_median"], candidate["encode_wall_s_median"]
        ),
        "decode_wall_speedup": speedup(
            baseline["decode_wall_s_median"], candidate["decode_wall_s_median"]
        ),
        "vae_wall_speedup": speedup(
            baseline["vae_wall_s_median"], candidate["vae_wall_s_median"]
        ),
        "encode_gpu_speedup": speedup(
            baseline["encode_gpu_s_median"], candidate["encode_gpu_s_median"]
        ),
        "decode_gpu_speedup": speedup(
            baseline["decode_gpu_s_median"], candidate["decode_gpu_s_median"]
        ),
        "vae_gpu_speedup": speedup(
            baseline["vae_gpu_s_median"], candidate["vae_gpu_s_median"]
        ),
    }
    failures = [
        {"arm": result["arm"], "reasons": result["gate"]["reasons"]}
        for result in results
        if not result["gate"]["passed"]
    ]
    summary = {
        "schema_version": 1,
        "experiment": "wan21_vae_ck_conv3d_v1",
        "note": "Each VAE arm ran on a different GPU; use ROTATION=1 to check card bias.",
        "workload": workload,
        "correctness_thresholds": thresholds,
        "arms": rows,
        "comparison": comparison,
        "gate": {"passed": not failures, "failures": failures},
    }
    atomic_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "arm               gpu offload CK calls  encode wall/gpu  "
        "decode wall/gpu  total wall  peak A/R MiB  decode cosine/maxabs"
    )
    for row in rows:
        print(
            f"{row['arm']:<18} {str(row['gpu_id']):>3} "
            f"{int(row['vae_cpu_offload']):>7} {row['ck_calls']:>8}  "
            f"{row['encode_wall_s_median']:7.3f}/{row['encode_gpu_s_median']:<7.3f}  "
            f"{row['decode_wall_s_median']:7.3f}/{row['decode_gpu_s_median']:<7.3f}  "
            f"{row['vae_wall_s_median']:9.3f}  "
            f"{row['peak_allocated_mib']:7.1f}/{row['peak_reserved_mib']:<7.1f}  "
            f"{format_metric(row['decode_cosine_min'], '.8f')}/"
            f"{format_metric(row['decode_max_abs_max'], '.6g')}"
        )
    print(
        "CK v1 resident vs PyTorch offload: "
        f"encode={comparison['encode_wall_speedup']:.3f}x, "
        f"decode={comparison['decode_wall_speedup']:.3f}x, "
        f"VAE total={comparison['vae_wall_speedup']:.3f}x (wall time)"
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.aggregate_root is not None:
        try:
            summary = aggregate(args.aggregate_root.resolve())
        except Exception:
            traceback.print_exc()
            return 1
        return 0 if summary["gate"]["passed"] else 1

    try:
        result = run_arm(args)
        log(args.arm, "runtune arm completed successfully")
        return 0 if result["gate"]["passed"] else 1
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output_dir / "failure.json",
            {
                "schema_version": 1,
                "experiment": "wan21_vae_ck_conv3d_v1",
                "arm": args.arm,
                "gpu_id": args.gpu_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        args.sync_dir.mkdir(parents=True, exist_ok=True)
        (args.sync_dir / "abort").touch()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
