#!/usr/bin/env python3
"""Four-arm Wan2.1 VAE T0 benchmark and result aggregator."""

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
ARM_ORDER = ("legacy", "miopen", "miopen_untiled", "optimized")


def log(arm: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [t0:{arm}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--sync-dir", type=Path)
    parser.add_argument("--participants", type=int, default=4)
    parser.add_argument("--arm", choices=ARM_ORDER)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--miopen", type=int, choices=(0, 1))
    parser.add_argument("--tiling", type=int, choices=(0, 1))
    parser.add_argument("--vae-cpu-offload", type=int, choices=(0, 1))
    parser.add_argument("--vae-path", type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.aggregate_root is not None:
        return args
    required = (
        "output_dir",
        "reference_dir",
        "sync_dir",
        "arm",
        "miopen",
        "tiling",
        "vae_cpu_offload",
        "vae_path",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing run arguments: " + ", ".join(missing))
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must satisfy (frames - 1) % 4 == 0")
    return args


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def wait_for(predicate, *, description: str, sync_dir: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if (sync_dir / "abort").exists():
            raise RuntimeError(f"another T0 arm failed while waiting for {description}")
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
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(tensor, temp)
    os.replace(temp, path)


def tensor_digest(torch, tensor, chunk_elements: int = 4_194_304) -> str:
    digest = hashlib.sha256()
    flattened = tensor.contiguous().view(-1).view(torch.uint16)
    for start in range(0, flattened.numel(), chunk_elements):
        digest.update(flattened[start : start + chunk_elements].numpy().tobytes())
    return digest.hexdigest()


def compare_tensors(torch, candidate, reference, chunk_elements: int = 2_097_152) -> dict[str, object]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}")
    candidate = candidate.contiguous().view(-1)
    reference = reference.contiguous().view(-1)
    count = candidate.numel()
    abs_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    dot = 0.0
    candidate_squared = 0.0
    reference_squared = 0.0
    exact = 0
    for start in range(0, count, chunk_elements):
        lhs_raw = candidate[start : start + chunk_elements]
        rhs_raw = reference[start : start + chunk_elements]
        exact += int((lhs_raw == rhs_raw).sum().item())
        lhs = lhs_raw.float()
        rhs = rhs_raw.float()
        diff = lhs - rhs
        abs_diff = diff.abs()
        abs_sum += float(abs_diff.sum().item())
        squared_sum += float((diff * diff).sum().item())
        max_abs = max(max_abs, float(abs_diff.max().item()))
        dot += float((lhs * rhs).sum().item())
        candidate_squared += float((lhs * lhs).sum().item())
        reference_squared += float((rhs * rhs).sum().item())
    mse = squared_sum / count
    denominator = math.sqrt(candidate_squared * reference_squared)
    return {
        "elements": count,
        "exact_percent": exact * 100.0 / count,
        "mean_abs": abs_sum / count,
        "max_abs": max_abs,
        "rmse": math.sqrt(mse),
        "cosine": dot / denominator if denominator else 1.0,
        "psnr_db_peak2": None if mse == 0 else 10.0 * math.log10(4.0 / mse),
    }


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
    start_memory = memory_snapshot(torch)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    end_memory = memory_snapshot(torch)
    return output, {
        "seconds": elapsed,
        "start": start_memory,
        "end": end_memory,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "host_max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def reference_metrics(torch, output, *, phase: str, round_index: int, args: argparse.Namespace) -> dict[str, object]:
    reference_path = args.reference_dir / f"round{round_index}_{phase}.pt"
    ready_path = reference_path.with_suffix(".ready")
    if args.arm == "legacy":
        log(args.arm, f"publishing round {round_index} {phase} reference")
        atomic_torch_save(torch, output, reference_path)
        ready_path.touch()
        reference = output
    else:
        log(args.arm, f"waiting for round {round_index} legacy {phase} reference")
        wait_for(
            ready_path.exists,
            description=f"legacy {phase} reference for round {round_index}",
            sync_dir=args.sync_dir,
            timeout=args.timeout_seconds,
        )
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    log(args.arm, f"comparing round {round_index} {phase} output element by element")
    metrics = compare_tensors(torch, output, reference)
    if reference is not output:
        del reference
    return metrics


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


def run_arm(args: argparse.Namespace) -> dict[str, object]:
    os.environ["PLATFORM"] = "amd_rocm"
    os.environ["DTYPE"] = "BF16"
    os.environ["LIGHTX2V_AMD_MIOPEN"] = str(args.miopen)
    os.environ["USE_CHANNELS_LAST_3D"] = "0"

    import torch

    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE

    torch.backends.cudnn.enabled = bool(args.miopen)
    if bool(torch.backends.cudnn.enabled) is not bool(args.miopen):
        raise RuntimeError("failed to apply requested MIOpen state")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    args.sync_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device_properties = torch.cuda.get_device_properties(0)
    log(
        args.arm,
        f"device={device_properties.name}, MIOpen={args.miopen}, tiling={args.tiling}, "
        f"VAE_CPU_OFFLOAD={args.vae_cpu_offload}",
    )

    log(args.arm, "loading Wan2.1 VAE")
    model, load_measurement = measure_cuda(
        torch,
        lambda: WanVAE(
            z_dim=args.latent_channels,
            vae_path=str(args.vae_path),
            dtype=torch.bfloat16,
            device="cpu" if args.vae_cpu_offload else "cuda",
            parallel=False,
            use_tiling=bool(args.tiling),
            cpu_offload=bool(args.vae_cpu_offload),
            load_from_rank0=False,
        ),
    )
    log(
        args.arm,
        f"model loaded in {load_measurement['seconds']:.3f}s; "
        f"peak allocated={load_measurement['peak_allocated_mib']:.1f} MiB",
    )
    barrier(args, "model_loaded")

    rounds = []
    with torch.inference_mode():
        for round_index in range(args.rounds):
            log(args.arm, f"round {round_index}: preparing 81-frame encoder input")
            video = build_video_input(torch, args, round_index)
            barrier(args, f"round{round_index}_encode_start")
            log(args.arm, f"round {round_index}: VAE encode start")
            encoded, encode_measurement = measure_cuda(torch, lambda: model.encode(video))
            log(
                args.arm,
                f"round {round_index}: VAE encode {encode_measurement['seconds']:.3f}s; "
                f"peak allocated={encode_measurement['peak_allocated_mib']:.1f} MiB",
            )
            encoded_cpu = encoded.detach().cpu().contiguous()
            encode_summary = tensor_summary(torch, encoded_cpu)
            encode_comparison = reference_metrics(
                torch, encoded_cpu, phase="encode", round_index=round_index, args=args
            )
            del video, encoded, encoded_cpu
            gc.collect()
            torch.cuda.empty_cache()

            log(args.arm, f"round {round_index}: preparing 21-frame decoder latent")
            latent = build_latent_input(torch, args, round_index)
            barrier(args, f"round{round_index}_decode_start")
            log(args.arm, f"round {round_index}: VAE decode start")
            decoded, decode_measurement = measure_cuda(torch, lambda: model.decode(latent))
            log(
                args.arm,
                f"round {round_index}: VAE decode {decode_measurement['seconds']:.3f}s; "
                f"peak allocated={decode_measurement['peak_allocated_mib']:.1f} MiB",
            )
            decoded_cpu = decoded.detach().cpu().contiguous()
            decode_summary = tensor_summary(torch, decoded_cpu)
            decode_comparison = reference_metrics(
                torch, decoded_cpu, phase="decode", round_index=round_index, args=args
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
                        "vs_legacy": encode_comparison,
                    },
                    "decode": {
                        "measurement": decode_measurement,
                        "output": decode_summary,
                        "vs_legacy": decode_comparison,
                    },
                }
            )
            barrier(args, f"round{round_index}_complete")

    result = {
        "experiment": "wan21_vae_t0",
        "arm": args.arm,
        "gpu_id": args.gpu_id,
        "flags": {
            "miopen": bool(args.miopen),
            "tiling": bool(args.tiling),
            "vae_cpu_offload": bool(args.vae_cpu_offload),
            "channels_last_3d": False,
        },
        "workload": {
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "latent_channels": args.latent_channels,
            "latent_frames": 1 + (args.frames - 1) // 4,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device_name": device_properties.name,
            "gcn_arch_name": getattr(device_properties, "gcnArchName", "unknown"),
            "device_total_memory_mib": device_properties.total_memory / MIB,
            "torch_miopen_switch": bool(torch.backends.cudnn.enabled),
        },
        "model_load": load_measurement,
        "rounds": rounds,
    }
    atomic_json(args.output_dir / "result.json", result)
    log(args.arm, "result.json written")
    return result


def median(values) -> float:
    return statistics.median(float(value) for value in values)


def flatten_result(result: dict[str, object]) -> dict[str, object]:
    rounds = result["rounds"]
    encode_seconds = [entry["encode"]["measurement"]["seconds"] for entry in rounds]
    decode_seconds = [entry["decode"]["measurement"]["seconds"] for entry in rounds]
    encode_peak = [entry["encode"]["measurement"]["peak_allocated_mib"] for entry in rounds]
    decode_peak = [entry["decode"]["measurement"]["peak_allocated_mib"] for entry in rounds]
    encode_reserved = [entry["encode"]["measurement"]["peak_reserved_mib"] for entry in rounds]
    decode_reserved = [entry["decode"]["measurement"]["peak_reserved_mib"] for entry in rounds]
    return {
        "arm": result["arm"],
        "gpu_id": result["gpu_id"],
        **result["flags"],
        "model_load_s": result["model_load"]["seconds"],
        "encode_s_median": median(encode_seconds),
        "decode_s_median": median(decode_seconds),
        "vae_s_median": median(encode_seconds) + median(decode_seconds),
        "encode_peak_allocated_mib": max(encode_peak),
        "decode_peak_allocated_mib": max(decode_peak),
        "peak_allocated_mib": max(encode_peak + decode_peak),
        "encode_peak_reserved_mib": max(encode_reserved),
        "decode_peak_reserved_mib": max(decode_reserved),
        "peak_reserved_mib": max(encode_reserved + decode_reserved),
        "encode_cosine_vs_legacy_min": min(entry["encode"]["vs_legacy"]["cosine"] for entry in rounds),
        "encode_max_abs_vs_legacy_max": max(entry["encode"]["vs_legacy"]["max_abs"] for entry in rounds),
        "decode_cosine_vs_legacy_min": min(entry["decode"]["vs_legacy"]["cosine"] for entry in rounds),
        "decode_max_abs_vs_legacy_max": max(entry["decode"]["vs_legacy"]["max_abs"] for entry in rounds),
    }


def speedup(rows_by_arm: dict[str, dict[str, object]], baseline: str, candidate: str, field: str) -> float:
    return float(rows_by_arm[baseline][field]) / float(rows_by_arm[candidate][field])


def aggregate(root: Path) -> dict[str, object]:
    results = []
    for result_path in sorted(root.glob("gpu*/result.json")):
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    found = {result["arm"] for result in results}
    if found != set(ARM_ORDER):
        raise ValueError(f"expected T0 arms {ARM_ORDER}, found {sorted(found)}")
    rows = sorted((flatten_result(result) for result in results), key=lambda row: ARM_ORDER.index(row["arm"]))
    rows_by_arm = {row["arm"]: row for row in rows}
    comparisons = {}
    for name, baseline, candidate in (
        ("miopen_only_vs_legacy", "legacy", "miopen"),
        ("untiled_increment", "miopen", "miopen_untiled"),
        ("resident_increment", "miopen_untiled", "optimized"),
        ("optimized_vs_legacy", "legacy", "optimized"),
    ):
        comparisons[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "encode_speedup": speedup(rows_by_arm, baseline, candidate, "encode_s_median"),
            "decode_speedup": speedup(rows_by_arm, baseline, candidate, "decode_s_median"),
            "vae_total_speedup": speedup(rows_by_arm, baseline, candidate, "vae_s_median"),
        }
    summary = {
        "experiment": "wan21_vae_t0",
        "note": "Each arm ran on a different GPU; rotate arms for a card-bias check.",
        "arms": rows,
        "comparisons": comparisons,
    }
    atomic_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("arm                 gpu  M T O   encode(s) decode(s) total(s) peak A/R(MiB)       decode cosine/maxabs")
    for row in rows:
        print(
            f"{row['arm']:<20} {str(row['gpu_id']):>3}  "
            f"{int(row['miopen'])} {int(row['tiling'])} {int(row['vae_cpu_offload'])}  "
            f"{row['encode_s_median']:9.3f} {row['decode_s_median']:9.3f} "
            f"{row['vae_s_median']:8.3f} "
            f"{row['peak_allocated_mib']:8.1f}/{row['peak_reserved_mib']:<8.1f}  "
            f"{row['decode_cosine_vs_legacy_min']:.8f}/{row['decode_max_abs_vs_legacy_max']:.6g}"
        )
    print("\nIncremental speedups (cross-GPU):")
    for name, comparison in comparisons.items():
        print(
            f"  {name:<24} encode {comparison['encode_speedup']:.3f}x, "
            f"decode {comparison['decode_speedup']:.3f}x, total {comparison['vae_total_speedup']:.3f}x"
        )
    return summary


def main() -> int:
    args = parse_args()
    if args.aggregate_root is not None:
        aggregate(args.aggregate_root)
        return 0
    try:
        result = run_arm(args)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "arm": args.arm,
            "gpu_id": args.gpu_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_json(args.output_dir / "failure.json", failure)
        args.sync_dir.mkdir(parents=True, exist_ok=True)
        (args.sync_dir / "abort").touch()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
