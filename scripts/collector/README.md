# GFX1201 Model Collectors

This directory contains diagnostic launchers and their analysis helpers. They
observe the frozen production workload and must not silently change model
precision, offload policy, buffer counts, tuned kernels, or parallel topology.

## LTX-2.3 B0

- `run_ltx23_weight_probe.sh`: runs the header-only collector through the exact
  production B0 launcher and container path. It therefore uses the same
  explicit image, two-GPU visibility, model/cache/result mounts and ROCm
  environment as inference. One-command collection:

  ```bash
  bash scripts/collector/run_ltx23_weight_probe.sh --image IMAGE
  ```

  It writes `ltx23_weight_probe.json` into the timestamped B0 result directory.
  Pass `--result-dir DIR` to choose that directory or `--sha256` for a full
  in-container artifact hash. The underlying `collect_ltx23_weight_probe.py`
  records every tensor name/dtype/shape, namespace variant, per-block schema,
  gate-logit compatibility and FP8 scale without loading tensor payloads.
- `collect_ltx23_host_evidence.py`: read-only host/GPU/source/model snapshots.
- Runtime schema-2 reports are emitted by LightX2V as
  `ltx23_collector.rank0.json` and `ltx23_collector.rank1.json`.
- `scripts/lib/monitor_run_memory.py --phase-schema ltx2` records host, cgroup,
  rank, and physical-VRAM evidence.
- `scripts/lib/summarize_ltx23_b0.py` validates the three preflights, exact
  two-rank GEMM probes, L1/L2/L3 routes, 8+3 steps, and the 1280x704
  121-frame/24-FPS audio-video output.

The launcher `scripts/run_ltx23_fp8_b0_704x1280_5s.sh` enables this collector
by default. Use `--no-collect` only for an explicit timing control. A
successful run waits for the memory monitor to finish and write its summary;
failure cleanup terminates it while retaining a failure snapshot.

## MiniMax-H3

- `run_minimax_h3_sp8_supercollect.sh`: SP8/Ulysses full-run evidence.
- `run_minimax_h3_sage_shape_sweep.sh`: SP4/SP8/CP2xU8-SP16 Sage contracts.
- `run_minimax_h3_megadiscovery.sh`: SP8 or CP2xU8-SP16 full-stack discovery.
- `collect_minimax_h3_host_evidence.py`: host, PCIe, KFD, NUMA, and storage snapshots.
- `benchmark_minimax_h3_collectives.py`: isolated production-shape communication probes.
- `summarize_minimax_h3_t1.py`: cross-rank H3 critical-path summary.
- `analyze_minimax_h3_megadiscovery.py`: unified JSON and Markdown diagnosis.

Run CP2xU8 SP16 MegaDiscovery with:

```bash
bash scripts/collector/run_minimax_h3_megadiscovery.sh \
  --image IMAGE \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The isolated probes run after inference in a separate process. Their synthetic
fusion variants and timing overhead never alter the production model process.
SP16 prints each active U8/CP2 measurement and checkpoints every completed
result. A timeout keeps those checkpoints and marks the unified report partial.
The historical root-level SuperCollect and Sage sweep shell entry points remain
compatibility wrappers.
