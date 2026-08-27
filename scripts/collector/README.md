# GFX1201 Model Collectors

This directory contains diagnostic launchers and their analysis helpers. They
observe the frozen production workload and must not silently change model
precision, offload policy, buffer counts, tuned kernels, or parallel topology.

## LTX-2.5 Day-0

- `run_ltx25_weight_census.sh` runs the no-GPU E0/E1 hard gates for the five
  BF16 split assets. It validates frozen size/SHA identities, Safetensors
  boundaries, the 48 x 84 Transformer schema, Gemma4 packed assets and
  cross-file version, component metadata, and the conservative two-slot memory
  account. It does not import Torch or initialize HIP.
- A full release-gate run is:

  ```bash
  bash scripts/collector/run_ltx25_weight_census.sh \
    --model-root /disk/ssd/difumodels-Dis/LTX-2.5 \
    --output-dir /disk/ssd/lightx2v-runtime/ltx25-day0 \
    --sha256
  ```

  The output directory receives `ltx25_weight_census.json`,
  `ltx25_weight_census.md`, and `ltx25_mock_memory.json`. A failed E0 retains
  both census reports and stops before E1.

- `run_ltx25_contract_collector.sh` automatically launches the in-image
  LTX-2.5 B0-reference route in discovery-only contract observation. It
  preserves the reference numerical/backend path, writes
  `ltx25_collector.l1.json` plus one
  `ltx25_collector.rankNNN.json` per denoise rank, and validates them with
  `validate_ltx25_contract_bundle.py`. A complete bundle is input for C1
  probes only; it is not a B0 acceptance or C1 PASS claim.

  ```bash
  bash scripts/collector/run_ltx25_contract_collector.sh
  ```

  Defaults are model root `/disk/ssd/difumodels-Dis/LTX-2.5`, GPUs `0,1`,
  two denoise ranks and a timestamped directory under
  `/disk/ssd/lightx2v-runtime/`. Missing Day-0 census evidence is generated
  automatically with full SHA verification. Options remain available only as
  overrides. The current executable video path is explicitly recorded as
  `conv_diagnostic`; it cannot qualify as official DiffVAE B0 acceptance.
  The in-container entry discovers LightX2V next to LightX2VRun first, then
  checks `/home/LightX2V` and `/workspace/LightX2V`; no path argument is
  required for either supported image layout. It also binds the discovered
  model root to the legacy `model_path` shell contract before sourcing
  LightX2V's `scripts/base/base.sh` under `set -u`.

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
