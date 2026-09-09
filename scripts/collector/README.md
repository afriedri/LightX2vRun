# GFX1201 Model Collectors

This directory contains diagnostic launchers and their analysis helpers. They
observe the frozen production workload and must not silently change model
precision, offload policy, buffer counts, tuned kernels, or parallel topology.

## LTX-2.5 Scale Entry and Optional Weight Audit

正常模型 scale 必须使用编号 10；它默认直接使用镜像内的 LightX2V 和 Aiter
AOT wheel，顺序运行 SP2/SP4/SP8/SP16，并统一收集内存、视频和 timing：

```bash
TIMING_RESULT_DIR=/disk/ssd/lightx2v-runtime/ltx25-scale-001 \
bash scripts/run_10_ltx25_c1_704x1280_121f_24fps_scale.sh \
  --image IMAGE
```

`LTX25-ScaleCollector.sh` 仅保留用于复现历史分布式 Video VAE 采证，不再作为
正常运行入口。Scale 运行不拉取或挂载源码，也不进行 Hot 选择。

- `run_ltx25_weight_census.sh` is an optional no-GPU audit for the five
  BF16 split assets. It validates frozen size/SHA identities, Safetensors
  boundaries, the 48 x 84 Transformer schema, Gemma4 packed assets and
  cross-file version, component metadata, and the conservative two-slot memory
  account. It does not import Torch or initialize HIP.
- Run it only when an explicit artifact audit is needed:

  ```bash
  bash scripts/collector/run_ltx25_weight_census.sh \
    --model-root /disk/ssd/difumodels-Dis/LTX-2.5 \
    --output-dir /disk/ssd/lightx2v-runtime/ltx25-day0 \
    --sha256
  ```

  The output directory receives `ltx25_weight_census.json`,
  `ltx25_weight_census.md`, and `ltx25_mock_memory.json`. A failed E0 retains
  both census reports and stops before E1. The runtime contract collector does
  not invoke or require this audit.

- `run_ltx25_contract_collector.sh` is the zero-argument Docker-host entry.
  It launches the standard LightX2V ROCm image and runs the LTX-2.5
  B0-reference route there in discovery-only contract observation. It
  preserves the reference numerical/backend path, writes
  `ltx25_collector.l1.json` plus one
  `ltx25_collector.rankNNN.json` per denoise rank, and validates them with
  `validate_ltx25_contract_bundle.py`. A complete bundle is input for C1
  probes only; it is not a B0 acceptance or C1 PASS claim.

  ```bash
  bash scripts/collector/run_ltx25_contract_collector.sh
  ```

  Defaults are model root `/disk/ssd/difumodels-Dis/LTX-2.5`, GPUs `0,1`,
  the image configured by `lightx2vModelrun`, two denoise ranks and a
  timestamped directory under `/disk/ssd/lightx2v-runtime/`. The host entry
  mounts the model read-only, the evidence directory writable, and LightX2VRun
  read-only. L1, SP2 L2/L3 and bundle validation run inside the image; no SHA,
  weight-schema census, VRAM handoff gate, or extra preflight is part of this
  collector path. The inner helper refuses direct host execution.
  `--image IMAGE` and `IMAGE=...` remain supported, and `DRY_RUN=1`
  prints the complete Docker contract. The current executable video path is
  explicitly recorded as `conv_diagnostic`; it cannot qualify as official
  DiffVAE B0 acceptance.

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

## H3 Ref2VA BF16 / INT8 ConvRot 权重采集

```bash
python3 scripts/collector/collect_minimax_h3_ref2v_weights.py \
  --model-root /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --output-dir /disk/ssd1/lightx2v-runtime/h3-ref2va-weight-inventory
```

固定采集未裁剪的 `minimax_h3_ref2va_bf16.safetensors` 与
`minimax_h3_ref2va_int8_convrot.safetensors`。复用现有 safetensors 解析器，
读取 header 和有上限的 ConvRot 整数标量，不读取权重矩阵、不初始化 GPU。
JSON/Markdown 包含文件结构、dtype、shape、groupsize 和两种格式差异。
这是资产采集，不是 GPU 推理或性能验收，也不加入运行时 preflight gate。
