# LightX2VRun

Docker launchers for the AMD ROCm GFX1201 LightX2V image. Scaled FP8 inference
supports 1, 2, 4, 8, and 16 GPUs without enabling classifier-free guidance.
The LoRA-FP8 launchers can merge the Wan2.2 high/low distill LoRAs into the
official DiT weights, quantize both models to scaled E4M3 FP8, and immediately
start inference.

Both the prebuilt scaled-FP8 and merged LoRA-FP8 launchers report synchronized
DiT progress. Each completed step uses the slowest rank's profiler duration,
which represents the distributed critical path. The final line reports DiT
total time, average `s/it`, and throughput in `it/s`; model loading, T5, and VAE
time are intentionally excluded. Upstream Wan warmup is enabled before measured
inference for every Distill/LoRA 1/2/4/8/16-GPU launcher; it covers the high- and
low-noise models without entering the measured progress samples. Set
`LIGHTX2V_WARMUP=0` to disable warmup or `LIGHTX2V_DIT_PROGRESS=0` to restore raw
torchrun output without the progress report.

GFX1201 images can bundle a committed fallback snapshot of these launchers at
`/opt/LightX2VRun`. The two in-image compatibility entries (Distill and LoRA,
both two-GPU) prefer the host checkout mounted at `/workspace/LightX2VRun` and
warn before falling back to the bundled snapshot. The bind mount used by
`lightx2vModelrun` therefore makes launcher-only updates available without an
image rebuild.

## Model layout

The default model root is `/disk/ssd/difumodels-Dis`:

```text
/disk/ssd/difumodels-Dis/
|-- Encoders/
|-- Wan2.2-Distill-Loras/
|-- Wan2.2-Distill-Models/
`-- Wan2.2-I2V-A14B/
    |-- high_noise_model/*.safetensors
    `-- low_noise_model/*.safetensors
```

`lightx2vModelrun` maps this host layout to the paths expected by the
LightX2V GFX1201 BF16 and FP8 configurations.

The LoRA-FP8 path requires the original official safetensors shards under both
noise-model directories. A directory containing only `config.json` is enough
for prebuilt scaled-FP8 inference, but it cannot be used for LoRA merging.

## Run

```bash
# One GPU
GPU_DEVICES=0 ./lightx2vModelrun --infer-fp8

# Scaled FP8 4-step I2V on GPUs 0 and 1
./lightx2vModelrun --infer-fp8

# Four GPUs (Ulysses SP=4)
GPU_DEVICES=0,1,2,3 ./lightx2vModelrun --infer-fp8

# Eight GPUs (Ulysses SP=8)
GPU_DEVICES=0,1,2,3,4,5,6,7 ./lightx2vModelrun --infer-fp8

# Sixteen GPUs (Ring SP=16)
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./lightx2vModelrun --infer-fp8

# Native SageAttention2; the same topology rule selects Ring on 16 GPUs
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./lightx2vModelrun --infer-sage-fp8

# BF16 4-step I2V
./lightx2vModelrun --infer-bf16

# Interactive container shell
./lightx2vModelrun
```

## Reproduce the RTX 4090 171-second workload

The upstream `171s` result is a single RTX 4090 running Wan2.1 I2V 14B at
720p for 81 frames with the 4-step distilled model and CFG disabled. The
published benchmark script uses `img_2.jpg`, the cooking prompt, block-level
CPU offload, FP8 DiT weights, and profiling output; the 720p table specifies
seed 1234. It is not a Wan2.2 MoE result.

The historical JSON kept 480p defaults with `720`/`1280` comments, so using the
current `configs/bench/lightx2v_6_distill.json` unchanged does not reproduce the
720p row. This launcher creates an explicit 720p runtime config and substitutes
the ROCm attention/GEMM backends while preserving the benchmark workload:

```bash
GPU_DEVICES=0 ./lightx2vModelrun --benchmark-wan21-4090
```

使用完全相同的 Wan2.1 单卡 720P/81 帧/4 步 workload 测原生 gfx1201
SageAttention2：

```bash
GPU_DEVICES=0 IMAGE=<包含-Aiter-ac27b0b26e-的镜像> \
  ./lightx2vModelrun --benchmark-wan21-4090-sage
```

这个入口只把 self-attention 从 FlyDSL BF16 改为
`aiter_fav3_sage_bf16_attn`；cross-attention、FP8 DiT 权重、seed、offload
比例和输入图均与原来的 4090 对标入口一致。

The default `OFFLOAD_RATIO=0.8` streams the first 32 DiT blocks and keeps the
last eight resident on the GPU, matching the historical schedule. To test
more residency on a 32 GB card, reduce it gradually; `OFFLOAD_RATIO=0` keeps
all 40 blocks resident but may run out of memory at 720p:

```bash
GPU_DEVICES=0 OFFLOAD_RATIO=0.8 ./lightx2vModelrun --benchmark-wan21-4090
```

Expected host model layout:

```text
/disk/ssd/difumodels-Dis/
`-- Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v/
    `-- distill_fp8/
        |-- google/umt5-xxl/*
        |-- block_0.safetensors ... block_39.safetensors
        |-- non_block.safetensors
        |-- diffusion_pytorch_model.safetensors.index.json
        |-- config.json
        |-- models_t5_umt5-xxl-enc-fp8.pth
        |-- clip-fp8.pth
        `-- Wan2.1_VAE.pth
```

The benchmark table links the ordinary
`lightx2v/Wan2.1-I2V-14B-720P-Lightx2v` base repository, but the historical
`run_lightx2v_6_distill.sh` explicitly uses the 4-step
`lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v` repository.
Only its `distill_fp8/` subtree is required. Those tensors use the same
per-channel FP8 weight layout required by `fp8-sgl`; the launcher changes the
runtime kernel from the CUDA-only Q8F implementation to GFX1201 Aiter rowwise
GEMM without requantizing the checkpoint.

The 2025 runner used `offload_ratio=0.8`, streaming 32 of 40 DiT blocks while
leaving the final eight resident. This GFX1201 branch restores that fractional
block residency while retaining the ROCm Aiter GEMM and attention backends.

Download just that subtree with ModelScope:

```bash
modelscope download \
  --model 'lightx2v/Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v' \
  --include \
    'distill_fp8/*.safetensors' \
    'distill_fp8/*.json' \
    'distill_fp8/*.pth' \
    'distill_fp8/google/umt5-xxl/*' \
    'distill_fp8/xlm-roberta-large/*' \
  --local_dir './Wan2.1-I2V-14B-720P-StepDistill-CfgDistill-Lightx2v'
```

The progress reporter compares DiT denoising time, excluding warmup, model
loading, encoders, VAE decoding, and video saving. The upstream number divided
by four is `42.75s/it`. Do not compare it with the earlier local `42.49s/it`
measurement: that was a four-GPU Wan2.2 run, not this single-GPU Wan2.1
workload. CUDA SageAttention versus ROCm Aiter, host offload bandwidth, and
software versions also prevent this from being a strict hardware-only
comparison.

## Merge LoRA, quantize, and infer

The first command performs the high-noise conversion and then the low-noise
conversion before inference. Conversion defaults to CPU because the converter
does not shard a 14B model across GPUs and a 32 GB card is likely to run out of
memory. Later runs validate and reuse the generated checkpoints.

```bash
# One GPU
GPU_DEVICES=0 ./lightx2vModelrun --infer-lora-fp8

# Two GPUs
GPU_DEVICES=0,1 ./lightx2vModelrun --infer-lora-fp8

# Four GPUs
GPU_DEVICES=0,1,2,3 ./lightx2vModelrun --infer-lora-fp8

# Eight GPUs
GPU_DEVICES=0,1,2,3,4,5,6,7 ./lightx2vModelrun --infer-lora-fp8

# Sixteen GPUs
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  ./lightx2vModelrun --infer-lora-fp8
```

By default the LoRAs are read from:

```text
/disk/ssd/difumodels-Dis/Wan2.2-Distill-Loras
```

The merged checkpoints and their provenance metadata are written to:

```text
/disk/ssd/difumodels-Dis/Wan2.2-Distill-Models/lora-fp8-strength1
```

If the LoRA directory contains more than one high/low version, select the pair
explicitly by file name:

```bash
HIGH_NOISE_LORA_FILENAME=wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
LOW_NOISE_LORA_FILENAME=wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
GPU_DEVICES=0,1 \
  ./lightx2vModelrun --infer-lora-fp8
```

The `*_1217.safetensors` files in the documented model layout are named
`wan2.2_t2v_*`; they are T2V LoRAs, not newer revisions of the I2V 1022 pair,
and must not be merged into the Wan2.2 I2V base.

Useful conversion controls:

```bash
# Build both checkpoints without starting inference
LORA_CONVERT_ONLY=1 GPU_DEVICES=0 ./lightx2vModelrun --infer-lora-fp8

# Rebuild an existing matching output
FORCE_LORA_CONVERT=1 GPU_DEVICES=0,1 ./lightx2vModelrun --infer-lora-fp8

# Different strength: use a different output directory to keep provenance clear
LORA_STRENGTH=0.8 \
LORA_FP8_OUTPUT_DIR=/disk/ssd/difumodels-Dis/Wan2.2-Distill-Models/lora-fp8-strength0.8 \
GPU_DEVICES=0,1 \
  ./lightx2vModelrun --infer-lora-fp8
```

Each conversion is accepted only when every adjustment applicable to the
official Wan2.2 I2V base was applied and the output contains E4M3 tensors with
matching scale tensors. The 1022 LoRAs also contain adjustments for a legacy
CLIP image-attention branch that is absent from the official latent-conditioned
I2V base and disabled by the runtime configuration; only those precisely
matched keys may be skipped. Any other missing key still stops conversion.
LoRA merging does not change GEMM shapes, so the existing GFX1201 Aiter rowwise
CK tuning remains applicable.

The default image is `ccr.ccs.tencentyun.com/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-FP8-260801-v2`. Generated videos
are written to `/disk/ssd/lightx2v-runtime/outputs`.

`lightx2vModelrun` mounts this checkout read-only at `/workspace/LightX2VRun`
and runs the matching script from `scripts/`. Changes to prompts, runtime JSON
overrides, or the 2/4/8/16 GPU launch scripts therefore do not require an image
rebuild. The image remains the fallback owner of the LightX2V Python package
and kernels.

The BF16-attention 16-GPU Ring path requires an image built from the LightX2V
ROCm branch with Aiter Triton `apply_with_lse()` support. The opt-in
`--infer-sage-fp8` path additionally requires Aiter revision
`ac27b0b26e78c864b5d771ea83daa37ce284d9f8`, where native gfx1201
SageAttention2 returns K-smoothing-corrected natural-log LSE, and LightX2V
revision `e761bd1f0596b2161f4ebeaf1c691d045d892bac`, where the ROCm adapter exposes
that LSE to Ring and Wan warmup releases its device cache. The launcher checks
both image labels before starting any rank. Those Aiter and LightX2V updates
require one image rebuild; launcher-only changes remain bind-mounted afterward.

The FP8 topology is:

```text
1 GPU:   Ulysses SP=1
2 GPUs:  Ulysses SP=2
4 GPUs:  Ulysses SP=4
8 GPUs:  Ulysses SP=8
16 GPUs: Ring SP=16
CFG:     disabled for every topology
```

Wan2.2 A14B has 40 attention heads. SP=1 keeps the same Ulysses configuration
without splitting the sequence; Ulysses also works for SP=2/4/8, but not SP=16
because 40 is not divisible by 16. The 16-GPU launcher uses LightX2V Ring
attention instead and does not create a second CFG replica group.

`--infer-fp8` remains the BF16 attention baseline. `--infer-sage-fp8` keeps
text/image cross-attention on the tuned Aiter Triton BF16 kernel and changes
only self-attention to `aiter_fav3_sage_bf16_attn`; on SP16, each Ring step
gets native Sage output plus natural-log LSE and merges all 16 K/V shards with
`logaddexp` weights.

The launcher fixes AITER dispatch to `GPU_ARCHS=gfx1201` and `CU_NUM=24`,
matching the rowwise CK tuning CSV embedded in the image.

Use environment variables to override the defaults when needed:

```bash
GPU_DEVICES=2,3 IMAGE=custom-lightx2v:latest ./lightx2vModelrun --infer-fp8

# Use a custom input from INPUT_DIR and generate 121 frames at 720p
GPU_DEVICES=0,1,2,3 \
IMAGE_PATH=/inputs/source.png \
TARGET_VIDEO_LENGTH=121 \
TARGET_HEIGHT=720 TARGET_WIDTH=1280 \
./lightx2vModelrun --infer-fp8
```

Run `./lightx2vModelrun --help` for all supported overrides.
