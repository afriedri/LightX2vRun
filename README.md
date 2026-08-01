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
time are intentionally excluded. This is the average for the current generation;
for a publishable benchmark, warm up with one complete generation and measure a
second identical generation. Set `LIGHTX2V_DIT_PROGRESS=0` to restore raw
torchrun output without this report.

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

# BF16 4-step I2V
./lightx2vModelrun --infer-bf16

# Interactive container shell
./lightx2vModelrun
```

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

The default image is `lightx2v-rocm:gfx1201-hvat-scratch`. Generated videos
are written to `/disk/ssd/lightx2v-runtime/outputs`.

`lightx2vModelrun` mounts this checkout read-only at `/workspace/LightX2VRun`
and runs the matching script from `scripts/`. Changes to prompts, runtime JSON
overrides, or the 2/4/8/16 GPU launch scripts therefore do not require an image
rebuild. The image remains the fallback owner of the LightX2V Python package
and kernels.

The first 16-GPU Ring run requires an image built from the LightX2V ROCm branch
that includes Aiter Triton `apply_with_lse()` support. After that one core image
update, launcher-only changes remain bind-mounted and need no rebuild.

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
