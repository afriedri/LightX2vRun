# LightX2VRun

Docker launchers for the AMD ROCm GFX1201 LightX2V image. Scaled FP8 inference
supports 2, 4, 8, and 16 GPUs without enabling classifier-free guidance.

## Model layout

The default model root is `/disk/ssd/difumodels-Dis`:

```text
/disk/ssd/difumodels-Dis/
|-- Encoders/
|-- Wan2.2-Distill-Models/
`-- Wan2.2-I2V-A14B/
```

`lightx2vModelrun` maps this host layout to the paths expected by the
LightX2V GFX1201 BF16 and FP8 configurations.

## Run

```bash
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
2 GPUs:  Ulysses SP=2
4 GPUs:  Ulysses SP=4
8 GPUs:  Ulysses SP=8
16 GPUs: Ring SP=16
CFG:     disabled for every topology
```

Wan2.2 A14B has 40 attention heads. Ulysses works for SP=2/4/8, but not SP=16
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
