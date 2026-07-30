# LightX2VRun

Docker launchers for the single-GPU AMD ROCm GFX1201 LightX2V image.

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
# Scaled FP8 4-step I2V
./lightx2vModelrun --infer-fp8

# BF16 4-step I2V
./lightx2vModelrun --infer-bf16

# Interactive container shell
./lightx2vModelrun
```

The default image is `lightx2v-rocm:gfx1201-hvat-scratch`. Generated videos
are written to `/disk/ssd/lightx2v-runtime/outputs`.

The launcher fixes AITER dispatch to `GPU_ARCHS=gfx1201` and `CU_NUM=24`,
matching the rowwise CK tuning CSV embedded in the image.

Use environment variables to override the defaults when needed:

```bash
GPU_DEVICES=1 IMAGE=custom-lightx2v:latest ./lightx2vModelrun --infer-fp8
```

Run `./lightx2vModelrun --help` for all supported overrides.
