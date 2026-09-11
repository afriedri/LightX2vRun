# MiniMax-H3 Ref2AV Full BF16 Integration and Multi-GPU Check

This entry uses the original `diffusion_models/minimax_h3_ref2va_bf16.safetensors` base model and does not load a LoRA.

For the complete AMD-facing setup, image, operator, execution, and acceptance procedure, see [MiniMax-H3 Full BF16 Ref2AV on gfx1201: Reproducible Integration and Operations Guide](minimax-h3-bf16-ref2av-gfx1201-reproduction-guide.md).

## Weights and sampling

The inventoried BF16 file is 66,280,487,368 bytes and contains 535 tensors: 522 BF16 and 13 FP32 tensors across 50 Core blocks. All keys, shapes, and dtypes match the original H3 architecture mapping.

The ComfyUI checkpoint must be converted to the existing native loader's key and layout contract: split contiguous Q/K/V, swap the SwiGLU layout from `[gate; value]` to `[value; gate]`, and rename the remaining tensors. The conversion does not quantize, merge a LoRA, or change BF16/FP32 dtypes. It reuses the public [Diffusers H3 conversion rules](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py). ComfyUI already stores contiguous Q/K/V, so the raw-checkpoint per-head reorder must not be applied; see [ComfyUI H3 attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).

The image performs this conversion on first use and writes one file per block to a persistent cache. Allow approximately 66.3 GB of additional disk space. Inference does not rearrange the complete checkpoint on every run. The cache identity contains the source path, size, mtime, header hash, and conversion version; the converter publishes the directory only after every output has been written. It deliberately does not require a full-file SHA scan.

The default AMD BF16 overlay uses `comfy_res_multistep`: the ComfyUI `simple` schedule plus a deterministic second-order update, video/audio shifts of 12/3, and `infer_steps=20`, meaning 20 model evaluations. It includes the audio-latent-to-video-timeline conversion and preserves native H3 model-input semantics. ComfyUI is not a runtime dependency. BF16 GEMM, AITER attention/RMS/RoPE, sequence parallelism, lazy block offload, and H4 operators remain HVAT implementations.

The legacy `reference_blend` and `training_euler` modes remain selectable. In those modes, `infer_steps` retains the older meaning of grid points including the endpoint. Reference images use `match` by default: two 2752x1536 images are each resized to 1344x768 and contribute 1,008 reference latent rows. The `max` and `diffusers` choices remain available through `H3_STEP_UPDATE` and `H3_REFERENCE_IMAGE_RESIZE_MODE`.

The dedicated entry defaults to 1344x768, 24 FPS, and 15 seconds. Native inference produces 362 aligned frames; finalization trims this to 360 encoded frames and 480,000 audio samples per 32-kHz channel. Encoding runs once. `--duration 5..15` changes the duration. `--smoke` changes only the canvas to 480x480; it does not shorten the requested duration.

Qwen, tokenizer, processor, video VAE, and audio VAE come from the complete native H3 directory. This entry does not load the Qwen or VAE single-file components from the ComfyUI directory.

## Operator and memory routes

| Component | Route |
| --- | --- |
| Core GEMM | BF16 `torch.mm` with the default ROCm BLAS heuristic; TunableOp disabled |
| Core attention | `aiter_fav3_sage_bf16_attn`; BF16 public input/output contract with internal low-precision acceleration |
| Core Q/K RMSNorm | `minimax_h3_aiter_triton_qk`, D128 |
| Core RoPE | `minimax_h3_triton_strict_rope`, rotary width 96 |
| Qwen, Refiner, and H4 attention | `aiter_triton_mha` |
| Residency | lazy DiT block offload; Qwen block offload, rank-0-only construction under SP, and unload after encoding; VAE offload plus tile parallelism |

Reference images add sequence rows, but the architecture remains 56 heads x D128. The implementation reuses upstream Ref2AV packing and the HVAT sequence partition. Operators do not depend on one fixed FL2AV sequence length. CPU regression covers the position/time-index partition for 1, 3, and 9 reference images at SP2/SP4/SP8/SP16 and the 29-evaluation block-buffer rotation. There is no per-step format adapter.

Earlier 2 x 32-GiB runs exposed OOMs in the output head and in whole-video VAE gathering. Both locations now operate in bounded chunks. The DiT payload is approximately 61.73 GiB; each Core block is approximately 1.2025 GiB, two GPU block buffers are approximately 2.405 GiB, and fixed tensors are approximately 1.605 GiB, for about 4.01 GiB of weights per rank. This is a weight-only calculation, not the runtime memory peak, and SP2 does not simply split every weight across two GPUs.

Qwen multimodal inputs, large references, Core activations and communication workspaces, and VAE encode/decode can increase the peak. DiT buffers remain allocated for reuse and cannot be assumed to disappear before the VAE starts. The full 20-evaluation base model performs much more I/O and computation than Turbo 4-step; Turbo timing is not a valid baseline.

## Execution

Use `scripts/run_10_minimax_h3_bf16_ref2v.sh`. The image, complete native H3 directory, ComfyUI BF16 root, GPUs, references, and prompt can be edited at the top of that script or overridden on the command line. The default `rooftop_2ref_15s` material is under `assets/minimax_h3/rooftop_15s/`: `Image Gen-1.png` is `<Picture 1>` and `RYAN-2.png` is `<Picture 2>`. The adjusted `prompt.txt` preserves the six-shot, 15-second rooftop fight and native audio-generation instructions; `prompt.original.txt` retains the source prompt.

Run the default full workload from the LightX2VRun repository:

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image lightx2v-rocm:hvat-gfx1201-R9600 \
  --collect
```

To verify the real 768P, 15-second, two-reference tensor and memory path before a full quality run, add `--infer-steps 3`. This preserves the real geometry and duration while reducing DiT evaluations. It is a functional smoke test, not a quality result. Four GPUs are recommended for the first end-to-end acceptance run:

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image lightx2v-rocm:hvat-gfx1201-R9600 \
  --gpus 0,1,2,3 --infer-steps 3 --collect
```

Example with explicit assets and a reduced canvas:

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image lightx2v-rocm:hvat-gfx1201-R9600 \
  --model-dir /disk/ssd1/difumodels/MiniMax-h3 \
  --comfy-dir /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --reference-image /path/to/reference.png \
  --prompt-file /path/to/ref2av-prompt.txt \
  --gpus 0,1 --duration 5 --smoke --infer-steps 3 --collect \
  --result-dir /disk/ssd1/lightx2v-runtime/outputs/h3-ref2av-smoke
```

Remove `--smoke --infer-steps 3` for a full base-model quality run. Repeat `--reference-image` for 1-9 references in prompt order. This profile does not require first/last frames or a LoRA directory.

The current LightX2VRun checkout is mounted read-only into the container. Conversion and inference execute inside the image. `CACHE_DIR` selects the persistent host cache; prepared BF16 weights are stored under `CACHE_DIR/h3_ref2av/`. The result directory contains the runtime config, preflight report, `run.log`, video, and `ffprobe.json`; `--collect` also writes one H3 lifecycle/operator report per rank.

The H3 release policy disables fallback for selected operators. `run.log` must contain `dispatch` events for the expected backends, and any `status=fallback` is a failed run. CPU tests do not prove that a GPU kernel passed.

To retain the legacy HVAT sampling and reference-resize behavior:

```bash
H3_STEP_UPDATE=reference_blend \
H3_REFERENCE_IMAGE_RESIZE_MODE=diffusers \
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image YOUR_REBUILT_IMAGE --gpus 0,1,2,3 --collect
```

The dedicated script defaults legacy modes to 30 grid points, or 29 model evaluations. Selecting the Comfy-compatible sampler does not copy the complete ComfyUI workflow: Qwen and both VAEs still come from the native H3 directory, and the full BF16 model contract remains unchanged.
