# MiniMax-H3 Full BF16 Ref2AV: 768P/15-Second Review (2026-09-09)

The target is the original `minimax_h3_ref2va_bf16.safetensors`, no LoRA, two reference images, 1344x768, 24 FPS, and 15 seconds. Earlier GPU runs exposed OOMs in the output head and VAE gathering. Both locations have been changed to bounded processing. A user-reported 2 x 32-GiB run subsequently completed; 4/8/16-GPU performance and quality remain to be measured.

## The two OOMs and their fixes

- The first failure came from a 1.40-GiB allocation in DiT output-head modulation followed by a 2.81-GiB whole-tensor FP32 conversion. The output head now processes each modality in blocks of 4,096 rows while preserving RMS, modulation order, and FP32 projection. The inventoried hidden width is 5,376, so each block has an 84-MiB FP32 input. An earlier note incorrectly used the attention inner width of 7,168; the implementation derives its size from the real tensor.
- The second failure occurred in the video VAE. Rank 1 failed while stacking all local tiles and rank 0 failed while allocating the receive buffer; each request was approximately 3.02 GiB. A 768P, 362-frame run contains 21 temporal slices x 28 spatial tiles = 588 tiles. The old SP2 path retained 294 decoded tiles per rank before sending them, and adding GPUs did not eliminate rank 0's whole-video accumulation.
- The fix allocates, decodes, and gathers one temporal slice at a time. Each SP2 rank handles 14 tiles per round, while each SP4 rank handles 7. The existing point-to-point gather, GPU spatial blend, temporal-overlap blend, and final crop remain in place. Every rank participates in each temporal slice; ranks without a tile use the existing barrier contract. With `return_cpu=True`, completed frames move to CPU immediately, and final RGB conversion is chunked on CPU rather than materializing the complete FP32 video on GPU.
- H4's FP16/D64 `aiter_triton_mha`, the S1797 tile contract, decoder, and projection operators are unchanged. More gather rounds and incremental D2H copies may add time. This change bounds memory; it does not claim zero overhead or a speedup.

## Sampling and HVAT ownership boundary

The implementation was checked against `workflow_templates@4c4b8cbc`, especially `templates/video_minimax_h3_r2v.json`, and the relevant ComfyUI `model_sampling.py`, `samplers.py`, `k_diffusion/sampling.py`, and `ldm/minimax/model.py` behavior.

- The new default is `comfy_res_multistep`: 20 sigmas sampled from the 1,000-point `simple` base schedule, followed by the endpoint, for exactly 20 model evaluations. Video/audio shifts are 12/3. The implementation includes carried audio state, velocity conversion, and final reconstruction inside the H3 scheduler; ComfyUI is not a runtime dependency.
- `reference_blend` and `training_euler` remain selectable. Their legacy grid-point counting semantics are unchanged. `H3_STEP_UPDATE=reference_blend` restores the former path; the dedicated Run entry then defaults to 30 grid points.
- Reference resize defaults to `match` and does not upscale. Each 2752x1536 source becomes 1344x768 and contributes 1,008 latent rows. The target video has 107,856 rows and audio has 1,206 rows. `H3_REFERENCE_IMAGE_RESIZE_MODE=diffusers` or `max` selects the older alternatives independently of the sampler.
- The original BF16 conversion cache, native Qwen and VAEs, reference packing, disabled CFG, lazy DiT block offload, rank-0 Qwen encode/unload, sequence parallelism, and VAE tile parallelism remain. The workflow's alternative weight formats, 480P/5-second workload, and Turbo LoRA are not used.
- Core continues to use `aiter_fav3_sage_bf16_attn`, Q/K RMS uses `minimax_h3_aiter_triton_qk`, and RoPE uses strict/no-FMA R96. GEMM remains BF16 Torch/ROCm BLAS with TunableOp disabled. Qwen, Refiner, and H4 remain on `aiter_triton_mha`. Selected H3 operator fallback remains disabled.
- The 362 native frames are cropped to 360 video frames and 480,000 samples per 32-kHz audio channel immediately before muxing, producing exactly 15 seconds while retaining native sound generation.

## Validation and limits

- Six offline trajectories generated from actual ComfyUI source cover 1, 2, 3, 7, and 20 evaluations, 12/3 shifts, and one alternative shift. The source-file SHA256 values were saved. Per-step video/audio states, endpoint reconstruction, and immutable reference-conditioning rows agree within tolerance. Per-request prepare/clear behavior and legacy step counting are covered.
- 104 focused LightX2V tests passed across the new sampler, legacy paths, output head, sequence parallelism, offload, reference indices, operator routes, collector, and output duration.
- Five real two-/four-process CPU Gloo decode tests passed. They compare parallel and serial assembly with zero error, cover idle ranks, verify 362-frame and padding crops, and assert that only one temporal slice is gathered at a time. These tests use a small deterministic decoder; they are not GPU neural-network or RCCL performance acceptance.
- The complete LightX2VRun test suite reported 248 passes at this review point. Additional focused tests also covered optional legacy sampling and environment forwarding.
- Device usage outside PyTorch's accounting in historical logs has not been attributed. The collector records stage-boundary device free/total, PyTorch allocated/reserved, and cumulative peaks. No log evidence identifies fragmentation, a specific library, or the driver as the cause.
- A user later reported an initial successful 2 x 32-GiB run. The complete 20-evaluation quality result, 4/8/16-GPU timing, and route-by-route GPU numerical acceptance should still be captured as release evidence.

## Reproduction command

First keep the real 768P, 15-second, two-image workload but reduce the DiT to three evaluations:

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image YOUR_RELEASE_IMAGE \
  --gpus 0,1,2,3 --infer-steps 3 --collect
```

After that passes, remove `--infer-steps 3` for the 20-evaluation quality run. Preserve `run.log`, `runtime_config.json`, `preflight.json`, `ffprobe.json`, the MP4, and all `h3_runcollector.rank*.json` files. A three-evaluation run validates the execution path, not output quality.
