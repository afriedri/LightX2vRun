# MiniMax-H3 Ref2AV Full BF16 and Turbo 8-Step Scaling (2026-09-11)

A user reported initial success on two 32-GiB GPUs. This workflow extends the same path to scale testing; 4/8/16-GPU generation, performance, and CP2 Sage numerical acceptance remain to be measured. The BF16 route preserves the complete original `minimax_h3_ref2va_bf16.safetensors` without LoRA. The Turbo route uses the fixed 8-step LoRA contract. Both sweeps default to the two-reference `rooftop_2ref_15s` profile and use nine references only when `--prompt-profile storeroom_courtyard_9ref_15s` is explicit. Both produce 768P, 24-FPS, 15-second output. Full BF16 defaults to 20 `comfy_res_multistep` evaluations; Turbo performs eight evaluations.

## Nine-reference material profile

The profile is `configs/minimax_h3/ref2av_storeroom_courtyard_9ref_15s.sh`; assets are under `assets/minimax_h3/storeroom_courtyard_9ref_15s/`. It exports nine absolute paths in `<Picture 1>` through `<Picture 9>` prompt order. Both the BF16 `run_11` and Turbo `run_13` entries select it through the same option:

```bash
--prompt-profile storeroom_courtyard_9ref_15s
```

Examples:

```bash
H3_INFER_STEPS=3 \
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --counts 2,4,8,16

bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --counts 2,4,8,16
```

Without `--prompt-profile`, the launchers load `configs/minimax_h3/ref2av_rooftop_2ref_15s.sh`. The compatibility spelling `--prompts-profile` is also accepted. The selected profile is recorded in `run_manifest.json` as `prompt_profile`, `material_name`, `material_config`, `prompt_file`, and `reference_images`.

## Entry and evidence collection

`scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh` reuses the FL2AV `run_timing_sweep.sh`; each arm calls `scripts/run_10_minimax_h3_bf16_ref2v.sh`. The host only schedules work. Weight conversion, preflight, and inference still execute in the image. Use an image with the current long-video/SP16 LightX2V integration and matching AITER wheel.

The AITER commits visible under the `yangecool` GitHub account are not the latest HVAT internal source used for the release wheel. Code merging is in progress; HVAT will provide the corresponding source patch after the merge completes. Do not use the public GitHub AITER branch to reproduce or validate the current wheel.

First exercise the complete 768P/15-second path without the reduced 480x480 canvas:

```bash
H3_INFER_STEPS=3 \
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE --counts 4,8,16
```

After all arms pass, remove `H3_INFER_STEPS=3` to perform the default 20 evaluations. Add the two-GPU baseline with `--counts 2,4,8,16`, or run one arm with `--gpu-count 16`. Every arm uses the same material, prompt, sampling contract, and default seed 42. The runtime produces 362 native frames and encodes 360. Quantized internal attention arithmetic across different parallel topologies is not expected to be bitwise identical.

The default physical groups are two GPUs `1,2`; four GPUs `15,14,13,12`; eight GPUs `0..7`; and sixteen GPUs `0..15`. Override these groups on an eight-GPU host, for example:

```bash
TIMING_GPU_GROUPS='0,1;0,1,2,3;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' \
H3_INFER_STEPS=3 \
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE --counts 4,8
```

`TIMING_RESULT_DIR` selects an empty result directory. By default, results are created under `/disk/ssd1/lightx2v-runtime/outputs/`.

Each sweep retains:

- root-level `run_manifest.json`, one `<count>gpu.log` per arm, `timings.csv`, `timings.md`, memory-monitor JSONL, and `sage_shapes.json`;
- separate `4gpu/`, `8gpu/`, and `16gpu/` directories containing the video, `runtime_config.json`, `preflight.json`, `ffprobe.json`, and all `h3_runcollector.rank*.json` files;
- the same collector and memory-monitor settings for every arm. Set `LIGHTX2V_H3_RUN_COLLECTOR=0` only for a consistently configured low-observer-overhead comparison;
- completed arms when a later arm fails. Shape-summary failures are warnings and do not discard raw videos or logs.

## Operator and partition review

| GPU count | Core topology | Heads per attention kernel | Distribution of 28 VAE tiles per temporal slice |
| ---: | --- | ---: | --- |
| 4 | Ulysses4 | 14 | 7 per rank |
| 8 | Ulysses8 | 7 | four ranks receive 4; four receive 3 |
| 16 | CP2 x Ulysses8 | 7 | twelve ranks receive 2; four receive 1 |

The 56 heads cannot be divided directly across Ulysses16, so SP16 uses CP2xU8. Ranks 0..7 and 8..15 form the two U8 groups, and corresponding ranks form CP2 columns. Keep U8 peers within the faster interconnect domain when topology permits. GPU count alone does not imply linear speedup.

- Source, config builder, container entry, and preflight accept native frame counts `17*n+5` from 124 through 362 at the existing 768P and 480P landscape sizes. FL2AV remains fixed at 124 frames. Shape, head-count, no-LoRA, unquantized-communication, and operator correctness gates remain active.
- SP4/SP8 use only `N % P` replicated condition rows. SP16 retains at least two auxiliary rows; when `N % 16` is 0 or 1, it retains 16 or 17 rows. Remaining condition and target tokens are partitioned together, and RoPE/timestep/AdaLN indices use the same order-preserving partition. SP2 retains its existing prefix strategy.
- Final DiT gather writes directly into the final hidden buffer through `all_gather_into_tensor`, avoiding a list gather followed by a second full-size concatenation. This removes approximately `N x 5376 x 2` bytes per rank. The existing output head still runs after gather; this is not a new model-sharding algorithm.
- Core explicitly selects native AITER gfx1201 SageAttention2 at D128. SP4 and SP8 use H14 and H7; SP16's square CP2 blocks use H7 and return LSE. AITER's B1/D128 selector accepts any positive sequence length and selects aligned or tail schedules from the real contract, so old FL2AV sequence constants are not reused.
- CP2 uses native Sage for large square blocks, AITER Triton Sage for the small rectangular auxiliary prefix, and LSE-aware merge. There is no failure-driven switch to SDPA. Quantization and merge error must be checked on GPU; a dense CPU comparison is insufficient.
- Q/K RMS, strict R96 RoPE, BF16 GEMM, and block offload are unchanged. Rank-0 Qwen encode, reference preparation, output head, and final media writing retain serial work, so end-to-end time will not scale linearly. The lazy loader keeps fixed tensors and two CPU plus two GPU Core-block buffers per rank; it does not pin a full private copy of the 33B model per rank. Concurrent disk reads and H2D copies can still become host-memory, storage, or PCIe bottlenecks.
- The VAE retains 21 temporal slices x 28 tiles, H4 S1797/D64 FP16 MHA, GPU spatial blending, and temporal blending. Completed frames move to CPU promptly. Uneven 28-tile distribution at SP8/SP16 is expected and does not change tile geometry merely to obtain even division.

## Validation and tuning handoff

Real CPU Gloo 4/8/16-process tests passed for attention, all-to-all, CP2 LSE merge, and final all-gather, including empty, odd, and 16/17-row auxiliary prefixes. VAE tests cover 2/4/8/16 processes, 28 tiles, uneven assignment, idle ranks, decode order, and blending. Long-video reference-index and LightX2VRun tests passed. These are not local GPU inference or tuning results.

The shape collector no longer assumes an even CP2 auxiliary prefix or a nonzero `N % 16`. It records the real rectangular Q length by context rank and uses runtime total-sequence metadata for isolated SP16 collection.

After the first GPU sweep, provide the entire result directory. Use `sage_shapes.json`, real M/N/K contracts, attention shapes, and stage timing to choose tuning targets. Separate native Sage time, A2A/Ring waits, weight-offload time, and rank-0 VAE gathering before changing an operator. Candidate priorities are H14/H7 long-sequence native Sage, the two square CP2 LSE kernels, and BF16 GEMMs at the shorter local row counts.
