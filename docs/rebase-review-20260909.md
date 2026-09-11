# LightX2V and LightX2VRun Upstream Integration Review (2026-09-09)

This integration preserves the HVAT model, weight, sampling, and runtime contracts while reusing upstream capabilities and infrastructure. It does not replace the five adapted models with upstream defaults. CPU and configuration regressions passed at the review point; GPU performance and video quality required renewed acceptance.

## Mainline and development branches

- Upstream: `git@gitcode.com:GitHub_Trending/li/LightX2V.git`.
- The fork's `main` matched upstream commit and tree `7425d1ec631528d1390d07d727ff7181465d5019` exactly.
- The LightX2V development branch is `hvat-gfx1201-R9600`; `main` remains a clean upstream mirror. Only obsolete feature branches on the primary GitCode remote were removed. The GitHub mirror and upstream were not modified.
- A pre-replay bundle was saved at `/home/yangecool/WorkSpace/Code/99-archive/lightx2v-20260909/LightX2V-before-upstream-rebase-20260909.bundle` and passed `git bundle verify`.

## Major upstream changes

Comparing old fork main `5c18e461` with new main `7425d1ec` yields 1,068 changed files. Because history was rewritten, the raw commit-set difference must not be interpreted as new functionality. After patch-equivalent duplicates were removed, the remaining difference was 83 upstream-side patches and 2 old-side patches.

| Area | Upstream change | Integration decision |
| --- | --- | --- |
| Wan | Unified distilled runner, DMD2/MeanFlow scheduler factory, CFG preset fix | Run adopts the new names and selects DMD2 explicitly while preserving HVAT FP8/LoRA, dual-model, and B0/B1 paths |
| LTX | Native LTX2.5 model, DurationHead, DiffVAE, ancestral sampler, component factory | A new entry dispatches the HVAT profile to the existing shared runner; it does not replace HVAT weights, sampler, or convolutional VAE |
| H3 | Pinned-host Qwen loading, TP bias fix, AdaLN cache, pre/post residency | Integrate the implementation while retaining FL2AV parameters and selected operators; new options remain disabled by default |
| H3 VAE | Spatiotemporal tile parallelism, load balancing, encode optimization | Retain the upstream structure and add idle-rank, output-order, and collector compatibility fixes |
| Common | Request/RNG reset, warmup moved into JSON, GQA SDPA, quantization rounding, FP8 communication | Reuse the base behavior and synchronize Run's config interface |
| Other | ROCm gfx1100/gfx1201 AITER fixes; XPU/Ascend/MLU improvements; training, deployment, and SwiftVR updates | Integrate without automatically replacing HVAT's selected backends |

## Model-by-model conclusions

| Model | Run entry | Preserved model and capability | Review focus |
| --- | --- | --- | --- |
| LTX2.3 | `ltx2` | Mixed FP8/BF16 weights, prompt cache, two-stage offload, sampling parameters | Shared preprocessing preserves HVAT Prompt AdaLN precision and position-grid rules |
| LTX2.5 | `ltx2_5` plus HVAT profile | Split assets, Gemma4, Conv VAE, existing sampler, two-slot loader, TunableOp, Sage, distributed VAE | The profile only selects the entry; it does not enable native DurationHead, DiffVAE, or ancestral sampling |
| Wan2.1 | `wan2.1` plus `distill_method=dmd2` | FP8 distilled weights, T5/CLIP, LoRA, AITER GEMM/attention, Ring/SP | Verify that the new factory still selects DMD2; warmup moves from an environment variable to JSON |
| Wan2.2 | `wan2.2_moe` plus `distill_method=dmd2` | High/low FP8 weights, LoRA, switch boundary, lazy B0/B1, CK Conv3D | Preserve construction and memory collection in the unified runner; B0/B1 matches the pre-replay implementation |
| MiniMax-H3 | `minimax_h3` | FL2AV frames/prompt, Turbo LoRA, 124 frames at 24 FPS, 50-block lazy loading, AITER routes, SP16 CP2xU8 | Fix block indexing, pageable loading for quantized Qwen, and VAE gather; inventory the real Ref2VA format first |

The complete LTX2.5 HVAT runtime config was compared field by field with its pre-replay form and differed only in `model_cls`. H3 SP1/SP2/SP4/SP8/SP16 final configs, including Run-normalized LoRA paths, matched their pre-replay equivalents. Model files were neither renamed nor moved.

## Performance review

- Excluded regressions included accidentally selecting the native LTX2.5 model/sampler, losing prompt cache or two-slot offload, adopting the new Wan name without DMD2, making a collector fetch a removed branch, and retaining the old Hot28 default image.
- Static verification showed unchanged Wan block offload and LTX TunableOp cache behavior. H3 lazy B1 rotation, FP8 quantization reuse, and AITER route regressions passed. The new SGL FP8 wrapper invokes the same backend without an extra tensor copy.
- H3's new spatiotemporal VAE tile count, rank-0 gather memory/time, one-time collective initialization on small tile sets, and the new request lifecycle's peak-memory/timing effects remained GPU acceptance items. CPU tests do not prove zero performance loss.
- Performance comparisons must hold weight files, material, prompt, seed, frame count, GPU topology, and collector settings constant. Compare phase timing, H2D, swap wait, peak memory, and actual operator dispatch. Upstream-native LTX2.5 results are not an HVAT-model baseline.

## Validation and image

- LightX2VRun: 207 tests passed, including 10 cross-repository entry/config tests; 49 shell scripts passed syntax checks.
- LightX2V: 140 focused tests passed; 5 GPU cases were skipped because no device was available. Static checks and `git diff --check` passed.
- H3 VAE: real four-process CPU Gloo tests verified gather order and idle ranks for 1, 2, 7, and 8 tiles. This is not RCCL/GPU acceptance.
- Docker/GPU end-to-end testing was unavailable on the review host, so no image was built or published during this review.
- Both repositories use `lightx2v-rocm:hvat-gfx1201-R9600` as the default tag. `dockerfiles/platforms/build_gfx1201.sh` produces that tag; Run accepts an explicit `--image` override.

The AITER commits currently visible under the `yangecool` GitHub account are not the latest HVAT internal enterprise source used to produce the release wheel. Code merging is in progress. The wheel embedded in the release image is the authoritative executable AITER artifact, and HVAT will provide the corresponding source patch after the merge is complete. The public GitHub AITER branch must not be used to reconstruct or validate that wheel in the interim.

## H3 Ref2VA weight inventory

The Run collector parses the specified unpruned BF16 safetensors file. It reads the header only. It does not read matrix payloads, import Torch, allocate a GPU, or enforce a full-file SHA gate.

Run on the weight host:

```bash
bash scripts/collector/run_minimax_h3_ref2v_weights.sh \
  --image YOUR_RELEASE_IMAGE \
  --model-root /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --asset bf16 \
  --output-dir /disk/ssd1/lightx2v-runtime/h3-ref2va-weight-inventory
```

The output `h3_ref2va_weights.json` and `.md` files contain BF16 checkpoint metadata, keys, shapes, dtypes, and file size. Inventory success is not model inference or performance acceptance.

The host launches Docker; the collector runs inside the image. Current Run and weights are mounted read-only, and reports are written to the host. `DRY_RUN=1` prints the mount and command. A header inventory can use an older image because it executes only the mounted standard-library collector, but actual inference and performance collection require the current LightX2V integration and matching AITER wheel.
