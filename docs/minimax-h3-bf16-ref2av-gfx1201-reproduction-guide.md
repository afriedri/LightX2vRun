# MiniMax-H3 Full BF16 Ref2AV on AMD Radeon AI PRO R9600 (gfx1201)

## Reproducible Integration, Image, Operation, and Acceptance Guide

| Field | Value |
| --- | --- |
| Audience | AMD Radeon product, enablement, performance, and operator-library engineering teams |
| Target | AMD Radeon AI PRO R9600, `gfx1201` / RDNA 4 |
| Workload | MiniMax-H3 full BF16 Reference-to-Video-and-Audio (Ref2AV), no LoRA |
| Reference output | 1344x768, 24 FPS, 15 seconds, two reference images |
| Software | LightX2V, LightX2VRun, AITER wheel, ROCm 7.2 validation baseline |
| Document status | Reproduction baseline, September 11, 2026 |

This is the primary handoff manual for reproducing the completed MiniMax-H3 full-BF16 Ref2AV adaptation. It covers the complete checkpoint transformation, operator integration, memory and loading changes, image construction and delivery, execution, evidence capture, and acceptance criteria.

## 1. Critical AITER source-availability notice

> **The AITER commits currently visible under the `yangecool` GitHub account, including its AITER branches, are not the latest HVAT internal enterprise source used to produce this release. Code merging is in progress. The AITER wheel embedded in the release image is the authoritative executable artifact. HVAT will provide the corresponding AITER source patch after the merge is complete.**

Consequences for this handoff:

- Do not use the public `yangecool/aiter` branch to reconstruct, compare, or validate the AITER wheel in the current release image.
- AMD can reproduce the application run immediately from the immutable release image and can audit the installed wheel, AOT module, tuned catalog, versions, and image labels.
- The provider-side image recipe compiles AITER to a wheel and installs only that wheel into the final runtime stage. The source tree is not required at runtime.
- Source-level AITER reproduction begins only after HVAT supplies the merge-complete patch corresponding to the release wheel.
- LightX2V and LightX2VRun source-level behavior, runtime configuration, weight conversion, launch commands, and evidence contracts remain reviewable now.

This separation is intentional: the release wheel is the current binary/operator contract; the later patch is the source-disclosure and source-rebuild contract.

## 2. What was adapted

The original Ref2AV BF16 transformer is delivered as one ComfyUI safetensors file. LightX2V expects a native component directory, split Q/K/V projections, a different SwiGLU half order, per-block lazy-loading files, and explicit backend routes. The completed integration adds the following end-to-end path:

```text
full ComfyUI BF16 Ref2AV checkpoint
  -> strict header/key/shape/dtype validation
  -> deterministic key and layout conversion
  -> atomic persistent block cache
  -> native LightX2V H3 loader
  -> lazy two-slot host/device block streaming
  -> gfx1201-specific AITER attention, RMSNorm, RoPE, and CK Conv3D routes
  -> Ulysses SP2/SP4/SP8 or CP2xU8 SP16
  -> bounded output head and temporal-slice VAE gathering
  -> 1344x768, 24-FPS, 15-second MP4 with generated audio
```

The base transformer remains BF16/FP32. It is not quantized and no LoRA is merged or loaded in this profile. SageAttention2 uses low-precision arithmetic internally, but its public LightX2V contract is BF16 Q/K/V to BF16 output.

## 3. Repository and release contract

Use these repository roles:

| Repository or artifact | Role | Release handling |
| --- | --- | --- |
| `LightX2V` | Model graph, BF16 converter, scheduler, lazy loading, operator registry/adapters, SP/VAE execution | Source revision recorded in image label and `/app/versions.txt` |
| `LightX2VRun` | Host launcher, runtime-config builder, preflight, evidence collector, prompt/material profiles | Public handoff repository; mounted read-only at run time |
| AITER wheel | Native gfx1201 SageAttention2, Triton MHA, CK Conv3D AOT module/catalog, runtime APIs | Installed in the image; authoritative until source patch delivery |
| Release image | Pinned ROCm/PyTorch/Triton/dependencies plus LightX2V and installed AITER wheel | Reproduce by immutable registry digest or image archive checksum |
| Model assets | Full BF16 Ref2AV transformer plus native Qwen/tokenizer/processor/VAEs | Supplied separately under the documented host layout |

The engineering heads observed for the September 11 handoff were:

```text
LightX2V branch:     hvat-gfx1201-R9600
LightX2V head:       97d9fef6e332ff67b0c251fbeb27780e5754e5f4
LightX2VRun branch:  hvat-gfx1201
AITER internal integration reference used by the local build tree:
                     693c3b6bc8be502d5cc8472683cb561239d727c5
```

The AITER value above is provenance for the local integration snapshot, not a claim that the public GitHub AITER branch is the latest internal source. The release image's installed wheel and labels take precedence. Record the exact values from every delivered image:

```bash
IMAGE='YOUR_RELEASE_IMAGE@sha256:YOUR_IMMUTABLE_DIGEST'

docker image inspect "$IMAGE" --format '{{.Id}}'
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}'
docker run --rm --entrypoint bash "$IMAGE" -lc 'cat /app/versions.txt'
```

Save this output next to the run evidence. Do not accept only a mutable tag.

## 4. Hardware and host prerequisites

### 4.1 Validated platform

- AMD Radeon AI PRO R9600 reporting `gfx1201`.
- ROCm 7.2 is the validation baseline. Preflight warns on another ROCm minor version rather than silently treating it as equivalent.
- The runtime contract is 24 WGP / 48 CU and requires `GPU_ARCHS=gfx1201` and `CU_NUM=24`.
- Supported world sizes are 1, 2, 4, 8, and 16 GPUs.
- SP2/SP4/SP8 use Ulysses. SP16 uses CP2xU8 UlyssesRing because 56 attention heads do not divide across Ulysses16.

For a first formal run, use four GPUs if available. A two-GPU run has completed in user testing, but four GPUs provide additional memory margin during initial validation.

### 4.2 Host software and devices

The host requires Docker, the ROCm kernel driver, `/dev/kfd`, and `/dev/dri`. The launcher uses host networking and IPC, adds the host's video group, and passes both ROCm device paths.

```bash
command -v docker
test -c /dev/kfd
test -d /dev/dri
docker info
```

The launch contract includes:

```text
--network=host
--ipc=host
--security-opt=seccomp=unconfined
--device=/dev/kfd
--device=/dev/dri
```

The user running Docker must be allowed to access the GPUs and all model, cache, input, and output paths.

### 4.3 Storage budget

Plan capacity for:

- 66,280,487,368 bytes for `minimax_h3_ref2va_bf16.safetensors`;
- approximately another 66.3 GB for the prepared transformer cache;
- native Qwen, tokenizer, processor, video VAE, and audio VAE components;
- persistent AITER, Triton, Torch-extension, and Hugging Face caches;
- result video and evidence for each GPU-count arm.

Do not place the generated cache on ephemeral container storage.

## 5. Obtain LightX2VRun

Use the GitHub repository intended for AMD access:

```bash
git clone git@github.com:yangecool/LightX2vRun.git
cd LightX2vRun
git branch --show-current
git rev-parse HEAD
git status --short
```

Use the release branch/commit communicated with the image. The launcher mounts the current checkout read-only, so a prompt or host-launcher update can be tested without rebuilding the image; a LightX2V or AITER operator change requires a new image.

## 6. Prepare model assets

The BF16 Ref2AV profile intentionally combines two sources:

1. a complete native H3 directory for shared Qwen/tokenizer/processor/VAEs;
2. the original full ComfyUI BF16 Ref2AV transformer file.

Recommended layout:

```text
/disk/ssd1/difumodels/MiniMax-h3/
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
└── audio_vae/

/disk/ssd1/difumodels/MiniMax-H3-ComfyUI/
└── diffusion_models/
    └── minimax_h3_ref2va_bf16.safetensors
```

The native directory may also contain `transformer/` and `transformer_ref/`; the full BF16 profile creates and uses its own prepared `transformer_ref` under the persistent cache.

The inventoried transformer contract is:

| Property | Expected value |
| --- | ---: |
| File size | 66,280,487,368 bytes |
| Tensor count | 535 |
| BF16 tensors | 522 |
| FP32 tensors | 13 |
| Core blocks | 50 |
| Refiner blocks | 2 |
| Attention heads | 56 |
| Head dimension | 128 |
| Hidden width | 5,376 |
| FFN width | 14,336 |

Record basic provenance before running. A full SHA256 is recommended for a formal artifact manifest even though the runtime cache intentionally avoids scanning the 66-GB payload on every start:

```bash
BF16_CKPT=/disk/ssd1/difumodels/MiniMax-H3-ComfyUI/diffusion_models/minimax_h3_ref2va_bf16.safetensors
stat --printf='%n %s bytes\n' "$BF16_CKPT"
sha256sum "$BF16_CKPT"
```

## 7. Complete-weight conversion

`LightX2V/tools/convert/minimax_h3_comfy.py` converts the original single-file transformer into the native LightX2V component contract. It is executed automatically inside the image by the Ref2AV launcher.

### 7.1 Validation before conversion

The converter:

- reads the embedded `metadata.config.transformer` and overlays recognized architecture fields on the fixed H3 defaults;
- requires the exact planned key set and rejects missing or unexpected tensors;
- validates every source shape;
- requires expected BF16/FP32 dtypes by layer family;
- verifies that `rope.inv_freq` exactly matches native recomputation before dropping that buffer;
- checks that the source checkpoint does not change during conversion.

This is a strict model-format conversion, not a best-effort rename.

### 7.2 Tensor mapping

| Source | Native LightX2V target | Operation |
| --- | --- | --- |
| `blocks.i.attn.qkv_proj.weight` | `transformer_blocks.i.attn.to_q/to_k/to_v.weight` | Split three contiguous dim-0 ranges of 56 x 128 rows; do not apply raw-checkpoint per-head reorder |
| `blocks.i.attn.out_proj.*` | `transformer_blocks.i.attn.to_out.0.*` | Rename |
| `blocks.i.mlp.fc1.weight` | `transformer_blocks.i.ff.net.0.proj.weight` | Swap the two dim-0 halves from `[gate; value]` to `[value; gate]` |
| `blocks.i.mlp.fc2.*` | `transformer_blocks.i.ff.net.2.*` | Rename |
| `token_refiner.blocks.*` | `token_refiner.refiner_blocks.*` | Rename, with the same QKV and FC1 rules |
| patch, condition, time, final norm/output | native `proj_in`, `context_embedder`, `time_embedder`, `norm_out`, and output projections | Deterministic rename |
| `rope.inv_freq` | none | Validate, then drop because native H3 recomputes it |

The transform never quantizes, dequantizes, merges a LoRA, or changes the stored BF16/FP32 dtype.

### 7.3 Cache and publication semantics

The cache key is derived from conversion version plus source absolute path, size, nanosecond mtime, and safetensors-header SHA256. A file lock serializes concurrent preparation. The converter writes into a temporary directory, rechecks the source identity, writes a manifest of output paths and sizes, and atomically renames the complete directory into place. An incomplete published cache is treated as an error and is never silently reused.

Expected structure:

```text
<CACHE_DIR>/h3_ref2av/
├── <24-character-source-key>.lock
└── <24-character-source-key>/
    ├── manifest.json
    └── transformer_ref/
        ├── config.json
        ├── fixed.safetensors
        ├── block_0.safetensors
        ├── ...
        └── block_49.safetensors
```

The per-run prepared model directory contains symlinks to the cached `transformer_ref` and the native shared components. Large shared weights are not copied.

Do not remove a cache during an active run. If the converter reports an incomplete generated directory, remove only the named generated directory after confirming that no process is using it, then rerun.

## 8. Image construction and AITER wheel delivery

### 8.1 Current provider-side image pipeline

The LightX2V image recipe is `dockerfiles/platforms/Dockerfile_gfx1201`; the build entry is `dockerfiles/platforms/build_gfx1201.sh`. It uses a multi-stage build:

1. start from a pinned `vllm/vllm-openai-rocm:v0.25.0-base` registry digest;
2. install the Triton build selected by the matching AITER integration;
3. build `amd-aiter` for `gfx1201`, 24 WGP, with `PREBUILD_KERNELS=1` and `PREBUILD_MODULES=module_ck_vae_conv3d`;
4. verify that exactly one wheel exists and that it contains the CK Conv3D AOT `.so`;
5. install that wheel into the clean final stage;
6. verify wheel version, AOT presence, tuned-catalog row count, and tuned-catalog SHA256;
7. install the pinned LightX2V runtime dependencies and editable LightX2V source;
8. embed the LightX2VRun launcher snapshot and write complete version provenance to `/app/versions.txt`.

The final image contains the installed AITER wheel payload, not a supported AITER source checkout. The internal source build context is unavailable to AMD until the merge-complete patch is delivered.

For the current provider-side tree, the build command is:

```bash
cd /path/to/LightX2V
AITER_SOURCE_DIR=/path/to/internal/aiter \
LIGHTX2V_RUN_SOURCE_DIR=/path/to/LightX2VRun \
IMAGE_TAG=lightx2v-rocm:hvat-gfx1201-R9600 \
bash dockerfiles/platforms/build_gfx1201.sh
```

This command documents how the delivered image was produced. It is not yet an AMD source-rebuild instruction because the matching AITER source patch is pending.

### 8.2 AMD-side executable reproduction now

Load or pull the immutable delivered image:

```bash
# Registry delivery
docker pull YOUR_RELEASE_IMAGE@sha256:YOUR_IMMUTABLE_DIGEST

# Or offline archive delivery
sha256sum lightx2v-gfx1201-h3-bf16.tar
docker load --input lightx2v-gfx1201-h3-bf16.tar
```

Verify the installed AITER artifact without relying on public AITER source:

```bash
docker run --rm --entrypoint bash "$IMAGE" -lc '
set -e
python3 -m pip show amd-aiter
python3 - <<"PY"
from importlib.metadata import distribution, version
from pathlib import Path
import aiter

dist = distribution("amd-aiter")
matches = [
    p for p in (dist.files or [])
    if str(p).startswith("aiter/jit/module_ck_vae_conv3d")
    and str(p).endswith(".so")
]
assert len(matches) == 1, matches
assert Path(dist.locate_file(matches[0])).is_file()
assert callable(getattr(aiter, "ck_conv3d_v1", None))
assert callable(getattr(aiter, "is_ck_conv3d_v1_prebuilt", None))
assert aiter.is_ck_conv3d_v1_prebuilt()
print("amd-aiter", version("amd-aiter"))
print("CK Conv3D AOT", dist.locate_file(matches[0]))
PY
cat /app/versions.txt
'
```

The formal delivery manifest should contain image digest/ID, archive SHA256 if applicable, installed `amd-aiter` version, wheel filename and SHA256, AOT module presence, tuned-catalog SHA256/row count, LightX2V revision, LightX2VRun revision, base-image digest, ROCm/PyTorch/Triton versions, and the pending-source-patch notice.

### 8.3 Source rebuild after patch delivery

After HVAT completes the merge and supplies the patch:

1. apply it to the exact AITER upstream/base commit named in the patch manifest;
2. initialize all AITER submodules at the recorded revisions;
3. verify a clean worktree and the patched commit/tree hash;
4. run `build_gfx1201.sh` with that source as `AITER_SOURCE_DIR`;
5. compare wheel SHA256, embedded AOT module, tuned catalog, image labels, `/app/versions.txt`, preflight, and the frozen workload output.

Do not substitute the currently visible public `yangecool/aiter` branch for these steps.

## 9. Runtime configuration and operator integration

The host launcher builds the runtime JSON from the published H3 config and `configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json`. The builder freezes performance-sensitive and correctness-sensitive values so a checkpoint config cannot silently replace the gfx1201 routes.

### 9.1 Frozen full-BF16 profile

| Field | Value |
| --- | --- |
| `task` / `h3_profile` | `ref2av` / `ref2av_bf16_base` |
| `dit_quant_scheme` | `Default` |
| LoRA | disabled; no `lora_configs` |
| `h3_step_update` | `comfy_res_multistep` |
| `infer_steps` | 20 model evaluations by default |
| video/audio flow shift | 12.0 / 3.0 |
| CFG | disabled |
| target | 1344x768, 362 native frames, 24 FPS, 15-second final output |
| loading | `lazy_load=true`, block CPU offload, text encoder unload after use |
| TunableOp | disabled |

### 9.2 Operator route table

| Model region | LightX2V route | AITER/runtime responsibility | Contract |
| --- | --- | --- | --- |
| H3 Core dense GEMM | default `MMWeight` / `torch.mm` | ROCm BLAS heuristic, observed hipBLASLt on hot shapes | BF16 activations/weights; TunableOp off |
| H3 Core self-attention | `aiter_fav3_sage_bf16_attn` | native gfx1201 SageAttention2 selector | BF16 Q/K/V/output, D128, with internal low-precision acceleration |
| H3 Core Q/K norm | `minimax_h3_aiter_triton_qk` | AITER/Triton normalization path | BF16, D128, H3 epsilon/rounding contract |
| H3 Core RoPE | `minimax_h3_triton_strict_rope` | strict no-FMA Triton kernel | BF16 `[L,H,128]`, rotary width 96, FP32 cos/sin |
| Qwen/H1 attention | `aiter_triton_mha` | canonical dense AITER Triton MHA | model-specific BF16 dense contract |
| token Refiner attention | `aiter_triton_mha` | canonical dense AITER Triton MHA | BF16 |
| H4 video VAE attention | `aiter_triton_mha` | canonical AITER Triton MHA | FP16, D64, S1797 tile path |
| H3 video VAE Conv3D | `Aiter_CK_Conv3d_v1` | prebuilt CK Conv3D AOT module, universal HOT and tuned catalog | BF16/FP16 legal contracts; explicit route reporting |

The LightX2V registry owns semantic selection, layout normalization, lifecycle, and fallback policy. AITER owns architecture/kernel legality and its schedule selector. Model code does not contain an exact H3 shape allowlist for Sage schedules.

The H3 release disables fallback for the selected critical routes. A configured route is not proof of execution; acceptance requires one `dispatch status=active` event per process rank at the real tensor boundary. Any `status=fallback` is a failed release run.

## 10. Loading and memory optimization

### 10.1 Transformer block streaming

The full DiT weight payload is approximately 61.73 GiB. It cannot be treated as conventionally resident on each 32-GiB GPU. The lazy loader keeps fixed tensors and uses two host plus two device block slots. While the GPU computes the current block, the next block can transfer and a later block can be prepared in host memory.

Approximate weight-only figures are:

| Item | Size |
| --- | ---: |
| One Core block | 1.2025 GiB |
| Two GPU block buffers | 2.405 GiB |
| Fixed transformer tensors | 1.605 GiB |
| Approximate per-rank retained weights | 4.01 GiB |

These are not peak-memory estimates. Activations, attention workspace, collectives, Qwen/reference tensors, VAE state, and allocator/runtime reservations are additional.

### 10.2 Qwen and phase ownership

With sequence parallelism, only rank 0 constructs and runs the text encoder. Qwen uses block offload and is unloaded after encoding. Prompt/reference conditioning is then partitioned with the target sequence. This avoids multiplying the complete text-encoder residency across ranks.

### 10.3 Bounded output head

The output head processes modalities and hidden rows in chunks of 4,096, preserving RMS, modulation order, and FP32 projection. This removed previous 1.40-GiB modulation and 2.81-GiB whole-FP32 temporary allocations.

### 10.4 Bounded video VAE gathering

The 15-second workload has 21 temporal slices and 28 spatial tiles per slice. Ranks decode and gather one temporal slice at a time. Completed frames move to CPU promptly; final RGB conversion is chunked on CPU. This avoids retaining or receiving all 588 tiles on GPU at once.

## 11. Run procedure

### 11.1 Set host locations

```bash
export IMAGE='YOUR_RELEASE_IMAGE@sha256:YOUR_IMMUTABLE_DIGEST'
export H3_MODEL_DIR=/disk/ssd1/difumodels/MiniMax-h3
export H3_COMFY_DIR=/disk/ssd1/difumodels/MiniMax-H3-ComfyUI
export RUNTIME_ROOT=/disk/ssd1/lightx2v-runtime
export CACHE_DIR=$RUNTIME_ROOT/cache
```

The repository's default two-image prompt profile is `assets/minimax_h3/rooftop_15s/`. Repeat `--reference-image` in prompt order to replace it with 1-9 references.

### 11.2 Inspect the Docker command without starting it

```bash
DRY_RUN=1 \
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image "$IMAGE" --gpus 0,1,2,3 --collect
```

Confirm that model and references are read-only mounts, the result/cache paths are writable, exactly the intended GPUs are visible, and the BF16 checkpoint is mounted at `/models/minimax_h3_comfy/diffusion_models/minimax_h3_ref2va_bf16.safetensors`.

### 11.3 Run explicit GPU preflight

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image "$IMAGE" --gpus 0,1,2,3 --preflight-only --collect
```

Preflight verifies:

- visible devices are exactly the requested count and report gfx1201;
- ROCm/PyTorch and the 24-WGP/48-CU environment contract;
- runtime JSON profile, geometry, sampler, no-LoRA state, and parallel topology;
- required model assets and reference inputs;
- LightX2V registry entries and disabled-fallback policy;
- AITER dense MHA and Sage APIs;
- CK Conv3D AOT module, public APIs, universal FP16/BF16 capability, and active catalog;
- standalone GPU kernels when requested by preflight-only mode;
- distributed-to-Sage adapter and SP16 LSE API when applicable;
- image/source provenance and collector schema when `--collect` is enabled.

Do not continue on an error. Preserve warnings in the evidence; do not delete them merely to obtain a clean summary.

### 11.4 Full-shape, reduced-evaluation validation

Run the real 1344x768, 15-second, two-reference path with three model evaluations:

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image "$IMAGE" --gpus 0,1,2,3 \
  --infer-steps 3 --collect \
  --result-dir "$RUNTIME_ROOT/outputs/h3-bf16-ref2av-fullshape-3step"
```

This intentionally retains the real sequence length, output head, all 21 VAE temporal slices, 28 tiles per slice, and exact 15-second muxing. It validates the end-to-end memory and operator path. It is not a quality baseline.

### 11.5 Formal 20-evaluation run

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image "$IMAGE" --gpus 0,1,2,3 \
  --collect \
  --result-dir "$RUNTIME_ROOT/outputs/h3-bf16-ref2av-768p-15s-release"
```

Do not use `--smoke` for formal acceptance. `--smoke` changes the canvas to 480x480 and is useful only for a smaller functional diagnosis.

### 11.6 Optional scaling sweep

```bash
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image "$IMAGE" --counts 2,4,8,16
```

Run all arms with identical prompt, references, seed, image digest, sampler, and collector setting. See [Full BF16 and Turbo 8-Step Scaling](h3-ref2av-scale.md) for physical GPU groups and SP16 details.

## 12. Evidence and acceptance gates

Each accepted run must retain:

```text
result-directory/
├── run_manifest.json
├── runtime_config.json
├── preflight.json
├── run.log
├── ffprobe.json
├── minimax_h3_ref2av.mp4
├── h3_runcollector.rank000.json
├── h3_runcollector.rank001.json
├── ... one file per rank
├── h3_t1_summary.json
└── CK Conv3D route snapshots/reports
```

Also retain the image-inspection output, `/app/versions.txt`, model SHA256 manifest, prompt file, reference-image hashes, host ROCm information, and physical GPU/topology inventory.

### 12.1 Mandatory pass criteria

- Process exit code is 0 on every rank.
- Preflight status is successful; requested GPU count and gfx1201 devices match.
- Runtime config says `ref2av_bf16_base`, `dit_quant_scheme=Default`, no LoRA, `comfy_res_multistep`, 20 evaluations for the formal run, shifts 12/3, 1344x768, and 362 native frames.
- Every critical route emits `dispatch status=active` on every applicable rank.
- No `status=fallback`, Python traceback, HIP error, RCCL error, OOM, NaN, Inf, or all-zero kernel output appears.
- CK Conv3D route reports show the prebuilt AITER path and no Torch fallback for supported VAE contracts.
- `ffprobe.json` shows 1344x768 video, 24 FPS, 360 encoded frames, and 15 seconds.
- Audio is present at 32 kHz and is 15 seconds.
- The MP4 is non-empty, decodes successfully, and passes human review for prompt/reference identity, motion continuity, absence of VAE seams, and synchronized/generated sound.
- The output and every evidence artifact are associated with the immutable image digest and input hashes.

### 12.2 Claims that the evidence does not support by itself

- A three-evaluation run does not establish final quality.
- CPU/Gloo tests do not establish GPU kernel correctness or RCCL performance.
- `route status=configured` does not prove the real tensor dispatched that backend.
- Successful weight inventory does not prove model inference.
- A mutable image tag does not identify the executed binary.
- Results from the Turbo 4-step or 8-step LoRA profile are not performance baselines for the full 20-evaluation BF16 model.
- Different GPU-count topologies are not expected to be bitwise identical because SageAttention2 contains quantized internal arithmetic and distributed merge order changes.

## 13. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `Docker image is unavailable` | tag/digest not pulled or loaded | Pull the immutable digest or load the verified archive |
| `/dev/kfd` or `/dev/dri` unavailable | host ROCm/device permissions | Fix host driver and Docker permissions before retrying |
| expected gfx1201, got another arch | wrong visible GPU set | Correct `--gpus`; do not bypass preflight |
| `GPU_ARCHS`/`CU_NUM` mismatch | stale environment or image | Require `gfx1201` and `24`; use the release launcher |
| BF16 checkpoint missing | wrong `--comfy-dir` layout | Place the file under `diffusion_models/` with the exact filename |
| shared component missing | incomplete `--model-dir` | Provide `text_encoder`, `tokenizer`, `processor`, `vae`, and `audio_vae` |
| BF16 key/shape/dtype mismatch | incompatible or modified checkpoint | Stop; inventory the actual checkpoint rather than relaxing validation |
| incomplete prepared cache | interrupted conversion or changed source | Ensure no active user, remove only the named generated cache directory, rerun |
| missing CK Conv3D prebuilt module | wrong AITER wheel/image | Use the matched release image; do not JIT-build at request time |
| `status=fallback` | operator contract not met or mismatched wheel/source | Treat as release failure; inspect the first dispatch reason |
| OOM in output head or VAE gather | image predates bounded-memory fixes or workload changed | Confirm LightX2V revision and real runtime geometry; retry first on four GPUs |
| output not exactly 15 seconds | changed frame/duration/sampling config or failed finalization | Verify 362 native frames, 360 encoded frames, 24 FPS, and 480,000 audio samples/channel |
| public AITER source differs from image | expected during merge | Use wheel/image provenance now; wait for the HVAT merge-complete patch for source rebuild |

## 14. Related documents

- [Full BF16 integration and run notes](h3-ref2av-bf16.md)
- [768P/15-second memory and sampling review](h3-ref2av-768p-review.md)
- [Full BF16 and Turbo scale workflow](h3-ref2av-scale.md)
- [Turbo 8-step LoRA integration](h3-ref2av-8step.md)
- [September 9 upstream integration review](rebase-review-20260909.md)
