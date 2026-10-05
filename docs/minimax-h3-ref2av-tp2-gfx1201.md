# MiniMax-H3 Ref2AV, TP2 on 2x AMD Radeon R9600D (gfx1201)

This entry reproduces the MiniMax-H3 full BF16 Ref2AV run on two AMD Radeon
R9600D GPUs (gfx1201, 24 CUs, 32 GiB each) with tensor parallelism (TP2):

- 1344x768, 362 frames at 24 FPS (15.08 s) plus 32 kHz stereo audio
- 20 denoising steps, seed 42, no LoRA, no CFG
- rooftop material (2 reference images) by default; the 9-reference storeroom
  material is also supported
- no quantization beyond the SageAttention kernel's own int8 QK / fp8 PV
  (`RADEON_CORESW_FP8=0`, `LIGHTX2V_H3_VAE_FP8=0`)

Unlike the Ulysses entries 10-13, this entry uses the TP2 path of the
`rocm/vllm-dev:gfx1201-h3-*` images (LightX2V in `/app/LightX2V`). It does not
use `lightx2vModelrun` or the ComfyUI checkpoint conversion. It reads the
native MiniMax-H3 model directory (HF layout, including `transformer_ref/`).

## Run

```bash
docker pull rocm/vllm-dev:gfx1201-h3-ho-sk7

# Measured configuration: GPUs 0,1, CPU/memory pinned to their NUMA node.
H3_MODEL_DIR=/path/to/MiniMax-H3 \
bash scripts/run_14_minimax_h3_ref2av_tp2_768p_15s.sh --gpus 0,1 --numa-node 0 \
    --result-dir "$PWD/outputs/h3_tp2_rooftop"
# -> outputs/h3_tp2_rooftop/minimax_h3_ref2av_tp2_20.{mp4,log}
```

Look up a GPU's NUMA node with
`cat /sys/bus/pci/devices/<pci-bdf>/numa_node` (`rocm-smi --showbus` lists
the BDFs). `DRY_RUN=1` prints the `docker run` command without starting it.

| Option / variable | Default | Meaning |
|---|---|---|
| `--image` / `IMAGE` | `rocm/vllm-dev:gfx1201-h3-ho-sk7` | Runtime image |
| `--gpus` / `GPU_DEVICES` | `0,1` | Exactly two physical HIP GPU ids |
| `--model-dir` / `H3_MODEL_DIR` | `/disk/ssd1/difumodels/MiniMax-h3` | MiniMax-H3 directory, mounted read-only at `/models/minimax_h3` |
| `--material` / `H3_MATERIAL` | `rooftop` | `rooftop` or `storeroom`, from `configs/minimax_h3/ref2av_*.sh` |
| `--reference-image`, `--prompt-file` | | Custom material (1-9 images, in prompt order) |
| `--result-dir` / `H3_RESULT_DIR` | `${OUTPUT_DIR:-$PWD/outputs}/minimax_h3_ref2av_tp2_<time>` | Host directory for the mp4 and log |
| `--infer-steps` / `H3_INFER_STEPS` | `20` | The image ships AdaLN caches for 1, 3 and 20 steps; other counts generate one first |
| `--sage-preset` / `H3_SAGE_PRESET` | `fast` | Attention core schedule, see below |
| `--numa-node` / `NUMA_NODE` | none | `--cpuset-cpus/--cpuset-mems` for that node |
| `DOCKER_NETWORK` | `bridge` | Container network; do not use `host` (see RCCL init failure) |
| `DRI_ALL` | `1` | `1`: all of `/dev/dri`, `--ipc=host`, `HIP_VISIBLE_DEVICES=<gpus>`. `0`: only the nodes in `H3_DRI_NODES`, `--ipc=private` |
| `CONTAINER_NAME`, `DOCKER_EXTRA_ARGS` | | Container name; extra `docker run` arguments |

The repository is mounted read-only at `/workspace/LightX2VRun`. The container
runs `scripts/lib/run_minimax_h3_ref2av_tp2_in_container.sh`, which sources
`scripts/lib/minimax_h3_gfx1201_rocm_env.sh`, writes the TP2 runtime config
from the image's `rooftop_15s/runtime_config.json` (override with
`H3_BASE_RUNTIME_CONFIG`), runs `torchrun --nproc_per_node=2 -m lightx2v.infer`,
and prints a summary: banners, per-rank step times and `Total ... s`.

## Attention core schedule

`AITER_GFX1201_SAGE_ASM` selects the gfx1201 SageAttention ASM core.
`AITER_GFX1201_SAGE_ASM_SCHED=core_a:N,core_b` runs `core_a` for the first N
attention calls and `core_b` for the rest. An empty schedule uses
`AITER_GFX1201_SAGE_ASM` for every call.

| `--sage-preset` | `AITER_GFX1201_SAGE_ASM` | `AITER_GFX1201_SAGE_ASM_SCHED` | Numerics |
|---|---|---|---|
| `fast` (default) | `esp321wsk7` | `esp321wsk7:250,esp321wsk7t2` | Exact for the first 250 calls (5 steps), approximate after |
| `exact` | `esp321wsk7` | (empty) | Bitwise identical to `esp321w` |
| `t2` | `esp321wsk7t2` | (empty) | Approximate on every call |
| `esp321w` | `esp321w` | (empty) | The image's ENV default, without block skipping |

- `esp321w` is the base core: 240 VGPRs (3 workgroups per WGP), x2 unroll,
  4-chain QK WMMA.
- `esp321wsk7` adds a skip for 16x16 blocks whose FP8 P values are all zero.
  Its output is bitwise identical to `esp321w`.
- `esp321wsk7t2` also skips blocks whose FP8 P values are all tiny (per-lane
  8-key sum < 2^-2). That adds about 2.4e-3 relative L2 error per attention
  call versus the exact core. Sage itself is about 5e-3 from exact softmax.

Explicit `AITER_GFX1201_SAGE_ASM` / `AITER_GFX1201_SAGE_ASM_SCHED` values in
the environment override the preset. Set `AITER_GFX1201_SAGE_ASM_SCHED=` to
force an empty schedule.

## Other switches

All are on by default in the in-container entry. Each can be overridden from
the host (`0` = off) and is passed into the container.

| Variable | Default | Effect |
|---|---|---|
| `FUSED_TAIL` (`RADEON_CORESW_FUSED_TAIL`) | `1` | Fused TP2 block tail with the IPC pull reducer; bitwise equal to the unfused tail |
| `AITER_GFX1201_DOWN_BLAS` | `1` | FFN down GEMM through hipBLASLt; 0.08% of outputs differ by 1 BF16 ulp from the aiter Triton kernel |
| `TE_PAGEABLE` | `1` | Pageable (not pinned) host staging of text-encoder weights |
| `VAE_ENC_PAR` | `1` | VAE encode split over both ranks |
| `H3_TEXT_ENCODER_TP` | `1` | Qwen3-VL text encoder TP2 (`0`: replicated encoder) |
| `LIGHTX2V_H3_VAE_DEC_BATCH` | `2` | Video VAE decode: tiles per call (bitwise) |
| `LIGHTX2V_H3_VAE_TRITON_ATTN` | `1` | Video VAE decode attention via Triton flash attention |
| `LIGHTX2V_H3_TE_PREFETCH_LATE` | `1` | Text encoder: issue the next-layer prefetch after the current layer |
| `LIGHTX2V_H3_TE_ANON_HOST` | `1` | Text encoder: copy pageable host weights to anonymous memory |
| `LIGHTX2V_H3_AUDIO_POLY_UPSAMPLE` | `1` | Audio VAE polyphase upsampler |
| `LIGHTX2V_H3_GC_FREEZE`, `_ASYNC_VAE_LOAD`, `_VAE_ENC_PREWARM` | `1` | gc.freeze, async VAE load, MIOpen prewarm of the VAE encoder |
| `LIGHTX2V_H3_DIT_PIN_SLAB` | `1` | DiT offload weights page-locked in 512 MiB slabs |
| `LIGHTX2V_H3_VAE_STREAM_SAVE` | `1` | Stream the parallel VAE decode into the mp4 encoder |

## Expected timings

Measured on 2x R9600D, 2026-10-01, 20 steps, `DRI_ALL=1`, bridge network,
NUMA-pinned. `Total` is LightX2V's end-to-end `Total Cost`: model load,
encoders, 20 DiT steps, VAE decode and save.

| Material | Preset | GPUs | Total (s) |
|---|---|---|---:|
| rooftop | `fast` | 0,1 | **2032.74** |
| rooftop | `t2` | 6,7 | 2061.07 |
| rooftop | `esp321w` | 0,1 / 6,7 | 2078.58 / 2087.09 |
| storeroom (9 refs) | `fast` | 0,1 | 2389.67 |

- For comparison, HVAT reports 756.18 s for this rooftop workload on 2x NVIDIA
  RTX 5090. The 2032.74 s run takes 2.69x as long (756.18 / 2032.74 = 37.2%).
- Run-to-run noise on a shared host is about +-1.5% (20-30 s). GPU pairs differ
  by a few seconds. On one pair (6,7) the exact skip core was about 1% slower
  than `esp321w`; there, `t2` was the fastest preset.
- The storeroom material's 9 references make the sequence 126,130 tokens
  instead of 114,660, so steps take about 15-25% longer.
- A full run takes about 35-45 minutes.

## Checking a run

A healthy log shows:

- `[aiter] gfx1201 sage ASM variant=<core> hsaco=attention_pv_early_<core>.hsaco` once per rank, and the schedule
- `gfx1201_sage_attention route=ASM_PV_EARLY ... shape=(1, 114660, 28, 128)` (rooftop)
- `[ACTIVE] FUSED_TAIL reducer=pull` and `[ACTIVE] TP2 fused block submitted ... fp8=[]`
- `Video VAE: fused Triton norm/RoPE/residual/SwiGLU ops on 36/36 decoder blocks (fp8_linears=False)`
- `pin slab: ... (failed chunks: 0)` and `text encoder: ... copied to anonymous memory`
- 20 `Run Dit every step cost` lines per rank, `output saved ... (362 frames, streamed)`, `tracebacks: 0`

The mp4 is h264 1344x768 with 362 frames and about 15.07 s of AAC audio. The
rooftop prompt ends with a cut to black, so the last frame is black.

Two runs of the same image are not bitwise identical. The VAE encode (MIOpen,
non-deterministic) differs run to run, and 20 steps amplify that. Two identical
runs differ at about 20 dB video PSNR, the same level as a CUDA-path reference
run versus a ROCm run. Compare quality by eye, not by PSNR.

## RCCL init failure with `--network=host`

Symptom: the warmup all_reduce fails with `hipIpcGetMemHandle failed : invalid
argument` (rccl `p2p_tmp.cc`) while another fresh ROCm container is running.

Cause: for IPC, ROCr binds an abstract unix socket named `@hsa<pid>` using the
container-local pid. Abstract sockets belong to the network namespace. In a
fresh container the torchrun ranks always get the same pids, so two
`--network=host` containers collide on the same names.

Fix: the launcher uses `--network=bridge`, which gives each container its own
namespace. `--network=none` does not work, because torchrun's c10d store cannot
resolve the container hostname. On the host,
`grep -a ' @hsa' /proc/net/unix` lists the names held in the host namespace.
`lightx2vModelrun` accepts `DOCKER_NETWORK=bridge` for the same reason.

## Image contents and source branches

`rocm/vllm-dev:gfx1201-h3-ho-sk7` is built in layers:

1. Base `lightx2v-rocm:gfx1201-h3-merged-v1`: aiter PR
   [#5900](https://github.com/ROCm/aiter/pull/5900) (gfx1201 SageAttention ASM
   and H3 TP2 kernels) and LightX2V PR
   [#1563](https://github.com/ModelTC/LightX2V/pull/1563) (MiniMax-H3 on
   gfx1201).
2. Two `git am` series on `/app/aiter` and `/app/LightX2V`. All code paths are
   opt-in in code and turned on through the image ENV and the launchers.
3. The `esp321wsk7`, `esp321wsk7t2` (and `esp321wsk7m5`) cores. These port the
   zero-P block skip onto the `esp321w` schedule, together with their entries
   in `aiter/ops/gfx1201/asm_attention.py`.

The image also contains a pre-seeded Triton cache for the new kernels and
AdaLN caches for 1, 3 and 20 steps.

The same changes are being upstreamed on these branches:

| Repository | Branch | Builds on | Contents |
|---|---|---|---|
| [ROCm/aiter](https://github.com/ROCm/aiter) | `h3/gfx1201-ho-sk7` | PR #5900 | Sage ASM cores esp321, esp321x/y/z, esp321w and esp321wsk7/t2 with the call schedule; IPC pull reducer and row-chunked fused TP2 block tail; opt-in tail flags; `AITER_GFX1201_DOWN_BLAS` |
| [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) | `h3/gfx1201-ho-sk7` | PR #1563 | Fused TP2 block tail with the pull reducer; video VAE decode batching + Triton attention; text-encoder late prefetch and anonymous host copy; audio polyphase upsampler; gc.freeze / async VAE load / VAE-encoder prewarm; DiT pin slab; streamed VAE decode into the mp4 encoder |

| Flag | Repository |
|---|---|
| `AITER_GFX1201_SAGE_ASM`, `AITER_GFX1201_SAGE_ASM_SCHED`, `AITER_GFX1201_DOWN_BLAS` | aiter |
| `RADEON_CORESW_FUSED_TAIL` | LightX2V (needs the aiter pull reducer) |
| `LIGHTX2V_H3_*`, `TE_PAGEABLE` / `VAE_ENC_PAR` / `H3_TEXT_ENCODER_TP` (runtime config keys) | LightX2V |

An image built from those two branches on the PR #5900 / #1563 base runs with
this entry unchanged (`--image <your-tag>`).
