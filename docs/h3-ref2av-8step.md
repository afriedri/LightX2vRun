# MiniMax-H3 Ref2AV Turbo 8-Step LoRA Integration

This entry dynamically applies the 8-step LoRA to the native `H3_MODEL_DIR/transformer_ref/` partition. It does not merge or quantize the base weights, does not run the ComfyUI BF16 converter, and does not change Ref2AV reference packing, long-video handling, sequence parallelism, SageAttention2, Conv3D, or the VAE route. The FL2AV 4-step and full BF16 Ref2AV profiles retain their existing behavior.

## Frozen contract

| Item | Ref2AV Turbo 8-step value |
| --- | --- |
| Profile | `ref2av_turbo_8step` |
| Base | `H3_MODEL_DIR/transformer_ref/`, the official Diffusers component loaded directly |
| LoRA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |
| Application | Dynamic, `strength=1.0`, `alpha=8` |
| Sampler | `training_euler` |
| Flow shift | video/audio = `12/3` |
| Step semantics | 9 configured sigma grid points, 8 model evaluations |
| Workload | 1344x768, 24 FPS, 15 seconds, 362 native frames, 360 encoded frames |
| Default parallelism | Four-GPU Ulysses SP4 plus VAE tile parallelism |

`alpha=8` agrees with the local upstream generic H3 8-step config, the existing Ref2AV Turbo config, and the runner's dynamic-LoRA contract. Do not copy FL2AV 4-step's `alpha=128`.

Upstream currently interprets `infer_steps=8` as eight evaluations. This branch preserves the older HVAT FL2AV contract, in which legacy `training_euler` interprets `infer_steps` as grid points including the endpoint. The profile therefore fixes the value at 9. Both the builder and preflight reject 8. Acceptance uses the eight `dit.denoise` evaluations observed in `run.log`.

Recommended host layout:

```text
/disk/ssd1/difumodels/MiniMax-h3/
├── transformer_ref/   # official Ref2VA component: config.json and weight shards
├── transformer/       # FL2AV component
├── text_encoder/
├── tokenizer/
├── processor/
├── vae/
└── audio_vae/

/disk/ssd/difumodels-Dis/Minimax-h3-Turbo/
└── minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
```

Download `transformer_ref/` as a component of the official MiniMax-H3 release structure. It is not the ComfyUI single-file checkpoint and does not require conversion. The LoRA is available from [Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors) or [ModelScope](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo). Use `--model-dir` and `--turbo-dir` for non-default locations.

## First end-to-end run

Use an image containing the current LightX2V integration and the matching AITER wheel. The AITER commits visible under the `yangecool` GitHub account are not the latest HVAT internal source used to build that wheel. Code merging is in progress, and HVAT will provide the corresponding source patch after the merge is complete. Treat the installed wheel and image provenance as the executable release contract.

Preflight parses the safetensors header without loading the complete LoRA and checks normalized keys, down/up pairs, two-dimensional shapes, ranks, dtypes, and optional alpha tensors:

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh \
  --preflight-only
```

Then run the complete four-GPU workload:

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh
```

Example with explicit image, weights, devices, references, and prompt:

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh \
  --image YOUR_RELEASE_IMAGE \
  --turbo-dir /path/to/Minimax-h3-Turbo \
  --gpus 0,1,2,3 \
  --reference-image /path/to/reference-1.png \
  --reference-image /path/to/reference-2.png \
  --prompt-file /path/to/prompt.txt
```

The profile rejects any sampler other than `training_euler` and any grid-point count other than 9. `--smoke` reduces the canvas and is not a quality acceptance run. The first formal acceptance should preserve 768P, 15 seconds, and production reference images.

## 2/4/8/16-GPU scale sweep

The dedicated sweep executes the same Ref2AV Turbo 8-step, 1344x768, 15-second workload at each selected GPU count and collects logs, memory, shapes, and timing summaries. Its current default image is `compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4`; use `--image` to select a release explicitly:

```bash
bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh
```

Use `--gpu-count 16` for only the 16-GPU arm. SP2/SP4/SP8 use Ulysses. SP16 uses `CP2xU8` UlyssesRing and requires an exclusive 16-GPU process group. `TIMING_GPU_GROUPS` overrides the default physical-device groups. Results are written under `${RUNTIME_ROOT}/outputs/minimax_h3_ref2av_turbo_8step_768p_15s_scale_<timestamp>/` by default.

## Shape evidence and optimization handoff

The existing H3 collector records each dynamic-LoRA down/up GEMM contract at the real operator boundary: M/K/N/rank, alpha, scale, strength, and GPU time. It also aggregates `dynamic_lora_gpu_ms` by denoising step. No `aiter-tune-gfx1201` change is required before the first run.

After the first end-to-end run, check:

- `preflight.json`: `lora_header.pair_count`, `ranks`, `dtypes`, and `config_alpha_required`;
- `run.log`: successful LoRA load/registration and exactly eight `dit.denoise` evaluations;
- every `h3_runcollector.rankNNN.json`: complete LoRA M/K/N/rank contracts;
- `h3_t1_summary.json`: dynamic-LoRA time and the absence of fallback, OOM, or unregistered targets;
- `ffprobe.json`: 1344x768 video and 32-kHz audio, both exactly 15 seconds.

Promote shapes into `aiter-tune-gfx1201` only if the collector demonstrates that dynamic-LoRA GEMM is a critical-path hotspot. If its contribution is small, retain the default ROCm BLAS heuristic rather than transferring the old FL2AV rank-128 result to a different checkpoint.
