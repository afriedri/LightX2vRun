# H3 Ref2AV Turbo 8-step LoRA 接入

本入口在 `H3_MODEL_DIR` 的原生 `transformer_ref/` 分区上动态加载 8-step LoRA，不合并、不量化基础权重，**不经 ComfyUI 权重转换**，也不改变 Ref2AV 的参考图 packing、长视频、SP、Sage、Conv3D 和 VAE 路径。FL2AV 4-step 与 Ref2AV 原始 BF16（Comfy 权重 + 转换）两个既有入口保持原行为。

## 冻结合约

| 项目 | Ref2AV Turbo 8-step |
|---|---|
| profile | `ref2av_turbo_8step` |
| base | `H3_MODEL_DIR/transformer_ref/`（官方 Diffusers 组件，直接加载） |
| LoRA | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` |
| LoRA 应用 | dynamic，`strength=1.0`，`alpha=8` |
| sampler | `training_euler` |
| flow shift | video/audio = `12/3` |
| 步数 | 配置写 9 个 sigma 网格点，当前 HVAT scheduler 实际执行 8 次模型计算 |
| 默认 workload | 1344x768，24 FPS，15 秒，362 个原生帧，编码输出 360 帧 |
| 默认并行 | 四卡 Ulysses SP4 + VAE tile parallel |

`alpha=8` 与本地 `upstream/main` 的通用 H3 8-step 配置、已有 Ref2AV Turbo 配置以及 runner 的动态 LoRA 提示一致。不要沿用 FL2AV 4-step 的 `alpha=128`。

上游当前 scheduler 的 `infer_steps=8` 表示 8 次计算；本分支为兼容既有 HVAT FL2AV 合约，legacy `training_euler` 的 `infer_steps` 表示包含终点的网格点数。因此本入口固定为 9，builder 和 preflight 都会拒绝改成 8，最终以 `run.log` 中 8 次 `dit.denoise` 为验收依据。

建议的宿主权重布局：

```text
/disk/ssd1/difumodels/MiniMax-h3/
├── transformer_ref/   # 官方 Ref2VA 组件：config.json + 权重分片
├── transformer/       # fl2av 任务使用（已存在）
├── text_encoder/ tokenizer/ processor/ vae/ audio_vae/

/disk/ssd/difumodels-Dis/Minimax-h3-Turbo/
└── minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors
```

`transformer_ref/` 从官方 MiniMax-H3 发布结构按组件下载（与 upstream README 的目录约定一致），不是 ComfyUI 单文件权重；本入口不做任何转换。LoRA 可从 [Hugging Face](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors) 或 [ModelScope](https://modelscope.cn/models/lightx2v/Minimax-h3-Turbo) 获取。实际目录不同时使用 `--model-dir` / `--turbo-dir`。

## 首次 E2E

需要使用包含当前 LightX2V 改动的新镜像。先做 preflight，它会读取 safetensors header 而不加载整份 LoRA，检查 key 归一化、down/up 配对、二维 shape、rank、dtype 和 alpha tensor 情况：

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh \
  --preflight-only
```

再运行完整四卡 E2E：

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh
```

入口默认启用 schema-2 collector。替换 GPU、权重或参考图示例：

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh \
  --image YOUR_REBUILT_IMAGE \
  --turbo-dir /path/to/Minimax-h3-Turbo \
  --gpus 0,1,2,3 \
  --reference-image /path/to/reference-1.png \
  --reference-image /path/to/reference-2.png \
  --prompt-file /path/to/prompt.txt
```

Turbo profile 会拒绝改变 `training_euler` 或 9 个网格点。`--smoke` 只降低画布，不代表质量验收；首轮正式验收应保留 768P、15 秒和真实参考图。

## 2/4/8/16 卡 Scale

专用入口会依次运行相同的 Ref2AV Turbo 8-step、1344x768、15 秒 workload，自动采集每档日志、显存、shape 和 timing 汇总。单次和 scale 入口默认使用 `compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4`，仍可用 `--image` 显式覆盖：

```bash
bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh
```

只验证 16 卡时使用 `--gpu-count 16`。2/4/8 卡使用 Ulysses；16 卡使用 `CP2xU8` UlyssesRing，并要求 16 个 GPU 构成独占的单进程组。可用 `TIMING_GPU_GROUPS` 覆盖默认物理卡组合，结果默认写入 `${RUNTIME_ROOT}/outputs/minimax_h3_ref2av_turbo_8step_768p_15s_scale_<timestamp>/`。

## Shape 与后续优化

需要收 shape，但不需要先修改 `aiter-tune-gfx1201`。现有 H3 collector 已在动态 LoRA 的实际算子边界记录每层 down/up GEMM 的 `M/K/N/rank`、alpha、scale、strength 和 GPU 时间，同时汇总每步 `dynamic_lora_gpu_ms`。首轮 E2E 后重点检查：

- `preflight.json` 中 `lora_header.pair_count`、`ranks`、`dtypes` 与 `config_alpha_required`；
- `run.log` 中动态 LoRA 成功加载/注册，且实际出现 8 次 `dit.denoise`；
- 各 rank 的 `h3_runcollector.rankNNN.json` 中 LoRA M/K/N/rank 完整；
- `h3_t1_summary.json` 中 `dynamic_lora_gpu_ms` 占比，且无 `fallback`、OOM 或未注册 target；
- `ffprobe.json` 同时包含 1344x768 视频和 32 kHz 音频，二者时长均为 15 秒。

只有 collector 证明动态 LoRA GEMM 是关键路径热点后，才把稳定的热 shape 带到 `aiter-tune-gfx1201` 做定向 tuning。若 LoRA 占比很低，先保留默认 ROCm BLAS heuristic，避免把 FL2AV rank-128 的旧结论直接套到新 checkpoint。
