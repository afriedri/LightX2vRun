# H3 BF16 Ref2V 768P 算子审查（2026-09-09）

目标：原始 `minimax_h3_ref2va_bf16.safetensors`，无 LoRA，双参考图，1344×768、24 FPS、15 秒，2×32 GiB gfx1201。**代码接入与 CPU 契约已核查；长序列 GPU 算子性能、数值表现和显存峰值尚未验收。**

依据历史 `minimax_h3_day3_opportunity_ledger.md`、`minimax_h3_day4_coding_diary.md` 与当前 LightX2V/aiter 代码交叉检查。保留较新的原生 GQA/varlen 和 Sage 能力分派，不恢复过时的有限形状白名单。

## 发现与修复

- Run 的 124 帧硬限制会阻止 15 秒请求，已仅为 Ref2AV 开放原生 124..362、`17*n+5` 范围。SP16 的原生 124 帧限制保留，不影响 SP2。
- 原生 15 秒须推理 362 帧；新增可选输出时长，在原有 H.264/AAC 编码前切成 360 帧及每声道 480,000 个 32 kHz 采样，不重复编码。未指定时长的既有输出保持不变。
- 旧方案允许 RMS/RoPE/H4 失败时回退 Torch；按当前要求，H3 Run 入口通过现有日志接口启用 `LIGHTX2V_NO_FALLBACK_MODELS=minimax_h3`，在执行回退算子之前报错。原设计的 Torch GEMM/CNN 正常保留；其他模型不受影响。preflight 检查原生 Sage/FlyDSL、Triton RMS 导入与此运行策略。
- Qwen SDPA 数值对照算子继承旧基类，缺少新版运行日志接口，构造路径会报错。改用现有平台 attention 基类，并恢复此前已跟不上构造签名/日志接口的 Qwen GQA、Vision varlen 测试。默认 aiter 路由保持不变。

## 算子逐段结论

| 环节 | 当前执行路径 | Ref2V 检查结论 |
|---|---|---|
| H1 Vision | Qwen 图像预处理、packed varlen → `aiter_triton_mha` | 两张图分别保留序列边界，无需改成拼接图；复用现有 varlen 契约 |
| H1 Text | Qwen 前 50 层，causal GQA → `aiter_triton_mha` | 保留原生 GQA，不物化重复 KV heads；rank0 编码后广播、卸载 |
| H2 参考图 | 原有视频 VAE CNN、256 像素 tile、seed-42 posterior | 每张图编码为参考 latent，不走首尾帧 anchor；此 CNN 是 native Torch 路径，并非全部算子都在 aiter |
| H3 Refiner | BF16 GEMM + `aiter_triton_mha` | 使用多模态 presentation；无 FL2V 固定长度要求 |
| H3 Core attention | `aiter_fav3_sage_bf16_attn` → aiter native gfx1201 Sage | SP2 交换后 28 heads、D128、dense non-causal self-attention；引用图改变 token 数，不改变算子契约。显式 native backend 不会静默退到另一种 attention |
| H3 QK RMS / RoPE | aiter Triton D128 RMS / strict Triton rotary96 | CPU 检查通过长序列位置及时间索引切分；选定内核的编译/执行失败会停止该次 H3 运行，禁止继续回退 Torch |
| H3 GEMM / FFN | BF16 `torch.mm`、原有 ROCm BLAS 选择 | 权重一次转换后直接用 native loader；不量化权重、不加载 LoRA、不增加每步权重重排；新 M 维耗时仍需采集 |
| H4 视频解码 | 原有 FP16/D64 VAE tile parallel + `aiter_triton_mha` | 实际 768P/362f tile 构造回归确认全部为 S1797、32 heads、D64，命中现有 Day4 配置；不转 BF16、不换 Sage，时长仅增加 tile 数量 |
| H5 / H6 | native audio VAE → 原有 H.264/AAC 编码 | 无外部音频输入仍会生成音频；编码前同步裁切，ffprobe 验证帧数、音视频时长和分辨率 |

aiter 的 `_select_sage_attention_v2_gfx1201_config` 按 B1/D128 能力和尾块情况选择 PAD8/KV-prefetch 调度，没有有限的 FL2V 形状白名单。`flash_attn_varlen_func` 在 gfx1201 显式走 Triton MHA。无需另写 Ref2V 专用 GPU 内核。

## 性能与双卡显存边界

两张原图均为 2752×1536，现有 `diffusers` resize 策略把它们放大到 3680×2048。每张贡献 7,360 行参考 latent，共 14,720 行。362 帧对应 107 个视频 latent 帧，即 107,856 行目标视频；音频为 603×2=1,206 行。因此 Core 总序列为 **123,782 + Qwen presentation 行数**，不能拿原 FL2V 小序列的测试耗时和峰值当作本次结果。

SP2 在 attention 前交换完整序列、拆分 heads，并非始终只持有半段序列。Q/K/V、通信缓冲、Sage 量化暂存、FFN 激活以及高分辨率参考图编码都占显存。之前约 4.01 GiB/卡仅是 DiT 常驻与双缓冲权重预算，不能据此保证 32 GiB 足够。

没有新增热循环适配或主机同步；输出裁切只创建视图。时长和参考序列增长本身会增加计算，dense attention 的开销不能按视频时长简单线性估算。保留参考图 resize 语义；如需缩图省显存，应作为明确的质量/性能选择单独测试。

## 验证及实机验收

Run 全量 CPU/命令替身测试 236 项通过。原有编码器的合成数据封装检查得到 360 帧、视频/音频均 15.0 秒；这不是模型生成的视频。LightX2V 定向 62 项（权重转换、长序列切分、Qwen GQA/varlen、RMS/RoPE、禁止回退、H4 tile、offload、输出）以及 aiter 配置 21 项通过；均不等于实际 GPU kernel 已通过。

已有 preflight attention 探针仅 257 行，适合查基本 kernel/通信可用性，不能证明本次长序列的显存和吞吐。按本次真实形状先采集：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image lightx2v-rocm:hvat-gfx1201-R9600 --infer-steps 3 --collect
```

这条命令保留双图、768P、15 秒，只减少模型评估次数。核对 `runtime_config.json`、`run.log` 中实际 `dispatch/fallback`、各 rank collector 的形状/阶段耗时/峰值，以及 `ffprobe.json`。完整视频生成去掉 `--infer-steps 3`；默认 30 个网格点、29 次模型计算。使用重建后的 LightX2V 镜像与更新后的 Run。


## 2026-09-09：首次 2×32 GiB 实跑的输出头 OOM

用户提供的日志定位到 `MiniMaxH3PostInfer.infer`：rank0 调制张量申请 1.40 GiB 失败（剩余 468 MiB）；rank1 将完整隐藏状态转 FP32 时申请 2.81 GiB 失败（剩余 1.57 GiB）。这次已经到达 DiT 输出头，尚不能证明后续去噪步骤、VAE 解码及整片生成通过。NCCL 清理警告是异常退出后的信息。

代码确认 SP2 和 SP4 都在输出头前 all-gather 完整序列。因此加到 4 卡会减少 Core 的部分激活和 attention heads，但不会把旧输出头的 2.81 GiB 临时张量减半；固定权重、参考图预处理也不会按卡数均分。不能承诺直接加卡就能解决。

修复沿用现有逐行 RMS、调制顺序和 FP32 输出投影，按各模态索引每 4096 行执行。隐藏宽度 7168 时，单块 FP32 输入为 112 MiB；省去对文本行的输出计算，以及给视频行算音频头、给音频行算视频头的无用投影。保留所有参考条件行的输出和既有顺序，不改 BF16 权重、算子 backend、参考图 resize、768P/15 秒。分块增加小 GEMM 调用数，GPU 时延及累计数值差异仍需实测，不能宣称逐位相同或零性能成本。

两卡设备占用与当前进程 PyTorch reserved 之间均有约 22 GiB 的差额。用户确认没有其他任务；现有日志不能确定其来自哪个运行库或驱动统计。新增 collector 各 stage 的 `memory_start` / `memory_end`，记录设备 free/total、PyTorch allocated/reserved、累计峰值和统计差额。只在 `--collect` 模式查询，不清缓存、不改变 allocator；不是阶段内设备峰值采样。

先用包含本次修复的新镜像，保留默认双图、768P、15 秒，执行 2 次模型计算和完整解码：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image YOUR_REBUILT_IMAGE --gpus 0,1 --infer-steps 3 --collect
```

如仍显存不足，用同一入口切 4 卡：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image YOUR_REBUILT_IMAGE --gpus 0,1,2,3 --infer-steps 3 --collect
```

通过后去掉 `--infer-steps 3`，恢复原始模型默认 30 个网格点。少步运行只验流程和峰值，不验出片质量。保存每个 rank 的 collector 和完整 `run.log`；输出头修复位于 LightX2V，单独更新宿主 Run 不会更新镜像内的代码。4 卡长序列切分、配置/preflight 和 Docker GPU 参数均补充 CPU 回归；没有本地 GPU 实测。

本次验证：LightX2V 定向 52 项通过（分块数值/临时行数、SP2/SP4 长序列、collector、offload 和音视频输出）；Run 定向 102 项通过（Ref2AV 配置、preflight、启动参数）。新增及修改的输出头测试通过 Ruff；两个仓库 `git diff --check` 通过。上述为 CPU/命令替身检查，不是两卡或四卡 GPU 验收。
