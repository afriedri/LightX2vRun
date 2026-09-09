# H3 Ref2AV 原始 BF16 接入与两卡检查

本入口使用 `diffusion_models/minimax_h3_ref2va_bf16.safetensors` 原始基础模型，不加载任何 LoRA。INT8 ConvRot 暂未接入。

## 权重与采样

用户回传 inventory 的 BF16 文件为 66,280,487,368 bytes，535 个张量（522 BF16、13 FP32），50 个 Core block。全部 key/shape/dtype 与原始 H3 架构映射对应。

ComfyUI checkpoint 需要改成现有 native loader 的 key/layout：连续 Q/K/V 拆分、SwiGLU `[gate; value]` 改成 `[value; gate]`，其余按名称映射，不量化、不合并 LoRA、不改变 BF16/FP32 dtype。转换复用 [Diffusers 的 H3 转换规则](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py)；ComfyUI 已有连续 Q/K/V，不能再执行原始分片的逐 head 重排，参见 [ComfyUI H3 attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)。

转换在镜像内首次执行，按 block 写入持久缓存，之后复用。需要额外约 66.3 GB 磁盘；不会每次采样临时重排整个权重。缓存用源文件路径、大小、mtime、header hash 和转换版本识别，完整写入后才发布；没有强制扫描全文件 SHA。

采样默认使用 AMD BF16 overlay 的 `comfy_res_multistep`：参考 ComfyUI 的 `simple` 时间表和确定性二阶更新，video/audio shift=12/3，`infer_steps=20` **表示 20 次模型计算**。包含音频 latent 到视频采样时间表的换算，保持 native H3 模型输入语义。ComfyUI 不作为运行依赖；BF16 GEMM、aiter attention/RMS/RoPE、SP、lazy block offload 和 H4 算子继续使用 HVAT 实现。

旧 `reference_blend` / `training_euler` 仍可选，旧模式的 `infer_steps` 继续表示含终点的网格点数。默认参考图选择已有 `match` 模式：两张 2752×1536 图片分别缩到 1344×768，每张贡献 1008 个参考 latent 行；`max` 和 `diffusers` 也保留。可通过 `H3_STEP_UPDATE`、`H3_REFERENCE_IMAGE_RESIZE_MODE` 覆盖，无需更换权重或算子。

专用脚本默认 1344×768、24 FPS、15 秒。原生推理对齐为 362 帧，编码前裁成 360 帧与 480,000 个 32 kHz 音频采样；只编码一次。`--duration 5..15` 可调整时长，`--smoke` 仅把画布改成 480×480，不改变时长。

Qwen、tokenizer、processor、视频/音频 VAE 复用之前的完整 native H3 目录。本入口暂不加载 ComfyUI 目录中的 Qwen/VAE 单文件。

## 算子与显存

| 部位 | 路径 |
|---|---|
| Core GEMM | 原有 BF16 `torch.mm`，ROCm BLAS 默认选择，TunableOp 关闭 |
| Core attention | `aiter_fav3_sage_bf16_attn`；保持现有 QK INT8 / PV FP8 的 Sage 内部计算方式 |
| Core Q/K RMSNorm | `minimax_h3_aiter_triton_qk`，D128 |
| Core RoPE | `minimax_h3_triton_strict_rope`，旋转宽度 96 |
| Qwen / refiner / H4 attention | 原有 `aiter_triton_mha` |
| 驻留策略 | DiT lazy + block offload；Qwen block offload、SP 时仅 rank0 构造并在编码后卸载；VAE offload + tile parallel |

参考图增加序列行数，架构仍为 56 heads × D128。复用上游 Ref2AV packing 和 HVAT SP 切分；算子不依赖固定 FL2V 序列长度。单张、3 张、9 张参考图的 SP2/4/8/16 切分与位置/时间索引已做 CPU 回归；29 次模型计算的 block 双缓冲轮转也已验证。未新增 GPU 算子或每步格式适配。

**两张 32 GiB 卡：旧输出头及后续 VAE 整段汇总均出现过 OOM，两处已改为分块处理，须重建镜像复测。** DiT 原始 payload 合计 61.73 GiB；每块约 1.2025 GiB，两个 GPU block 缓冲约 2.405 GiB，固定部分约 1.605 GiB，合计约 4.01 GiB/卡。该数字仅计权重，不是运行显存峰值；SP2 不把全部权重简单平分。

Qwen 多模态输入、较大的参考图、Core 激活/通信临时张量和 VAE 编解码可能抬高峰值。现有 DiT buffer 为后续复用而保留，不能假设进入 VAE 时显存归零。原始模型步数增加会增加时间与磁盘传输次数，不能拿 Turbo 4step 耗时作为基线。本机没有可用 GPU/Docker 实测，不能保证 2×32 GiB 的 768p 多参考图运行成功。

## 执行

更新两仓并重建 LightX2V 镜像。旧 ReleaseFinal 镜像只适用于前面的无 GPU 权重 inventory；本入口需要新镜像内包含本次转换工具及当前 native H3 代码。

日常专用入口为 `scripts/run_10_minimax_h3_bf16_ref2v.sh`。镜像、完整 native H3 目录、ComfyUI BF16 根目录、GPU、参考图数组和 prompt 路径集中在脚本顶部，可直接编辑，也可用命令行覆盖。默认素材已设为用户提供的 `promptsandpics.zip`，存放在 `assets/minimax_h3/rooftop_15s/`：`Image Gen-1.png` 对应 `<Picture 1>`，`RYAN-2.png` 对应 `<Picture 2>`；使用调整后的 `prompt.txt`，保留原文 `prompt.original.txt`。仅去掉不存在的外部音频引用，保留六镜头、15 秒屋顶打斗、喘息、音效和配乐描述，音频由 H3 生成。默认新采样执行 20 次模型计算，默认 GPU `0,1`；本轮优先使用 `--gpus 0,1,2,3` 做完整流程验收，不强求两卡。

在权重机器的 Run 仓库，使用默认素材完整生成：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh --image lightx2v-rocm:hvat-gfx1201-R9600 --collect
```

镜像必须包含新采样器和 VAE 按时间片汇总修复；Run 使用当前宿主仓库的只读挂载。要先核对 **768P、15 秒、双图实际形状** 的算子/显存，在上面命令后加 `--infer-steps 3`，保留完整画布和时长；新采样模式计算 3 次模型，不代表出片质量。

替换素材、显式跑较小 smoke 的示例：

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

少步骤 smoke 只用于检查运行，不代表出片质量。完整基础采样去掉 `--smoke --infer-steps 3`。增加参考图时重复 `--reference-image`，顺序对应 prompt；最多 9 张，不需要首尾帧或 LoRA 目录。

Run 只读挂载进镜像，转换和推理均在容器内执行。默认两卡 `0,1`；`CACHE_DIR` 决定宿主机持久缓存目录，权重缓存在其 `h3_ref2av/` 下。输出目录保留 runtime config、preflight、`run.log`、视频与 ffprobe；`--collect` 另生成各 rank 的 H3 生命周期/算子报告。H3 入口禁止选定算子回退；`run.log` 中 `dispatch` 记录实际 backend，出现 `fallback` 原因后本次运行失败；CPU 测试不代表真实 GPU kernel 已通过。

本次算子审查、测试范围与待实测项目见 [768P Ref2V 审查报告](h3-ref2av-768p-review.md)。用户已提供远端真实 BF16 推理的输出头及 VAE 汇总 OOM 日志；修复后的远端完整推理尚未验收。可用 `--gpus 0,1,2,3` 切换 SP4，详细分析及复测命令见上述报告。

专用脚本复用 `run_minimax_h3_ref2av_bf16.sh` 和现有镜像内执行链；两入口保持同一套权重转换、算子、offload 和 collector。显式重复 `--reference-image` 会替换脚本顶部的默认图片列表。`--help` 显示有效默认路径，`DRY_RUN=1` 可核对 Docker 挂载；`--collect` 按需启用，不默认增加 profiling 开销。


保留旧 HVAT 采样和参考图处理的运行方式：

```bash
H3_STEP_UPDATE=reference_blend H3_REFERENCE_IMAGE_RESIZE_MODE=diffusers \
bash scripts/run_10_minimax_h3_bf16_ref2v.sh --image YOUR_REBUILT_IMAGE --gpus 0,1,2,3 --collect
```

专用脚本检测旧采样模式后默认使用 30 个网格点（29 次计算），也可用 `--infer-steps` 覆盖。选择 Comfy 采样并不意味着复制 ComfyUI 整套工作流；Qwen/VAEs 仍用原完整 native H3 目录，不切换成模板中的 INT8/NVFP4 权重。
