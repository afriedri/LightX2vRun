# H3 Ref2AV 原始 BF16 接入与两卡检查

本入口使用 `diffusion_models/minimax_h3_ref2va_bf16.safetensors` 原始基础模型，不加载任何 LoRA。INT8 ConvRot 暂未接入。

## 权重与采样

用户回传 inventory 的 BF16 文件为 66,280,487,368 bytes，535 个张量（522 BF16、13 FP32），50 个 Core block。全部 key/shape/dtype 与原始 H3 架构映射对应。

ComfyUI checkpoint 需要改成现有 native loader 的 key/layout：连续 Q/K/V 拆分、SwiGLU `[gate; value]` 改成 `[value; gate]`，其余按名称映射，不量化、不合并 LoRA、不改变 BF16/FP32 dtype。转换复用 [Diffusers 的 H3 转换规则](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py)；ComfyUI 已有连续 Q/K/V，不能再执行原始分片的逐 head 重排，参见 [ComfyUI H3 attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)。

转换在镜像内首次执行，按 block 写入持久缓存，之后复用。需要额外约 66.3 GB 磁盘；不会每次采样临时重排整个权重。缓存用源文件路径、大小、mtime、header hash 和转换版本识别，完整写入后才发布；没有强制扫描全文件 SHA。

采样采用现有 HVAT `configs/minimax_h3/minimax_h3_ref2av.json` 的基础模型参数：video/audio shift=12/3、`reference_blend`、`infer_steps=30`（30 个网格点，29 次模型计算）。支持 `--infer-steps` 显式调整，不继承 Turbo 4step 参数。测试几何沿用 gfx1201 的 124f@24fps、1344×768；`--smoke` 为 480×480。

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

**两张 32 GiB 卡：静态权重预算可行，完整峰值未验收。** DiT 原始 payload 合计 61.73 GiB；每块约 1.2025 GiB，两个 GPU block 缓冲约 2.405 GiB，固定部分约 1.605 GiB，合计约 4.01 GiB/卡。该数字仅计权重，不是运行显存峰值；SP2 不把全部权重简单平分。

Qwen 多模态输入、较大的参考图、Core 激活/通信临时张量和 VAE 编解码可能抬高峰值。现有 DiT buffer 为后续复用而保留，不能假设进入 VAE 时显存归零。原始模型步数增加会增加时间与磁盘传输次数，不能拿 Turbo 4step 耗时作为基线。本机没有可用 GPU/Docker 实测，不能保证 2×32 GiB 的 768p 多参考图运行成功。

## 执行

更新两仓并重建 LightX2V 镜像。旧 ReleaseFinal 镜像只适用于前面的无 GPU 权重 inventory；本入口需要新镜像内包含本次转换工具及当前 native H3 代码。

日常专用入口为 `scripts/run_10_minimax_h3_bf16_ref2v.sh`。镜像、完整 native H3 目录、ComfyUI BF16 根目录、GPU、参考图数组和 prompt 路径集中在脚本顶部，可直接编辑，也可用命令行覆盖。默认素材为 `/disk/ssd1/lightx2v-runtime/inputs/h3-ref2v/reference.png` 和 `prompt.txt`，需要换成自己的素材；不自动套用 FL2V 首尾帧或旧 prompt。完整采样默认 30 个网格点，默认 GPU `0,1`。

在权重机器的 Run 仓库，先用两卡、单参考图、少量基础模型步骤检查算子/显存：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image lightx2v-rocm:hvat-gfx1201-R9600 \
  --model-dir /disk/ssd1/difumodels/MiniMax-h3 \
  --comfy-dir /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --reference-image /path/to/reference.png \
  --prompt-file /path/to/ref2av-prompt.txt \
  --gpus 0,1 --smoke --infer-steps 3 --collect \
  --result-dir /disk/ssd1/lightx2v-runtime/outputs/h3-ref2av-smoke
```

少步骤 smoke 只用于检查运行，不代表出片质量。完整基础采样去掉 `--smoke --infer-steps 3`。增加参考图时重复 `--reference-image`，顺序对应 prompt；最多 9 张，不需要首尾帧或 LoRA 目录。

Run 只读挂载进镜像，转换和推理均在容器内执行。默认两卡 `0,1`；`CACHE_DIR` 决定宿主机持久缓存目录，权重缓存在其 `h3_ref2av/` 下。输出目录保留 runtime config、preflight、`run.log`、视频与 ffprobe；`--collect` 另生成各 rank 的 H3 生命周期/算子报告。`run.log` 中的 `dispatch` / `fallback` 用于核实实际 backend；CPU 测试不代表真实 GPU kernel 已通过。

验证结果：Run 全量 219 项通过；LightX2V 权重格式 3 项、算子路由/Ref2AV 切分/lazy offload 定向 41 项通过。所有结果均为 CPU/命令替身验证；未执行远端真实 BF16 的转换与推理。

专用脚本复用 `run_minimax_h3_ref2av_bf16.sh` 和现有镜像内执行链；两入口保持同一套权重转换、算子、offload 和 collector。显式重复 `--reference-image` 会替换脚本顶部的默认图片列表。`--help` 显示有效默认路径，`DRY_RUN=1` 可核对 Docker 挂载；`--collect` 按需启用，不默认增加 profiling 开销。
