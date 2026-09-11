# H3 Ref2AV INT8 ConvRot 单权重采集

目标文件只有：

```text
/disk/ssd1/difumodels/MiniMax-H3-ComfyUI/
└── diffusion_models/
    └── minimax_h3_ref2va_int8_convrot.safetensors
```

该文件不能直接替换现有 `H3_MODEL_DIR/transformer_ref/`。ComfyUI checkpoint
使用 fused QKV/FC1 key 和 ConvRot sidecar，LightX2V runtime 使用拆分后的原生
key；接入前必须确认实际 tensor header、metadata、量化覆盖范围、
`weight_scale` shape 和 `convrot_groupsize` 值。

## INT8-only collector

在权重机器的最新 `LightX2VRun` 仓库执行：

```bash
bash scripts/collector/run_minimax_h3_ref2v_weights.sh \
  --image compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4 \
  --model-root /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --asset int8-convrot \
  --output-dir /disk/ssd1/lightx2v-runtime/h3-ref2va-int8-inventory
```

collector 兼容 ComfyUI 的两代表示：新版逐层 `.comfy_quant` uint8 JSON，
以及旧版 safetensors `_quantization_metadata.layers`。它只读取 header、有上限
的 `.comfy_quant` 和兼容 LightX2V 旧格式的 `convrot_groupsize` 整数标量，
不读取权重矩阵、不反量化、不初始化 Torch/GPU。完成后保留并回传：

```text
/disk/ssd1/lightx2v-runtime/h3-ref2va-int8-inventory/
├── h3_ref2va_weights.json   # 接入转换器所需的完整 tensor inventory
└── h3_ref2va_weights.md     # 人工快速检查摘要
```

成功标准：命令退出码为 0，JSON 顶层 `status` 为 `ok`，
`selected_assets` 仅包含 `int8_convrot`，且 `assets.int8_convrot` 中存在
完整的 `tensor_inventory`，`quantization_format_counts` 包含
`int8_tensorwise`，且 `convrot_groupsize_counts` 非空。

另外检查 `assets.int8_convrot.conversion_readiness`：它会逐层关联
`.comfy_quant`、`.weight` 和 `.weight_scale`，验证 I8/F32 dtype、逐输出通道
scale shape、groupsize 对输入维的整除关系、50 个 Core block、2 个 Refiner
block 和 H3 输入输出锚点。它还会核对当前 LightX2V 的 mixed-precision
边界：50 个 Core block 每块的 QKV、attention output、FC1、FC2、AdaLN 共
250 个源 projection 必须带 ConvRot metadata，而 Refiner 与 pre/post 不能被
INT8 化。collector 同时核对五类 Core projection 的标准 shape；源端 250 层
经 QKV 拆分后应得到 350 个 LightX2V INT8 linear。`status=ready` 才表示
header 具备进入转换器开发的条件；这仍不是 GPU 数值或画质验收。

## 采集后的接入边界

inventory 确认真实格式后，再完成以下改动：

1. 把 ComfyUI fused QKV/FC1 以及对应的 `weight_scale`、
   `convrot_groupsize` 转成 LightX2V 原生 key，并生成可缓存的分块 checkpoint；
2. runtime 配置启用 `dit_quantized=true`、`dit_quant_scheme=int8-convrot`，
   `dit_quantized_ckpt` 指向准备后的权重；
3. 转换缓存按量化 lazy loader 的约定生成 `non_block.safetensors` 和 50 个
   `block_N.safetensors`，并检查镜像中的 `comfy-kitchen.int8_linear`；
4. 继续动态加载现有 BF16 8-step LoRA，保持 `alpha=8`、`strength=1.0`、
   `training_euler` 和 9 个 sigma 网格点（8 次模型计算）；
5. 依次做 header/preflight、单卡 smoke、四卡 E2E，再做 2/4/8/16 卡 scale。

## 预计的 ComfyUI → LightX2V 转换

### 复用之前的 ComfyUI 兼容管线

现有 BF16 Ref2AV 已经不是从零实现：
`tools/convert/minimax_h3_comfy.py` 负责 Comfy 单文件的 key/layout 转换和
原子缓存，Run 已有 `H3_COMFY_DIR` 只读挂载、`h3_ref2av` 持久缓存、共享
Qwen/VAE 软链接，以及参考图 packing。LightX2V 还已有独立实现的
`comfy_res_multistep` 和 Comfy 离线轨迹回归。因此 INT8 路线建议复用两条
既有管线的交集：

```text
Comfy INT8 单文件
  → 扩展现有 minimax_h3_comfy.py 的量化转换分支
  → 复用 BF16 Ref2AV 的挂载/缓存/参考输入
  → 接入现有 Turbo 8-step 的 LoRA/scheduler/SP/VAE/collector
```

具体复用边界：

| 之前的能力 | 是否复用 | 本次处理 |
|---|---|---|
| Comfy 单文件挂载、源签名、原子转换缓存 | 是 | 扩展现有 converter，不另建一套缓存框架 |
| H3 QKV/FC1 key 映射 | 是 | 对 I8 weight 与 F32 scale 同步执行相同输出维变换 |
| Ref2AV 参考图 packing、Qwen、video/audio VAE | 是 | 继续使用 LightX2V 已验收实现和 `H3_MODEL_DIR` 共享组件 |
| Turbo 8-step LoRA、`training_euler`、SP2/4/8/16 | 是 | 沿用当前任务 12/13 合约，只替换 base checkpoint |
| LightX2V `comfy_res_multistep` sampler | 是，作为 base 基线 | 已有 Comfy `simple + res_multistep(eta=0)` 兼容实现；当前 8-step LoRA 首轮仍保持 `training_euler` |
| Comfy `UNETLoader`、`ModelPatcher`、运行时 offload | 否 | 会绕开 LightX2V 的 lazy block、Sage、SP、VAE parallel 和 collector |
| workflow 中 pruned base 与 4-step Comfy LoRA | 否 | 与当前未裁剪 base、8-step LightX2V LoRA 不是同一权重合约 |

### `comfy_res_multistep` 的精确复用合约

`workflow_templates/templates/video_minimax_h3_r2v.json` 当前实际连线是：

```text
ModelSamplingMiniMaxH3(video shift=12, audio shift=3)
  → BasicScheduler(scheduler=simple, steps=20, denoise=1)
  → KSamplerSelect(res_multistep)
  → SamplerCustomAdvanced
```

因此 INT8 base 基线必须固定为 `simple + res_multistep`，20 次模型计算、
`denoise=1`、`eta=0`、`cfg_pp=false`。模板说明中提到的 `beta`/`normal`
只是可选画质建议，不是该 workflow 当前执行路径，不能放进首轮基线；需要时另建
实验 profile。

LightX2V 已有独立的 `comfy_res_multistep` 实现和 ComfyUI 离线轨迹 fixture，
无需新增 sampler 或依赖 ComfyUI runtime。复用时必须同时保持以下 H3 双流语义，
不能只对齐 video sigma 表：

1. 按 Comfy `simple_scheduler` 从 1000 点 shifted video sigma 表取 20 个点，
   再追加终点 0；`infer_steps=20` 在此 profile 中表示 20 次模型计算；
2. audio sigma 从同一 base sigma 以 audio shift 3 闭式导出，不单独再抽一张
   离散表；sampler 内部用 `sigma_audio / sigma_video` carry audio state；
3. 第一次与终点使用 Euler，中间步分别为 video/audio 保留上一轮 x0，执行
   二阶 `res_multistep`；
4. 只更新 target video/audio rows，参考图、参考视频和参考音频 packing 出来的
   condition rows 在全部 20 步中保持不变；
5. 每次 `prepare()` 和 `clear()` 都清空 video/audio 历史与 audio carried state，
   避免常驻服务把上一请求的二阶状态带入下一请求。

这些语义已由
`lightx2v_platform/test/test_minimax_h3_comfy_sampling.py` 覆盖。INT8 接入只应
替换 base 权重准备和 Core linear kernel，不修改该 scheduler、参考输入 packing、
SP 或 VAE 路径。

建议最终保留两个用途明确的 profile：

| profile | LoRA | sampler/steps | 用途 |
|---|---|---|---|
| `ref2av_int8_convrot_base` | 无 | `simple + comfy_res_multistep`，20 次模型计算，`denoise=1`、`eta=0`、shift 12/3 | 精确对齐当前 Comfy workflow/BF16 base 管线，先验证 INT8 base 的画质与数值 |
| `ref2av_turbo_8step_int8_convrot` | 现有 BF16 8-step LoRA | `training_euler`，9 个网格点/8 次模型计算，shift 12/3 | 最终 Turbo 和 2/4/8/16 卡 scale |

不建议首轮直接把 8-step LoRA 与 `comfy_res_multistep` 混用：这会同时改变
base dtype/layout 和采样轨迹，出现漂移时无法区分是转换、量化还是 sampler
导致。等两个 profile 分别通过后，可以额外开放实验 profile 做同权重 A/B，
但不能替代发布合约。

建议把以下字段作为两个 profile 的 preflight 冻结合约，避免把“UI steps”和
“sigma 网格点”两套计数语义混用：

```json
{
  "ref2av_int8_convrot_base": {
    "h3_step_update": "comfy_res_multistep",
    "infer_steps": 20,
    "video_flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "enable_cfg": false,
    "lora_configs": []
  },
  "ref2av_turbo_8step_int8_convrot": {
    "h3_step_update": "training_euler",
    "infer_steps": 9,
    "video_flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "enable_cfg": false,
    "lora": "existing BF16 8-step, alpha=8, strength=1.0"
  }
}
```

这里 base 的 `infer_steps=20` 是 20 次计算；Turbo 的 `infer_steps=9` 是包含
终点 0 的 9 个网格点，对应 8 次计算。运行日志和 scale 报告应同时打印
`h3_step_update`、配置值与实际模型调用次数。

ComfyUI 仓库是 GPLv3，LightX2V 是 Apache-2.0；这里复用已公开的文件格式、
张量语义和运行结果作为兼容规范，不直接复制 ComfyUI 的 loader/patcher
实现。具体代码继续放在 LightX2V 已有的独立转换与调度框架中。

根据本机 `/home/yangecool/WorkSpace/Code/ComfyUI` 当前源码，`UNETLoader`
先从 `.comfy_quant` 恢复每层 mixed-precision 配置，再由 H3 模型直接使用
fused QKV 和 fused FC1。LightX2V 的 H3 模型 key 不同，但 ConvRot 的旋转发生
在输入维，以下输出维切分/重排不需要反量化：

| ComfyUI 源层 | LightX2V 目标层 | weight/scale 操作 |
|---|---|---|
| `blocks.i.attn.qkv_proj` | `transformer_blocks.i.attn.to_q/to_k/to_v` | weight 与逐通道 scale 都沿 dim0 等分三份 |
| `blocks.i.attn.out_proj` | `transformer_blocks.i.attn.to_out.0` | 直接重命名 |
| `blocks.i.mlp.fc1` | `transformer_blocks.i.ff.net.0.proj` | weight 与 scale 的 gate/value 两半同步交换 |
| `blocks.i.mlp.fc2` | `transformer_blocks.i.ff.net.2` | 直接重命名 |
| `blocks.i.adaln_proj.linear` | `transformer_blocks.i.adaln_proj.linear` | 直接重命名，bias 原样保留 |

每个源 `.comfy_quant` 必须是 `format=int8_tensorwise`、`convrot=true`。
转换后不保留该 JSON tensor，而是为每个目标 INT8 linear 生成 LightX2V 所需的
`.convrot_groupsize` 标量；`.weight` 保持 I8，`.weight_scale` 保持 F32。
Refiner、patch/text/time 投影和输出头继续走现有 BF16/FP32 映射。

预计的缓存布局：

```text
<cache>/<source-signature>/transformer_ref/
├── config.json
├── non_block.safetensors
├── block_0.safetensors
├── ...
└── block_49.safetensors
```

运行配置新增独立 profile（建议名
`ref2av_turbo_8step_int8_convrot`），设置：

```json
{
  "dit_quantized": true,
  "dit_quant_scheme": "int8-convrot",
  "dit_quantized_ckpt": "<prepared>/transformer_ref"
}
```

`H3_MODEL_DIR` 仍提供 Qwen、tokenizer、processor、video/audio VAE；
`H3_COMFY_DIR` 提供唯一的
`diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors`；
`H3_TURBO_DIR` 继续提供现有 8-step BF16 LoRA。LoRA 不合入 I8 权重，而是在
`MMWeightWint8ConvRot.apply()` 中以 BF16 动态分支加到 base 输出上，所以仍可
保持 `alpha=8`、`strength=1.0`、9 个 sigma 网格点和 8 次模型计算。

2/4/8 卡 Ulysses 与 16 卡 CP2xU8 只切序列、不切 DiT 权重，预计无需改变
并行拓扑；但必须重新验证每个 rank 都命中 `comfy-kitchen.int8_linear`、没有
BF16 fallback，并做输出画质/音视频时长对照。

本机 ComfyUI 源码也给出了格式依据：`comfy/model_detection.py` 从
`blocks.0.attn.qkv_proj.weight` 和 `blocks.0.mlp.fc1.weight` 检测 H3 架构；
`comfy/ops.py` 把 `int8_tensorwise` 的 `weight_scale`、`convrot` 和
`convrot_groupsize` 交给 INT8 kernel；`comfy/lora.py` 再把 MiniMax H3 LoRA
映射到 `diffusion_model.*.weight`。因此这些量化 sidecar 必须跟随 fused
projection 一起正确转换，不能只重命名 `.weight`。

`workflow_templates/templates/video_minimax_h3_r2v.json` 可证明 ComfyUI 的
INT8 ConvRot base 能与 BF16 LoRA 组合，但模板使用的是 pruned base、4-step
ComfyUI LoRA 和 `res_multistep/simple`。它不能替代本文件与现有 8-step LoRA
组合的实际 inventory 和 E2E 验收。
