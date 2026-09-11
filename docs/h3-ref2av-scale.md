# H3 Ref2AV：BF16 与 Turbo 8-step scale（2026-09-11）

用户反馈两张 32 GiB 卡初步成功。本轮复用该路径扩展 scale；4/8/16 卡 GPU 出片、性能与 CP2 Sage 数值仍待实测。BF16 权重保持完整原始 `minimax_h3_ref2va_bf16.safetensors`、无 LoRA，Turbo 链路使用固定 8-step LoRA 合约。两条 scale 默认保持 `rooftop_2ref_15s` 双参考图配置；通过 `--prompt-profile` 显式选择 `storeroom_courtyard_9ref_15s` 才使用九图。输出均为 768P / 24 FPS / 15 秒；BF16 默认 Comfy res_multistep 20 次计算，Turbo 为 8 次模型计算。

## 九参考图素材配置

九图配置为 `configs/minimax_h3/ref2av_storeroom_courtyard_9ref_15s.sh`，素材位于 `assets/minimax_h3/storeroom_courtyard_9ref_15s/`。配置按 `<Picture 1>` 到 `<Picture 9>` 的 prompt 顺序导出九条绝对路径。BF16 的 `run_11` 和 Turbo 的 `run_13` 都通过共享参数选择同一配置：

```bash
--prompt-profile storeroom_courtyard_9ref_15s
```

BF16 与 Turbo 九图 scale 示例：

```bash
H3_INFER_STEPS=3 bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_REBUILT_IMAGE \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --counts 2,4,8,16

bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --counts 2,4,8,16
```

不传 `--prompt-profile` 时加载 `configs/minimax_h3/ref2av_rooftop_2ref_15s.sh`，继续使用原双图屋顶素材。兼容别名 `--prompts-profile`。所选 profile 会写入 `run_manifest.json` 的 `prompt_profile`、`material_name`、`material_config`、`prompt_file` 和 `reference_images` 字段。

## 入口与采集

Run 新入口 `scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh` 复用 FL2AV 的 `run_timing_sweep.sh`，每个卡数调用现有 `run_10_minimax_h3_bf16_ref2v.sh`。宿主只调度，模型转换、preflight 和推理仍在镜像里运行。需要更新 LightX2V、Run 并重建镜像；旧 V3 不包含长视频 SP16 扩展。

先验完整流程（不使用缩小到 480×480 的 smoke）：

```bash
H3_INFER_STEPS=3 bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_REBUILT_IMAGE --counts 4,8,16
```

通过后去掉 `H3_INFER_STEPS=3` 跑默认 20 次；`--counts 2,4,8,16` 可加入两卡基线，`--gpu-count 16` 可单测。所有卡数使用相同素材、prompt、采样和默认 seed=42；362 个原生帧在封装前裁为 360 帧，声音精确 15 秒。不同并行拓扑下的量化数值不保证逐位一致。

卡组沿用 FL2AV 的轮换：2 卡 `1,2`；4 卡 `15,14,13,12`；8 卡 `0..7`；16 卡 `0..15`。只有 8 卡的机器先指定 `--counts 4,8` 并覆盖卡组：

```bash
TIMING_GPU_GROUPS='0,1;0,1,2,3;0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' \
H3_INFER_STEPS=3 bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_REBUILT_IMAGE --counts 4,8
```

`TIMING_RESULT_DIR` 指定空的结果目录；默认位于 `/disk/ssd1/lightx2v-runtime/outputs/`。每次 sweep 保留：

- 根目录：`run_manifest.json`、各卡数 `4gpu.log` 等、`timings.csv/md`、内存监控 JSONL、`sage_shapes.json`。
- `4gpu/`、`8gpu/`、`16gpu/`：各自的视频、`runtime_config.json`、`preflight.json`、`ffprobe.json`、所有 rank 的 `h3_runcollector.rank*.json`，不会相互覆盖。
- 默认启用现有 collector 和内存监控。`LIGHTX2V_H3_RUN_COLLECTOR=0` 可做较少测量干扰的性能复跑；比较时所有卡数必须使用同样的采集设置。
- 失败停止 sweep 并保留已完成结果。形状汇总失败会明确警告，原视频和日志保留。

## 算子与切片审查

| 卡数 | Core 拓扑 | 每个 attention 核的头数 | 每片 28 个 VAE tile 的分配 |
|---|---|---|---|
| 4 | Ulysses4 | 14 | 每卡 7 |
| 8 | Ulysses8 | 7 | 4 卡各 4、4 卡各 3 |
| 16 | CP2 × Ulysses8 | 7 | 12 卡各 2、4 卡各 1 |

56 个头不能直接均分到 Ulysses16，所以继续复用现有 CP2×U8。16 卡 rank 0..7、8..15 分别形成 U8 组，对应 rank 组成 CP2 列；单机运行时应尽量把组内卡放在通信较快的连接域，不能只按卡数预期线性加速。

- 源码、配置构建器、容器入口、preflight 统一接受 `17*n+5`、124..362 的原生帧数及既有 768P/480P 横屏尺寸。FL2AV 保持固定 124 帧；保留形状、头数、无 LoRA、无量化通信与算子正确性约束。
- 4/8/16 卡对长视频使用已有 minimal auxiliary prefix。总序列为 N 时，SP4/8 只重复 N%P 个条件行；SP16 至少保留 2 行，余数 0/1 时分别保留 16/17 行。其余条件 token 与目标 token 一起切分，RoPE/timestep/AdaLN 索引同步切分，顺序不变。两卡的既有前缀策略不变。
- DiT 最终通过 `all_gather_into_tensor` 直接写入最终 hidden 缓冲区，去掉 list gather 再 cat 产生的第二份完整 hidden；每卡少一个约 `N×5376×2` 字节副本。仍需汇总后运行既有输出头，未引入新的模型分片算法。
- 主体继续显式选择 aiter native gfx1201 SageAttention2，D128，4/8 卡分别 H14/H7；16 卡方形分块 H7 并返回 LSE。aiter 的 B1/D128 内部分派已覆盖任意正序列长度：整齐边界与尾部不同调度，不需要把旧 FL2AV 的具体 S 值硬套到 Ref2V。
- CP2 继续使用已有的组合路径：大块方形 attention 用 native Sage，小量辅助前缀的矩形 attention 用 aiter Triton Sage，再做 LSE 合并。这是现有明确分工，不新增失败后切换 SDPA 的路径。GPU 上须检验量化与合并后的误差；CPU dense 对照不能替代这项验证。
- Q/K RMS、strict R96 RoPE、BF16 GEMM、block offload 不变。Qwen rank0 编码、参考准备、最终输出头和音视频写出仍存在串行或重复部分，整体耗时不会按卡数简单等比缩短。lazy loader 只保留固定张量以及每 rank 两个 CPU、两个 GPU Core block 缓冲，不把整套 33B 权重复制到每 rank 的 pinned RAM；多 rank 同时读权重和从 host 搬运仍可能受内存/PCIe 带宽限制。
- VAE 保持 21 个时间片、每片 28 个 tile、H4 S1797/D64 FP16 MHA 及现有 GPU 拼接和时间融合，完成帧及时转 CPU。单片 28 个任务在 SP8/SP16 不整除是正常现象，不通过改变 tile 尺寸冒险换算子形状。

## 验证与 tune 交接

已通过真实 CPU Gloo 4/8/16 进程 attention+A2A+CP2 LSE+最终 all-gather 对照，覆盖空、奇数及 16/17 行辅助前缀；VAE 通过 2/4/8/16 进程、28 tile、不均匀分配和空闲 rank 的解码顺序/融合测试。长视频参考索引及 Run 全量测试通过。未进行本机 GPU 推理或调优。

修复了旧形状采集器的两项假设：CP2 辅助前缀并非总是偶数，N%16=0/1 也不能保留零/一行。现在按 context rank 记录真实矩形 Q 长度，并利用运行时总序列元数据支持单独 SP16 采集。

第一次实测后提供整个 sweep 结果目录。优先根据 `sage_shapes.json` 与 rank collector 中的实际 M/N/K、attention 形状及阶段耗时确定 tune 对象：H14/H7 长序列 native Sage、CP2 两个方形 LSE kernel，以及本地行数缩短后的 BF16 GEMM。先区分算子耗时、A2A/Ring 等待、权重 offload 和 VAE rank0 汇总，不预设更换算子一定更快。
