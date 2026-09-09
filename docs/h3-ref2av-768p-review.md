# H3 BF16 Ref2V：768P/15 秒审查报告（2026-09-09）

目标保持原始 `minimax_h3_ref2va_bf16.safetensors`、无 LoRA、双参考图、1344×768 / 24 FPS / 15 秒。GPU 端已先后暴露输出头与 VAE 汇总 OOM；当前两处已修复，**尚未完成修复后的真实 GPU 出片验收**。不以两卡跑通为硬目标，本轮先四卡验收。

## 本次两处 OOM

- 第一处：DiT 输出头调制申请 1.40 GiB、整段 FP32 转换申请 2.81 GiB 失败。输出头已经按模态和每 4096 行分块，保留 RMS、调制次序和 FP32 投影。权重 inventory 中隐藏宽度为 **5376**，单块 FP32 输入 **84 MiB**；此前写成 7168 / 112 MiB，是把 attention 内部宽度当成输出头隐藏宽度，现已更正。分块代码根据实际张量工作，不依赖这个注释数字。
- 第二处：新日志到达视频 VAE，rank1 `torch.stack(local_tiles)`、rank0 `receive_buffer` 各申请 3.02 GiB 失败。768P / 362 原生帧共有 21 个时间片，每片 28 个 tile，共 588 个。旧 SP2 每卡积累 294 个解码 tile 再发送；SP4 虽然各 147 个，rank0 仍汇总全部结果，加卡不能消除全片积累。
- 修复：逐时间片分配、解码和汇总，SP2 每轮每卡 14 个，SP4 每轮每卡 7 个。复用现有 P2P 汇总、GPU 空间拼接、时间重叠融合及尾帧裁切；所有 rank 参与每个时间片，空闲 rank 按既有 barrier 规则处理。已完成的帧在 `return_cpu=True` 时立即移到 CPU；最终 RGB 转换在 CPU 分块执行，不在 GPU 拼出整片再转 FP32。
- H4 的 FP16/D64 `aiter_triton_mha`、S1797 tile、decoder 和投影算子不变。汇总轮数增多及分片 D2H 可能影响时间，未宣称零开销或更快；目的是限制峰值。

## 采样借鉴与 HVAT 定制边界

参考本机 `workflow_templates@4c4b8cbc` 的 `templates/video_minimax_h3_r2v.json` 以及 ComfyUI `model_sampling.py`、`samplers.py`、`k_diffusion/sampling.py`、`ldm/minimax/model.py`。

- 新默认：`comfy_res_multistep`，`simple` 的 1000 点基础表抽取 20 个 sigma 并追加终点，**20 次计算**；video/audio shift=12/3。包含音频 carried latent、速度转换和最终还原。实现限定在 H3 scheduler，不引入 ComfyUI 运行依赖。
- 可选旧路径：`reference_blend` / `training_euler` 及原网格点计数语义保持，Run 不把新采样设为唯一允许配置。使用 `H3_STEP_UPDATE=reference_blend` 可恢复；专用脚本相应默认 30 个网格点。
- 参考图默认复用现有 `match`，不放大输入；两张 2752×1536 → 1344×768，每张 1008 个参考 latent 行，旧 diffusers 每张 7360 行。目标视频行仍为 107856，音频行仍为 1206。`H3_REFERENCE_IMAGE_RESIZE_MODE=diffusers` 或 `max` 可选旧策略；采样方式和参考图策略独立。
- 保留原始 BF16 权重转换缓存、native Qwen/VAEs、既有 reference packing、CFG 关闭、lazy block offload、Qwen rank0 编码/卸载、SP 与 VAE tile parallel。不会套用模板的 pruned INT8、NVFP4、480P/5 秒规格，也不加载 Turbo LoRA。
- Core 继续 `aiter_fav3_sage_bf16_attn`；Q/K RMS 继续 `minimax_h3_aiter_triton_qk`；RoPE 继续 strict/no-FMA R96；GEMM 继续原 BF16 Torch/ROCm BLAS，TunableOp 关闭。Qwen/refiner/H4 继续 `aiter_triton_mha`。H3 禁止算子回退的运行策略保持。
- 362 原生帧在既有封装前裁为 360 帧和每声道 480000 个 32 kHz 采样，视频输出仍是精确 15 秒；输入素材、prompt 及直接生成声音保持。

## 验证与限制

- 从 ComfyUI 实际源码执行生成 6 组离线轨迹，覆盖 1/2/3/7/20 次计算、12/3 与其他 shift。保存源文件 SHA256；H3 scheduler 的逐步视频/音频、终点还原和不更新参考条件行通过容差对照。请求间多步历史清理与旧采样计数通过。
- LightX2V 定向 **104 项**通过，覆盖新采样、旧路径、输出头、SP、offload、参考索引、算子路由、collector 和输出时长。
- 另有真实 CPU Gloo 两/四进程解码测试 **5 项**通过：并行结果与原串行拼接零误差、闲置 rank 不死锁、362 帧及 padding 裁切正确、每次只汇总一个时间片。使用小型确定性 decoder；不是 GPU 神经网络推理或 NCCL 性能验收。
- Run 全量 **248 项**通过；随后增加旧采样可选与 env 转发覆盖，相关定向 **105 项**通过。
- 用户确认无其他任务，但日志中的设备占用与 PyTorch reserved 差额仍未归因。collector 已记录阶段起止设备 free/total、PyTorch allocated/reserved 和累计峰值；没有日志依据把差额归咎于碎片、某个运行库或驱动。
- 两卡/四卡真实峰值、全片质量及 20 次采样的 GPU 耗时均待实测。CPU 轨迹一致不代表整条 ComfyUI 与 HVAT 视频逐位相同。

## 新镜像复测

LightX2V 修改在镜像内，必须重建；宿主 Run 更新会只读挂载进容器。先四卡，保留实际 768P/15 秒/双图，运行三次计算及完整解码：

```bash
bash scripts/run_10_minimax_h3_bf16_ref2v.sh \
  --image YOUR_REBUILT_IMAGE --gpus 0,1,2,3 --infer-steps 3 --collect
```

通过后去掉 `--infer-steps 3`，默认执行 20 次。三次运行只验流程，不验出片质量。保存 `run.log`、`runtime_config.json` 和所有 rank 的 `h3_runcollector.rank*.json`，据实测决定是否再试两卡，不强求。
