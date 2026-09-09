# LightX2V / Run 上游接入 review（2026-09-09）

本轮以 HVAT 与 Run 的模型、权重、采样和运行契约为准，复用上游能力及基础设施。没有把五个已适配模型统一替换成上游默认模型。CPU/配置回归通过；GPU 性能与视频质量尚未重新验收。

## main 与分支

- 指定上游：`git@gitcode.com:GitHub_Trending/li/LightX2V.git`。
- fork `main` 已精确对齐 `7425d1ec631528d1390d07d727ff7181465d5019`，提交与树相同。
- LightX2V 开发分支：`hvat-gfx1201-R9600`；保留 `main` 用于纯上游同步。仅清理 GitCode 主远端旧功能分支，GitHub 镜像与上游不修改。
- 重放前的完整历史归档：`/home/yangecool/WorkSpace/Code/99-archive/lightx2v-20260909/LightX2V-before-upstream-rebase-20260909.bundle`，已通过 `git bundle verify`。

## 上游主要更新

比较旧 fork main `5c18e461` 与新 main `7425d1ec`：1,068 个文件变化。新旧历史存在改写，按 patch 等价排除重复提交后，差异为上游侧 83 个、旧侧 2 个；不能把原始提交差集全部视为新增功能。

| 范围 | 上游更新 | 本轮接入方式 |
|---|---|---|
| Wan | 蒸馏 runner 合并、DMD2/MeanFlow 调度工厂、CFG 预设修复 | Run 使用新名称并显式指定 DMD2，保留 HVAT FP8/LoRA、双模型与 B0/B1 |
| LTX | 原生 LTX2.5 模型/DurationHead/DiffVAE/ancestral sampler、组件工厂 | 新入口按 HVAT profile 分派到已有共用 runner；不替换 HVAT 权重、采样器和 Conv VAE |
| H3 | ConvRot INT8、Qwen host pinning、TP bias 修复、AdaLN 缓存、pre/post 驻留 | 合入实现，保持现有 FL2V 参数和已选算子；新选项不默认启用 |
| H3 VAE | 时空 tile 并行、负载均衡、encode 优化 | 保留上游结构，补闲置 rank、输出顺序和 collector 适配 |
| 通用 | 请求状态/RNG 重置、warmup 配置移入 JSON、GQA SDPA、量化取整、FP8 通信优化 | 复用基础实现，Run 同步配置接口 |
| 其他 | ROCm gfx1100/gfx1201 Aiter 修复、XPU/Ascend/MLU 优化、训练/部署/SwiftVR 更新 | 合入上游，不自动切换 HVAT 的已选后端 |

## 五个模型逐项结论

| 模型 | Run 新入口 | 保留的模型与特色能力 | review 重点 |
|---|---|---|---|
| LTX2.3 | `ltx2` | 原有 FP8/BF16 混合权重、prompt cache、两阶段卸载、采样参数 | 共用预处理保持 HVAT Prompt AdaLN 精度及位置网格规则 |
| LTX2.5 | `ltx2_5` + HVAT profile | Run 原有 split assets、Gemma4、Conv VAE、既有采样器、两槽加载、TunableOp、Sage、分布式 VAE | profile 仅做入口分派；不引入原生 DurationHead/DiffVAE 或 ancestral sampler |
| Wan2.1 | `wan2.1` + `distill_method=dmd2` | 原 FP8 蒸馏权重、T5/CLIP、LoRA、Aiter GEMM/Attention、Ring/SP | 校验新工厂仍选 DMD2；warmup 从原环境变量写入 JSON |
| Wan2.2 | `wan2.2_moe` + `distill_method=dmd2` | 原 high/low FP8 权重、LoRA、切换边界、lazy/B0/B1、CK Conv3D | 保留统一 runner 内的构造/内存采集；B0/B1 实现与重放前相同 |
| H3 | `minimax_h3` | FL2V 首尾帧/prompt、原 Turbo LoRA、124f/24fps、50-block lazy、Aiter 路由、SP16 CP2×U8 | 修复 block index、量化 Qwen pageable 加载和 VAE gather；Ref2VA 先采实际权重结构 |

LTX2.5 的完整 HVAT 运行配置与重放前逐项比较，除 `model_cls` 外一致。H3 的最终配置在 SP1/2/4/8/16 均与重放前一致，包括 Run 归一化后的 LoRA 路径。模型文件没有重命名或移动。

## 性能审查

- 已排除的明显退化：误用原生 LTX2.5 模型/采样器、丢失 prompt cache、丢失两槽 offload、Wan 新名称却未启用 DMD2、collector 拉已删除分支、使用旧 Hot28 默认镜像。
- 静态确认：Wan block offload 实现、LTX TunableOp 缓存未变；H3 lazy B1 轮转、FP8 quant reuse、Aiter 路由回归通过。SGL FP8 新包装仍调用同一后端，不新增张量拷贝。
- 待实测：H3 新时空 VAE 的 tile 数、rank0 gather 显存和耗时；少 tile 场景新增一次 collective 初始化；新请求生命周期对峰值内存和耗时的影响。不能据 CPU 测试宣称性能零损失。
- 比较时固定同一权重、素材、prompt、seed、帧数、GPU 拓扑及 collector 开关，观察推理阶段耗时、H2D、swap wait、显存峰值和实际算子路由。历史 LTX2.5 的上游原生模型成绩不能作为 HVAT 模型基线。

## 验证与镜像

- Run：207 项通过，其中 10 项跨仓入口/配置验证；49 个 shell 脚本语法通过。
- LightX2V：140 项定向回归通过，5 项 GPU 用例因无 GPU 跳过；变更文件静态检查和 `git diff --check` 通过。
- H3 VAE：真实 4 进程 CPU Gloo 验证 1/2/7/8 tiles 的 gather 顺序和闲置 rank；不是 RCCL/GPU 验收。
- 本机无法执行 Docker/GPU E2E，本轮未构建或发布镜像。
- 两仓统一默认镜像名：`lightx2v-rocm:hvat-gfx1201-R9600`。构建脚本 `dockerfiles/platforms/build_gfx1201.sh` 默认产出此 tag；Run 支持显式 `--image` 覆盖。
- gfx1201 运行依赖补入 `comfy-kitchen==0.2.15`，用于上游 ConvRot 实现；不替换 ROCm Torch。INT8 ConvRot 的真实 gfx1201 性能仍待测。

## H3 Ref2VA 权重采集

复用 Run 现有 safetensors 解析/边界校验，采集指定的未裁剪 BF16 与 INT8 ConvRot 两个文件。只读取 header 和最多 64 KiB 的 ConvRot 整数标量元数据；不读取矩阵 payload，不加载 Torch，不分配 GPU，也不做强制 SHA gate。

在权重机器的 Run 仓库执行：

```bash
python3 scripts/collector/collect_minimax_h3_ref2v_weights.py \
  --model-root /disk/ssd1/difumodels/MiniMax-H3-ComfyUI \
  --output-dir /disk/ssd1/lightx2v-runtime/h3-ref2va-weight-inventory
```

输出 `h3_ref2va_weights.json` 和 `.md`，包含两个文件的 metadata、keys、shape、dtype、大小、ConvRot groupsize 及差异。该机器当前不可达，真实采集和完整 Ref2VA 推理尚未执行，不能把小样本测试当作真实权重验收。
