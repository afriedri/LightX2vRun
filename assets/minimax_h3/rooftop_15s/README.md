# H3 Ref2V 默认素材

来自用户提供的 `promptsandpics.zip`。PNG 与原始 prompt 按原始字节保存。

| 输入顺序 | 文件 | prompt 对应 |
|---|---|---|
| 1 | Image Gen-1.png | `<Picture 1>` / `<Subject 1>` |
| 2 | RYAN-2.png | `<Picture 2>` / `<Subject 2>` |

`prompt.original.txt` 为原文；`prompt.txt` 为执行版本：去掉外部 `<Audio 1>/<Audio 2>` 引用及不适用的 keyframe 标记，保留六段镜头时序、打斗声、喘息和配乐，修正排版及 `s trike` 拼写。没有音频输入文件。

默认入口：`scripts/run_10_minimax_h3_bf16_ref2v.sh`，原始 BF16、无 LoRA、1344×768、24 FPS、15 秒、GPU 0/1。两张图是角色参考，均不作为首尾帧锁定。
