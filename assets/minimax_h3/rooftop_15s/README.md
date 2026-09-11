# Default MiniMax-H3 Ref2AV Material

These assets came from the user-supplied `promptsandpics.zip`. The PNG files
and original prompt are preserved byte for byte.

| Input order | File | Prompt reference |
|---|---|---|
| 1 | Image Gen-1.png | `<Picture 1>` / `<Subject 1>` |
| 2 | RYAN-2.png | `<Picture 2>` / `<Subject 2>` |

`prompt.original.txt` is the source. `prompt.txt` is the executable version: it
removes external `<Audio 1>/<Audio 2>` references and inapplicable keyframe
markers, preserves the six-shot timeline, fight sounds, breathing, and music,
and fixes formatting plus the `s trike` typo. There is no audio input file.

The default entry is `scripts/run_10_minimax_h3_bf16_ref2v.sh`: full BF16, no
LoRA, 1344x768, 24 FPS, 15 seconds, and GPUs 0/1. Both images are character
references; neither is a locked first or last frame.
