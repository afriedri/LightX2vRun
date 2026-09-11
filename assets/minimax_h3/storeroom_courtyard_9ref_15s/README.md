# H3 Ref2AV nine-reference storeroom courtyard materials

Source: user-provided `9.zip`. The PNG payloads and original prompt bytes are preserved; only the filenames were changed to English.

| Input order | Runtime filename | Original archive name | Prompt role |
|---|---|---|---|
| 1 | `wang_hao_front.png` | image 1 | `<Picture 1>` / Wang Hao |
| 2 | `wang_hao_full.png` | image 2 | `<Picture 2>` / Wang Hao |
| 3 | `chen_fan_front.png` | image 3 | `<Picture 3>` / Chen Fan |
| 4 | `chen_fan_full.png` | image 4 | `<Picture 4>` / Chen Fan |
| 5 | `discipline_disciple.png` | image 5 | `<Picture 5>` / discipline disciple |
| 6 | `su_xue_front.png` | image 6 | `<Picture 6>` / Su Xue |
| 7 | `su_xue_full.png` | image 7 | `<Picture 7>` / Su Xue |
| 8 | `character_layout.png` | image 8 | `<Picture 8>` / character blocking |
| 9 | `zhao_xiaoman.png` | image 9 | `<Picture 9>` / Zhao Xiaoman |

`prompt.original.txt` is the byte-preserved source prompt. `prompt.txt` is the runtime version: image mentions are mapped to ordered `<Picture N>` references, the unavailable external audio reference is removed, broken line wraps and the shot-4 label are repaired, and the source scene/dialogue/quality constraints are retained.

The material configuration is `configs/minimax_h3/ref2av_storeroom_courtyard_9ref_15s.sh`. Both numbered Ref2AV scale entries select it with `--prompt-profile storeroom_courtyard_9ref_15s`; without that option they retain the original two-reference rooftop profile.
