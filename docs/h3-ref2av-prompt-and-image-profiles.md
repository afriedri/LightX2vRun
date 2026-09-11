# MiniMax-H3 Ref2AV Prompt and Reference-Image Profiles

This document describes only prompt and reference-image storage and routing in
`LightX2VRun`. Model weights, sampling, operators, and scale acceptance remain
documented in `h3-ref2av-bf16.md`, `h3-ref2av-8step.md`, and
`h3-ref2av-scale.md`.

## Current profiles

| Prompt profile | References | Asset directory | Scale behavior |
| --- | ---: | --- | --- |
| `rooftop_2ref_15s` | 2 | `assets/minimax_h3/rooftop_15s/` | Default when `--prompt-profile` is omitted |
| `storeroom_courtyard_9ref_15s` | 9 | `assets/minimax_h3/storeroom_courtyard_9ref_15s/` | Selected explicitly with `--prompt-profile storeroom_courtyard_9ref_15s` |

The prompt profile is independent of the other two profile-like inputs:

| Input | Meaning |
| --- | --- |
| `--image IMAGE` | Docker image containing the LightX2V runtime and operator stack |
| `H3_PROFILE` | Model execution contract: `ref2av_bf16_base` or `ref2av_turbo_8step` |
| `--prompt-profile PROFILE` | Prompt file plus ordered reference-image set used by a scale entry |

`--image` never selects a reference picture. Both BF16 and Turbo scales accept
the same prompt profiles.

## On-disk layout

The original two-reference profile is stored as:

```text
assets/minimax_h3/rooftop_15s/
├── Image Gen-1.png          # <Picture 1>
├── RYAN-2.png               # <Picture 2>
├── prompt.original.txt      # source prompt
├── prompt.txt               # runtime prompt
└── README.md                # source and ordering notes

configs/minimax_h3/
└── ref2av_rooftop_2ref_15s.sh
```

The nine-reference profile uses English filenames throughout:

```text
assets/minimax_h3/storeroom_courtyard_9ref_15s/
├── wang_hao_front.png       # <Picture 1>
├── wang_hao_full.png        # <Picture 2>
├── chen_fan_front.png       # <Picture 3>
├── chen_fan_full.png        # <Picture 4>
├── discipline_disciple.png  # <Picture 5>
├── su_xue_front.png         # <Picture 6>
├── su_xue_full.png          # <Picture 7>
├── character_layout.png     # <Picture 8>, blocking/layout reference
├── zhao_xiaoman.png         # <Picture 9>
├── prompt.original.txt      # byte-preserved source prompt
├── prompt.txt               # runtime prompt
└── README.md                # source and ordering notes

configs/minimax_h3/
└── ref2av_storeroom_courtyard_9ref_15s.sh
```

PNG payloads are not resized or re-encoded in the repository. A profile's
shell config derives absolute paths from its own repository location, exports
`H3_PROMPT_FILE`, and exports the ordered paths as newline-separated
`H3_REFERENCE_IMAGES`. File order in the config, rather than alphabetical
filename order, defines `<Picture 1>` through `<Picture N>`.

`prompt.original.txt` preserves the received text for provenance.
`prompt.txt` is the executable version. For the nine-reference profile it maps
the original image mentions to `<Picture 1>` through `<Picture 9>`, repairs
broken wrapping and shot labels, and removes the unavailable external audio
reference. Scene, dialogue, sound-effect, no-music, no-subtitle, and visual
quality requirements are retained.

## Scale-entry reference chain

The two numbered scale entries are:

- BF16 base: `scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh`
- Turbo 8-step: `scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh`

They share the following routing path:

```text
run_11 or run_13
  -> scripts/lib/run_timing_sweep.sh
  -> parse --prompt-profile (default: rooftop_2ref_15s)
  -> source configs/minimax_h3/ref2av_<profile>.sh
  -> export H3_PROMPT_FILE + ordered H3_REFERENCE_IMAGES
  -> run_10 (BF16) or run_12 (Turbo) for every selected GPU count
  -> scripts/run_minimax_h3_ref2av_bf16.sh
  -> lightx2vModelrun
  -> read-only Docker mounts /inputs/h3_reference_0 ... h3_reference_N
  -> IMAGE_PATH=/inputs/h3_reference_0,...,/inputs/h3_reference_N
```

`run_timing_sweep.sh` selects `run_10` for `H3_PROFILE=ref2av_bf16_base` and
`run_12` for `H3_PROFILE=ref2av_turbo_8step`. Those wrappers convert the
newline-separated image list into repeated `--reference-image` arguments. The
shared Ref2AV launcher validates that there are 1-9 references, rejects commas
or newlines inside an individual path, resolves host paths to absolute paths,
and reads `H3_PROMPT_FILE` into `PROMPT`.

`lightx2vModelrun` preserves list order. Reference number 1 is mounted
read-only at `/inputs/h3_reference_0`, reference number 2 at
`/inputs/h3_reference_1`, and so on. It sends the comma-separated container
paths through `IMAGE_PATH`; the in-container Ref2AV runner splits that value
back into the same ordered reference list before image preparation and
packing. Reference images are conditioning inputs, not first/last-frame locks.

The root `run_manifest.json` records the resolved selection as
`prompt_profile`, `material_name`, `material_config`, `prompt_file`, and
`reference_images`. Every GPU-count arm in one sweep receives the same values.

## Commands

BF16 scale with the default two-reference profile:

```bash
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE \
  --gpu-count 2
```

BF16 scale with nine references:

```bash
H3_INFER_STEPS=3 \
bash scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh \
  --image YOUR_RELEASE_IMAGE \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --gpu-count 2
```

Turbo scale with the default two-reference profile:

```bash
bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh \
  --gpu-count 2
```

Turbo scale with nine references:

```bash
bash scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh \
  --prompt-profile storeroom_courtyard_9ref_15s \
  --gpu-count 2
```

The compatibility spelling `--prompts-profile` is accepted, but new commands
should use the canonical singular spelling `--prompt-profile`.

The `--prompt-profile` option belongs to the scale layer. For a direct
`run_10` or `run_12` invocation, pass the prompt and all images explicitly:

```bash
bash scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh \
  --prompt-file /absolute/path/to/prompt.txt \
  --reference-image /absolute/path/to/picture-1.png \
  --reference-image /absolute/path/to/picture-2.png
```

Repeat `--reference-image` in prompt order, up to nine times. Explicit
`--reference-image` arguments replace the wrapper's complete default list.

## Adding another prompt profile

Keep each workload self-contained:

1. Create `assets/minimax_h3/<english_profile_name>/`.
2. Store original images without re-encoding and use English filenames.
3. Add `prompt.original.txt`, executable `prompt.txt`, and an ordering
   `README.md`.
4. Add `configs/minimax_h3/ref2av_<english_profile_name>.sh` that exports one
   prompt path and 1-9 ordered reference paths.
5. Add the profile name to the `--prompt-profile` allow-list in
   `scripts/lib/run_timing_sweep.sh`.
6. Extend `tests/test_h3_ref2av_scale.py` to verify the exact filename order,
   prompt markers, BF16/Turbo propagation, and manifest fields.

Do not overwrite an existing material directory to publish a new workload.
Changing image order without changing the prompt's `<Picture N>` mapping is a
different input contract and must be reviewed as such.
