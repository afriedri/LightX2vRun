# Deprecated scripts

Nothing under this directory is called by the six numbered timing entries.

- `completed_wan21/` and `completed_wan22/` preserve completed profiling,
  runtune, and Experiment-A/B evidence. They are reference-only and are not
  maintained as executable workflows.
- `wan22_legacy/` preserves the former general FP8, Sage, and LoRA launchers.
  They may still be useful for code archaeology, but do not implement the
  current lazy/B1 production policy.
- `diagnostics/` contains the still-useful read-only memory monitor and bundle
  packaging/uploader. These tools remain covered by focused tests but are not
  part of ordinary timing runs.

New work must use `scripts/run_01_*.sh`, `run_02_*.sh`, or
`run_04_*.sh` through `run_07_*.sh` and their shared `scripts/lib/` code.
