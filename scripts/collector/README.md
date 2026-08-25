# MiniMax-H3 Collectors

This directory contains diagnostic launchers and their analysis helpers. They
observe the frozen production workload and must not silently change model
precision, offload policy, buffer counts, tuned kernels, or parallel topology.

- `run_minimax_h3_sp8_supercollect.sh`: SP8/Ulysses full-run evidence.
- `run_minimax_h3_sage_shape_sweep.sh`: SP4/SP8/Ring16 Sage contracts.
- `run_minimax_h3_megadiscovery.sh`: SP8 or Ring16 full-stack discovery.
- `collect_minimax_h3_host_evidence.py`: host, PCIe, KFD, NUMA, and storage snapshots.
- `benchmark_minimax_h3_collectives.py`: isolated production-shape communication probes.
- `summarize_minimax_h3_t1.py`: cross-rank H3 critical-path summary.
- `analyze_minimax_h3_megadiscovery.py`: unified JSON and Markdown diagnosis.

Run Ring16 MegaDiscovery with:

```bash
bash scripts/collector/run_minimax_h3_megadiscovery.sh \
  --image IMAGE \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The isolated probes run before inference in a separate process. Their synthetic
fusion variants and timing overhead never alter the production model process.
The historical root-level SuperCollect and Sage sweep shell entry points remain
compatibility wrappers.
