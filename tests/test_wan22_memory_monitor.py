from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = REPO_ROOT / "scripts/monitor_wan22_memory.py"
SPEC = importlib.util.spec_from_file_location("monitor_wan22_memory", MONITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def _sample(timestamp: str, monotonic: float, ranks: list[int]):
    return {
        "timestamp_utc": timestamp,
        "monotonic_seconds": monotonic,
        "phase": "dit",
        "container": {
            "ranks": [
                {
                    "rank": rank,
                    "local_rank": rank,
                    "container_pid": 100 + rank,
                    "host_pid": 1000 + rank,
                    "Rss_bytes": 1000,
                    "cpus_allowed_list": f"{rank * 2}-{rank * 2 + 1}",
                    "mems_allowed_list": str(rank % 2),
                }
                for rank in ranks
            ],
            "cgroup": {"current_bytes": 1234, "peak_bytes": 0},
        },
        "rank_sums": {"Rss_bytes": 1000 * len(ranks)},
        "host": {"used_excluding_available_bytes": 2000},
        "gpus": [],
    }


def test_summary_records_rank_exit_observation_window():
    summary = MONITOR.Summary()
    summary.add(_sample("2026-08-13T00:00:00.000+00:00", 10.0, [0, 1]))
    summary.add(_sample("2026-08-13T00:00:01.000+00:00", 11.0, [0]))
    summary.close_active_ranks(
        "2026-08-13T00:00:02.000+00:00",
        12.0,
        "container_stopped",
    )

    lifecycle = summary.as_dict()["rank_lifecycle"]
    assert lifecycle["1"]["last_seen_monotonic_seconds"] == 10.0
    assert lifecycle["1"]["exit_observed_monotonic_seconds"] == 11.0
    assert lifecycle["1"]["exit_observation"] == "rank_missing_from_probe"
    assert lifecycle["0"]["exit_observation"] == "container_stopped"
    assert lifecycle["0"]["cpus_allowed_list"] == "0-1"
    assert lifecycle["0"]["mems_allowed_list"] == "0"


def test_zero_kernel_peak_keeps_sampled_current_separate():
    summary = MONITOR.Summary()
    summary.add(_sample("2026-08-13T00:00:00.000+00:00", 10.0, [0]))
    payload = summary.as_dict()

    assert payload["kernel_reported_cgroup_peak_bytes"] == 0
    assert payload["max_observed_cgroup_current_bytes"] == 1234
    assert payload["physical_gpu_total_bytes"] == {}


def test_gpu_index_parser_rejects_duplicates_and_preserves_order():
    assert MONITOR.parse_gpu_indices("15,14,13,12") == (15, 14, 13, 12)
    try:
        MONITOR.parse_gpu_indices("0,0")
    except Exception as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate GPU indices were accepted")


def test_phase_tracker_uses_all_rank_memory_boundaries(tmp_path):
    log = tmp_path / "run.log"
    tracker = MONITOR.PhaseTracker(log, expected_ranks=2)
    log.write_text(
        "[WanRunTune] device_memory label=transformer_shell_ready rank=0 allocated=1 reserved=2 free=3\n"
        "[WanRunTune] device_memory label=transformer_shell_ready rank=1 allocated=1 reserved=2 free=3\n"
        "[WanRunTune] device_memory label=transformer_high_ready rank=0 allocated=1 reserved=2 free=3\n"
        "[WanRunTune] device_memory label=transformer_high_ready rank=1 allocated=1 reserved=2 free=3\n",
        encoding="utf-8",
    )
    assert tracker.update() == "dit_high"

    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            "[WanRunTune] device_memory label=transformer_low_ready rank=0 allocated=1 reserved=2 free=3\n"
            "[WanRunTune] device_memory label=transformer_low_ready rank=1 allocated=1 reserved=2 free=3\n"
        )
    assert tracker.update() == "dit_low"

    with log.open("a", encoding="utf-8") as stream:
        stream.write("[Profile] Rank 0 - Level1_Log Run VAE Decoder cost 1.0 seconds\n")
    assert tracker.update() == "vae_decode"
