from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/lib/mock_ltx23_b0_memory.py"
SPEC = importlib.util.spec_from_file_location("mock_ltx23_b0_memory", SCRIPT)
MOCK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOCK)


def write_header_only_checkpoint(path: Path) -> None:
    header = {
        "__metadata__": {
            "model_version": "2.3.0",
            "config": json.dumps(
                {
                    "transformer": {
                        "num_layers": 48,
                        "num_attention_heads": 32,
                        "attention_head_dim": 128,
                    }
                }
            ),
        }
    }
    offset = 0

    def add(name: str, dtype: str, shape: list[int], item_size: int) -> None:
        nonlocal offset
        size = item_size
        for value in shape:
            size *= value
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    add("model.diffusion_model.patchify_proj.weight", "BF16", [16, 16], 2)
    add("model.diffusion_model.video_vae.synthetic.weight", "BF16", [32, 32], 2)
    for block in range(48):
        fp8 = 4 <= block <= 46
        for index in range(34):
            layer = (
                "model.diffusion_model.transformer_blocks."
                f"{block}.linear_{index:02d}"
            )
            add(f"{layer}.weight", "F8_E4M3" if fp8 else "BF16", [8, 8], 1 if fp8 else 2)
            if fp8:
                add(f"{layer}.input_scale", "F32", [1], 4)
                add(f"{layer}.weight_scale", "F32", [1], 4)

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def test_mock_exactly_counts_mixed_double_pools_without_tensor_data(tmp_path):
    checkpoint = tmp_path / "header-only.safetensors"
    write_header_only_checkpoint(checkpoint)
    report = MOCK.build_report(checkpoint)

    assert report["checkpoint"]["header_only_input"] is True
    assert report["block_layout"]["bytes_per_bf16_block"] == 34 * 8 * 8 * 2
    assert report["block_layout"]["bytes_per_fp8_block"] == 34 * (8 * 8 + 4 + 4)
    expected_pool = 2 * (
        report["block_layout"]["bytes_per_bf16_block"]
        + report["block_layout"]["bytes_per_fp8_block"]
    )
    assert report["residency"]["gpu_block_pool_bytes"] == expected_pool
    assert report["residency"]["pinned_host_block_pool_bytes"] == expected_pool
    assert report["residency"]["gpu_non_block_dit_bytes"] == 16 * 16 * 2
    assert report["residency"]["phase_unloaded_component_bytes"] == 32 * 32 * 2
    assert report["workload"]["stages"]["stage1"]["local_video_tokens"] == 1760
    assert report["workload"]["stages"]["stage2"]["local_video_tokens"] == 7040
    assert report["workload"]["stages"]["stage2"]["ulysses_attention_tokens_per_rank"] == 14080
    assert report["workload"]["stages"]["stage2"]["ulysses_attention_heads_per_rank"] == 16
    assert report["status"] == "PASS"


def test_mock_fails_when_budget_exceeds_per_card_capacity(tmp_path):
    checkpoint = tmp_path / "header-only.safetensors"
    write_header_only_checkpoint(checkpoint)
    report = MOCK.build_report(
        checkpoint,
        card_bytes=1024**3,
        dynamic_budget_gib=1.0,
        runtime_reserve_mib=0,
    )
    assert report["gate"]["static_fit"] == "PASS"
    assert report["gate"]["budgeted_fit"] == "FAIL"
    assert report["status"] == "FAIL"
