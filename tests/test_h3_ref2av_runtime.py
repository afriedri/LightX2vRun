import json
import os
from pathlib import Path
import subprocess

import pytest
from test_minimax_h3_runtime_config import BUILDER

ROOT = Path(__file__).resolve().parents[1]
LIGHTX2V = Path(os.environ.get('LIGHTX2V_SOURCE', ROOT.parent / 'LightX2V-ROCm-GFX1201/LightX2V'))


@pytest.mark.parametrize('world', [1,2,4,8,16])
def test_ref_base_profile_has_no_lora_and_preserves_backend_policy(world):
    if not LIGHTX2V.exists(): pytest.skip('cross-repository test needs LIGHTX2V_SOURCE')
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/minimax_h3_ref2av.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json').read_text())
    config = BUILDER.build(publish, overlay, smoke=False, task='ref2av', seq_parallel_size=world)
    assert config['task'] == 'ref2av' and config['infer_steps'] == 30
    assert config['video_flow_shift'] == 12 and config['audio_flow_shift'] == 3
    assert not config.get('lora_configs') and not config['lora_dynamic_apply']
    assert config['h3_step_update'] == 'reference_blend'
    assert config['lazy_load']
    assert config['reference_image_resize_mode'] == 'diffusers'
    assert config['offload_granularity'] == 'block' and not config['tensor_parallel']
    assert config['h3_core_attn_type'] == 'aiter_fav3_sage_bf16_attn'
    assert config['text_encoder_rank0_only'] == (world > 1)
    assert 'lora_configs' not in overlay


def test_ref_docker_launcher_maps_ordered_references_and_current_run(tmp_path):
    script = ROOT / 'scripts/run_minimax_h3_ref2av_bf16.sh'
    env = {**os.environ, 'DRY_RUN':'1', 'PROMPT':'reference composition', 'CACHE_DIR':str(tmp_path/'cache')}
    result = subprocess.run(['bash', str(script), '--image','example:h3-new',
        '--reference-image', str(tmp_path/'first image.png'), '--reference-image', str(tmp_path/'second.png'),
        '--comfy-dir',str(tmp_path/'comfy'), '--result-dir',str(tmp_path/'out'), '--collect'],
        text=True,capture_output=True,env=env)
    assert result.returncode == 0, result.stderr
    # Docker arguments are shell-quoted in dry-run output.
    import shlex
    lines = result.stdout.splitlines()
    command = next(line for line in lines if line.startswith('DRY RUN: docker '))
    args = shlex.split(command.removeprefix("DRY RUN: "))
    assert 'H3_TASK=ref2av' in args
    assert 'IMAGE_PATH=/inputs/h3_reference_0,/inputs/h3_reference_1' in args
    assert f'type=bind,source={tmp_path}/first image.png,target=/inputs/h3_reference_0,readonly' in args
    assert f'type=bind,source={tmp_path}/comfy,target=/models/minimax_h3_comfy,readonly' in args
    assert 'LIGHTX2V_H3_RUN_COLLECTOR=1' in args
    assert any('target=/workspace/LightX2VRun' in arg for arg in args)
    assert not any('target=/models/minimax_h3_turbo' in arg for arg in args)
    assert not any('target=/inputs/minimax_h3_last_frame' in arg for arg in args)


def test_ref_artifacts_do_not_require_turbo_or_last_frame(tmp_path):
    from test_minimax_h3_preflight import PREFLIGHT, make_artifacts
    model, turbo, first, last = make_artifacts(tmp_path)
    (model / 'transformer').rename(model / 'transformer_ref')
    last.unlink()
    for path in turbo.iterdir(): path.unlink()
    turbo.rmdir()
    result = PREFLIGHT.check_artifacts(model, turbo, first, last, expected_world_size=2, task='ref2av')
    assert result['turbo'] is None
    assert 'reference_image_1' in result and 'last_frame' not in result


def test_ref_base_allows_step_override_but_rejects_lora(tmp_path):
    from test_minimax_h3_preflight import PREFLIGHT
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/minimax_h3_ref2av.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json').read_text())
    config = BUILDER.build(publish, overlay, smoke=True, task='ref2av', seq_parallel_size=2, infer_steps=3)
    assert config['infer_steps'] == 3 and config['target_width'] == 480
    path = tmp_path / 'runtime.json'; path.write_text(json.dumps(config))
    assert PREFLIGHT.check_runtime_config(path, 2)['lora_dynamic_apply'] is False
    overlay['lora_configs'] = [{'path':'stale-turbo.safetensors'}]
    with pytest.raises(SystemExit, match='does not load LoRA'):
        BUILDER.build(publish, overlay, smoke=True, task='ref2av')
