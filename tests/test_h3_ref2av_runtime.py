import json
import os
from pathlib import Path
import subprocess

import pytest
from test_minimax_h3_runtime_config import BUILDER

ROOT = Path(__file__).resolve().parents[1]
LIGHTX2V = Path(os.environ.get('LIGHTX2V_SOURCE', ROOT.parent / 'LightX2V'))


@pytest.mark.parametrize('world', [1,2,4,8,16])
def test_ref_base_profile_has_no_lora_and_preserves_backend_policy(world):
    if not LIGHTX2V.exists(): pytest.skip('cross-repository test needs LIGHTX2V_SOURCE')
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/minimax_h3_ref2av.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json').read_text())
    config = BUILDER.build(publish, overlay, smoke=False, task='ref2av', seq_parallel_size=world)
    assert config['task'] == 'ref2av' and config['infer_steps'] == 20
    assert config['video_flow_shift'] == 12 and config['audio_flow_shift'] == 3
    assert not config.get('lora_configs') and not config['lora_dynamic_apply']
    assert config['h3_step_update'] == 'comfy_res_multistep'
    assert config['lazy_load']
    assert config['reference_image_resize_mode'] == 'match'
    assert config['offload_granularity'] == 'block' and not config['tensor_parallel']
    assert config['h3_core_attn_type'] == 'aiter_fav3_sage_bf16_attn'
    assert config['text_encoder_rank0_only'] == (world > 1)
    assert 'lora_configs' not in overlay


@pytest.mark.parametrize('world', [1, 2, 4, 8, 16])
def test_ref_turbo_8step_profile_composes_full_base_with_dynamic_lora(tmp_path, world):
    from test_minimax_h3_preflight import PREFLIGHT
    if not LIGHTX2V.exists(): pytest.skip('cross-repository test needs LIGHTX2V_SOURCE')
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/dmd/minimax_h3_ref2av_8step.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_8step_768p_gfx1201.json').read_text())
    config = BUILDER.build(
        publish,
        overlay,
        smoke=False,
        task='ref2av',
        profile='ref2av_turbo_8step',
        seq_parallel_size=world,
        target_video_length=362,
        output_duration=15,
    )
    assert config['h3_profile'] == 'ref2av_turbo_8step'
    assert config['infer_steps'] == 9 and config['h3_step_update'] == 'training_euler'
    assert config['video_flow_shift'] == 12 and config['audio_flow_shift'] == 3
    assert config['lora_dynamic_apply'] and config['lazy_load']
    assert config['lora_configs'] == [{
        'path': '/models/minimax_h3_turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',
        'strength': 1.0,
        'alpha': 8,
    }]
    assert config['h3_core_attn_type'] == 'aiter_fav3_sage_bf16_attn'
    path = tmp_path / f'ref2av-turbo-sp{world}.json'
    path.write_text(json.dumps(config))
    assert PREFLIGHT.check_runtime_config(path, world)['h3_profile'] == 'ref2av_turbo_8step'


def test_ref_turbo_8step_rejects_sampling_contract_override():
    if not LIGHTX2V.exists(): pytest.skip('cross-repository test needs LIGHTX2V_SOURCE')
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/dmd/minimax_h3_ref2av_8step.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_8step_768p_gfx1201.json').read_text())
    with pytest.raises(SystemExit, match='infer_steps must be 9'):
        BUILDER.build(
            publish,
            overlay,
            smoke=False,
            task='ref2av',
            profile='ref2av_turbo_8step',
            infer_steps=8,
        )


@pytest.mark.parametrize('gpus', ['0,1', '0,1,2,3', ','.join(map(str, range(8))), ','.join(map(str, range(16)))])
def test_ref_docker_launcher_maps_ordered_references_and_current_run(tmp_path, gpus):
    script = ROOT / 'scripts/run_minimax_h3_ref2av_bf16.sh'
    env = {**os.environ, 'DRY_RUN':'1', 'PROMPT':'reference composition', 'CACHE_DIR':str(tmp_path/'cache'),
           'H3_PROFILE':'ref2av_bf16_base',
           'H3_STEP_UPDATE':'reference_blend', 'H3_REFERENCE_IMAGE_RESIZE_MODE':'max'}
    result = subprocess.run(['bash', str(script), '--image','example:h3-new',
        '--reference-image', str(tmp_path/'first image.png'), '--reference-image', str(tmp_path/'second.png'),
        '--comfy-dir',str(tmp_path/'comfy'), '--result-dir',str(tmp_path/'out'), '--collect', '--duration', '15', '--gpus', gpus],
        text=True,capture_output=True,env=env)
    assert result.returncode == 0, result.stderr
    # Docker arguments are shell-quoted in dry-run output.
    import shlex
    lines = result.stdout.splitlines()
    command = next(line for line in lines if line.startswith('DRY RUN: docker '))
    args = shlex.split(command.removeprefix("DRY RUN: "))
    assert 'H3_TASK=ref2av' in args
    assert 'H3_STEP_UPDATE=reference_blend' in args
    assert 'H3_REFERENCE_IMAGE_RESIZE_MODE=max' in args
    assert f'HIP_VISIBLE_DEVICES={gpus}' in args
    assert 'H3_OUTPUT_DURATION=15' in args and 'TARGET_VIDEO_LENGTH=362' in args
    assert 'IMAGE_PATH=/inputs/h3_reference_0,/inputs/h3_reference_1' in args
    assert f'type=bind,source={tmp_path}/first image.png,target=/inputs/h3_reference_0,readonly' in args
    assert f'type=bind,source={tmp_path}/comfy,target=/models/minimax_h3_comfy,readonly' in args
    assert 'LIGHTX2V_H3_RUN_COLLECTOR=1' in args
    assert 'LIGHTX2V_VAE_CONV3D_BACKEND=Aiter_CK_Conv3d_v1' in args
    assert 'LIGHTX2V_CK_CONV3D_PROFILE_DIR=/workspace/LightX2V/save_results/ck_conv3d_routes' in args
    assert any('target=/workspace/LightX2VRun' in arg for arg in args)
    assert not any('target=/models/minimax_h3_turbo' in arg for arg in args)
    assert not any('target=/inputs/minimax_h3_last_frame' in arg for arg in args)


def test_ref_turbo_docker_launcher_mounts_base_and_8step_lora(tmp_path):
    import shlex

    script = ROOT / 'scripts/run_minimax_h3_ref2av_bf16.sh'
    env = {
        **os.environ,
        'DRY_RUN': '1',
        'PROMPT': 'reference composition',
        'CACHE_DIR': str(tmp_path / 'cache'),
        'H3_PROFILE': 'ref2av_turbo_8step',
        'H3_STEP_UPDATE': 'training_euler',
        'H3_INFER_STEPS': '9',
    }
    result = subprocess.run(
        [
            'bash', str(script), '--image', 'example:h3-turbo',
            '--reference-image', str(tmp_path / 'reference.png'),
            '--model-dir', str(tmp_path / 'model'),
            '--turbo-dir', str(tmp_path / 'turbo'),
            '--result-dir', str(tmp_path / 'out'), '--duration', '15', '--gpus', '0,1,2,3',
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith('DRY RUN: docker '))
    args = shlex.split(command.removeprefix('DRY RUN: '))
    assert 'H3_TASK=ref2av' in args
    assert 'H3_PROFILE=ref2av_turbo_8step' in args
    assert 'H3_STEP_UPDATE=training_euler' in args and 'H3_INFER_STEPS=9' in args
    assert f'type=bind,source={tmp_path}/model,target=/models/minimax_h3,readonly' in args
    assert f'type=bind,source={tmp_path}/turbo,target=/models/minimax_h3_turbo,readonly' in args
    assert not any('target=/models/minimax_h3_comfy' in arg for arg in args)
    assert f'type=bind,source={tmp_path}/cache/h3_ref2av,target=/workspace/.cache/h3_ref2av' not in args
    assert 'H3 Ref2AV base:' in result.stdout
    assert 'H3 Ref2AV BF16:' not in result.stdout
    assert 'H3 prepared cache:' not in result.stdout


def test_ref_turbo_modelrun_does_not_require_comfy_environment(tmp_path):
    import shlex

    env = {
        key: value
        for key, value in os.environ.items()
        if key != 'H3_COMFY_DIR'
    }
    env.update({
        'DRY_RUN': '1',
        'IMAGE': 'example:h3-turbo',
        'LIGHTX2V_H3_EXPLICIT_IMAGE': '1',
        'H3_TASK': 'ref2av',
        'H3_PROFILE': 'ref2av_turbo_8step',
        'H3_MODEL_DIR': str(tmp_path / 'model'),
        'H3_TURBO_DIR': str(tmp_path / 'turbo'),
        'H3_REFERENCE_IMAGES': str(tmp_path / 'reference.png'),
        'PROMPT': 'reference composition',
        'GPU_DEVICES': '0',
        'INPUT_DIR': str(tmp_path / 'inputs'),
        'OUTPUT_DIR': str(tmp_path / 'outputs'),
        'CACHE_DIR': str(tmp_path / 'cache'),
    })
    result = subprocess.run(
        ['bash', str(ROOT / 'lightx2vModelrun'), '--run-minimax-h3-gfx1201'],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith('DRY RUN: docker '))
    args = shlex.split(command.removeprefix('DRY RUN: '))
    assert not any('minimax_h3_comfy' in arg for arg in args)
    assert 'H3 Ref2AV base:' in result.stdout
    assert 'H3 Ref2AV BF16:' not in result.stdout
    assert 'H3 prepared cache:' not in result.stdout


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


@pytest.mark.parametrize("world", [2, 4, 8, 16])
@pytest.mark.parametrize("duration", range(5, 16))
def test_ref_duration_roundtrip_to_preflight(tmp_path, duration, world):
    from test_minimax_h3_preflight import PREFLIGHT
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/minimax_h3_ref2av.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json').read_text())
    frames = ((duration * 24 - 5 + 16) // 17) * 17 + 5
    config = BUILDER.build(publish, overlay, smoke=False, task='ref2av', seq_parallel_size=world,
                           target_video_length=frames, output_duration=duration)
    assert config['target_video_length'] == frames
    assert config['h3_output_duration'] == duration
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(config))
    assert PREFLIGHT.check_runtime_config(path, world)['task'] == 'ref2av'


@pytest.mark.parametrize("kwargs, message", [
    ({'target_video_length': 360}, '17.n.5'),
    ({'target_video_length': 379}, '17.n.5'),
    ({'target_video_length': 362, 'output_duration': 5}, 'does not match'),
    ({'target_video_length': 362, 'task': 'fl2av'}, 'FL2AV'),
])
def test_ref_rejects_invalid_timing_before_weight_preparation(kwargs, message):
    with pytest.raises(SystemExit, match=message):
        BUILDER.build({}, {}, smoke=False, **{'task': 'ref2av', **kwargs})


def test_dedicated_ref_defaults_are_supplied_rooftop_assets_and_15s(tmp_path):
    import sys
    import shlex
    capture = tmp_path / 'model-run.sh'
    capture.write_text(f"exec {shlex.quote(sys.executable)} -c 'import json,os; print(json.dumps(dict(os.environ)))'\n")
    env = {key: value for key, value in os.environ.items()
           if key not in ('PROMPT', 'H3_REFERENCE_IMAGE', 'H3_PROMPT_FILE', 'H3_OUTPUT_DURATION')}
    env['MODEL_RUN'] = str(capture)
    script = ROOT / 'scripts/run_10_minimax_h3_bf16_ref2v.sh'
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['TARGET_VIDEO_LENGTH'] == '362' and values['H3_OUTPUT_DURATION'] == '15'
    assert values['H3_PROFILE'] == 'ref2av_bf16_base'
    refs = values['H3_REFERENCE_IMAGES'].splitlines()
    assert [Path(ref).name for ref in refs] == ['Image Gen-1.png', 'RYAN-2.png']
    assert all(Path(ref).is_file() for ref in refs)
    prompt = values['PROMPT']
    assert '<Picture 1>' in prompt and '<Picture 2>' in prompt and '<Audio' not in prompt
    assert '15-second' in prompt and '[Shot 6]' in prompt
    result = subprocess.run(['bash', str(script), '--reference-image', '/tmp/new.png', '--duration', '5'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['TARGET_VIDEO_LENGTH'] == '124' and values['H3_OUTPUT_DURATION'] == '5'
    assert values['H3_REFERENCE_IMAGES'] == '/tmp/new.png'


def test_dedicated_ref_turbo_defaults_to_exact_8step_contract(tmp_path):
    import shlex
    import sys

    capture = tmp_path / 'model-run.sh'
    capture.write_text(f"exec {shlex.quote(sys.executable)} -c 'import json,os; print(json.dumps(dict(os.environ)))'\n")
    env = {
        key: value for key, value in os.environ.items()
        if key not in ('IMAGE', 'PROMPT', 'H3_REFERENCE_IMAGE', 'H3_PROMPT_FILE', 'H3_OUTPUT_DURATION', 'H3_INFER_STEPS', 'H3_STEP_UPDATE')
    }
    env['MODEL_RUN'] = str(capture)
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh')],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values['H3_PROFILE'] == 'ref2av_turbo_8step'
    assert values['H3_STEP_UPDATE'] == 'training_euler' and values['H3_INFER_STEPS'] == '9'
    assert values['TARGET_VIDEO_LENGTH'] == '362' and values['H3_OUTPUT_DURATION'] == '15'
    assert values['GPU_DEVICES'] == '0,1,2,3'
    assert values['IMAGE'] == 'compiler.hvat-ai.com:8001/lightx2vrocm/lightx2v-rocm:gfx1201-R9600-260910-H3-Ref2V-Turbo8WithFull-V4'
    assert values['LIGHTX2V_H3_RUN_COLLECTOR'] == '1'
    assert values['H3_TURBO_DIR'] == '/disk/ssd/difumodels-Dis/Minimax-h3-Turbo'


def test_public_help_lists_ref2av_base_scale_and_turbo_entries():
    result = subprocess.run(
        ['bash', str(ROOT / 'lightx2vModelrun'), '--help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'scripts/run_10_minimax_h3_bf16_ref2v.sh --image IMAGE' in result.stdout
    assert 'scripts/run_11_minimax_h3_bf16_ref2v_768p_15s_scale.sh --image IMAGE' in result.stdout
    assert 'scripts/run_12_minimax_h3_ref2av_8step_768p_15s.sh --image IMAGE' in result.stdout
    assert 'scripts/run_13_minimax_h3_ref2av_8step_768p_15s_scale.sh [--image IMAGE]' in result.stdout


@pytest.mark.parametrize("mode,steps", [("reference_blend",30), ("training_euler",5), ("comfy_res_multistep",20)])
def test_ref_sampling_alternatives_preserve_hvat_kernels(tmp_path, mode, steps):
    from test_minimax_h3_preflight import PREFLIGHT
    publish = json.loads((LIGHTX2V / 'configs/minimax_h3/minimax_h3_ref2av.json').read_text())
    overlay = json.loads((LIGHTX2V / 'configs/platforms/amd_rocm/minimax_h3_ref2av_bf16_gfx1201.json').read_text())
    config = BUILDER.build(publish, overlay, smoke=False, task='ref2av', seq_parallel_size=4,
                           step_update=mode, infer_steps=steps, reference_image_resize_mode='diffusers')
    assert config['h3_step_update'] == mode and config['infer_steps'] == steps
    assert config['reference_image_resize_mode'] == 'diffusers'
    assert config['h3_core_attn_type'] == 'aiter_fav3_sage_bf16_attn'
    assert config['h3_core_rope_type'] == 'minimax_h3_triton_strict_rope'
    assert config['offload_granularity'] == 'block' and config['lazy_load']
    path = tmp_path/'runtime.json'; path.write_text(json.dumps(config))
    assert PREFLIGHT.check_runtime_config(path, 4)['task'] == 'ref2av'
