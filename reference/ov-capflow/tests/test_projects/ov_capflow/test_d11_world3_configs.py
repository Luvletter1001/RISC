from pathlib import Path

from mmengine import Config


CONFIG_DIR = Path('configs/ov_capflow/dotav2')
CONTROL = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x_world3_control.py')
CANDIDATE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch1_rare4x_world3_d11_content_first600.py')
V2_CONTROL = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world3_v2_control.py')
V2_CANDIDATE = (
    'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
    'scale1024_batch2_rare4x_world3_v2_d11_content_first600.py')
CONTROL_CHECKPOINT = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_compatible.pth')
CANDIDATE_CHECKPOINT = (
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d11_content_first600_v2.pth')


def _load(name):
    return Config.fromfile(CONFIG_DIR / name).to_dict()


def test_d11_world3_configs_have_matched_runtime_protocol():
    control = _load(CONTROL)
    candidate = _load(CANDIDATE)

    assert tuple(control['physical_gpus']) == (3, 4, 5)
    assert tuple(candidate['physical_gpus']) == (0, 1, 2)
    assert control['selected_world_size'] == 3
    assert candidate['selected_world_size'] == 3
    for cfg in (control, candidate):
        assert cfg['model']['num_queries'] == 600
        assert cfg['model']['train_query_groups'] == 3
        assert cfg['model']['bbox_head']['matching_query_groups'] == 3
        assert cfg['train_dataloader']['batch_size'] == 1
        assert cfg['train_dataloader']['batch_sampler'][
            'update_count_multiple'] == 3
        assert cfg['optim_wrapper']['accumulative_counts'] == 3
        assert (
            cfg['train_dataloader']['batch_size']
            * cfg['selected_world_size']
            * cfg['optim_wrapper']['accumulative_counts']
        ) == 9
        assert cfg['randomness'] == {
            'seed': 20260716,
            'deterministic': False,
            'diff_rank_seed': False,
        }
        assert cfg['resume'] is False

    assert control['load_from'] == CONTROL_CHECKPOINT
    assert candidate['load_from'] == CANDIDATE_CHECKPOINT
    assert control['d11_pair_id'] == candidate['d11_pair_id']
    assert control['d11_role'] == 'control'
    assert candidate['d11_role'] == 'candidate'
    assert control['d11_only_scientific_delta'] == 'none'
    assert candidate['d11_only_scientific_delta'] == (
        'initialize_q600_content_queries_from_raw_ogc_first600')


def test_d11_candidate_diff_is_restricted_to_checkpoint_paths_and_metadata():
    control = _load(CONTROL)
    candidate = _load(CANDIDATE)

    allowed_top_level = {
        'physical_gpus',
        'd11_role',
        'd11_only_scientific_delta',
        'load_from',
        'work_dir',
    }
    for key in allowed_top_level:
        control.pop(key)
        candidate.pop(key)

    control['train_dataloader']['batch_sampler'].pop('audit_path')
    candidate['train_dataloader']['batch_sampler'].pop('audit_path')

    assert candidate == control


def test_d11_world3_v2_uses_exact_cover_compatible_matched_protocol():
    control = _load(V2_CONTROL)
    candidate = _load(V2_CANDIDATE)

    assert tuple(control['physical_gpus']) == (3, 4, 5)
    assert tuple(candidate['physical_gpus']) == (0, 1, 2)
    assert control['selected_world_size'] == 3
    assert candidate['selected_world_size'] == 3
    for cfg in (control, candidate):
        assert cfg['train_dataloader']['batch_size'] == 2
        assert cfg['train_dataloader']['batch_sampler'][
            'update_count_multiple'] == 1
        assert cfg['optim_wrapper']['accumulative_counts'] == 1
        assert (
            cfg['train_dataloader']['batch_size']
            * cfg['selected_world_size']
            * cfg['optim_wrapper']['accumulative_counts']
        ) == 6
        assert cfg['randomness'] == {
            'seed': 20260716,
            'deterministic': False,
            'diff_rank_seed': False,
        }
        assert cfg['resume'] is False

    # 266 full six-sample updates plus a final 2/1/1 rank allocation.
    dataset_size = 1600
    full_update_size = 2 * 3
    assert divmod(dataset_size, full_update_size) == (266, 4)
    assert 3 <= dataset_size % full_update_size <= full_update_size

    assert control['load_from'] == CONTROL_CHECKPOINT
    assert candidate['load_from'] == CANDIDATE_CHECKPOINT
    assert control['d11_v2_pair_id'] == candidate['d11_v2_pair_id']
    assert control['d11_v2_role'] == 'control'
    assert candidate['d11_v2_role'] == 'candidate'


def test_d11_world3_v2_candidate_has_only_allowed_pair_differences():
    control = _load(V2_CONTROL)
    candidate = _load(V2_CANDIDATE)

    allowed_top_level = {
        'physical_gpus',
        'd11_v2_role',
        'd11_v2_only_scientific_delta',
        'load_from',
        'work_dir',
    }
    for key in allowed_top_level:
        control.pop(key)
        candidate.pop(key)

    control['train_dataloader']['batch_sampler'].pop('audit_path')
    candidate['train_dataloader']['batch_sampler'].pop('audit_path')

    assert candidate == control
