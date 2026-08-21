import os
from pathlib import Path

import pytest
import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import HOOKS, MODELS
from mmrotate.utils import register_all_modules


@pytest.mark.skipif(
    os.getenv('RUN_OVCAPFLOW_INTEGRATION') != '1',
    reason='real HRSC integration is opt-in')
@pytest.mark.parametrize('config_name', [
    'ov_capflow_swin-t_hrsc_c0_native_10e.py',
    'ov_capflow_swin-t_hrsc_c1_fusion_5e.py',
    'ov_capflow_swin-t_hrsc_c1_parent_preserving_5e.py',
    'ov_capflow_swin-t_hrsc_c2_balanced_5e.py',
    'ov_capflow_swin-t_hrsc_c3_null_5e.py',
])
def test_real_hrsc_batch_forward_backward(config_name):
    assert torch.cuda.is_available()
    cfg = Config.fromfile(Path('configs/ov_capflow/hrsc') / config_name)
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model).cuda().train()
    if 'parent_preserving' in config_name:
        freeze_hook_cfg = next(
            item for item in cfg.custom_hooks
            if item.get('type') == 'FreezeExceptHook')
        trainable = HOOKS.build(freeze_hook_cfg).apply(model)
        assert trainable
        assert all('.semantic_fusion.' in name for name in trainable)
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(dataloader))
    batch = model.data_preprocessor(batch, training=True)
    losses = model.loss(batch['inputs'], batch['data_samples'])
    total = sum(value for value in losses.values()
                if torch.is_tensor(value) and value.requires_grad)
    assert losses
    assert all(torch.isfinite(value).all() for value in losses.values()
               if torch.is_tensor(value))
    total.backward()
    active = [name for name, parameter in model.named_parameters()
              if parameter.requires_grad and parameter.grad is not None]
    assert active
    if 'c0_native' in config_name:
        assert not any(name.startswith('loss_null_') for name in losses)
        assert 'loss_capacity_mass' not in losses
    if 'c1_fusion' in config_name:
        assert model.bbox_head.last_balance_stats is None
    if 'c2_balanced' in config_name:
        assert model.bbox_head.last_balance_stats is not None
    if 'c3_null' in config_name:
        assert {'loss_null_matched', 'loss_null_unmatched',
                'loss_null_mass', 'loss_gate_order'} <= set(losses)
