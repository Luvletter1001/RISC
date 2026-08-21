import copy
import hashlib
import json
import pickle
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from mmengine.logging import MessageHub
from mmengine.optim import build_optim_wrapper
from mmengine.optim.scheduler import StepLR as MMEngineStepLR
from mmengine.utils import apply_to
from mmrotate.registry import HOOKS

import projects.OVCapFlow.ov_capflow.d13n_mode_hook as d13n_mode_hook_module
from projects.OVCapFlow.ov_capflow import D13NOptimWrapperConstructor
from projects.OVCapFlow.ov_capflow.d13n_mode_hook import (
    D13NParentEvalModeHook, _equal_serialized, _validate_adamw_moment,
    hash_named_tensor_state)
from projects.OVCapFlow.ov_capflow.existence_residual import ExistenceResidual
from projects.OVCapFlow.ov_capflow.freeze_except_hook import (
    apply_freeze_except)


ALLOWLIST = [r'^bbox_head\.existence_residual\.(weight|bias)$']
EXPECTED_NAMES = [
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
]
CFG_TEXT = 'model = dict(d13n=True)\n'


class StringSubclass(str):
    pass


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


class MappingSubclass(dict):
    pass


class FloatSubclass(float):
    pass


class EvilFloat(float):

    equality_calls = 0

    def __eq__(self, other):
        type(self).equality_calls += 1
        return True

    def __ne__(self, other):
        type(self).equality_calls += 1
        return False


class EvilEquality:

    def __ne__(self, other):
        return False


class TrackingDefaultDict(defaultdict):

    def reset_tracking(self):
        self.len_calls = 0
        self.iter_calls = 0
        self.getitem_calls = 0

    def __len__(self):
        self.len_calls += 1
        return super().__len__()

    def __iter__(self):
        self.iter_calls += 1
        return super().__iter__()

    def __getitem__(self, key):
        self.getitem_calls += 1
        return super().__getitem__(key)


class ToyHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.parent_projection = nn.Linear(256, 256)
        self.parent_dropout = nn.Dropout(0.9)
        self.existence_residual = ExistenceResidual(256)
        self.last_matching_group_matched_counts = None
        self.last_existence_logit_quantiles = None
        self.last_existence_residual_quantiles = None
        self.last_existence_clamp_hit_counts = None

    def parent_features(self, inputs):
        return self.parent_dropout(self.parent_projection(inputs))


class ToyDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('storage_probe_buffer', torch.zeros(1))
        self.backbone = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.Dropout(0.8))
        self.encoder = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0.8))
        self.bbox_head = ToyHead()

    def parent_features(self, inputs):
        return self.bbox_head.parent_features(
            self.encoder(self.backbone(inputs)))


class WrappedModel:

    def __init__(self, module):
        self.module = module


def _build(role='candidate', wrapped=False, optimizer_cfg=None):
    torch.manual_seed(17)
    model = ToyDetector()
    matched = apply_freeze_except(model, ALLOWLIST)
    assert matched == EXPECTED_NAMES
    wrapped_model = WrappedModel(model) if wrapped else model
    if optimizer_cfg is None:
        optimizer_cfg = dict(type='AdamW', lr=1e-3)
    optim_wrapper = build_optim_wrapper(wrapped_model, dict(
        constructor='D13NOptimWrapperConstructor',
        optimizer=optimizer_cfg))
    runner = SimpleNamespace(
        model=wrapped_model,
        optim_wrapper=optim_wrapper,
        message_hub=MessageHub.get_instance(
            'd13n-test-{}-{}-{}'.format(role, wrapped, id(model))),
        epoch=0,
        iter=3,
        cfg=SimpleNamespace(pretty_text=CFG_TEXT),
    )
    hook = D13NParentEvalModeHook(role=role)
    return model, runner, hook


def _optimizer_step(model, runner, candidate=True):
    inputs = torch.linspace(-1.0, 1.0, 512).reshape(2, 256)
    hidden = model.parent_features(inputs).detach()
    logits = model.bbox_head.existence_residual(hidden)
    loss = logits.square().mean() + logits.mean() if candidate else (
        model.bbox_head.existence_residual.weight.sum() * 0.0 +
        model.bbox_head.existence_residual.bias.sum() * 0.0)
    runner.optim_wrapper.zero_grad()
    runner.optim_wrapper.backward(loss)
    runner.optim_wrapper.step()


def _checkpoint(model, runner):
    return {
        'meta': {
            'epoch': runner.epoch + 1,
            'iter': runner.iter,
            'cfg': runner.cfg.pretty_text,
        },
        'state_dict': {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        'optimizer': copy.deepcopy(apply_to(
            runner.optim_wrapper.state_dict(),
            lambda value: hasattr(value, 'cpu'),
            lambda value: value.cpu())),
    }


def _initialized(role='candidate', wrapped=False, optimizer_cfg=None):
    model, runner, hook = _build(
        role=role, wrapped=wrapped, optimizer_cfg=optimizer_cfg)
    hook.before_train_iter(runner, batch_idx=0)
    _optimizer_step(model, runner, candidate=role == 'candidate')
    return model, runner, hook, _checkpoint(model, runner)


def _sync_checkpoint_optimizer(runner, checkpoint):
    checkpoint['optimizer'] = copy.deepcopy(apply_to(
        runner.optim_wrapper.state_dict(),
        lambda value: hasattr(value, 'cpu'),
        lambda value: value.cpu()))


def _live_optimizer_state_slots(runner):
    optimizer = runner.optim_wrapper.optimizer
    parameters = [group['params'][0] for group in optimizer.param_groups]
    return [optimizer.state[parameter] for parameter in parameters]


def _inference_clone(tensor):
    with torch.inference_mode():
        value = tensor.clone()
    assert value.is_inference()
    return value


def _noncanonical_singleton_stride_like(tensor):
    assert tensor.shape == torch.Size([1])
    value = torch.empty_strided(
        tensor.shape, (7,), dtype=tensor.dtype, device=tensor.device)
    value.copy_(tensor)
    assert value.is_contiguous()
    assert value._base is None
    assert value.storage_offset() == 0
    assert value.storage().size() == value.numel()
    assert value.stride() == (7,)
    return value


def _snapshot(model):
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()}


def _parent_snapshot(model):
    return {name: value for name, value in _snapshot(model).items()
            if name not in EXPECTED_NAMES}


def _assert_same(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def _inject_adapter_alias(model, alias_kind):
    if alias_kind == 'parameter':
        model.alias = nn.Module()
        model.alias.register_parameter(
            'stolen', model.bbox_head.existence_residual.weight)
    else:
        model.alias = model.bbox_head.existence_residual


def _legacy_freeze_without_alias_audit(model):
    expected = set(EXPECTED_NAMES)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected)


@pytest.mark.parametrize('alias_kind', ['parameter', 'module'])
def test_freeze_and_constructor_reject_adapter_registration_alias(alias_kind):
    model = ToyDetector()
    _inject_adapter_alias(model, alias_kind)
    before = {id(parameter): parameter.requires_grad
              for parameter in model.parameters()}
    with pytest.raises(RuntimeError, match='alias|registration'):
        apply_freeze_except(model, ALLOWLIST)
    after = {id(parameter): parameter.requires_grad
             for parameter in model.parameters()}
    assert after == before

    _legacy_freeze_without_alias_audit(model)
    with pytest.raises(RuntimeError, match='alias|registration'):
        build_optim_wrapper(model, dict(
            constructor='D13NOptimWrapperConstructor',
            optimizer=dict(type='AdamW', lr=1e-3)))


@pytest.mark.parametrize('alias_kind', ['parameter', 'module'])
def test_checkpoint_rejects_adapter_alias_injected_after_build(alias_kind):
    model, runner, hook = _build()
    hook.before_train_iter(runner, batch_idx=0)
    _optimizer_step(model, runner, candidate=True)
    _inject_adapter_alias(model, alias_kind)
    checkpoint = _checkpoint(model, runner)
    with pytest.raises(RuntimeError, match='alias|registration'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('alias_kind', ['parameter', 'module'])
def test_frozen_parent_alias_is_legal_and_excluded_from_optimizer(alias_kind):
    model = ToyDetector()
    parent_parameter = model.backbone[0].weight
    if alias_kind == 'parameter':
        model.parent_alias = nn.Module()
        model.parent_alias.register_parameter('shared', parent_parameter)
    else:
        model.parent_alias = model.backbone[0]
    assert apply_freeze_except(model, ALLOWLIST) == EXPECTED_NAMES
    assert parent_parameter.requires_grad is False
    optim_wrapper = build_optim_wrapper(model, dict(
        constructor='D13NOptimWrapperConstructor',
        optimizer=dict(type='AdamW', lr=1e-3)))
    optimizer_parameters = [
        parameter for group in optim_wrapper.optimizer.param_groups
        for parameter in group['params']]
    assert all(parameter is not parent_parameter
               for parameter in optimizer_parameters)
    assert len(optimizer_parameters) == 2


def test_default_mmengine_constructor_includes_frozen_parent_parameters():
    model = ToyDetector()
    apply_freeze_except(model, ALLOWLIST)
    optim_wrapper = build_optim_wrapper(
        model, dict(optimizer=dict(type='AdamW', lr=1e-3)))
    optimizer_parameters = [
        parameter for group in optim_wrapper.optimizer.param_groups
        for parameter in group['params']]
    assert len(optimizer_parameters) == len(list(model.parameters()))
    assert any(parameter is model.backbone[0].weight
               for parameter in optimizer_parameters)
    assert model.backbone[0].weight.requires_grad is False


def test_registered_d13n_constructor_contains_only_allowlisted_objects():
    assert D13NOptimWrapperConstructor.__name__ == (
        'D13NOptimWrapperConstructor')
    model, runner, _ = _build()
    optimizer_parameters = [parameter for group in
                            runner.optim_wrapper.optimizer.param_groups
                            for parameter in group['params']]
    expected = [dict(model.named_parameters())[name]
                for name in EXPECTED_NAMES]
    assert len(optimizer_parameters) == 2
    assert all(actual is wanted for actual, wanted in
               zip(optimizer_parameters, expected))
    assert sum(parameter.numel() for parameter in optimizer_parameters) == 257


def test_d13n_constructor_normalizes_configdict_optimizer_groups():
    _, runner, hook, checkpoint = _initialized(
        optimizer_cfg=ConfigDict(type='AdamW', lr=1e-3))
    assert type(runner.optim_wrapper.optimizer.param_groups) is list
    assert all(type(group) is dict
               for group in runner.optim_wrapper.optimizer.param_groups)
    hook.before_save_checkpoint(runner, checkpoint)
    assert checkpoint['meta']['d13n_integrity']['schema'] == (
        'd13n-checkpoint-integrity-v1')


@pytest.mark.parametrize('mutation', ['extra_trainable', 'missing_trainable'])
def test_d13n_constructor_rejects_trainable_allowlist_drift(mutation):
    model = ToyDetector()
    apply_freeze_except(model, ALLOWLIST)
    if mutation == 'extra_trainable':
        model.backbone[0].weight.requires_grad_(True)
    else:
        model.bbox_head.existence_residual.bias.requires_grad_(False)
    with pytest.raises(RuntimeError, match='trainable'):
        build_optim_wrapper(model, dict(
            constructor='D13NOptimWrapperConstructor',
            optimizer=dict(type='AdamW', lr=1e-3)))


@pytest.mark.parametrize('paramwise_cfg', [
    {}, {'bypass_duplicate': False},
    {'custom_keys': {'existence_residual': {'lr_mult': 2.0}}}])
def test_d13n_constructor_rejects_caller_paramwise_override(paramwise_cfg):
    model = ToyDetector()
    apply_freeze_except(model, ALLOWLIST)
    with pytest.raises(ValueError, match='paramwise_cfg'):
        build_optim_wrapper(model, dict(
            constructor='D13NOptimWrapperConstructor',
            paramwise_cfg=paramwise_cfg,
            optimizer=dict(type='AdamW', lr=1e-3)))


@pytest.mark.parametrize('wrapped', [False, True])
def test_mode_hook_preserves_root_training_and_only_trains_adapter(wrapped):
    model, runner, hook = _build(wrapped=wrapped)
    model.train()
    hook.before_train_iter(runner, batch_idx=0)
    assert model.training is True
    assert model.bbox_head.training is False
    assert all(child.training is False for child in model.children())
    assert model.bbox_head.existence_residual.training is True
    training_modules = [name for name, module in model.named_modules()
                        if name and module.training]
    assert training_modules == ['bbox_head.existence_residual']
    first_modes = [(name, module.training)
                   for name, module in model.named_modules()]
    hook.before_train_iter(runner, batch_idx=0)
    assert [(name, module.training)
            for name, module in model.named_modules()] == first_modes


def test_parent_features_are_bitwise_repeatable_after_mode_enforcement():
    model, runner, hook = _build()
    model.train()
    hook.before_train_iter(runner, batch_idx=0)
    inputs = torch.linspace(-3.0, 3.0, 512).reshape(2, 256)
    first = model.parent_features(inputs)
    second = model.parent_features(inputs)
    assert torch.equal(first, second)


@pytest.mark.parametrize('role,expect_head_change', [
    ('candidate', True), ('control', False)])
def test_one_step_changes_only_candidate_adapter(role, expect_head_change):
    model, runner, hook = _build(role=role)
    hook.before_train_iter(runner, batch_idx=0)
    before_parent = _parent_snapshot(model)
    before_head = {name: _snapshot(model)[name] for name in EXPECTED_NAMES}
    _optimizer_step(model, runner, candidate=role == 'candidate')
    _assert_same(before_parent, _parent_snapshot(model))
    after = _snapshot(model)
    changes = [not torch.equal(before_head[name], after[name])
               for name in EXPECTED_NAMES]
    assert any(changes) is expect_head_change


def test_after_train_iter_publishes_detached_fixed_scalar_telemetry():
    model, runner, hook = _build()
    head = model.bbox_head
    head.last_matching_group_matched_counts = torch.tensor(
        [[1, 2, 3], [3, 4, 5]])
    head.last_existence_logit_quantiles = torch.tensor([
        [-2.0, 0.0, 2.0], [-3.0, 1.0, 3.0], [-4.0, 2.0, 4.0]])
    head.last_existence_residual_quantiles = torch.tensor([
        [-.2, 0., .2], [-.3, .1, .3], [-.4, .2, .4]])
    head.last_existence_clamp_hit_counts = torch.tensor([7, 11])
    hook.after_train_iter(runner, batch_idx=0)
    expected = {
        'd13n/matched_count_g0': 2.0,
        'd13n/matched_count_g1': 3.0,
        'd13n/matched_count_g2': 4.0,
        'd13n/existence_logit_g1_q50': 1.0,
        'd13n/existence_residual_g2_q95': .4,
        'd13n/clamp_hit_lower': 7.0,
        'd13n/clamp_hit_upper': 11.0,
    }
    for key, value in expected.items():
        history = runner.message_hub.get_scalar(key)
        assert history.current() == pytest.approx(value)
        assert not torch.is_tensor(history.current())


def test_after_train_iter_rejects_missing_or_graph_telemetry():
    model, runner, hook = _build()
    with pytest.raises(RuntimeError, match='telemetry'):
        hook.after_train_iter(runner, batch_idx=0)
    model.bbox_head.last_matching_group_matched_counts = torch.ones(
        1, 3, requires_grad=True)
    model.bbox_head.last_existence_logit_quantiles = torch.ones(3, 3)
    model.bbox_head.last_existence_residual_quantiles = torch.ones(3, 3)
    model.bbox_head.last_existence_clamp_hit_counts = torch.ones(2)
    with pytest.raises(RuntimeError, match='detached'):
        hook.after_train_iter(runner, batch_idx=0)


def test_named_tensor_hash_is_order_stable_and_content_bound():
    left = {'b': torch.tensor([2.0]), 'a': torch.tensor([1, 3])}
    right = {'a': left['a'].clone(), 'b': left['b'].clone()}
    assert hash_named_tensor_state(left) == hash_named_tensor_state(right)
    right['a'][0] += 1
    assert hash_named_tensor_state(left) != hash_named_tensor_state(right)
    renamed = {'c': left['a'].clone(), 'b': left['b'].clone()}
    assert hash_named_tensor_state(left) != hash_named_tensor_state(renamed)


def test_before_save_checkpoint_binds_live_optimizer_and_hashes_integrity():
    model, runner, hook, checkpoint = _initialized(wrapped=True)
    hook.before_save_checkpoint(runner, checkpoint)
    integrity = checkpoint['meta']['d13n_integrity']
    assert integrity['schema'] == 'd13n-checkpoint-integrity-v1'
    assert integrity['role'] == 'candidate'
    assert integrity['epoch'] == 1
    assert integrity['iter'] == 3
    expected_groups = [[EXPECTED_NAMES[0]], [EXPECTED_NAMES[1]]]
    assert integrity['optimizer_parameter_names_by_group'] == expected_groups
    assert integrity['live_optimizer_group_sizes'] == [1, 1]
    assert integrity['serialized_optimizer_group_sizes'] == [1, 1]
    serialized_ids = integrity['serialized_optimizer_parameter_ids_by_group']
    assert [len(group) for group in serialized_ids] == [1, 1]
    assert len(set(parameter_id for group in serialized_ids
                   for parameter_id in group)) == 2
    assert integrity['initialized_optimizer_state_slots'] == 2
    assert integrity['optimizer_class'].endswith('.AdamW')
    assert set(checkpoint['optimizer']) == {
        'state', 'param_groups', 'base_param_settings'}
    live_base = runner.optim_wrapper.base_param_settings
    serialized_base = checkpoint['optimizer']['base_param_settings']
    expected_group_fields = {
        'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach',
        'maximize', 'capturable'}
    assert set(live_base) == expected_group_fields
    assert set(serialized_base) == expected_group_fields
    for base in (live_base, serialized_base):
        assert type(base['params']) is torch.Tensor
        assert base['params'].shape == torch.Size([1])
        assert base['params'].dtype == torch.float32
        assert base['params'].device.type == 'cpu'
        assert not base['params'].requires_grad
        assert torch.equal(base['params'], torch.zeros(1))
    for parameter_state in checkpoint['optimizer']['state'].values():
        assert list(parameter_state) == ['step', 'exp_avg', 'exp_avg_sq']
        assert torch.is_tensor(parameter_state['step'])
        assert parameter_state['step'].shape == torch.Size([])
        assert parameter_state['step'].dtype == torch.float32
        assert parameter_state['step'].device.type == 'cpu'
    assert integrity['parent_state_sha256'] == hash_named_tensor_state({
        name: value for name, value in checkpoint['state_dict'].items()
        if name not in EXPECTED_NAMES})
    assert integrity['adapter_state_sha256'] == hash_named_tensor_state({
        name: checkpoint['state_dict'][name] for name in EXPECTED_NAMES})
    assert integrity['config_sha256'] == hashlib.sha256(
        checkpoint['meta']['cfg'].encode('utf-8')).hexdigest()
    unhashed = dict(integrity)
    digest = unhashed.pop('integrity_sha256')
    payload = json.dumps(
        unhashed, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    assert digest == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('mutation', [
    'unknown', 'duplicate', 'missing', 'reordered', 'added'])
def test_before_save_rejects_live_optimizer_binding_mutations(mutation):
    model, runner, hook, checkpoint = _initialized()
    groups = runner.optim_wrapper.optimizer.param_groups
    if mutation == 'unknown':
        groups[0]['params'][0] = nn.Parameter(
            torch.zeros_like(groups[0]['params'][0]))
    elif mutation == 'duplicate':
        groups[1]['params'][0] = groups[0]['params'][0]
    elif mutation == 'missing':
        del groups[1]['params'][0]
    elif mutation == 'reordered':
        groups[0]['params'][0], groups[1]['params'][0] = (
            groups[1]['params'][0], groups[0]['params'][0])
    else:
        groups[0]['params'].append(model.backbone[0].weight)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'delete_param_id', 'swap_param_ids', 'add_param_id', 'delete_state',
    'add_state', 'swap_states', 'nonfinite_state', 'wrong_state_shape',
    'bool_state_key'])
def test_before_save_rejects_serialized_optimizer_mutations(mutation):
    _, runner, hook, checkpoint = _initialized()
    groups = checkpoint['optimizer']['param_groups']
    parameter_ids = [group['params'][0] for group in groups]
    state = checkpoint['optimizer']['state']
    if mutation == 'delete_param_id':
        del groups[-1]['params'][0]
    elif mutation == 'swap_param_ids':
        groups[0]['params'][0], groups[1]['params'][0] = (
            groups[1]['params'][0], groups[0]['params'][0])
    elif mutation == 'add_param_id':
        groups[0]['params'].append(99)
    elif mutation == 'delete_state':
        del state[parameter_ids[-1]]
    elif mutation == 'add_state':
        state[99] = {'step': torch.tensor(1.)}
    elif mutation == 'swap_states':
        state[parameter_ids[0]], state[parameter_ids[1]] = (
            state[parameter_ids[1]], state[parameter_ids[0]])
    elif mutation == 'nonfinite_state':
        state[parameter_ids[0]]['exp_avg'].flatten()[0] = float('nan')
    elif mutation == 'wrong_state_shape':
        state[parameter_ids[0]]['exp_avg'] = torch.zeros(3)
    else:
        state[True] = state.pop(parameter_ids[1])
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'missing_exp_avg', 'negative_exp_avg_sq', 'extra_state_field',
    'wrong_exp_avg_dtype'])
def test_before_save_rejects_synchronized_adamw_state_schema_mutations(
        mutation):
    _, runner, hook, checkpoint = _initialized()
    state = _live_optimizer_state_slots(runner)[0]
    if mutation == 'missing_exp_avg':
        del state['exp_avg']
    elif mutation == 'negative_exp_avg_sq':
        state['exp_avg_sq'].fill_(-1)
    elif mutation == 'extra_state_field':
        state['unexpected'] = torch.zeros_like(state['exp_avg'])
    else:
        state['exp_avg'] = state['exp_avg'].double()
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize(('step', 'case'), [
    (True, 'bool'),
    ('1', 'non_numeric'),
    (torch.tensor(0.), 'zero'),
    (torch.tensor(-1.), 'negative'),
    (torch.tensor(1.5), 'fractional'),
    (torch.tensor(float('nan')), 'nonfinite'),
    (torch.tensor([1.]), 'nonscalar'),
], ids=lambda value: value if isinstance(value, str) else None)
def test_before_save_rejects_invalid_synchronized_adamw_step(step, case):
    del case
    _, runner, hook, checkpoint = _initialized()
    _live_optimizer_state_slots(runner)[0]['step'] = step
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_rejects_synchronized_empty_optimizer_group():
    _, runner, hook, checkpoint = _initialized()
    new_group = copy.copy(runner.optim_wrapper.optimizer.param_groups[0])
    new_group['params'] = []
    runner.optim_wrapper.optimizer.param_groups.append(new_group)
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_rejects_non_adamw_optimizer():
    _, runner, hook, checkpoint = _initialized(
        optimizer_cfg=dict(type='SGD', lr=1e-3, momentum=0.9))
    with pytest.raises(RuntimeError, match='AdamW'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_serialized_adamw_moment_uses_cpu_for_non_cpu_parameter():
    parameter = nn.Parameter(torch.empty(3, device='meta'))
    serialized = torch.zeros(3, dtype=parameter.dtype)
    _validate_adamw_moment(
        serialized, parameter, 'exp_avg', 'serialized')
    with pytest.raises(RuntimeError, match='device'):
        _validate_adamw_moment(
            serialized, parameter, 'exp_avg', 'live')


@pytest.mark.parametrize('mutation', [
    'missing_betas', 'amsgrad_true', 'invalid_betas', 'invalid_eps',
    'invalid_lr_type', 'invalid_weight_decay_type', 'capturable_true',
    'unknown_field', 'different_group_values'])
def test_before_save_rejects_synchronized_adamw_group_corruption(mutation):
    _, runner, hook, checkpoint = _initialized()
    groups = runner.optim_wrapper.optimizer.param_groups
    if mutation == 'missing_betas':
        for group in groups:
            del group['betas']
    elif mutation == 'amsgrad_true':
        for group in groups:
            group['amsgrad'] = True
    elif mutation == 'invalid_betas':
        for group in groups:
            group['betas'] = (0.9, 1.0)
    elif mutation == 'invalid_eps':
        for group in groups:
            group['eps'] = 0.0
    elif mutation == 'invalid_lr_type':
        for group in groups:
            group['lr'] = True
    elif mutation == 'invalid_weight_decay_type':
        for group in groups:
            group['weight_decay'] = '0.01'
    elif mutation == 'capturable_true':
        for group in groups:
            group['capturable'] = True
    elif mutation == 'unknown_field':
        for group in groups:
            group['unexpected'] = 1
    else:
        groups[1]['lr'] = groups[0]['lr'] * 2
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'outer_groups', 'group', 'params', 'base', 'slot'])
def test_before_save_rejects_non_builtin_live_optimizer_containers(mutation):
    _, runner, hook, checkpoint = _initialized()
    optimizer = runner.optim_wrapper.optimizer
    if mutation == 'outer_groups':
        optimizer.param_groups = ListSubclass(optimizer.param_groups)
    elif mutation == 'group':
        optimizer.param_groups[0] = DictSubclass(optimizer.param_groups[0])
    elif mutation == 'params':
        optimizer.param_groups[0]['params'] = ListSubclass(
            optimizer.param_groups[0]['params'])
    elif mutation == 'base':
        runner.optim_wrapper.base_param_settings = DictSubclass(
            runner.optim_wrapper.base_param_settings)
    else:
        parameter = optimizer.param_groups[0]['params'][0]
        optimizer.state[parameter] = DictSubclass(optimizer.state[parameter])
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'plain_dict', 'none_factory', 'list_factory'])
def test_before_save_rejects_noncanonical_live_optimizer_state_outer(
        mutation):
    _, runner, hook, checkpoint = _initialized()
    optimizer = runner.optim_wrapper.optimizer
    items = list(optimizer.state.items())
    if mutation == 'plain_dict':
        optimizer.state = dict(items)
    elif mutation == 'none_factory':
        optimizer.state = defaultdict(None, items)
    else:
        optimizer.state = defaultdict(list, items)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_rejects_state_subclass_before_overloaded_access():
    _, runner, hook, checkpoint = _initialized()
    optimizer = runner.optim_wrapper.optimizer
    state = TrackingDefaultDict(dict)
    for parameter, slot in optimizer.state.items():
        state[parameter] = slot
    state.reset_tracking()
    optimizer.state = state
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)
    assert state.len_calls == 0
    assert state.iter_calls == 0
    assert state.getitem_calls == 0


def test_before_save_rejects_live_optimizer_state_order_before_state_dict(
        monkeypatch):
    _, runner, hook, checkpoint = _initialized()
    optimizer = runner.optim_wrapper.optimizer
    optimizer.state = defaultdict(
        dict, reversed(list(optimizer.state.items())))

    def forbidden_state_dict():
        pytest.fail('state_dict must not run before raw state topology audit')

    monkeypatch.setattr(
        runner.optim_wrapper, 'state_dict', forbidden_state_dict)
    with pytest.raises(RuntimeError, match='live AdamW state'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('group_index', [0, 1])
@pytest.mark.parametrize('field', ['step', 'exp_avg', 'exp_avg_sq'])
def test_before_save_rejects_live_state_dict_slot_value_drift(
        monkeypatch, group_index, field):
    _, runner, hook, checkpoint = _initialized()
    live_state = copy.deepcopy(runner.optim_wrapper.state_dict())
    parameter_id = live_state['param_groups'][group_index]['params'][0]
    value = live_state['state'][parameter_id][field].clone()
    value.reshape(-1)[0].add_(1)
    live_state['state'][parameter_id][field] = value
    monkeypatch.setattr(
        runner.optim_wrapper, 'state_dict', lambda: live_state)
    checkpoint['optimizer'] = copy.deepcopy(live_state)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'wrapper_top', 'state', 'outer_groups', 'group', 'params', 'base', 'slot'])
def test_before_save_rejects_non_builtin_synchronized_state_dict_containers(
        monkeypatch, mutation):
    _, runner, hook, checkpoint = _initialized()
    live_state = runner.optim_wrapper.state_dict()
    if mutation == 'wrapper_top':
        live_state = DictSubclass(live_state)
    elif mutation == 'state':
        live_state['state'] = DictSubclass(live_state['state'])
    elif mutation == 'outer_groups':
        live_state['param_groups'] = ListSubclass(live_state['param_groups'])
    elif mutation == 'group':
        live_state['param_groups'][0] = DictSubclass(
            live_state['param_groups'][0])
    elif mutation == 'params':
        live_state['param_groups'][0]['params'] = ListSubclass(
            live_state['param_groups'][0]['params'])
    elif mutation == 'base':
        base = DictSubclass(runner.optim_wrapper.base_param_settings)
        runner.optim_wrapper.base_param_settings = base
        live_state['base_param_settings'] = base
    else:
        parameter_id = next(iter(live_state['state']))
        live_state['state'][parameter_id] = DictSubclass(
            live_state['state'][parameter_id])
    monkeypatch.setattr(
        runner.optim_wrapper, 'state_dict', lambda: live_state)
    checkpoint['optimizer'] = copy.deepcopy(live_state)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'lr_int', 'eps_int', 'weight_decay_int', 'initial_lr_int',
    'first_beta_int', 'second_beta_int', 'lr_float_subclass'])
def test_before_save_rejects_non_builtin_synchronized_adamw_floats(mutation):
    _, runner, hook, checkpoint = _initialized()
    settings = [*runner.optim_wrapper.optimizer.param_groups,
                runner.optim_wrapper.base_param_settings]
    if mutation == 'lr_int':
        for setting in settings:
            setting['lr'] = 1
    elif mutation == 'eps_int':
        for setting in settings:
            setting['eps'] = 1
    elif mutation == 'weight_decay_int':
        for setting in settings:
            setting['weight_decay'] = 0
    elif mutation == 'initial_lr_int':
        for setting in settings:
            setting['initial_lr'] = 1
    elif mutation == 'first_beta_int':
        for setting in settings:
            setting['betas'] = (0, 0.999)
    elif mutation == 'second_beta_int':
        for setting in settings:
            setting['betas'] = (0.9, 0)
    else:
        for setting in settings:
            setting['lr'] = FloatSubclass(0.001)
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_rejects_evil_serialized_float_before_equality():
    _, runner, hook, checkpoint = _initialized()
    EvilFloat.equality_calls = 0
    for group in checkpoint['optimizer']['param_groups']:
        group['lr'] = EvilFloat(123.0)
    checkpoint['optimizer']['base_param_settings']['lr'] = EvilFloat(123.0)
    with pytest.raises(RuntimeError, match='AdamW lr'):
        hook.before_save_checkpoint(runner, checkpoint)
    assert EvilFloat.equality_calls == 0


def test_before_save_accepts_synchronized_optional_initial_lr():
    _, runner, hook, checkpoint = _initialized()
    for group in runner.optim_wrapper.optimizer.param_groups:
        group['initial_lr'] = group['lr']
    runner.optim_wrapper.base_param_settings['initial_lr'] = (
        runner.optim_wrapper.base_param_settings['lr'])
    _sync_checkpoint_optimizer(runner, checkpoint)
    hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'negative_lr', 'zero_eps', 'extra_field', 'missing_field',
    'frozen_parent_parameter', 'wrong_dummy_dtype', 'wrong_dummy_shape',
    'nonzero_dummy', 'trainable_dummy', 'different_from_real_groups'])
def test_before_save_rejects_synchronized_base_settings_corruption(mutation):
    model, runner, hook, checkpoint = _initialized()
    base = runner.optim_wrapper.base_param_settings
    if mutation == 'negative_lr':
        base['lr'] = -1.0
    elif mutation == 'zero_eps':
        base['eps'] = 0.0
    elif mutation == 'extra_field':
        base['unexpected'] = 1
    elif mutation == 'missing_field':
        del base['betas']
    elif mutation == 'frozen_parent_parameter':
        base['params'] = model.backbone[0].weight
    elif mutation == 'wrong_dummy_dtype':
        base['params'] = base['params'].double()
    elif mutation == 'wrong_dummy_shape':
        base['params'] = torch.zeros(2)
    elif mutation == 'nonzero_dummy':
        base['params'] = torch.ones(1)
    elif mutation == 'trainable_dummy':
        base['params'] = torch.zeros(1, requires_grad=True)
    else:
        base['betas'] = (0.8, 0.99)
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'parameter_step', 'parameter_moment', 'step_grad', 'moment_grad',
    'dummy_grad', 'invalid_moment_layout'])
def test_before_save_rejects_optimizer_autograd_tensor_corruption(mutation):
    _, runner, hook, checkpoint = _initialized()
    state = _live_optimizer_state_slots(runner)[0]
    base_dummy = runner.optim_wrapper.base_param_settings['params']
    if mutation == 'parameter_step':
        state['step'] = nn.Parameter(state['step'].clone())
    elif mutation == 'parameter_moment':
        state['exp_avg'] = nn.Parameter(state['exp_avg'].clone())
    elif mutation == 'step_grad':
        state['step'].grad = torch.ones_like(state['step'])
    elif mutation == 'moment_grad':
        state['exp_avg'].grad = torch.ones_like(state['exp_avg'])
    elif mutation == 'dummy_grad':
        base_dummy.grad = torch.ones_like(base_dummy)
    else:
        state['exp_avg'] = state['exp_avg'].to_sparse()
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'expanded_moment', 'moment_view', 'moment_offset', 'negative_moment_view',
    'step_view', 'base_view'])
def test_before_save_rejects_noncanonical_live_optimizer_tensor(mutation):
    _, runner, hook, checkpoint = _initialized()
    state = _live_optimizer_state_slots(runner)[0]
    base = runner.optim_wrapper.base_param_settings
    if mutation == 'expanded_moment':
        state['exp_avg'] = torch.zeros(1).expand_as(state['exp_avg'])
    elif mutation == 'moment_view':
        backing = state['exp_avg'].clone()
        state['exp_avg'] = backing.view_as(state['exp_avg'])
    elif mutation == 'moment_offset':
        original = state['exp_avg']
        backing = torch.empty(original.numel() + 1)
        view = backing[1:].view_as(original)
        view.copy_(original)
        state['exp_avg'] = view
    elif mutation == 'negative_moment_view':
        backing = -state['exp_avg'].clone()
        state['exp_avg'] = torch._neg_view(backing)
    elif mutation == 'step_view':
        state['step'] = state['step'].clone().view(())
    else:
        base['params'] = base['params'].clone().view(1)
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'moment_view', 'moment_offset', 'negative_moment_view', 'base_view'])
def test_before_save_rejects_noncanonical_serialized_optimizer_tensor(
        mutation):
    _, runner, hook, checkpoint = _initialized()
    state = next(iter(checkpoint['optimizer']['state'].values()))
    base = checkpoint['optimizer']['base_param_settings']
    if mutation == 'moment_view':
        backing = state['exp_avg'].clone()
        state['exp_avg'] = backing.view_as(state['exp_avg'])
    elif mutation == 'moment_offset':
        original = state['exp_avg']
        backing = torch.empty(original.numel() + 1)
        view = backing[1:].view_as(original)
        view.copy_(original)
        state['exp_avg'] = view
    elif mutation == 'negative_moment_view':
        backing = -state['exp_avg'].clone()
        state['exp_avg'] = torch._neg_view(backing)
    else:
        base['params'] = base['params'].clone().view(1)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'live_base', 'live_bias_moment',
    'serialized_base', 'serialized_bias_moment'])
def test_before_save_rejects_noncanonical_singleton_optimizer_stride(
        mutation):
    _, runner, hook, checkpoint = _initialized()
    if mutation == 'live_base':
        base = runner.optim_wrapper.base_param_settings
        base['params'] = _noncanonical_singleton_stride_like(base['params'])
        _sync_checkpoint_optimizer(runner, checkpoint)
    elif mutation == 'live_bias_moment':
        state = _live_optimizer_state_slots(runner)[1]
        state['exp_avg'] = _noncanonical_singleton_stride_like(
            state['exp_avg'])
        _sync_checkpoint_optimizer(runner, checkpoint)
    elif mutation == 'serialized_base':
        base = checkpoint['optimizer']['base_param_settings']
        base['params'] = _noncanonical_singleton_stride_like(base['params'])
    else:
        state = list(checkpoint['optimizer']['state'].values())[1]
        state['exp_avg'] = _noncanonical_singleton_stride_like(
            state['exp_avg'])
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_adamw_moment_rejects_conjugate_view():
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.complex64))
    value = torch.tensor([1 + 2j, 3 + 4j]).conj()
    assert value.is_conj()
    with pytest.raises(RuntimeError, match='optimizer'):
        _validate_adamw_moment(value, parameter, 'exp_avg', 'live')


def test_expanded_moment_crashes_real_resume_step_without_publication_guard():
    _, source_runner, _, checkpoint = _initialized()
    source_state = _live_optimizer_state_slots(source_runner)[0]
    source_state['exp_avg'] = torch.zeros(1).expand_as(
        source_state['exp_avg'])
    _sync_checkpoint_optimizer(source_runner, checkpoint)

    target_model, target_runner, _ = _build()
    target_runner.optim_wrapper.load_state_dict(
        copy.deepcopy(checkpoint['optimizer']))
    with pytest.raises(RuntimeError, match='single memory location'):
        _optimizer_step(target_model, target_runner, candidate=True)


@pytest.mark.parametrize('mutation', [
    'shared_moments', 'cross_slot_steps', 'dummy_buffer', 'dummy_buffer_view',
    'dummy_state_identity', 'dummy_state_view'])
def test_before_save_rejects_optimizer_storage_aliases(mutation):
    model, runner, hook, checkpoint = _initialized()
    slots = _live_optimizer_state_slots(runner)
    if mutation == 'shared_moments':
        slots[1]['exp_avg_sq'] = slots[1]['exp_avg']
    elif mutation == 'cross_slot_steps':
        slots[1]['step'] = slots[0]['step']
    elif mutation == 'dummy_buffer':
        runner.optim_wrapper.base_param_settings['params'] = (
            model.storage_probe_buffer)
    elif mutation == 'dummy_buffer_view':
        runner.optim_wrapper.base_param_settings['params'] = (
            model.storage_probe_buffer.view(1))
    elif mutation == 'dummy_state_identity':
        slots[1]['exp_avg'].zero_()
        runner.optim_wrapper.base_param_settings['params'] = (
            slots[1]['exp_avg'])
    else:
        slots[1]['exp_avg'].zero_()
        runner.optim_wrapper.base_param_settings['params'] = (
            slots[1]['exp_avg'].view(1))
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'shared_moment_storage', 'cross_slot_step_storage',
    'dummy_state_storage', 'dummy_checkpoint_buffer'])
def test_before_save_rejects_serialized_optimizer_storage_aliases(mutation):
    _, runner, hook, checkpoint = _initialized()
    state = checkpoint['optimizer']['state']
    parameter_ids = list(state)
    if mutation == 'shared_moment_storage':
        slot = state[parameter_ids[1]]
        exp_avg = slot['exp_avg']
        exp_avg_sq = slot['exp_avg_sq']
        backing = torch.cat((exp_avg.reshape(-1), exp_avg_sq.reshape(-1)))
        slot['exp_avg'] = backing[:exp_avg.numel()].view_as(exp_avg)
        slot['exp_avg_sq'] = backing[exp_avg.numel():].view_as(exp_avg_sq)
    elif mutation == 'cross_slot_step_storage':
        backing = torch.tensor([1., 1.])
        state[parameter_ids[0]]['step'] = backing[0]
        state[parameter_ids[1]]['step'] = backing[1]
    elif mutation == 'dummy_state_storage':
        exp_avg = state[parameter_ids[1]]['exp_avg']
        backing = torch.cat((torch.zeros(1), exp_avg.reshape(-1)))
        checkpoint['optimizer']['base_param_settings']['params'] = backing[:1]
        state[parameter_ids[1]]['exp_avg'] = backing[1:].view_as(exp_avg)
    else:
        checkpoint['optimizer']['base_param_settings']['params'] = (
            checkpoint['state_dict']['storage_probe_buffer'])
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_valid_optimizer_state_storage_is_pairwise_disjoint():
    model, runner, hook, checkpoint = _initialized()
    hook.before_save_checkpoint(runner, checkpoint)
    live_tensors = [runner.optim_wrapper.base_param_settings['params']]
    for slot in _live_optimizer_state_slots(runner):
        live_tensors.extend(
            slot[field] for field in ('step', 'exp_avg', 'exp_avg_sq'))
    live_storage = {
        (str(tensor.device), tensor.storage().data_ptr())
        for tensor in live_tensors
    }
    model_storage = {
        (str(tensor.device), tensor.storage().data_ptr())
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.layout == torch.strided and tensor.numel() > 0
    }
    assert len(live_storage) == 7
    assert live_storage.isdisjoint(model_storage)


@pytest.mark.parametrize('field', [
    'step', 'exp_avg', 'exp_avg_sq', 'base_dummy'])
def test_before_save_rejects_synchronized_inference_optimizer_tensor(field):
    _, runner, hook, checkpoint = _initialized()
    if field == 'base_dummy':
        base = runner.optim_wrapper.base_param_settings
        base['params'] = _inference_clone(base['params'])
    else:
        state = _live_optimizer_state_slots(runner)[0]
        state[field] = _inference_clone(state[field])
    _sync_checkpoint_optimizer(runner, checkpoint)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('field', [
    'step', 'exp_avg', 'exp_avg_sq', 'base_dummy'])
def test_before_save_rejects_serialized_inference_optimizer_tensor(field):
    _, runner, hook, checkpoint = _initialized()
    if field == 'base_dummy':
        base = checkpoint['optimizer']['base_param_settings']
        base['params'] = _inference_clone(base['params'])
    else:
        state = next(iter(checkpoint['optimizer']['state'].values()))
        state[field] = _inference_clone(state[field])
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', ['missing_field', 'extra_field'])
def test_before_save_rejects_synchronized_wrapper_topology_corruption(
        monkeypatch, mutation):
    _, runner, hook, checkpoint = _initialized()
    live_state = copy.deepcopy(runner.optim_wrapper.state_dict())
    if mutation == 'missing_field':
        del live_state['base_param_settings']
    else:
        live_state['unexpected'] = 1
    monkeypatch.setattr(
        runner.optim_wrapper, 'state_dict', lambda: live_state)
    checkpoint['optimizer'] = copy.deepcopy(live_state)
    with pytest.raises(RuntimeError, match='optimizer'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_valid_optimizer_checkpoint_load_restores_base_settings():
    _, source_runner, hook, checkpoint = _initialized()
    for group in source_runner.optim_wrapper.optimizer.param_groups:
        group['initial_lr'] = group['lr']
    source_runner.optim_wrapper.base_param_settings['initial_lr'] = (
        source_runner.optim_wrapper.base_param_settings['lr'])
    _sync_checkpoint_optimizer(source_runner, checkpoint)
    hook.before_save_checkpoint(source_runner, checkpoint)

    target_model, target_runner, target_hook = _build()
    target_runner.optim_wrapper.base_param_settings['lr'] = 0.5
    target_runner.optim_wrapper.load_state_dict(
        copy.deepcopy(checkpoint['optimizer']))
    restored = target_runner.optim_wrapper.base_param_settings
    assert set(restored) == set(
        source_runner.optim_wrapper.base_param_settings)
    for field in set(restored) - {'params'}:
        assert restored[field] == (
            source_runner.optim_wrapper.base_param_settings[field])
    assert torch.equal(restored['params'], torch.zeros(1))
    target_hook.before_train_iter(target_runner, batch_idx=0)
    _optimizer_step(target_model, target_runner, candidate=True)
    target_hook.before_save_checkpoint(
        target_runner, _checkpoint(target_model, target_runner))


def test_mmengine_scheduler_save_resume_resave_resume_chain():
    def sync_base(runner):
        groups = runner.optim_wrapper.optimizer.param_groups
        base = runner.optim_wrapper.base_param_settings
        base['lr'] = groups[0]['lr']
        base['initial_lr'] = groups[0]['initial_lr']

    source_model, source_runner, source_hook = _build()
    optimizer = source_runner.optim_wrapper.optimizer
    scheduler = MMEngineStepLR(
        source_runner.optim_wrapper, step_size=1, gamma=0.5)
    source_hook.before_train_iter(source_runner, batch_idx=0)
    _optimizer_step(source_model, source_runner, candidate=True)
    scheduler.step()
    for group in optimizer.param_groups:
        assert type(group['lr']) is float
        assert type(group['initial_lr']) is float
        assert type(group['eps']) is float
        assert type(group['weight_decay']) is float
        assert type(group['betas']) is tuple
        assert all(type(beta) is float for beta in group['betas'])
    sync_base(source_runner)
    first_checkpoint = _checkpoint(source_model, source_runner)
    first_checkpoint['param_schedulers'] = [
        copy.deepcopy(scheduler.state_dict())]
    source_hook.before_save_checkpoint(source_runner, first_checkpoint)

    resumed_model, resumed_runner, resumed_hook = _build()
    resumed_model.load_state_dict(
        copy.deepcopy(first_checkpoint['state_dict']))
    resumed_runner.optim_wrapper.load_state_dict(
        copy.deepcopy(first_checkpoint['optimizer']))
    resumed_scheduler = MMEngineStepLR(
        resumed_runner.optim_wrapper, step_size=1, gamma=0.5)
    resumed_scheduler.load_state_dict(
        copy.deepcopy(first_checkpoint['param_schedulers'][0]))
    assert resumed_scheduler.state_dict() == (
        first_checkpoint['param_schedulers'][0])
    resumed_hook.before_train_iter(resumed_runner, batch_idx=0)
    _optimizer_step(resumed_model, resumed_runner, candidate=True)
    resumed_scheduler.step()
    sync_base(resumed_runner)
    second_checkpoint = _checkpoint(resumed_model, resumed_runner)
    second_checkpoint['param_schedulers'] = [
        copy.deepcopy(resumed_scheduler.state_dict())]
    resumed_hook.before_save_checkpoint(resumed_runner, second_checkpoint)

    final_model, final_runner, final_hook = _build()
    final_model.load_state_dict(copy.deepcopy(second_checkpoint['state_dict']))
    final_runner.optim_wrapper.load_state_dict(
        copy.deepcopy(second_checkpoint['optimizer']))
    final_scheduler = MMEngineStepLR(
        final_runner.optim_wrapper, step_size=1, gamma=0.5)
    final_scheduler.load_state_dict(
        copy.deepcopy(second_checkpoint['param_schedulers'][0]))
    assert final_scheduler.state_dict() == (
        second_checkpoint['param_schedulers'][0])
    final_hook.before_train_iter(final_runner, batch_idx=0)
    _optimizer_step(final_model, final_runner, candidate=True)
    final_scheduler.step()
    sync_base(final_runner)
    final_hook.before_save_checkpoint(
        final_runner, _checkpoint(final_model, final_runner))


@pytest.mark.parametrize('mutation', [
    'negative_epoch', 'negative_iter', 'epoch_mismatch', 'iter_mismatch',
    'cfg_mismatch', 'missing_runner_cfg'])
def test_before_save_rejects_runtime_provenance_mutations(mutation):
    _, runner, hook, checkpoint = _initialized()
    if mutation == 'negative_epoch':
        runner.epoch = -1
        checkpoint['meta']['epoch'] = 0
    elif mutation == 'negative_iter':
        runner.iter = -1
        checkpoint['meta']['iter'] = -1
    elif mutation == 'epoch_mismatch':
        checkpoint['meta']['epoch'] += 1
    elif mutation == 'iter_mismatch':
        checkpoint['meta']['iter'] += 1
    elif mutation == 'cfg_mismatch':
        checkpoint['meta']['cfg'] += '# drift\n'
    else:
        del runner.cfg
    with pytest.raises(RuntimeError, match='D13-N'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'runner_cfg_subclass', 'checkpoint_cfg_subclass', 'evil_checkpoint_cfg'])
def test_before_save_rejects_non_builtin_checkpoint_config_types(mutation):
    _, runner, hook, checkpoint = _initialized()
    if mutation == 'runner_cfg_subclass':
        runner.cfg.pretty_text = StringSubclass(CFG_TEXT)
        checkpoint['meta']['cfg'] = runner.cfg.pretty_text
    elif mutation == 'checkpoint_cfg_subclass':
        checkpoint['meta']['cfg'] = StringSubclass(CFG_TEXT)
    else:
        checkpoint['meta']['cfg'] = EvilEquality()
    with pytest.raises(RuntimeError, match='D13-N'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'delete_adapter', 'rename_adapter', 'swap_adapter', 'add_state_tensor'])
def test_before_save_rejects_checkpoint_model_state_mutations(mutation):
    _, runner, hook, checkpoint = _initialized()
    state = checkpoint['state_dict']
    if mutation == 'delete_adapter':
        del state[EXPECTED_NAMES[1]]
    elif mutation == 'rename_adapter':
        state[EXPECTED_NAMES[1] + '_renamed'] = state.pop(EXPECTED_NAMES[1])
    elif mutation == 'swap_adapter':
        first = state[EXPECTED_NAMES[0]].clone()
        state[EXPECTED_NAMES[0]] = state[EXPECTED_NAMES[1]].clone()
        state[EXPECTED_NAMES[1]] = first
    else:
        state['unexpected'] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='state_dict'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'delete_field', 'rename_field', 'swap_names', 'add_field',
    'bad_self_hash'])
def test_before_save_rejects_existing_integrity_metadata_mutations(mutation):
    _, runner, hook, checkpoint = _initialized()
    hook.before_save_checkpoint(runner, checkpoint)
    integrity = checkpoint['meta']['d13n_integrity']
    if mutation == 'delete_field':
        del integrity['config_sha256']
    elif mutation == 'rename_field':
        integrity['configuration_sha256'] = integrity.pop('config_sha256')
    elif mutation == 'swap_names':
        integrity['optimizer_parameter_names_by_group'].reverse()
    elif mutation == 'add_field':
        integrity['unexpected'] = True
    else:
        integrity['integrity_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='integrity'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_accepts_unchanged_existing_integrity_metadata():
    _, runner, hook, checkpoint = _initialized()
    hook.before_save_checkpoint(runner, checkpoint)
    first = copy.deepcopy(checkpoint['meta']['d13n_integrity'])
    hook.before_save_checkpoint(runner, checkpoint)
    assert checkpoint['meta']['d13n_integrity'] == first


def test_before_save_rejects_present_none_integrity_metadata():
    _, runner, hook, checkpoint = _initialized()
    checkpoint['meta']['d13n_integrity'] = None
    with pytest.raises(RuntimeError, match='integrity'):
        hook.before_save_checkpoint(runner, checkpoint)


@pytest.mark.parametrize('mutation', [
    'bool_epoch', 'float_state_slots', 'float_nested_group_size',
    'bool_nested_state_id', 'reordered_keys'])
def test_before_save_rejects_equal_valued_integrity_type_or_order_mutations(
        mutation):
    _, runner, hook, checkpoint = _initialized()
    hook.before_save_checkpoint(runner, checkpoint)
    integrity = checkpoint['meta']['d13n_integrity']
    if mutation == 'bool_epoch':
        integrity['epoch'] = True
    elif mutation == 'float_state_slots':
        integrity['initialized_optimizer_state_slots'] = 2.0
    elif mutation == 'float_nested_group_size':
        integrity['live_optimizer_group_sizes'][0] = 1.0
    elif mutation == 'bool_nested_state_id':
        integrity['serialized_optimizer_parameter_ids_by_group'][1][0] = True
    else:
        checkpoint['meta']['d13n_integrity'] = dict(
            reversed(list(integrity.items())))
    with pytest.raises(RuntimeError, match='integrity'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_before_save_rejects_nested_list_subclass_with_valid_self_hash():
    _, runner, hook, checkpoint = _initialized()
    hook.before_save_checkpoint(runner, checkpoint)
    integrity = checkpoint['meta']['d13n_integrity']
    integrity['live_optimizer_group_sizes'] = ListSubclass(
        integrity['live_optimizer_group_sizes'])
    unhashed = dict(integrity)
    unhashed.pop('integrity_sha256')
    payload = json.dumps(
        unhashed, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    integrity['integrity_sha256'] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(RuntimeError, match='integrity'):
        hook.before_save_checkpoint(runner, checkpoint)


def test_serialized_equality_rejects_mapping_subclass():
    assert not _equal_serialized(
        {'nested': [1]}, MappingSubclass({'nested': [1]}))


@pytest.mark.parametrize('role', [
    'candidate ', 'CONTROL', '', None, 1, StringSubclass('candidate')])
def test_mode_hook_rejects_noncanonical_roles(role):
    with pytest.raises((TypeError, ValueError), match='role'):
        D13NParentEvalModeHook(role=role)


@pytest.mark.parametrize('role', [
    StringSubclass('candidate'), 'invalid', None])
def test_before_save_revalidates_mutated_hook_role(role):
    _, runner, hook, checkpoint = _initialized()
    try:
        hook.role = role
    except (AttributeError, TypeError, ValueError):
        assert hook.role == 'candidate'
        assert 'd13n_integrity' not in checkpoint['meta']
        return
    with pytest.raises((TypeError, ValueError, RuntimeError), match='role'):
        hook.before_save_checkpoint(runner, checkpoint)
    assert 'd13n_integrity' not in checkpoint['meta']


@pytest.mark.parametrize(('configured_role', 'mutated_role'), [
    ('candidate', 'control'), ('control', 'candidate')])
def test_mode_hook_rejects_cross_arm_public_role_mutation(
        configured_role, mutated_role):
    _, runner, hook, checkpoint = _initialized(role=configured_role)
    assert hook.role == configured_role
    try:
        hook.role = mutated_role
    except (AttributeError, TypeError, ValueError):
        assert hook.role == configured_role
        assert 'd13n_integrity' not in checkpoint['meta']
        return
    with pytest.raises(RuntimeError, match='role'):
        hook.before_save_checkpoint(runner, checkpoint)
    assert 'd13n_integrity' not in checkpoint['meta']


@pytest.mark.parametrize(('configured_role', 'mutated_role'), [
    ('candidate', 'control'), ('control', 'candidate')])
def test_role_authority_ignores_synchronized_instance_field_drift(
        configured_role, mutated_role):
    _, runner, hook, checkpoint = _initialized(role=configured_role)
    hook.__dict__.update({
        'role': mutated_role,
        '_role': mutated_role,
        'configured_role': mutated_role,
        '_configured_role': mutated_role,
    })
    assert hook.role == configured_role
    hook.before_save_checkpoint(runner, checkpoint)
    assert checkpoint['meta']['d13n_integrity']['role'] == configured_role


def test_before_save_fails_closed_without_construction_role_authority():
    _, runner, hook, checkpoint = _initialized()
    authority = getattr(
        d13n_mode_hook_module, '_D13N_ROLE_BY_HOOK', None)
    assert authority is not None
    del authority[hook]
    with pytest.raises(RuntimeError, match='role authority'):
        hook.before_save_checkpoint(runner, checkpoint)
    assert 'd13n_integrity' not in checkpoint['meta']


@pytest.mark.parametrize('role', ['candidate', 'control'])
def test_mode_hook_public_role_inspection_preserves_config_value(role):
    hook = D13NParentEvalModeHook(role=role)
    assert type(hook.role) is str
    assert hook.role == role


@pytest.mark.parametrize('role', ['candidate', 'control'])
def test_registry_build_preserves_role_authority(role):
    hook = HOOKS.build(dict(type='D13NParentEvalModeHook', role=role))
    assert type(hook) is D13NParentEvalModeHook
    assert hook.role == role


@pytest.mark.parametrize('roundtrip', ['deepcopy', 'pickle'])
def test_hook_roundtrip_reconstructs_role_from_external_authority(roundtrip):
    _, runner, _, checkpoint = _initialized(role='candidate')
    hook = D13NParentEvalModeHook(role='candidate')
    hook.__dict__.update({
        'role': 'control',
        '_role': 'control',
        'configured_role': 'control',
        '_configured_role': 'control',
    })
    if roundtrip == 'deepcopy':
        restored = copy.deepcopy(hook)
    else:
        restored = pickle.loads(pickle.dumps(hook))
    assert restored is not hook
    assert restored.role == 'candidate'
    restored.before_save_checkpoint(runner, checkpoint)
    assert checkpoint['meta']['d13n_integrity']['role'] == 'candidate'
