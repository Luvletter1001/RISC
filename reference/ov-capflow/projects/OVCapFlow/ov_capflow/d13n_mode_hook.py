import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from weakref import WeakKeyDictionary

import torch
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper

from mmrotate.registry import HOOKS

from .freeze_except_hook import (D13N_PARAMETER_NAMES,
                                 require_d13n_trainable_parameters)


_ADAPTER_NAMES = D13N_PARAMETER_NAMES
_QUANTILE_NAMES = ('q05', 'q50', 'q95')
_ADAMW_GROUP_FIELDS = frozenset({
    'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach',
    'maximize', 'capturable'
})
_OPTIM_WRAPPER_STATE_FIELDS = frozenset({
    'state', 'param_groups', 'base_param_settings'
})
_ADAMW_STATE_FIELDS = ('step', 'exp_avg', 'exp_avg_sq')
_D13N_ROLE_BY_HOOK = WeakKeyDictionary()


def _validate_d13n_role(role):
    if type(role) is not str:
        raise TypeError('role must be control or candidate')
    if role not in ('control', 'candidate'):
        raise ValueError('role must be control or candidate')
    return role


def _require_d13n_role_authority(hook):
    try:
        role = _D13N_ROLE_BY_HOOK[hook]
    except KeyError as error:
        raise RuntimeError(
            'D13-N hook role authority is missing') from error
    return _validate_d13n_role(role)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _canonical_json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def hash_named_tensor_state(state):
    """Hash tensor state including its names, dtypes, shapes, and raw bytes."""
    if not isinstance(state, Mapping):
        raise TypeError('tensor state must be a mapping')
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise TypeError('tensor state must map string names to tensors')
        value = tensor.detach().cpu().contiguous()
        header = json.dumps({
            'name': name,
            'dtype': str(value.dtype),
            'shape': list(value.shape),
        }, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True).encode('utf-8')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, byteorder='big'))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, byteorder='big'))
        digest.update(raw)
    return digest.hexdigest()


def _require_checkpoint_model_state(model, checkpoint_state):
    if not isinstance(checkpoint_state, Mapping):
        raise RuntimeError('checkpoint state_dict must be a mapping')
    live_state = model.state_dict()
    if list(checkpoint_state) != list(live_state):
        raise RuntimeError('checkpoint state_dict keys differ from live model')
    for name, live_value in live_state.items():
        saved_value = checkpoint_state[name]
        if (not torch.is_tensor(saved_value) or
                saved_value.shape != live_value.shape or
                saved_value.dtype != live_value.dtype or
                not torch.equal(saved_value.detach().cpu(),
                                live_value.detach().cpu())):
            raise RuntimeError(
                'checkpoint state_dict tensor differs from live model: '
                f'{name}')


def _equal_serialized(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and
                left.shape == right.shape and left.dtype == right.dtype and
                torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if (not isinstance(left, Mapping) or not isinstance(right, Mapping) or
                type(left) is not type(right)):
            return False
        left_keys = list(left)
        right_keys = list(right)
        return (
            len(left_keys) == len(right_keys) and
            all(type(left_key) is type(right_key) and left_key == right_key
                for left_key, right_key in zip(left_keys, right_keys)) and
            all(_equal_serialized(left[left_key], right[right_key])
                for left_key, right_key in zip(left_keys, right_keys)))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_equal_serialized(a, b)
                    for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _validate_canonical_optimizer_tensor(value, source):
    canonical_stride = []
    running_stride = 1
    if type(value) is torch.Tensor:
        for size in reversed(value.shape):
            canonical_stride.append(running_stride)
            running_stride *= size
        canonical_stride = tuple(reversed(canonical_stride))
    if (type(value) is not torch.Tensor or value.requires_grad or
            value.grad is not None or value.is_inference() or
            value.layout != torch.strided or not value.is_contiguous() or
            value.stride() != canonical_stride or
            value._base is not None or value.storage_offset() != 0 or
            value.is_conj() or value.is_neg() or value.numel() <= 0):
        raise RuntimeError(
            f'optimizer {source} tensor is not canonical owned storage')
    try:
        storage_size = value.storage().size()
    except Exception as error:
        raise RuntimeError(
            f'optimizer {source} tensor storage is inaccessible') from error
    if storage_size != value.numel():
        raise RuntimeError(
            f'optimizer {source} tensor storage size is not canonical')


def _validate_adamw_step(value, source):
    _validate_canonical_optimizer_tensor(value, f'{source} AdamW step')
    if (value.ndim != 0 or
            value.dtype != torch.float32 or value.device.type != 'cpu'):
        raise RuntimeError(
            f'optimizer {source} AdamW step must be an exact no-grad CPU '
            'float32 scalar Tensor')
    if not torch.isfinite(value).item():
        raise RuntimeError(f'optimizer {source} AdamW step is non-finite')
    scalar = value.item()
    if scalar <= 0 or scalar != int(scalar):
        raise RuntimeError(
            f'optimizer {source} AdamW step must be a positive integer')


def _validate_adamw_moment(value, parameter, field, source):
    if source == 'live':
        expected_device = parameter.device
    elif source == 'serialized':
        expected_device = torch.device('cpu')
    else:
        raise RuntimeError('optimizer AdamW state source is invalid')
    _validate_canonical_optimizer_tensor(
        value, f'{source} AdamW {field}')
    if (value.shape != parameter.shape or
            value.dtype != parameter.dtype or
            value.device != expected_device):
        raise RuntimeError(
            f'optimizer {source} AdamW {field} exact Tensor/autograd/'
            'shape/dtype/device contract is invalid')
    if not torch.isfinite(value).all().item():
        raise RuntimeError(
            f'optimizer {source} AdamW {field} is non-finite')
    if field == 'exp_avg_sq' and not torch.ge(value, 0).all().item():
        raise RuntimeError(
            f'optimizer {source} AdamW exp_avg_sq must be nonnegative')


def _validate_adamw_state_slot(parameter_state, parameter, source):
    expected_fields = set(_ADAMW_STATE_FIELDS)
    if (type(parameter_state) is not dict or
            any(type(field) is not str for field in parameter_state) or
            set(parameter_state) != expected_fields):
        raise RuntimeError(
            f'optimizer {source} AdamW state fields must be exact')
    _validate_adamw_step(parameter_state['step'], source)
    _validate_adamw_moment(
        parameter_state['exp_avg'], parameter, 'exp_avg', source)
    _validate_adamw_moment(
        parameter_state['exp_avg_sq'], parameter, 'exp_avg_sq', source)


def _validate_live_adamw_state(state, expected_parameters):
    if (type(state) is not defaultdict or
            state.default_factory is not dict):
        raise RuntimeError(
            'optimizer live AdamW state must be exact defaultdict(dict)')
    state_parameters = list(state.keys())
    if (len(state_parameters) != len(expected_parameters) or
            any(actual is not expected for actual, expected in zip(
                state_parameters, expected_parameters))):
        raise RuntimeError(
            'optimizer live AdamW state parameter topology/order is invalid')
    slots = [state[parameter] for parameter in expected_parameters]
    for slot, parameter in zip(slots, expected_parameters):
        _validate_adamw_state_slot(slot, parameter, 'live')
    return slots


def _equal_optimizer_tensor_value(left, right):
    return (type(left) is torch.Tensor and type(right) is torch.Tensor and
            left.shape == right.shape and left.dtype == right.dtype and
            left.device == right.device and
            torch.equal(left.detach().cpu(), right.detach().cpu()))


def _strided_storage_key(tensor, source):
    if (not torch.is_tensor(tensor) or tensor.layout != torch.strided or
            tensor.numel() <= 0):
        raise RuntimeError(
            f'optimizer {source} tensor has invalid strided storage')
    try:
        pointer = tensor.storage().data_ptr()
    except Exception as error:
        raise RuntimeError(
            f'optimizer {source} tensor storage is inaccessible') from error
    if type(pointer) is not int or pointer == 0:
        raise RuntimeError(
            f'optimizer {source} tensor storage pointer is invalid')
    return tensor.device.type, tensor.device.index, pointer


def _forbidden_tensor_identity_storage(tensors, source):
    identities = set()
    storage_keys = set()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        identities.add(id(tensor))
        if tensor.layout == torch.strided and tensor.numel() > 0:
            storage_keys.add(_strided_storage_key(tensor, source))
    return identities, storage_keys


def _validate_optimizer_tensor_storage(
        state_tensors, base_dummy, forbidden_tensors, source):
    if len(state_tensors) != 6:
        raise RuntimeError(
            f'optimizer {source} must expose exactly six state tensors')
    forbidden_identities, forbidden_storage = (
        _forbidden_tensor_identity_storage(
            forbidden_tensors, f'{source} forbidden model state'))
    audited = [('base_param_settings.params', base_dummy), *state_tensors]
    identities = set()
    storage_keys = set()
    for name, tensor in audited:
        identity = id(tensor)
        storage_key = _strided_storage_key(tensor, f'{source} {name}')
        if identity in identities or storage_key in storage_keys:
            raise RuntimeError(
                f'optimizer {source} tensors must be pairwise storage-disjoint')
        if (identity in forbidden_identities or
                storage_key in forbidden_storage):
            raise RuntimeError(
                f'optimizer {source} tensor aliases registered model state')
        identities.add(identity)
        storage_keys.add(storage_key)


def _validate_finite_float(value, field, source, *, positive=False):
    if (type(value) is not float or not math.isfinite(value) or
            (value <= 0 if positive else value < 0)):
        domain = (
            'finite and positive' if positive else 'finite and nonnegative')
        raise RuntimeError(
            f'optimizer {source} AdamW {field} must be an exact built-in '
            f'float that is {domain}')


def _validate_adamw_settings(group, source):
    if type(group) is not dict:
        raise RuntimeError(
            f'optimizer {source} group/base settings must be an exact dict')
    fields = set(group)
    if (any(type(field) is not str for field in group) or fields not in (
            _ADAMW_GROUP_FIELDS,
            _ADAMW_GROUP_FIELDS | {'initial_lr'})):
        raise RuntimeError(
            f'optimizer {source} AdamW group schema is invalid')
    _validate_finite_float(group['lr'], 'lr', source)
    betas = group['betas']
    if (type(betas) is not tuple or len(betas) != 2 or
            any(type(beta) is not float or not math.isfinite(beta) or
                beta < 0 or beta >= 1
                for beta in betas)):
        raise RuntimeError(
            f'optimizer {source} AdamW betas must be an exact tuple of '
            'built-in finite floats in [0, 1)')
    _validate_finite_float(group['eps'], 'eps', source, positive=True)
    _validate_finite_float(
        group['weight_decay'], 'weight_decay', source)
    if group['amsgrad'] is not False:
        raise RuntimeError(
            f'optimizer {source} AdamW amsgrad must be exactly False')
    if group['foreach'] is not None:
        raise RuntimeError(
            f'optimizer {source} AdamW foreach must be exactly None')
    if group['maximize'] is not False:
        raise RuntimeError(
            f'optimizer {source} AdamW maximize must be exactly False')
    if group['capturable'] is not False:
        raise RuntimeError(
            f'optimizer {source} AdamW capturable must be exactly False')
    if 'initial_lr' in group:
        _validate_finite_float(
            group['initial_lr'], 'initial_lr', source)


def _validate_adamw_group(group, source):
    _validate_adamw_settings(group, source)
    if type(group['params']) is not list or len(group['params']) != 1:
        raise RuntimeError(
            f'optimizer {source} AdamW group topology must be [1, 1]')


def _validate_adamw_groups(groups, source):
    if type(groups) is not list or len(groups) != 2:
        raise RuntimeError(
            f'optimizer {source} AdamW group topology must be exactly [1, 1]')
    for group in groups:
        _validate_adamw_group(group, source)
    first_fields = set(groups[0]) - {'params'}
    second_fields = set(groups[1]) - {'params'}
    if (first_fields != second_fields or
            any(not _equal_serialized(groups[0][field], groups[1][field])
                for field in first_fields)):
        raise RuntimeError(
            f'optimizer {source} AdamW group settings must be identical')


def _validate_adamw_base_settings(base, groups, source):
    _validate_adamw_settings(base, source)
    dummy = base['params']
    _validate_canonical_optimizer_tensor(
        dummy, f'{source} base_param_settings.params')
    if (dummy.shape != torch.Size([1]) or
            dummy.dtype != torch.float32 or dummy.device.type != 'cpu' or
            not torch.equal(dummy, torch.zeros(1))):
        raise RuntimeError(
            f'optimizer {source} base params must be the exact CPU float32 '
            'zero dummy tensor')
    base_fields = set(base) - {'params'}
    for group in groups:
        group_fields = set(group) - {'params'}
        if (base_fields != group_fields or
                any(not _equal_serialized(base[field], group[field])
                    for field in base_fields)):
            raise RuntimeError(
                f'optimizer {source} base settings must match real groups')


def _validate_wrapper_state(state, source):
    if (type(state) is not dict or
            any(type(field) is not str for field in state) or
            set(state) != _OPTIM_WRAPPER_STATE_FIELDS):
        raise RuntimeError(
            f'optimizer {source} wrapper state schema is invalid')
    if type(state['state']) is not dict:
        raise RuntimeError(
            f'optimizer {source} state must be an exact dict')
    if any(type(slot) is not dict for slot in state['state'].values()):
        raise RuntimeError(
            f'optimizer {source} state slots must be exact dicts')


def _validate_optimizer_state_dict_payload(
        payload, live_groups, source, tensor_source):
    _validate_wrapper_state(payload, source)
    groups = payload['param_groups']
    state = payload['state']
    _validate_adamw_groups(groups, source)
    _validate_adamw_base_settings(
        payload['base_param_settings'], groups, source)

    parameter_ids_by_group = []
    id_to_parameter = {}
    for live_group, saved_group in zip(live_groups, groups):
        parameter_ids = saved_group['params']
        if any(type(parameter_id) is not int
               for parameter_id in parameter_ids):
            raise RuntimeError(
                f'optimizer {source} parameter IDs must be exact integers')
        parameter_ids_by_group.append(list(parameter_ids))
        for parameter_id, parameter in zip(
                parameter_ids, live_group['params']):
            if parameter_id in id_to_parameter:
                raise RuntimeError(
                    f'optimizer {source} parameter IDs must be unique')
            id_to_parameter[parameter_id] = parameter

    flattened_ids = [
        parameter_id for group in parameter_ids_by_group
        for parameter_id in group]
    if (len(flattened_ids) != 2 or
            any(type(parameter_id) is not int for parameter_id in state) or
            list(state) != flattened_ids):
        raise RuntimeError(
            f'optimizer {source} must have exact ordered integer state IDs')

    state_tensors = []
    for parameter_id in flattened_ids:
        parameter_state = state[parameter_id]
        _validate_adamw_state_slot(
            parameter_state, id_to_parameter[parameter_id], tensor_source)
        state_tensors.extend(
            (f'{parameter_id}.{field}', parameter_state[field])
            for field in _ADAMW_STATE_FIELDS)
    return groups, state, parameter_ids_by_group, state_tensors


def _optimizer_integrity(
        model, optim_wrapper, serialized_optimizer, checkpoint_state):
    if not isinstance(optim_wrapper, OptimWrapper):
        raise RuntimeError('D13-N optimizer must be an MMEngine OptimWrapper')
    optimizer = optim_wrapper.optimizer
    if type(optimizer) is not torch.optim.AdamW:
        raise RuntimeError('D13-N optimizer must be exact torch.optim.AdamW')
    _validate_adamw_groups(optimizer.param_groups, 'live')
    live_base = getattr(optim_wrapper, 'base_param_settings', None)
    _validate_adamw_base_settings(
        live_base, optimizer.param_groups, 'live authority')
    audited_parameters = require_d13n_trainable_parameters(model)
    name_by_identity = {
        id(parameter): name
        for name, parameter in zip(_ADAPTER_NAMES, audited_parameters)
    }
    names_by_group = []
    live_parameters = []
    seen_identities = set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group.get('params', []):
            identity = id(parameter)
            if identity not in name_by_identity:
                raise RuntimeError('optimizer contains an unknown live tensor')
            if identity in seen_identities:
                raise RuntimeError(
                    'optimizer contains a duplicate live tensor')
            seen_identities.add(identity)
            group_names.append(name_by_identity[identity])
            live_parameters.append(parameter)
        names_by_group.append(group_names)
    expected_names_by_group = [[name] for name in _ADAPTER_NAMES]
    if names_by_group != expected_names_by_group:
        raise RuntimeError(
            'optimizer groups must contain exact ordered D13-N adapter names')
    if sum(parameter.numel() for parameter in live_parameters) != 257:
        raise RuntimeError(
            'optimizer must contain exactly 257 trainable scalars')
    live_state_slots = _validate_live_adamw_state(
        optimizer.state, live_parameters)
    live_state_tensors = []
    for parameter, parameter_state in zip(
            live_parameters, live_state_slots):
        parameter_name = name_by_identity[id(parameter)]
        live_state_tensors.extend(
            (f'{parameter_name}.{field}', parameter_state[field])
            for field in _ADAMW_STATE_FIELDS)
    _validate_optimizer_tensor_storage(
        live_state_tensors, live_base['params'],
        [*model.parameters(), *model.buffers()], 'live')

    live_serialized = optim_wrapper.state_dict()
    _, live_serialized_state, live_serialized_ids, _ = (
        _validate_optimizer_state_dict_payload(
            live_serialized, optimizer.param_groups, 'live state_dict',
            'live'))
    if not _equal_serialized(
            live_serialized['base_param_settings'], live_base):
        raise RuntimeError(
            'optimizer live state_dict base differs from live authority')
    for raw_slot, parameter_ids in zip(
            live_state_slots, live_serialized_ids):
        saved_slot = live_serialized_state[parameter_ids[0]]
        if any(not _equal_optimizer_tensor_value(
                raw_slot[field], saved_slot[field])
                for field in _ADAMW_STATE_FIELDS):
            raise RuntimeError(
                'optimizer live state_dict slot differs from raw live state')

    groups, state, serialized_ids_by_group, serialized_state_tensors = (
        _validate_optimizer_state_dict_payload(
            serialized_optimizer, optimizer.param_groups, 'serialized',
            'serialized'))
    if not _equal_serialized(serialized_optimizer, live_serialized):
        raise RuntimeError(
            'serialized optimizer differs from the live optimizer state')
    _validate_optimizer_tensor_storage(
        serialized_state_tensors,
        serialized_optimizer['base_param_settings']['params'],
        checkpoint_state.values(), 'serialized')
    return {
        'optimizer_parameter_names_by_group': names_by_group,
        'optimizer_class': (
            f'{optimizer.__class__.__module__}.'
            f'{optimizer.__class__.__qualname__}'),
        'live_optimizer_group_sizes': [
            len(group['params']) for group in optimizer.param_groups],
        'serialized_optimizer_group_sizes': [
            len(group) for group in serialized_ids_by_group],
        'serialized_optimizer_parameter_ids_by_group': serialized_ids_by_group,
        'initialized_optimizer_state_slots': len(state),
    }


def _checkpoint_runtime_provenance(runner, meta):
    runner_epoch = getattr(runner, 'epoch', None)
    runner_iteration = getattr(runner, 'iter', None)
    if (type(runner_epoch) is not int or runner_epoch < 0 or
            type(runner_iteration) is not int or runner_iteration < 0):
        raise RuntimeError(
            'D13-N runner epoch/iter must be nonnegative integers')
    runner_config = getattr(runner, 'cfg', None)
    config = getattr(runner_config, 'pretty_text', None)
    if type(config) is not str or not config:
        raise RuntimeError(
            'D13-N runner cfg.pretty_text must be a non-empty string')
    epoch = meta.get('epoch')
    iteration = meta.get('iter')
    if type(epoch) is not int or epoch != runner_epoch + 1:
        raise RuntimeError(
            'D13-N checkpoint epoch must equal runner.epoch + 1')
    if type(iteration) is not int or iteration != runner_iteration:
        raise RuntimeError(
            'D13-N checkpoint iter must equal runner.iter')
    checkpoint_config = meta.get('cfg')
    if (type(checkpoint_config) is not str or
            checkpoint_config != config):
        raise RuntimeError(
            'D13-N checkpoint cfg must equal runner.cfg.pretty_text')
    return epoch, iteration, config


def _validate_existing_integrity(existing, expected):
    if type(existing) is not dict:
        raise RuntimeError(
            'existing D13-N checkpoint integrity must be an exact dict')
    existing_keys = list(existing)
    expected_keys = list(expected)
    if (len(existing_keys) != len(expected_keys) or
            any(type(existing_key) is not type(expected_key) or
                existing_key != expected_key
                for existing_key, expected_key in zip(
                    existing_keys, expected_keys))):
        raise RuntimeError(
            'existing D13-N checkpoint integrity schema/order is invalid')
    digest = existing.get('integrity_sha256')
    if (type(digest) is not str or len(digest) != 64 or
            any(character not in '0123456789abcdef' for character in digest)):
        raise RuntimeError(
            'existing D13-N checkpoint integrity digest is not lowercase hex')
    unhashed = {
        key: value for key, value in existing.items()
        if key != 'integrity_sha256'
    }
    try:
        recomputed = _canonical_json_sha256(unhashed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            'existing D13-N checkpoint integrity is not canonical JSON') \
            from error
    if digest != recomputed:
        raise RuntimeError(
            'existing D13-N checkpoint integrity self-hash is invalid')
    if not _equal_serialized(existing, expected):
        raise RuntimeError(
            'existing D13-N checkpoint integrity has typed value drift')


@HOOKS.register_module()
class D13NParentEvalModeHook(Hook):
    """Keep the frozen parent deterministic and bind optimizer integrity."""

    def __init__(self, role):
        _D13N_ROLE_BY_HOOK[self] = _validate_d13n_role(role)

    @property
    def role(self):
        return _require_d13n_role_authority(self)

    def __deepcopy__(self, memo):
        restored = type(self)(self.role)
        memo[id(self)] = restored
        return restored

    def __reduce__(self):
        return type(self), (self.role, )

    @staticmethod
    def _require_head(model):
        head = getattr(model, 'bbox_head', None)
        adapter = getattr(head, 'existence_residual', None)
        if head is None or adapter is None:
            raise RuntimeError('D13-N existence_residual head is required')
        return head, adapter

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = _unwrap_model(runner.model)
        head, adapter = self._require_head(model)
        model.training = True
        for child in model.children():
            child.eval()
        if head.training:
            raise RuntimeError(
                'D13-N parent bbox_head must remain in eval mode')
        adapter.train(True)

    @staticmethod
    def _require_telemetry_tensor(value, name, shape_tail):
        if not torch.is_tensor(value):
            raise RuntimeError(f'D13-N telemetry {name} is missing')
        if value.requires_grad or value.grad_fn is not None:
            raise RuntimeError(f'D13-N telemetry {name} must be detached')
        if tuple(value.shape[-len(shape_tail):]) != tuple(shape_tail):
            raise RuntimeError(f'D13-N telemetry {name} has invalid shape')
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f'D13-N telemetry {name} must be finite')
        return value.cpu()

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        model = _unwrap_model(runner.model)
        head, _ = self._require_head(model)
        matched = self._require_telemetry_tensor(
            head.last_matching_group_matched_counts,
            'matching_group_matched_counts', (3, ))
        if matched.ndim != 2 or matched.shape[0] < 1:
            raise RuntimeError(
                'D13-N telemetry matching counts must have shape [B,3]')
        logit_quantiles = self._require_telemetry_tensor(
            head.last_existence_logit_quantiles,
            'existence_logit_quantiles', (3, 3))
        residual_quantiles = self._require_telemetry_tensor(
            head.last_existence_residual_quantiles,
            'existence_residual_quantiles', (3, 3))
        clamp_hits = self._require_telemetry_tensor(
            head.last_existence_clamp_hit_counts,
            'existence_clamp_hit_counts', (2, ))
        if logit_quantiles.ndim != 2 or residual_quantiles.ndim != 2:
            raise RuntimeError(
                'D13-N quantile telemetry must have shape [3,3]')
        if clamp_hits.ndim != 1:
            raise RuntimeError('D13-N clamp telemetry must have shape [2]')

        for group_index, value in enumerate(matched.float().mean(dim=0)):
            runner.message_hub.update_scalar(
                f'd13n/matched_count_g{group_index}', value.item())
        for prefix, quantiles in (
                ('existence_logit', logit_quantiles),
                ('existence_residual', residual_quantiles)):
            for group_index in range(3):
                for quantile_index, quantile_name in enumerate(
                        _QUANTILE_NAMES):
                    runner.message_hub.update_scalar(
                        f'd13n/{prefix}_g{group_index}_{quantile_name}',
                        quantiles[group_index, quantile_index].item())
        runner.message_hub.update_scalar(
            'd13n/clamp_hit_lower', clamp_hits[0].item())
        runner.message_hub.update_scalar(
            'd13n/clamp_hit_upper', clamp_hits[1].item())

    def before_save_checkpoint(self, runner, checkpoint):
        role = _require_d13n_role_authority(self)
        if not isinstance(checkpoint, dict):
            raise RuntimeError('D13-N checkpoint must be a dictionary')
        model = _unwrap_model(runner.model)
        self._require_head(model)
        require_d13n_trainable_parameters(model)
        state_dict = checkpoint.get('state_dict')
        _require_checkpoint_model_state(model, state_dict)
        optimizer_fields = _optimizer_integrity(
            model, runner.optim_wrapper, checkpoint.get('optimizer'),
            state_dict)

        meta = checkpoint.get('meta')
        if not isinstance(meta, dict):
            raise RuntimeError('D13-N checkpoint meta must be a dictionary')
        epoch, iteration, config = _checkpoint_runtime_provenance(runner, meta)
        if any(name not in state_dict for name in _ADAPTER_NAMES):
            raise RuntimeError(
                'D13-N checkpoint state_dict lacks adapter tensors')
        adapter_state = {name: state_dict[name] for name in _ADAPTER_NAMES}
        parent_state = {name: value for name, value in state_dict.items()
                        if name not in _ADAPTER_NAMES}
        integrity = {
            'schema': 'd13n-checkpoint-integrity-v1',
            'role': role,
            'epoch': epoch,
            'iter': iteration,
            **optimizer_fields,
            'parent_state_sha256': hash_named_tensor_state(parent_state),
            'adapter_state_sha256': hash_named_tensor_state(adapter_state),
            'config_sha256': hashlib.sha256(
                config.encode('utf-8')).hexdigest(),
        }
        integrity['integrity_sha256'] = _canonical_json_sha256(integrity)
        if 'd13n_integrity' in meta:
            _validate_existing_integrity(meta['d13n_integrity'], integrity)
        meta['d13n_integrity'] = integrity
