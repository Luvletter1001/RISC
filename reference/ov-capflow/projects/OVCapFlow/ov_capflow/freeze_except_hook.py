import re

from mmengine.hooks import Hook
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS

from mmrotate.registry import HOOKS


D13N_PARAMETER_NAMES = (
    'bbox_head.existence_residual.weight',
    'bbox_head.existence_residual.bias',
)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def all_parameter_registration_paths(model):
    """Return every parameter path, including shared-module aliases.

    PyTorch 1.12 deduplicates ``named_parameters`` and has no public
    ``remove_duplicate`` argument for that method. Traversing every module
    path without module deduplication resets the parameter memo per module.
    The direct registry fallback also exposes two names for one parameter in
    the same module, which the public method still deduplicates.
    """
    registrations = []
    for module_path, module in model.named_modules(remove_duplicate=False):
        public_names = set()
        for local_name, parameter in module.named_parameters(recurse=False):
            public_names.add(local_name)
            full_name = (f'{module_path}.{local_name}'
                         if module_path else local_name)
            registrations.append((full_name, parameter))
        for local_name, parameter in module._parameters.items():
            if parameter is None or local_name in public_names:
                continue
            full_name = (f'{module_path}.{local_name}'
                         if module_path else local_name)
            registrations.append((full_name, parameter))
    return registrations


def _registrations_by_identity(model):
    grouped = {}
    for path, parameter in all_parameter_registration_paths(model):
        identity = id(parameter)
        if identity not in grouped:
            grouped[identity] = {'parameter': parameter, 'paths': []}
        grouped[identity]['paths'].append(path)
    return grouped


def require_d13n_trainable_parameters(model):
    """Audit every registration path and return the two canonical tensors."""
    grouped = _registrations_by_identity(model)
    path_entries = {}
    for entry in grouped.values():
        for path in entry['paths']:
            path_entries.setdefault(path, []).append(entry)

    expected = []
    for name in D13N_PARAMETER_NAMES:
        entries = path_entries.get(name, [])
        if len(entries) != 1:
            raise RuntimeError(
                f'D13-N canonical parameter registration is invalid: {name}')
        entry = entries[0]
        if entry['paths'] != [name]:
            raise RuntimeError(
                f'D13-N adapter parameter has a registration alias: {name}')
        expected.append(entry['parameter'])
    if len({id(parameter) for parameter in expected}) != 2:
        raise RuntimeError('D13-N canonical parameter registrations alias')

    trainable_names = []
    for entry in grouped.values():
        if not entry['parameter'].requires_grad:
            continue
        paths = entry['paths']
        if len(paths) != 1:
            raise RuntimeError(
                'D13-N trainable parameter has multiple registration aliases')
        trainable_names.append(paths[0])
    if trainable_names != list(D13N_PARAMETER_NAMES):
        raise RuntimeError(
            'D13-N trainable parameters must be exact ordered adapter names')
    if sum(parameter.numel() for parameter in expected) != 257:
        raise RuntimeError(
            'D13-N trainable parameters must be two distinct 257-scalar '
            'adapter tensors')
    return expected


def apply_freeze_except(model, trainable_patterns):
    """Freeze all parameters except a fail-closed regular-expression list."""
    if not trainable_patterns:
        raise ValueError('trainable_patterns must be non-empty')
    patterns = [re.compile(pattern) for pattern in trainable_patterns]
    grouped = _registrations_by_identity(model)
    matched = []
    freeze_plan = []
    for entry in grouped.values():
        decisions = [any(pattern.search(path) for pattern in patterns)
                     for path in entry['paths']]
        if len(set(decisions)) != 1:
            raise RuntimeError(
                'parameter registration alias crosses the freeze allowlist')
        enabled = decisions[0]
        if enabled and len(entry['paths']) != 1:
            raise RuntimeError(
                'trainable parameter has multiple registration aliases')
        freeze_plan.append((entry['parameter'], enabled))
        if enabled:
            matched.extend(entry['paths'])
    if not matched:
        raise RuntimeError('trainable patterns matched no parameters')
    for parameter, enabled in freeze_plan:
        parameter.requires_grad_(enabled)
    return matched


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class D13NOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Build an optimizer containing only the frozen D13-N adapter."""

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        if paramwise_cfg is not None:
            raise ValueError(
                'D13NOptimWrapperConstructor forbids caller paramwise_cfg')
        super().__init__(
            optim_wrapper_cfg,
            paramwise_cfg={'bypass_duplicate': True})

    def __call__(self, model):
        unwrapped = _unwrap_model(model)
        expected = require_d13n_trainable_parameters(unwrapped)
        optim_wrapper = super().__call__(model)
        optim_wrapper.optimizer.param_groups = [
            dict(group) for group in optim_wrapper.optimizer.param_groups]
        audited = require_d13n_trainable_parameters(unwrapped)
        if any(before is not after
               for before, after in zip(expected, audited)):
            raise RuntimeError(
                'D13-N parameter registrations changed during construction')
        actual = [
            parameter
            for group in optim_wrapper.optimizer.param_groups
            for parameter in group['params']
        ]
        if (len(actual) != 2 or
                any(got is not wanted
                    for got, wanted in zip(actual, audited))):
            raise RuntimeError(
                'D13-N optimizer parameters differ from ordered adapter '
                'tensors')
        return optim_wrapper


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze every parameter except an explicitly matched allowlist."""

    def __init__(self, trainable_patterns):
        self.trainable_patterns = list(trainable_patterns)

    def apply(self, model):
        return apply_freeze_except(model, self.trainable_patterns)

    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') \
            else runner.model
        matched = self.apply(model)
        runner.logger.info('trainable_parameters=%s', matched)
