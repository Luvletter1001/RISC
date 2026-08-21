#!/usr/bin/env python3
"""Run the fail-closed, zero-update Stage-0 audit for D12.

The audit verifies the complete checkpoint delta, resolved config parity,
raw normalized Q600 predictions on one fixed real batch, finite real-GT
losses and gradients without an optimizer step, and the existing strict and
open-vocabulary audits.  Final JSON outputs are published without replacement.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

from projects.OVCapFlow.tools.prepare_cleanstart_checkpoint import (
    convert_groundingdino_state_dict,
    reject_forbidden_namespaces,
    sha256_file,
    unwrap_model_state,
)
from projects.OVCapFlow.tools.prepare_d12_checkpoint_pair import (
    AUTHORIZED_POSITION_COUNT,
    publish_artifacts_no_clobber,
    validate_pair_delta,
)


SATURATION_EPSILON = 1e-6
MAX_SATURATION_FRACTION = 0.05
MAX_SATURATION_INCREASE = 0.05
MIN_MEDIAN_AREA_RATIO = 1.0 / 16.0
MAX_MEDIAN_AREA_RATIO = 16.0
MAX_P99_AREA = 1.0
MAX_TOTAL_LOSS_RATIO = 20.0
MAX_GRADIENT_NORM_RATIO = 100.0
MAX_QUERY_AREA = 50_000_000
EXPECTED_QUERIES = 600
SEED = 20260716
DEFAULT_MANIFEST = Path(
    'work_dirs/dotav2_cleanstart/checkpoints/'
    'groundingdino_swint_ogc_q600_d12_w5_pair_manifest.json')


def _selected_keys(
        state: Mapping[str, Tensor], predicate) -> list[str]:
    return sorted(key for key in state if predicate(key))


def _keys_equal(
        control: Mapping[str, Tensor],
        candidate: Mapping[str, Tensor],
        keys: Iterable[str]) -> bool:
    return all(torch.equal(control[key], candidate[key]) for key in keys)


def compare_complete_states(
        control: Mapping[str, Tensor],
        candidate: Mapping[str, Tensor],
        converted_source: Mapping[str, Tensor]) -> dict[str, Any]:
    """Validate the D12 delta plus named state families required by the spec."""
    report = validate_pair_delta(control, candidate, converted_source)
    families = {
        'query_state': _selected_keys(
            control,
            lambda key: 'query_initializer.query_embedding' in key),
        'reference_state': _selected_keys(
            control,
            lambda key: 'query_initializer.reference_embedding' in key),
        'dn_state': _selected_keys(
            control,
            lambda key: key.startswith('dn_query_generator.')),
        'classification_bias': _selected_keys(
            control,
            lambda key: (
                'cls_branches.' in key and key.endswith('.bias'))),
    }
    family_checks = {}
    for name, keys in families.items():
        equal = _keys_equal(control, candidate, keys)
        if not equal:
            raise ValueError(name + ' differs outside allowed transport')
        family_checks[name + '_equal'] = True
        family_checks[name + '_keys'] = keys
    report.update(family_checks)
    report['pass'] = True
    return report


def summarize_normalized_cxywh(
        boxes: Tensor,
        epsilon: float = SATURATION_EPSILON) -> dict[str, Any]:
    """Validate and summarize raw normalized rotated boxes."""
    if not isinstance(boxes, Tensor):
        raise TypeError('boxes must be a tensor')
    if boxes.ndim < 2 or boxes.shape[-1] != 5:
        raise ValueError('boxes last dimension must be exactly 5')
    if boxes.numel() == 0:
        raise ValueError('boxes must be non-empty')
    boxes = boxes.detach().float().cpu()
    if not torch.isfinite(boxes).all().item():
        raise ValueError('all box values must be finite')
    cxywh = boxes[..., :4]
    if not ((cxywh >= 0).all() and (cxywh <= 1).all()):
        raise ValueError('normalized cxywh must lie in [0, 1]')
    if not (cxywh[..., 2:4] > 0).all().item():
        raise ValueError('normalized width and height must be strictly positive')

    saturation = ((cxywh <= epsilon) | (cxywh >= 1 - epsilon)).float()
    areas = (cxywh[..., 2] * cxywh[..., 3]).reshape(-1)
    return {
        'prediction_count': int(areas.numel()),
        'coordinate_count': int(cxywh.numel()),
        'saturation_fraction': float(saturation.mean().item()),
        'median_area': float(torch.quantile(areas, 0.5).item()),
        'p99_area': float(torch.quantile(areas, 0.99).item()),
        'min_cxywh': float(cxywh.min().item()),
        'max_cxywh': float(cxywh.max().item()),
        'all_finite': True,
        'all_cxywh_in_unit_interval': True,
        'all_width_height_positive': True,
    }


def _require_q600(summary: Mapping[str, Any], role: str) -> None:
    counts = [int(value) for value in summary.get('per_image_counts', ())]
    if not counts or any(count != EXPECTED_QUERIES for count in counts):
        raise ValueError(role + ' prediction is not strict Q600')


def enforce_prediction_pair(
        candidate: Mapping[str, Any],
        control: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen zero-update box-distribution thresholds."""
    _require_q600(candidate, 'candidate')
    _require_q600(control, 'control')
    for role, summary in (('candidate', candidate), ('control', control)):
        for key in (
                'all_finite',
                'all_cxywh_in_unit_interval',
                'all_width_height_positive'):
            if not bool(summary.get(key)):
                raise ValueError(role + ' box validity check failed: ' + key)

    candidate_saturation = float(candidate['saturation_fraction'])
    control_saturation = float(control['saturation_fraction'])
    if (not math.isfinite(candidate_saturation) or
            candidate_saturation > MAX_SATURATION_FRACTION):
        raise ValueError('candidate saturation fraction exceeds 0.05')
    saturation_increase = candidate_saturation - control_saturation
    if (not math.isfinite(saturation_increase) or
            saturation_increase > MAX_SATURATION_INCREASE):
        raise ValueError('candidate saturation increase exceeds 0.05')

    candidate_median = float(candidate['median_area'])
    control_median = float(control['median_area'])
    if not (math.isfinite(candidate_median) and candidate_median > 0):
        raise ValueError('candidate median area must be finite and positive')
    if not (math.isfinite(control_median) and control_median > 0):
        raise ValueError('control median area must be finite and positive')
    median_ratio = candidate_median / control_median
    if not MIN_MEDIAN_AREA_RATIO <= median_ratio <= MAX_MEDIAN_AREA_RATIO:
        raise ValueError('candidate/control median area ratio is outside bounds')

    candidate_p99 = float(candidate['p99_area'])
    if not math.isfinite(candidate_p99) or candidate_p99 > MAX_P99_AREA:
        raise ValueError('candidate p99 area exceeds 1')
    return {
        'pass': True,
        'candidate_saturation_fraction': candidate_saturation,
        'control_saturation_fraction': control_saturation,
        'saturation_increase': saturation_increase,
        'median_area_ratio': median_ratio,
        'candidate_p99_area': candidate_p99,
    }


def _finite_loss_dict(stats: Mapping[str, Any]) -> bool:
    losses = stats.get('individual_losses', {})
    return (
        isinstance(losses, Mapping)
        and bool(losses)
        and all(math.isfinite(float(value)) for value in losses.values()))


def enforce_loss_gradient_pair(
        candidate: Mapping[str, Any],
        control: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen real-GT no-optimizer loss/gradient thresholds."""
    if not _finite_loss_dict(candidate) or not _finite_loss_dict(control):
        raise ValueError('candidate and control must have finite losses')
    candidate_total = float(candidate['total_loss'])
    control_total = float(control['total_loss'])
    if not math.isfinite(control_total) or control_total <= 0:
        raise ValueError('positive control total loss is required')
    if not math.isfinite(candidate_total) or candidate_total <= 0:
        raise ValueError('positive finite candidate total loss is required')
    total_ratio = candidate_total / control_total
    if total_ratio > MAX_TOTAL_LOSS_RATIO:
        raise ValueError('candidate total loss ratio exceeds 20')

    candidate_grad = float(candidate['gradient_norm'])
    control_grad = float(control['gradient_norm'])
    if not math.isfinite(control_grad) or control_grad <= 0:
        raise ValueError('positive control gradient norm is required')
    if not math.isfinite(candidate_grad) or candidate_grad <= 0:
        raise ValueError('positive finite candidate gradient norm is required')
    gradient_ratio = candidate_grad / control_grad
    if gradient_ratio > MAX_GRADIENT_NORM_RATIO:
        raise ValueError('candidate gradient norm ratio exceeds 100')

    for role, stats in (('candidate', candidate), ('control', control)):
        if not bool(stats.get('gradients_finite')):
            raise ValueError(role + ' must have finite gradients')
        if not bool(stats.get('parameters_unchanged')):
            raise ValueError(role + ' parameter changed without optimizer')
        if not bool(stats.get('prediction_succeeds_after_backward')):
            raise ValueError(role + ' prediction after backward failed')
        query_area = int(stats.get('max_estimated_query_area', -1))
        if query_area < 0 or query_area > MAX_QUERY_AREA:
            raise ValueError(role + ' query area exceeds frozen budget')
    return {
        'pass': True,
        'total_loss_ratio': total_ratio,
        'gradient_norm_ratio': gradient_ratio,
    }


def assert_deterministic_tensors(
        first: Sequence[Tensor], second: Sequence[Tensor]) -> bool:
    """Require bitwise equality for a repeated tensor-producing operation."""
    if len(first) != len(second):
        raise ValueError('repeated inference is not deterministic')
    for first_value, second_value in zip(first, second):
        if (not isinstance(first_value, Tensor)
                or not isinstance(second_value, Tensor)
                or not torch.equal(first_value, second_value)):
            raise ValueError('repeated inference is not deterministic')
    return True


def publish_json_no_clobber(report: Mapping[str, Any], output: Path) -> None:
    payload = (
        json.dumps(report, indent=2, sort_keys=True) + '\n').encode('utf-8')
    publish_artifacts_no_clobber([(Path(output), payload)])


def _load_model_only_state(path: Path) -> OrderedDict[str, Tensor]:
    checkpoint = torch.load(str(path), map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise TypeError('checkpoint root must be a mapping')
    if set(checkpoint) != {'state_dict'}:
        raise ValueError('D12 checkpoint must contain only state_dict')
    state = checkpoint['state_dict']
    if not isinstance(state, Mapping):
        raise TypeError('state_dict must be a mapping')
    normalized = OrderedDict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError('state_dict must map string keys to tensors')
        normalized[key] = value.detach().cpu()
    reject_forbidden_namespaces(normalized)
    return normalized


def _validate_manifest(
        manifest_path: Path,
        source: Path,
        config: Path,
        control_checkpoint: Path,
        candidate_checkpoint: Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    expected = {
        'source': (Path(source), manifest['source']),
        'config': (Path(config), manifest['config']),
        'control': (
            Path(control_checkpoint), manifest['outputs']['control']),
        'candidate': (
            Path(candidate_checkpoint), manifest['outputs']['candidate']),
    }
    for label, (path, record) in expected.items():
        if Path(record['path']).resolve() != path.resolve():
            raise ValueError(label + ' manifest path mismatch')
        if record['sha256'] != sha256_file(path):
            raise ValueError(label + ' manifest SHA256 mismatch')
        if int(record['size_bytes']) != path.stat().st_size:
            raise ValueError(label + ' manifest size mismatch')
    transport = manifest['transport']
    if int(transport['authorized_position_count']) != (
            AUTHORIZED_POSITION_COUNT):
        raise ValueError('manifest authorized position count mismatch')
    for name in (
            'query_transport',
            'reference_transport',
            'dn_transport',
            'classification_transport'):
        if transport[name] != {'enabled': False}:
            raise ValueError('manifest enables forbidden transport: ' + name)
    if not all(bool(value) for value in manifest['checks'].values()):
        raise ValueError('manifest contains a failed pair check')
    return {
        'pass': True,
        'path': str(Path(manifest_path).resolve()),
        'sha256': sha256_file(manifest_path),
        'git_commit': manifest['git_commit'],
        'actual_unequal_position_count': int(
            transport['actual_unequal_position_count']),
    }


def _normalized_config_for_pair(cfg: Config) -> dict[str, Any]:
    normalized = cfg.to_dict()
    for key in (
            'physical_gpus',
            'd12_role',
            'd12_only_scientific_delta',
            'load_from',
            'work_dir'):
        normalized.pop(key, None)
    normalized['train_dataloader']['batch_sampler'].pop('audit_path', None)
    return normalized


def _validate_config_pair(
        candidate_cfg: Config, control_cfg: Config) -> dict[str, Any]:
    if _normalized_config_for_pair(candidate_cfg) != (
            _normalized_config_for_pair(control_cfg)):
        raise ValueError('candidate/control resolved configs are not paired')
    for cfg in (candidate_cfg, control_cfg):
        if int(cfg.model.num_queries) != EXPECTED_QUERIES:
            raise ValueError('resolved config is not Q600')
        if int(cfg.model.train_query_groups) != 3:
            raise ValueError('resolved config must use three train groups')
        if int(cfg.model.bbox_head.matching_query_groups) != 3:
            raise ValueError('resolved head must use three matching groups')
        if int(cfg.model.decoder.num_layers) != 6:
            raise ValueError('resolved decoder must have six layers')
        if int(cfg.selected_world_size) != 5:
            raise ValueError('resolved config must use world size five')
        if int(cfg.train_dataloader.batch_size) != 2:
            raise ValueError('resolved config must use per-rank batch two')
        sampler = cfg.train_dataloader.batch_sampler
        if sampler.type != 'DNQueryBudgetBatchSampler':
            raise ValueError('resolved config lost typed batch sampler')
        if int(sampler.num_matching_queries) != 1800:
            raise ValueError('sampler must budget three Q600 groups')
        if int(sampler.num_dn_queries) != 100:
            raise ValueError('sampler DN budget must remain 100')
        if int(sampler.max_query_area) != MAX_QUERY_AREA:
            raise ValueError('sampler query-area budget changed')
        if int(sampler.update_count_multiple) != 1:
            raise ValueError('sampler update multiple must be one')
        if int(cfg.optim_wrapper.accumulative_counts) != 1:
            raise ValueError('gradient accumulation must be one')
        if bool(cfg.resume):
            raise ValueError('D12 config must not resume')
    return {
        'pass': True,
        'candidate_physical_gpus': list(candidate_cfg.physical_gpus),
        'control_physical_gpus': list(control_cfg.physical_gpus),
        'world_size': 5,
        'per_rank_batch_size': 2,
        'global_batch_size': 10,
        'dataset_size': 1600,
        'updates_per_epoch': 160,
        'num_queries': 600,
        'train_query_groups': 3,
        'sampler_matching_query_budget': 1800,
        'dn_query_budget': 100,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_real_batch_with_gt(cfg: Config):
    dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader_cfg.batch_sampler.audit_path = None
    dataloader = Runner.build_dataloader(dataloader_cfg)
    for _, raw_batch in zip(range(len(dataloader)), dataloader):
        total_gt = sum(
            len(sample.gt_instances)
            for sample in raw_batch['data_samples'])
        if total_gt > 0:
            return raw_batch, total_gt
    raise RuntimeError('no real non-empty-GT batch found')


def _build_loaded_model(
        cfg: Config,
        state: Mapping[str, Tensor],
        device: str):
    model = MODELS.build(copy.deepcopy(cfg.model))
    model.init_weights()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError('strict checkpoint load reported incompatible keys')
    return model.to(device)


def _parameter_versions(model) -> dict[str, tuple[int, int]]:
    return {
        name: (parameter.data_ptr(), int(parameter._version))
        for name, parameter in model.named_parameters()
    }


def _versions_unchanged(
        model, before: Mapping[str, tuple[int, int]]) -> bool:
    after = _parameter_versions(model)
    return after == dict(before)


def _flatten_losses(losses: Mapping[str, Any]) -> tuple[
        dict[str, float], Tensor]:
    scalars = {}
    total_terms = []
    for name, raw_value in losses.items():
        values = (
            list(raw_value)
            if isinstance(raw_value, (list, tuple))
            else [raw_value])
        tensors = [value for value in values if isinstance(value, Tensor)]
        if len(tensors) != len(values) or not tensors:
            continue
        scalar = sum(value.mean() for value in tensors)
        scalars[name] = float(scalar.detach().cpu().item())
        if 'loss' in name and scalar.requires_grad:
            total_terms.append(scalar)
    if not total_terms:
        raise ValueError('model returned no differentiable loss')
    total = sum(total_terms)
    return scalars, total


def _gradient_stats(model) -> tuple[float, bool]:
    squared_norm = 0.0
    finite = True
    active = 0
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        active += 1
        gradient = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(gradient).all().item())
        squared_norm += float(
            gradient.float().norm(2).detach().cpu().item()) ** 2
    if active == 0:
        raise ValueError('backward produced no active gradients')
    return math.sqrt(squared_norm), finite


def _estimated_query_area(
        cfg: Config, raw_batch: Mapping[str, Any]) -> int:
    batch_size = len(raw_batch['data_samples'])
    max_gt = max(
        len(sample.gt_instances) for sample in raw_batch['data_samples'])
    num_dn = int(cfg.train_dataloader.batch_sampler.num_dn_queries)
    groups = max(1, num_dn // max(1, max_gt))
    denoising_queries = 2 * max_gt * groups
    matching_queries = int(
        cfg.train_dataloader.batch_sampler.num_matching_queries)
    return batch_size * (matching_queries + denoising_queries) ** 2


def _prediction_with_raw_boxes(model, batch_inputs, data_samples):
    captured = []

    def capture_raw(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            raise ValueError('bbox head output contract changed')
        captured.append(output[1][-1].detach().cpu().clone())

    handle = model.bbox_head.register_forward_hook(capture_raw)
    try:
        with torch.no_grad():
            predictions = model.predict(
                batch_inputs,
                copy.deepcopy(data_samples),
                rescale=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('expected exactly one raw bbox-head forward')
    raw_boxes = captured[0]
    if raw_boxes.ndim != 3 or raw_boxes.shape[-2:] != (
            EXPECTED_QUERIES, 5):
        raise ValueError('raw prediction is not [batch, 600, 5]')
    return predictions, raw_boxes


def _run_role_model(
        *,
        role: str,
        cfg: Config,
        state: Mapping[str, Tensor],
        raw_batch: Mapping[str, Any],
        device: str,
        seed: int) -> dict[str, Any]:
    _seed_everything(seed)
    model = _build_loaded_model(cfg, state, device)
    versions = _parameter_versions(model)

    model.eval()
    eval_batch = model.data_preprocessor(
        copy.deepcopy(raw_batch), training=False)
    first_predictions, first_raw = _prediction_with_raw_boxes(
        model, eval_batch['inputs'], eval_batch['data_samples'])
    second_predictions, second_raw = _prediction_with_raw_boxes(
        model, eval_batch['inputs'], eval_batch['data_samples'])
    assert_deterministic_tensors([first_raw], [second_raw])
    counts = [
        len(sample.pred_instances) for sample in first_predictions]
    scores_finite = all(
        torch.isfinite(sample.pred_instances.scores).all().item()
        for sample in first_predictions)
    boxes_finite = all(
        torch.isfinite(sample.pred_instances.bboxes).all().item()
        for sample in first_predictions)
    box_summary = summarize_normalized_cxywh(first_raw)
    box_summary.update(
        per_image_counts=counts,
        scores_finite=scores_finite,
        rescaled_boxes_finite=boxes_finite,
        deterministic_raw_boxes=True)

    model.train()
    train_batch = model.data_preprocessor(
        copy.deepcopy(raw_batch), training=True)
    model.zero_grad(set_to_none=True)
    losses = model.loss(
        train_batch['inputs'], train_batch['data_samples'])
    individual_losses, total = _flatten_losses(losses)
    if not torch.isfinite(total).all().item():
        raise ValueError(role + ' total loss is non-finite')
    total.backward()
    gradient_norm, gradients_finite = _gradient_stats(model)

    model.eval()
    after_predictions, after_raw = _prediction_with_raw_boxes(
        model, eval_batch['inputs'], eval_batch['data_samples'])
    after_counts = [
        len(sample.pred_instances) for sample in after_predictions]
    after_succeeds = (
        after_counts == [EXPECTED_QUERIES] * len(after_counts)
        and bool(torch.isfinite(after_raw).all().item()))
    loss_stats = {
        'individual_losses': individual_losses,
        'total_loss': float(total.detach().cpu().item()),
        'gradient_norm': gradient_norm,
        'gradients_finite': gradients_finite,
        'parameters_unchanged': _versions_unchanged(model, versions),
        'prediction_succeeds_after_backward': after_succeeds,
        'max_estimated_query_area': _estimated_query_area(cfg, raw_batch),
    }

    del eval_batch, train_batch, first_raw, second_raw, after_raw, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        'role': role,
        'prediction': box_summary,
        'loss_gradient': loss_stats,
    }


def _audit_final_path(output: Path, role: str, audit: str) -> Path:
    return output.with_name(f'{output.stem}_{role}_{audit}.json')


def _run_external_audit(
        *,
        script: Path,
        config: Path,
        checkpoint: Path,
        output: Path,
        device: str) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output) or list(
            output.parent.glob(output.name + '.pending.*')):
        raise FileExistsError('output collision: ' + str(output))
    temporary = output.with_name(
        f'{output.name}.subprocess.{os.getpid()}')
    if os.path.lexists(temporary):
        raise FileExistsError('output collision: ' + str(temporary))
    command = [
        sys.executable,
        str(script),
        str(config),
        '--checkpoint',
        str(checkpoint),
        '--output',
        str(temporary),
        '--device',
        device,
    ]
    subprocess.run(command, check=True)
    report = json.loads(temporary.read_text(encoding='utf-8'))
    if not bool(report.get('pass')):
        raise ValueError(script.name + ' reported failure')
    publish_json_no_clobber(report, output)
    temporary.unlink()
    return {
        'pass': True,
        'path': str(output.resolve()),
        'sha256': sha256_file(output),
        'report': report,
    }


def run_preflight(
        *,
        candidate_config: Path,
        control_config: Path,
        candidate_checkpoint: Path,
        control_checkpoint: Path,
        source: Path,
        manifest: Path,
        device: str,
        seed: int,
        output: Path,
        strict_script: Path,
        open_vocabulary_script: Path) -> dict[str, Any]:
    """Execute the full D12 Stage-0 audit and publish one passing report."""
    output = Path(output).expanduser().resolve()
    candidate_config = Path(candidate_config).expanduser().resolve()
    control_config = Path(control_config).expanduser().resolve()
    candidate_checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    control_checkpoint = Path(control_checkpoint).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve()
    strict_script = Path(strict_script).expanduser().resolve()
    open_vocabulary_script = Path(
        open_vocabulary_script).expanduser().resolve()
    if os.path.lexists(output) or list(
            output.parent.glob(output.name + '.pending.*')):
        raise FileExistsError('output collision: ' + str(output))

    register_all_modules(init_default_scope=True)
    candidate_cfg = Config.fromfile(str(candidate_config))
    control_cfg = Config.fromfile(str(control_config))
    import_modules_from_strings(**control_cfg.custom_imports)
    config_report = _validate_config_pair(candidate_cfg, control_cfg)
    manifest_report = _validate_manifest(
        manifest,
        source,
        control_config,
        control_checkpoint,
        candidate_checkpoint)

    source_checkpoint = torch.load(str(source), map_location='cpu')
    source_state = unwrap_model_state(source_checkpoint)
    reject_forbidden_namespaces(source_state)
    converted_source = convert_groundingdino_state_dict(source_state)
    reject_forbidden_namespaces(converted_source)
    control_state = _load_model_only_state(control_checkpoint)
    candidate_state = _load_model_only_state(candidate_checkpoint)
    state_report = compare_complete_states(
        control_state, candidate_state, converted_source)

    _seed_everything(seed)
    raw_batch, total_gt = _first_real_batch_with_gt(control_cfg)
    candidate_model_report = _run_role_model(
        role='candidate',
        cfg=candidate_cfg,
        state=candidate_state,
        raw_batch=raw_batch,
        device=device,
        seed=seed)
    control_model_report = _run_role_model(
        role='control',
        cfg=control_cfg,
        state=control_state,
        raw_batch=raw_batch,
        device=device,
        seed=seed)
    prediction_gate = enforce_prediction_pair(
        candidate_model_report['prediction'],
        control_model_report['prediction'])
    loss_gradient_gate = enforce_loss_gradient_pair(
        candidate_model_report['loss_gradient'],
        control_model_report['loss_gradient'])

    audits = {}
    for role, config, checkpoint in (
            ('candidate', candidate_config, candidate_checkpoint),
            ('control', control_config, control_checkpoint)):
        audits[role] = {
            'strict': _run_external_audit(
                script=strict_script,
                config=config,
                checkpoint=checkpoint,
                output=_audit_final_path(output, role, 'strict'),
                device=device),
            'open_vocabulary': _run_external_audit(
                script=open_vocabulary_script,
                config=config,
                checkpoint=checkpoint,
                output=_audit_final_path(output, role, 'open_vocabulary'),
                device=device),
        }

    report = {
        'schema_version': 1,
        'pass': True,
        'seed': int(seed),
        'device': device,
        'zero_optimizer_steps': True,
        'fixed_real_batch_total_gt': int(total_gt),
        'manifest': manifest_report,
        'config_pair': config_report,
        'complete_state_delta': state_report,
        'candidate': candidate_model_report,
        'control': control_model_report,
        'prediction_gate': prediction_gate,
        'loss_gradient_gate': loss_gradient_gate,
        'integrity_audits': audits,
        'thresholds': {
            'max_saturation_fraction': MAX_SATURATION_FRACTION,
            'max_saturation_increase': MAX_SATURATION_INCREASE,
            'median_area_ratio': [
                MIN_MEDIAN_AREA_RATIO, MAX_MEDIAN_AREA_RATIO],
            'max_candidate_p99_area': MAX_P99_AREA,
            'max_total_loss_ratio': MAX_TOTAL_LOSS_RATIO,
            'max_gradient_norm_ratio': MAX_GRADIENT_NORM_RATIO,
            'max_query_area': MAX_QUERY_AREA,
        },
    }
    publish_json_no_clobber(report, output)
    report['output'] = {
        'path': str(output),
        'sha256': sha256_file(output),
    }
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidate-config', required=True, type=Path)
    parser.add_argument('--control-config', required=True, type=Path)
    parser.add_argument('--candidate-checkpoint', required=True, type=Path)
    parser.add_argument('--control-checkpoint', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--strict-script',
        type=Path,
        default=root / 'projects/OVCapFlow/tools/audit_strict_inference.py')
    parser.add_argument(
        '--open-vocabulary-script',
        type=Path,
        default=root / 'projects/OVCapFlow/tools/audit_open_vocabulary.py')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(
        candidate_config=args.candidate_config,
        control_config=args.control_config,
        candidate_checkpoint=args.candidate_checkpoint,
        control_checkpoint=args.control_checkpoint,
        source=args.source,
        manifest=args.manifest,
        device=args.device,
        seed=args.seed,
        output=args.output,
        strict_script=args.strict_script,
        open_vocabulary_script=args.open_vocabulary_script)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
