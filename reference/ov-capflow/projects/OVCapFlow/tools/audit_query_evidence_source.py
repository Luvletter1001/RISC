import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


SOURCE_SCHEMA = 'ov-capflow-query-evidence-source-v1'
LEVEL_INDEX = 2
TOKEN_BUDGET = 600
PRIOR_T = 0.145148646
PRIOR_E = 0.210711449
PRIOR_E_SAME_LABEL = 0.203709
SPATIAL_SHUFFLE_SEED = 2026071601
SEMANTIC_SHUFFLE_SEED = 2026071602


@dataclass(frozen=True)
class SourceGateThresholds:
    min_reachability: float = 0.25
    min_prior_margin: float = 0.03
    min_placebo_drop: float = 0.05
    max_base_novel_gap: float = 0.10


def _require_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f'{name} must be finite')


def build_level_centers(spatial_shape, valid_ratios):
    height, width = (int(value) for value in spatial_shape)
    if height <= 0 or width <= 0:
        raise ValueError('spatial shape must be positive')
    ratios = torch.as_tensor(valid_ratios, dtype=torch.float32)
    if ratios.ndim != 2 or ratios.shape[1] != 2:
        raise ValueError('valid_ratios must have shape [B,2]')
    _require_finite(ratios, 'valid_ratios')
    if not torch.all((ratios > 0) & (ratios <= 1)).item():
        raise ValueError('valid_ratios must lie in (0,1]')
    ys = (torch.arange(height, device=ratios.device, dtype=torch.float32)
          + 0.5) / height
    xs = (torch.arange(width, device=ratios.device, dtype=torch.float32)
          + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    raster = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
    return raster / ratios[:, None, :]


def semantic_token_evidence(token_logits, text_token_mask):
    logits = torch.as_tensor(token_logits).float()
    mask = torch.as_tensor(text_token_mask, dtype=torch.bool,
                           device=logits.device)
    if logits.ndim != 3 or mask.ndim != 2:
        raise ValueError('token logits/mask must have shapes [B,S,T]/[B,T]')
    if logits.shape[0] != mask.shape[0] or logits.shape[2] != mask.shape[1]:
        raise ValueError('token logits and text mask shapes disagree')
    if not mask.any(dim=1).all().item():
        raise ValueError('every sample needs a valid prompt token')
    valid_values = logits.masked_select(mask[:, None, :].expand_as(logits))
    _require_finite(valid_values, 'valid token logits')
    return logits.masked_fill(~mask[:, None, :], -torch.inf).amax(dim=-1)


def choose_evidence_indices(evidence, valid_mask, budget=TOKEN_BUDGET):
    values = torch.as_tensor(evidence).float()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool,
                            device=values.device)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError('evidence and valid mask must share shape [B,S]')
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError('budget must be a positive integer')
    _require_finite(values, 'evidence')
    valid_counts = valid.sum(dim=1)
    if not torch.all(valid_counts > 0).item():
        raise ValueError('each sample needs a valid spatial token')
    count = min(budget, values.shape[1], int(valid_counts.min().item()))
    selected = []
    for batch_index in range(values.shape[0]):
        candidates = torch.nonzero(valid[batch_index], as_tuple=False)[:, 0]
        order = sorted(
            candidates.tolist(),
            key=lambda index: (-float(values[batch_index, index]), index))
        selected.append(torch.tensor(
            order[:count], dtype=torch.long, device=values.device))
    return torch.stack(selected, dim=0)


def _make_placebo_logits_with_valid_mask(token_logits, centers, mode, seed,
                                         valid_mask):
    logits = torch.as_tensor(token_logits).clone()
    points = torch.as_tensor(centers).clone()
    if logits.ndim != 3 or points.ndim != 3:
        raise ValueError('placebo inputs must have shapes [B,S,T]/[B,S,2]')
    if logits.shape[:2] != points.shape[:2] or points.shape[2] != 2:
        raise ValueError('placebo input shapes disagree')
    valid = torch.as_tensor(valid_mask)
    if valid.shape != logits.shape[:2]:
        raise ValueError('placebo valid mask must have shape [B,S]')
    if valid.device != logits.device or points.device != logits.device:
        raise ValueError('placebo inputs and valid mask must share a device')
    valid = valid.to(dtype=torch.bool)
    if not valid.any(dim=1).all().item():
        raise ValueError('each placebo sample needs a valid spatial token')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    token_count = logits.shape[2]
    if mode == 'spatial':
        for batch_index in range(logits.shape[0]):
            indices = torch.nonzero(
                valid[batch_index], as_tuple=False)[:, 0]
            permutation = torch.randperm(
                len(indices), generator=generator).to(indices.device)
            points[batch_index, indices] = points[
                batch_index, indices[permutation]].clone()
    elif mode == 'semantic':
        for batch_index in range(logits.shape[0]):
            indices = torch.nonzero(
                valid[batch_index], as_tuple=False)[:, 0]
            for token_index in range(token_count):
                permutation = torch.randperm(
                    len(indices), generator=generator).to(indices.device)
                logits[batch_index, indices, token_index] = logits[
                    batch_index, indices[permutation], token_index].clone()
    elif mode == 'uniform':
        logits[valid] = 0
    else:
        raise ValueError(f'unknown placebo mode: {mode}')
    return logits, points


def make_placebo_logits(token_logits, centers, mode, seed):
    logits = torch.as_tensor(token_logits)
    if logits.ndim != 3:
        raise ValueError('placebo inputs must have shapes [B,S,T]/[B,S,2]')
    valid = torch.ones(
        logits.shape[:2], dtype=torch.bool, device=logits.device)
    return _make_placebo_logits_with_valid_mask(
        token_logits, centers, mode, seed, valid)


def geometry_reachability(gt_rboxes, evidence_centers):
    boxes = torch.as_tensor(gt_rboxes).float()
    centers = torch.as_tensor(evidence_centers).float()
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError('gt_rboxes must have shape [G,5]')
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('evidence_centers must have shape [S,2]')
    _require_finite(boxes, 'gt_rboxes')
    _require_finite(centers, 'evidence centers')
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    if centers.numel() == 0:
        return torch.zeros(boxes.shape[0], dtype=torch.bool,
                           device=boxes.device)
    delta = centers[None, :, :] - boxes[:, None, :2]
    cosine = torch.cos(boxes[:, 4])[:, None]
    sine = torch.sin(boxes[:, 4])[:, None]
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    normalized = torch.maximum(
        2 * local_x.abs() / boxes[:, None, 2],
        2 * local_y.abs() / boxes[:, None, 3])
    return (normalized <= 1).any(dim=1)


def normalized_points_to_original_pixels(points, img_shape, scale_factor):
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError('points must be a finite [N,2] array')
    img_height, img_width = int(img_shape[0]), int(img_shape[1])
    scales = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
    if scales.size not in (2, 4) or not np.isfinite(scales).all():
        raise ValueError('scale_factor must contain two or four finite values')
    scale_x, scale_y = float(scales[0]), float(scales[1])
    if img_height <= 0 or img_width <= 0 or scale_x <= 0 or scale_y <= 0:
        raise ValueError('image shape and scale factors must be positive')
    result = array.copy()
    result[:, 0] = result[:, 0] * img_width / scale_x
    result[:, 1] = result[:, 1] * img_height / scale_y
    return result


def source_gate_decision(metrics, thresholds):
    required = ('overall', 'spatial_shuffle', 'semantic_shuffle',
                'base14', 'novel4', 'finite', 'hashes_match')
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f'missing source gate field: {missing[0]}')
    numeric = {name: float(metrics[name]) for name in required[:5]}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError('source gate metrics must be finite')
    epsilon = 1e-12
    gates = {
        'reachability_at_least_25pct': (
            numeric['overall'] + epsilon >= thresholds.min_reachability),
        'beats_prior_e_by_3pp': (
            numeric['overall'] - PRIOR_E + epsilon >=
            thresholds.min_prior_margin),
        'placebo_sensitivity': (
            numeric['overall'] - numeric['spatial_shuffle'] + epsilon >=
            thresholds.min_placebo_drop and
            numeric['overall'] - numeric['semantic_shuffle'] + epsilon >=
            thresholds.min_placebo_drop),
        'novel_base_gap_at_most_10pp': (
            numeric['base14'] - numeric['novel4'] <=
            thresholds.max_base_novel_gap + epsilon),
        'finite_reproducible_provenance': (
            metrics['finite'] is True and metrics['hashes_match'] is True),
    }
    return {
        'status': 'PASS' if all(gates.values()) else 'FAIL',
        'gates': gates,
        'placebo_subchecks': {
            'spatial_shuffle_loses_5pp': (
                numeric['overall'] - numeric['spatial_shuffle'] + epsilon >=
                thresholds.min_placebo_drop),
            'semantic_shuffle_loses_5pp': (
                numeric['overall'] - numeric['semantic_shuffle'] + epsilon >=
                thresholds.min_placebo_drop),
        },
        'failed_gates': [name for name, passed in gates.items() if not passed],
    }


def write_report_no_replace(path, payload):
    required = {'schema', 'status', 'decision', 'frozen_contract',
                'provenance', 'counts', 'reachability', 'placebos', 'strata',
                'empty_images', 'finite_checks'}
    if set(payload) != required or payload['schema'] != SOURCE_SCHEMA:
        raise ValueError('source report schema is incomplete or changed')
    encoded = (json.dumps(payload, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def format_results_tsv_row(payload, report_sha256):
    gates = payload['decision']['gates']
    gate_text = ','.join(
        f'{name}={str(value).lower()}' for name, value in sorted(gates.items()))
    report_path = payload.get('provenance', {}).get(
        'report_path',
        '.lab/workspace/exp-8-qaf-source-v1/source_report.json')
    return ('8-D160-QAF-SOURCE\tresearch/dotav2-cleanstart-ov-e2e-ap70\t'
            '8-D159-QAF-SPEC\t9dee60d80c562a998348ce822e19dbd42ba99cb9\t'
            f"{payload['status'].lower()}\t{gate_text};"
            f"overall={payload['reachability']['overall']};"
            f'report={report_path};'
            f'report_sha256={report_sha256}\t'
            'source-gate-decision\t0\tfrozen E24 source preflight\n')


def validate_level_token_count(spatial_shape, observed_token_count):
    height, width = (int(value) for value in spatial_shape)
    expected = height * width
    if height <= 0 or width <= 0 or observed_token_count != expected:
        raise ValueError(
            f'level token count mismatch: expected {expected}, '
            f'got {observed_token_count}')


def require_file_sha256(path, expected_sha256):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('expected SHA256 must be lowercase hex')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f'SHA256 mismatch for {path}: {digest}')
    return digest


def require_unique_image_ids(image_ids):
    seen = set()
    for image_id in image_ids:
        if not isinstance(image_id, str) or not image_id:
            raise ValueError('image id must be a non-empty string')
        if image_id in seen:
            raise ValueError(f'duplicate image id: {image_id}')
        seen.add(image_id)


EXPECTED_SHA256 = {
    'config': 'f47af897b26c85a6f30fe6ecbc0a59144622d2e4e64cdd0e9346f037a0e31691',
    'proxy_config': 'f5cbb5d8e216d5a3f9d34e74596275ca03fd638e8b85c55486deb819e7d19c5d',
    'checkpoint': 'a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8',
    'canonical_dump': '39c1d6d3193cbc9e7f8d4e6daace9fc95df86c0d0fc7f1be7870cb13ee092c45',
    'dense_manifest': 'cf187eedba4f19703a77475639887147e4e1285b28f7db4c91686c0296030594',
    'proxy_manifest': 'a00b945ddd0a769008d57145feba28382fb9e56f5d6f1427413220f8008e1a2e',
    'train_manifest': '1457c641d91a7e0bf26a62f0cd6c70c71d9e9c6df5a2137fe4b73b8e8fc05290',
    'center_analyzer': '07ecd6cdce83c012987883f5ce6f7b9ea4c9d60de30253bc8fdd1139454866b6',
    'diagnostics_core': 'bd29dc88650e0db545479c60adaa6a9ee64da1734d3a1756c7cd787c7269c92d',
}


def _read_json(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain one JSON object')
    return payload


def _load_config(path):
    from mmengine import Config
    from mmengine.utils import import_modules_from_strings
    from mmrotate.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(path))
    import_modules_from_strings(**cfg.custom_imports)
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    return cfg


def _build_dataset(cfg):
    from mmengine.runner import Runner

    return Runner.build_dataloader(cfg.val_dataloader).dataset


def _load_frozen_model(cfg, checkpoint_path, device):
    from mmrotate.registry import MODELS

    model = MODELS.build(cfg.model).to(device).eval()
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'checkpoint mismatch: missing={} unexpected={}'.format(
                incompatible.missing_keys, incompatible.unexpected_keys))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('frozen source audit found a trainable parameter')
    return model


def _dataset_image_id(dataset, index):
    info = dataset.get_data_info(int(index))
    image_id = info.get('img_id')
    if image_id is None:
        image_id = Path(info['img_path']).stem
    return str(image_id)


def _make_replay_batch(model, dataset, dataset_index, batch_size):
    from mmengine.dataset import pseudo_collate

    if batch_size < 1:
        raise ValueError('replay batch size must be positive')
    raw = pseudo_collate(
        [dataset[int(dataset_index)] for _ in range(batch_size)])
    return model.data_preprocessor(raw, training=False)


def _prepare_prompt(model, batch_inputs, batch_data_samples):
    text_prompts = [sample.text for sample in batch_data_samples]
    enhanced = [sample.get('caption_prompt', None)
                for sample in batch_data_samples]
    tokens_positive = [sample.get('tokens_positive', None)
                       for sample in batch_data_samples]
    custom_entities = batch_data_samples[0].get('custom_entities', False)
    prepared = [
        model.get_tokens_positive_and_prompts(
            prompt, custom_entities, enhancement, positive)
        for prompt, enhancement, positive in zip(
            text_prompts, enhanced, tokens_positive)
    ]
    positive_maps, captions, _, entities = zip(*prepared)
    if any(isinstance(caption, list) for caption in captions):
        raise ValueError('source preflight forbids chunked prompts')
    text_dict = model.language_model(list(captions))
    if model.text_feat_map is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
    for sample, positive_map in zip(batch_data_samples, positive_maps):
        if positive_map is None:
            raise ValueError('source preflight requires a class positive map')
        sample.token_positive_map = positive_map
    if len(batch_inputs) != len(positive_maps):
        raise ValueError('prompt batch and image batch disagree')
    return text_dict, positive_maps, entities


def _canonical_positive_signature(positive_map, entities):
    canonical_map = {
        str(int(label)): [int(value) for value in torch.as_tensor(
            token_ids).reshape(-1).tolist()]
        for label, token_ids in sorted(
            positive_map.items(), key=lambda item: int(item[0]))
    }
    encoded = json.dumps(
        {'positive_map': canonical_map,
         'class_order': [str(value) for value in entities]},
        sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _gather_rows(values, indices):
    suffix = values.shape[2:]
    gather_index = indices.reshape(
        indices.shape + (1,) * len(suffix)).expand(indices.shape + suffix)
    return torch.gather(values, 1, gather_index)


def _extract_registered_level(model, batch, level_index, budget,
                              spatial_seed, semantic_seed):
    from projects.OVCapFlow.ov_capflow.calibration import (
        grounding_logits_to_class_log_scores,
    )

    batch_inputs = batch['inputs']
    samples = batch['data_samples']
    with torch.no_grad():
        text_dict, positive_maps, entities = _prepare_prompt(
            model, batch_inputs, samples)
        visual_features = model.extract_feat(batch_inputs)
        encoder_inputs, _ = model.pre_transformer(
            visual_features, samples)
        encoder_outputs = model.forward_encoder(
            **encoder_inputs, text_dict=text_dict)

        memory = encoder_outputs['memory']
        memory_mask = encoder_outputs['memory_mask']
        if memory_mask is None:
            memory_mask = torch.zeros(
                memory.shape[:2], dtype=torch.bool, device=memory.device)
        spatial_shapes = encoder_inputs['spatial_shapes']
        level_start_index = encoder_inputs['level_start_index']
        valid_ratios = encoder_inputs['valid_ratios']
        memory_text = encoder_outputs['memory_text']
        text_token_mask = encoder_outputs['text_token_mask']
        if (spatial_shapes.shape != (4, 2) or
                level_start_index.shape != (4,) or
                valid_ratios.shape != (memory.shape[0], 4, 2)):
            raise ValueError('encoder geometry does not match four levels')

        height = int(spatial_shapes[level_index, 0].item())
        width = int(spatial_shapes[level_index, 1].item())
        start = int(level_start_index[level_index].item())
        stop = start + height * width
        if (level_index + 1 < len(level_start_index) and
                stop != int(level_start_index[level_index + 1].item())):
            raise ValueError('registered level offsets disagree')
        level_memory = memory[:, start:stop]
        level_mask = memory_mask[:, start:stop]
        validate_level_token_count((height, width), level_memory.shape[1])
        centers = build_level_centers(
            (height, width), valid_ratios[:, level_index])
        valid = (~level_mask & (centers >= 0).all(dim=-1) &
                 (centers <= 1).all(dim=-1))

        full_token_logits = model.bbox_head.cls_branches[
            model.decoder.num_layers](
                level_memory, memory_text, text_token_mask)
        text_length = int(text_token_mask.shape[1])
        if (full_token_logits.shape[:2] != level_memory.shape[:2] or
                full_token_logits.shape[2] < text_length):
            raise ValueError('classification branch shape mismatch')
        token_logits = full_token_logits[..., :text_length]

        variants = {
            'primary': (token_logits, centers),
            'spatial_shuffle': _make_placebo_logits_with_valid_mask(
                token_logits, centers, 'spatial', spatial_seed, valid),
            'semantic_shuffle': _make_placebo_logits_with_valid_mask(
                token_logits, centers, 'semantic', semantic_seed, valid),
            'uniform': _make_placebo_logits_with_valid_mask(
                token_logits, centers, 'uniform', semantic_seed, valid),
        }
        results = {}
        for name, (variant_logits, variant_centers) in variants.items():
            evidence = semantic_token_evidence(
                variant_logits, text_token_mask)
            selected = choose_evidence_indices(evidence, valid, budget)
            selected_logits = _gather_rows(variant_logits, selected)
            selected_centers = _gather_rows(variant_centers, selected)
            normalized_mass = torch.softmax(
                evidence.masked_fill(~valid, -torch.inf), dim=-1)
            selected_mass = torch.gather(
                normalized_mass, 1, selected).sum(dim=-1)
            labels = []
            for batch_index, positive_map in enumerate(positive_maps):
                class_logs = grounding_logits_to_class_log_scores(
                    selected_logits[batch_index], positive_map)
                labels.append(class_logs.argmax(dim=-1))
            results[name] = {
                'centers': selected_centers,
                'labels': torch.stack(labels),
                'evidence': torch.gather(evidence, 1, selected),
                'selected_mass': selected_mass,
                'selected_count': int(selected.shape[1]),
            }
        signature = _canonical_positive_signature(
            positive_maps[0], entities[0])
        if any(_canonical_positive_signature(mapping, entity) != signature
               for mapping, entity in zip(positive_maps, entities)):
            raise ValueError('prompt mapping changed inside a replay batch')
    return results, signature


def _new_counter(modes, strata):
    return {
        mode: {stratum: {'hit': 0, 'same_hit': 0, 'total': 0}
               for stratum in strata}
        for mode in modes
    }


def _rate(hit, total):
    if total <= 0:
        raise ValueError('registered reachability denominator is empty')
    return float(hit / total)


def _stratum_masks(record, geometry_ids, classes, base_classes,
                   novel_classes):
    labels = record.gt_labels[geometry_ids].astype(np.int64, copy=False)
    names = np.asarray([classes[int(label)] for label in labels])
    boxes = record.gt_boxes[geometry_ids]
    sqrt_area = np.sqrt(boxes[:, 2] * boxes[:, 3])
    return {
        'overall': np.ones(len(geometry_ids), dtype=np.bool_),
        'le8px': sqrt_area <= 8.0,
        'small_vehicle': names == 'small-vehicle',
        'gt_count_over_600': np.full(
            len(geometry_ids), len(record.gt_boxes) > 600, dtype=np.bool_),
        'base14': np.isin(names, tuple(base_classes)),
        'novel4': np.isin(names, tuple(novel_classes)),
    }


def _accumulate_dense(args, model, dataset, records, dense_manifest,
                      class_manifest):
    from projects.OVCapFlow.tools.analyze_dense400_center_availability import (
        normalized_center_distances,
    )
    from projects.OVCapFlow.tools.analyze_dense400_center_controls import (
        _record_edges,
    )

    modes = ('primary', 'spatial_shuffle', 'semantic_shuffle', 'uniform')
    strata = ('overall', 'le8px', 'small_vehicle',
              'gt_count_over_600', 'base14', 'novel4')
    counters = _new_counter(modes, strata)
    selected_counts = []
    selected_mass = []
    already_reachable_mass = []
    signature = None
    replay_batch_size = int(args.replay_batch_size)

    entries = dense_manifest['records']
    if len(entries) != 400 or len(records) != 400:
        raise ValueError('Dense400 must contain exactly 400 records')
    require_unique_image_ids([str(entry['img_id']) for entry in entries])
    for entry, record in zip(entries, records):
        image_id = str(entry['img_id'])
        dataset_index = int(entry['dataset_index'])
        if (_dataset_image_id(dataset, dataset_index) != image_id or
                record.img_id != image_id or
                len(record.gt_boxes) != int(entry['gt_count'])):
            raise ValueError(f'Dense400 join mismatch for {image_id}')
        batch = _make_replay_batch(
            model, dataset, dataset_index, replay_batch_size)
        observed = [str(sample.img_id) for sample in batch['data_samples']]
        if any(value != image_id for value in observed):
            raise ValueError(f'preprocessed image id mismatch for {image_id}')
        extracted, observed_signature = _extract_registered_level(
            model, batch, args.level_index, args.token_budget,
            args.spatial_shuffle_seed, args.semantic_shuffle_seed)
        if signature is None:
            signature = observed_signature
        elif signature != observed_signature:
            raise ValueError('prompt mapping changed across Dense400')

        meta = batch['data_samples'][0].metainfo
        geometry = _record_edges(record, 0.5)['geometry_miss']
        geometry_ids = np.flatnonzero(geometry)
        masks = _stratum_masks(
            record, geometry_ids, class_manifest['classes'],
            class_manifest['base_classes'],
            class_manifest['novel_classes'])
        gt_boxes = record.gt_boxes[geometry_ids]
        gt_labels = record.gt_labels[geometry_ids]
        for mode in modes:
            result = extracted[mode]
            points = normalized_points_to_original_pixels(
                result['centers'][0].float().cpu().numpy(),
                meta['img_shape'], meta['scale_factor'])
            labels = result['labels'][0].cpu().numpy()
            distances = normalized_center_distances(points, gt_boxes)
            inside = distances <= 1.0
            any_reached = inside.any(axis=1)
            same_reached = (
                inside & (gt_labels[:, None] == labels[None, :])).any(axis=1)
            for stratum, mask in masks.items():
                counters[mode][stratum]['hit'] += int(
                    np.count_nonzero(any_reached & mask))
                counters[mode][stratum]['same_hit'] += int(
                    np.count_nonzero(same_reached & mask))
                counters[mode][stratum]['total'] += int(np.count_nonzero(mask))

            if mode == 'primary':
                selected_counts.append(result['selected_count'])
                selected_mass.append(float(result['selected_mass'][0].item()))
                reachable_gt = record.gt_boxes[~geometry]
                if len(reachable_gt):
                    occupied = (normalized_center_distances(
                        points, reachable_gt) <= 1.0).any(axis=0)
                    mass = result['selected_mass'].new_tensor(
                        result['evidence'][0]).softmax(dim=0)
                    already_reachable_mass.append(float(
                        mass[torch.as_tensor(occupied, device=mass.device)]
                        .sum().item()))
    return {
        'counters': counters,
        'selected_counts': selected_counts,
        'selected_mass': selected_mass,
        'already_reachable_mass': already_reachable_mass,
        'positive_map_sha256': signature,
    }


def _proxy_empty_diagnostic(args, model, proxy_dataset, proxy_manifest,
                            expected_signature):
    entries = [entry for entry in proxy_manifest['records']
               if entry.get('s1_split') == 'val']
    registered_ids = [str(value) for value in proxy_manifest['s1']['val_stems']]
    entry_ids = [str(entry['stem']) for entry in entries]
    dataset_ids = [
        _dataset_image_id(proxy_dataset, index)
        for index in range(len(proxy_dataset))
    ]
    require_unique_image_ids(registered_ids)
    require_unique_image_ids(dataset_ids)
    if entry_ids != registered_ids or set(dataset_ids) != set(registered_ids):
        raise ValueError('proxy validation identities differ from manifest')
    dataset_index_by_id = {
        image_id: index for index, image_id in enumerate(dataset_ids)
    }
    empty_entries = [entry for entry in entries
                     if int(entry['gt_count']) == 0]
    if len(empty_entries) != 120:
        raise ValueError('proxy manifest must register exactly 120 empties')
    masses = []
    counts = []
    for entry in empty_entries:
        image_id = str(entry['stem'])
        dataset_index = dataset_index_by_id[image_id]
        batch = _make_replay_batch(
            model, proxy_dataset, dataset_index, 1)
        if str(batch['data_samples'][0].img_id) != image_id:
            raise ValueError(f'proxy preprocessing join failed for {image_id}')
        extracted, signature = _extract_registered_level(
            model, batch, args.level_index, args.token_budget,
            args.spatial_shuffle_seed, args.semantic_shuffle_seed)
        if signature != expected_signature:
            raise ValueError('proxy prompt mapping differs from Dense400')
        primary = extracted['primary']
        masses.append(float(primary['selected_mass'][0].item()))
        counts.append(primary['selected_count'])
    return {
        'registered_empty_images': 120,
        'evaluated_empty_images': len(masses),
        'selected_softmax_mass_mean': float(np.mean(masses)),
        'selected_softmax_mass_min': float(np.min(masses)),
        'selected_softmax_mass_max': float(np.max(masses)),
        'selected_tokens_min': int(min(counts)),
        'selected_tokens_max': int(max(counts)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_full24e_'
        'scale1024_rare4x_gpu89_batch2.py'))
    parser.add_argument('--proxy-config', type=Path, default=Path(
        'configs/ov_capflow/dotav2/'
        'ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped_'
        'scale1024_batch1_rare4x.py'))
    parser.add_argument('--checkpoint', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/'
        'full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/'
        'epoch_24.pth'))
    parser.add_argument('--canonical-dump', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/'
        'eval_t7_epoch24_raw13833_gpu2389_dump/predictions.pkl'))
    parser.add_argument('--dense-manifest', type=Path, default=Path(
        'docs/project_history/exp_20260723_e24_route_audit/evidence/'
        'dense400_manifest.json'))
    parser.add_argument('--proxy-manifest', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json'))
    parser.add_argument('--train-manifest', type=Path, default=Path(
        'work_dirs/dotav2_cleanstart/subsets/seed20260715_rare4x/'
        'train/manifest.json'))
    parser.add_argument('--level-index', type=int, default=LEVEL_INDEX)
    parser.add_argument('--token-budget', type=int, default=TOKEN_BUDGET)
    parser.add_argument('--spatial-shuffle-seed', type=int,
                        default=SPATIAL_SHUFFLE_SEED)
    parser.add_argument('--semantic-shuffle-seed', type=int,
                        default=SEMANTIC_SHUFFLE_SEED)
    parser.add_argument('--output', type=Path, default=Path(
        '.lab/workspace/exp-8-qaf-source-v1/source_report.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replay-batch-size', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    return parser


def run_source_preflight(args):
    from projects.OVCapFlow.tools.analyze_dense400_center_controls import (
        load_dense400,
    )

    if (args.level_index != LEVEL_INDEX or
            args.token_budget != TOKEN_BUDGET or
            args.spatial_shuffle_seed != SPATIAL_SHUFFLE_SEED or
            args.semantic_shuffle_seed != SEMANTIC_SHUFFLE_SEED):
        raise ValueError('CLI differs from the frozen source contract')
    sidecar = args.output.with_name('source_results.tsv')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('source report or sidecar already exists')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('preflight output directory must pre-exist')

    paths = {
        'config': args.config,
        'proxy_config': args.proxy_config,
        'checkpoint': args.checkpoint,
        'canonical_dump': args.canonical_dump,
        'dense_manifest': args.dense_manifest,
        'proxy_manifest': args.proxy_manifest,
        'train_manifest': args.train_manifest,
        'center_analyzer': Path(
            'projects/OVCapFlow/tools/analyze_dense400_center_controls.py'),
        'diagnostics_core': Path(
            'projects/OVCapFlow/tools/dotav2_q600_diagnostics.py'),
    }
    provenance_files = {}
    for name, path in paths.items():
        digest = require_file_sha256(path, EXPECTED_SHA256[name])
        provenance_files[name] = {'path': str(path), 'sha256': digest}

    dense_manifest = _read_json(args.dense_manifest)
    proxy_manifest = _read_json(args.proxy_manifest)
    records, loaded_manifest = load_dense400(
        args.canonical_dump, args.dense_manifest)
    if loaded_manifest != dense_manifest:
        raise ValueError('Dense400 manifest changed during load')
    if (sum(len(record.gt_boxes) for record in records) != 123888 or
            sum(int(_record['gt_count'])
                for _record in dense_manifest['records']) != 123888):
        raise ValueError('Dense400 GT anchor must be exactly 123888')

    cfg = _load_config(args.config)
    proxy_cfg = _load_config(args.proxy_config)
    model = _load_frozen_model(cfg, args.checkpoint, args.device)
    dense_dataset = _build_dataset(cfg)
    proxy_dataset = _build_dataset(proxy_cfg)
    dense = _accumulate_dense(
        args, model, dense_dataset, records, dense_manifest, proxy_manifest)
    primary_total = dense['counters']['primary']['overall']['total']
    if primary_total != 61068:
        raise ValueError('canonical geometry-miss anchor must be 61068')
    empty = _proxy_empty_diagnostic(
        args, model, proxy_dataset, proxy_manifest,
        dense['positive_map_sha256'])

    primary = dense['counters']['primary']
    metrics = {
        'overall': _rate(primary['overall']['hit'],
                         primary['overall']['total']),
        'spatial_shuffle': _rate(
            dense['counters']['spatial_shuffle']['overall']['hit'],
            primary['overall']['total']),
        'semantic_shuffle': _rate(
            dense['counters']['semantic_shuffle']['overall']['hit'],
            primary['overall']['total']),
        'base14': _rate(primary['base14']['hit'], primary['base14']['total']),
        'novel4': _rate(primary['novel4']['hit'], primary['novel4']['total']),
        'finite': True,
        'hashes_match': True,
    }
    decision = source_gate_decision(metrics, SourceGateThresholds())
    strata = {
        name: {
            'reached': values['hit'],
            'same_label_reached': values['same_hit'],
            'total': values['total'],
            'reachability': _rate(values['hit'], values['total']),
            'same_label_reachability': _rate(
                values['same_hit'], values['total']),
        }
        for name, values in primary.items()
    }
    payload = {
        'schema': SOURCE_SCHEMA,
        'status': decision['status'],
        'decision': decision,
        'frozen_contract': {
            'design_commit':
            '9dee60d80c562a998348ce822e19dbd42ba99cb9',
            'encoder_level_index': LEVEL_INDEX,
            'encoder_stride': 32,
            'token_budget': TOKEN_BUDGET,
            'prior_t': PRIOR_T,
            'prior_e': PRIOR_E,
            'prior_e_same_label': PRIOR_E_SAME_LABEL,
            'spatial_shuffle_seed': SPATIAL_SHUFFLE_SEED,
            'semantic_shuffle_seed': SEMANTIC_SHUFFLE_SEED,
        },
        'provenance': {
            'command': [sys.executable] + sys.argv,
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'cuda_device': str(args.device),
            'report_path': str(args.output),
            'files': provenance_files,
            'positive_map_and_class_order_sha256':
            dense['positive_map_sha256'],
        },
        'counts': {
            'dense_images': len(records),
            'dense_gt': 123888,
            'geometry_miss_gt': primary_total,
            'selected_tokens_min': int(min(dense['selected_counts'])),
            'selected_tokens_max': int(max(dense['selected_counts'])),
            'empty_proxy_images': empty['evaluated_empty_images'],
        },
        'reachability': {
            'overall': metrics['overall'],
            'same_label': _rate(
                primary['overall']['same_hit'], primary_total),
            'prior_t': PRIOR_T,
            'prior_e': PRIOR_E,
            'prior_e_same_label': PRIOR_E_SAME_LABEL,
        },
        'placebos': {
            mode: {
                'reachability': _rate(
                    dense['counters'][mode]['overall']['hit'], primary_total),
                'same_label_reachability': _rate(
                    dense['counters'][mode]['overall']['same_hit'],
                    primary_total),
            }
            for mode in ('spatial_shuffle', 'semantic_shuffle', 'uniform')
        },
        'strata': strata,
        'empty_images': empty,
        'finite_checks': {
            'all_metrics_finite': all(math.isfinite(float(value))
                                      for value in metrics.values()),
            'selected_evidence_mass_mean': float(np.mean(
                dense['selected_mass'])),
            'already_reachable_selected_mass_mean': float(np.mean(
                dense['already_reachable_mass'])),
            'all_parameters_frozen': True,
            'optimizer_constructed': False,
            'detections_emitted': False,
        },
    }
    report_sha256 = write_report_no_replace(args.output, payload)
    with sidecar.open('x', encoding='utf-8') as stream:
        stream.write(format_results_tsv_row(payload, report_sha256))
    return payload, report_sha256


def verify_existing_report(path):
    report_bytes = Path(path).read_bytes()
    payload = json.loads(report_bytes)
    required = {'schema', 'status', 'decision', 'frozen_contract',
                'provenance', 'counts', 'reachability', 'placebos', 'strata',
                'empty_images', 'finite_checks'}
    gates = payload.get('decision', {}).get('gates', {})
    if (set(payload) != required or payload.get('schema') != SOURCE_SCHEMA or
            len(gates) != 5 or
            any(type(value) is not bool for value in gates.values())):
        raise ValueError('stored source report violates the locked schema')
    expected_status = 'PASS' if all(gates.values()) else 'FAIL'
    if payload.get('status') != expected_status:
        raise ValueError('stored source status disagrees with its gates')
    digest = hashlib.sha256(report_bytes).hexdigest()
    sidecar = Path(path).with_name('source_results.tsv').read_text(
        encoding='utf-8')
    if f'report_sha256={digest}' not in sidecar:
        raise ValueError('stored sidecar does not bind the report digest')
    print(json.dumps({
        'status': expected_status,
        'gates': gates,
        'placebo_subchecks': payload['decision']['placebo_subchecks'],
        'report_sha256': digest,
    }, sort_keys=True))
    return 0 if expected_status == 'PASS' else 3


def main():
    args = build_parser().parse_args()
    try:
        if args.verify_only:
            return verify_existing_report(args.output)
        payload, digest = run_source_preflight(args)
    except Exception as error:
        print(f'infrastructure/provenance failure: {error}', file=sys.stderr)
        return 2
    print(json.dumps(
        {'status': payload['status'], 'report_sha256': digest},
        sort_keys=True))
    return 0 if payload['status'] == 'PASS' else 3


if __name__ == '__main__':
    raise SystemExit(main())
