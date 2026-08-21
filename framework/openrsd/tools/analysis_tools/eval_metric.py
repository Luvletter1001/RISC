# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from typing import List, MutableMapping, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import DATASETS


def _normalize_img_id(img_id: Union[str, int, np.integer]) -> str:
    if isinstance(img_id, (int, np.integer)):
        return str(int(img_id))
    return str(img_id)


def _empty_gt_instances() -> dict:
    return {
        'labels': torch.zeros((0, ), dtype=torch.long),
        'bboxes': torch.zeros((0, 5), dtype=torch.float32),
    }


def _empty_ignored_instances() -> dict:
    return {
        'labels': torch.zeros((0, ), dtype=torch.long),
        'bboxes': torch.zeros((0, 5), dtype=torch.float32),
    }


def _normalize_empty_sample(sample: dict, pred_img_id) -> dict:
    sample = dict(sample)
    if not sample.get('gt_instances'):
        sample['gt_instances'] = _empty_gt_instances()
    if not sample.get('ignored_instances'):
        sample['ignored_instances'] = _empty_ignored_instances()
    if 'img_id' not in sample:
        sample['img_id'] = pred_img_id
    return sample


def merge_predictions_with_ground_truth(
        cfg: Config,
        predictions: Sequence[MutableMapping]) -> List[dict]:
    """Attach GT from ``test_dataloader`` so DOTAMetric receives full samples.

    Pickled outputs from ``DumpDetResults`` strip ``gt_instances``; offline
    evaluation must merge them back with the validation set (same order is not
    required; matching is by ``img_id``).
    """
    pred_map = {}
    for p in predictions:
        pid = _normalize_img_id(p['img_id'])
        if pid in pred_map:
            raise ValueError(f'Duplicate img_id in predictions: {pid}')
        pred_map[pid] = p

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    gt_by_id = {}
    for batch in dataloader:
        for ds in batch['data_samples']:
            d = ds.to_dict()
            gt_by_id[_normalize_img_id(d['img_id'])] = d

    merged = []
    for pid in sorted(pred_map.keys()):
        pred = pred_map[pid]
        if pid in gt_by_id:
            sample = _normalize_empty_sample(gt_by_id[pid], pred['img_id'])
        else:
            # e.g. ``filter_empty_gt=True`` drops images with no GT; keep preds.
            sample = {
                'img_id': pred['img_id'],
                'gt_instances': _empty_gt_instances(),
                'ignored_instances': _empty_ignored_instances(),
            }
        sample['pred_instances'] = pred['pred_instances']
        merged.append(sample)

    if len(merged) != len(predictions):
        raise RuntimeError(
            f'Merged length {len(merged)} != predictions {len(predictions)}')

    only_pred = set(pred_map.keys()) - set(gt_by_id.keys())
    only_gt = set(gt_by_id.keys()) - set(pred_map.keys())
    if only_gt:
        raise ValueError(
            f'{len(only_gt)} dataset image(s) missing from pkl (e.g. {next(iter(only_gt))})')
    if only_pred:
        warnings.warn(
            f'offline_eval: {len(only_pred)} pkl image(s) have no GT in '
            f'dataloader (likely filter_empty_gt); scored with empty GT.',
            stacklevel=2)

    return merged


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--predictions-only',
        action='store_true',
        help='Do not merge GT from the dataloader (fails for DumpDetResults '
        'pkls unless each sample already contains gt_instances).')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    predictions = mmengine.load(args.pkl_results)

    if args.predictions_only:
        samples = predictions
    else:
        samples = merge_predictions_with_ground_truth(cfg, predictions)

    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    eval_results = evaluator.offline_evaluate(samples)
    print(eval_results)


if __name__ == '__main__':
    main()
