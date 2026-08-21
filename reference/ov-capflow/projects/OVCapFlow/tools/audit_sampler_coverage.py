#!/usr/bin/env python3
"""Replay a configured DN budget sampler and save exact coverage evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from projects.OVCapFlow.ov_capflow import (DNQueryBudgetBatchSampler,
                                           publish_json_noreplace)


class _RankSamplerView:

    def __init__(self, dataset, rank, world_size, seed, epoch):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle = True


def _build_sampler_coverage_report(
        dataset,
        batch_size: int,
        batch_sampler_cfg: Mapping,
        seed: int,
        epoch: int,
        world_size: int) -> dict:
    """Materialize every rank without invoking sampler audit side effects."""
    if world_size < 1:
        raise ValueError('world_size must be positive')
    required = ('num_matching_queries', 'num_dn_queries', 'max_query_area')
    missing = [key for key in required if key not in batch_sampler_cfg]
    if missing:
        raise ValueError('missing batch sampler fields: {}'.format(missing))

    rank_batches = []
    report = None
    for rank in range(world_size):
        sampler = _RankSamplerView(
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            seed=seed,
            epoch=epoch)
        batch_sampler = DNQueryBudgetBatchSampler(
            sampler=sampler,
            batch_size=int(batch_size),
            num_matching_queries=int(
                batch_sampler_cfg['num_matching_queries']),
            num_dn_queries=int(batch_sampler_cfg['num_dn_queries']),
            max_query_area=int(batch_sampler_cfg['max_query_area']),
            update_count_multiple=int(
                batch_sampler_cfg.get('update_count_multiple', 1)),
            audit_path=None)
        plan, rank_report = batch_sampler._build_plan()
        rank_batches.append([
            list(update_rank_batches[rank])
            for update_rank_batches in plan
        ])
        if rank == 0:
            report = dict(rank_report)

    if report is None:
        raise RuntimeError('sampler audit produced no report')
    update_counts = [len(batches) for batches in rank_batches]
    if len(set(update_counts)) != 1:
        raise RuntimeError(
            'rank update counts differ: {}'.format(update_counts))
    if any(not batch for batches in rank_batches for batch in batches):
        raise RuntimeError('sampler emitted an empty local batch')

    coverage = [
        index for batches in rank_batches for batch in batches
        for index in batch
    ]
    expected = list(range(len(dataset)))
    if sorted(coverage) != expected or len(coverage) != len(set(coverage)):
        raise RuntimeError('sampler coverage is not exact and non-overlapping')
    if report['duplicate_count'] != 0 or report['missing_count'] != 0:
        raise RuntimeError('sampler self-audit reports missing/duplicate data')

    report.update({
        'num_matching_queries': int(
            batch_sampler_cfg['num_matching_queries']),
        'num_dn_queries': int(batch_sampler_cfg['num_dn_queries']),
        'rank_update_counts': update_counts,
        'rank_sample_counts': [
            sum(len(batch) for batch in batches) for batches in rank_batches
        ],
    })
    return report


def audit_sampler_coverage(
        dataset,
        batch_size: int,
        batch_sampler_cfg: Mapping,
        seed: int,
        epoch: int,
        world_size: int,
        output: Path) -> dict:
    """Materialize every rank and publish exact coverage evidence once."""
    report = _build_sampler_coverage_report(
        dataset=dataset,
        batch_size=batch_size,
        batch_sampler_cfg=batch_sampler_cfg,
        seed=seed,
        epoch=epoch,
        world_size=world_size)
    publish_json_noreplace(Path(output), report)
    return report


def audit_config(
        config: Path,
        world_size: int,
        epoch: int,
        output: Path,
        seed: Optional[int] = None) -> dict:
    """Build a configured dataset and replay its exact batch sampler."""
    from mmengine import Config
    from mmengine.utils import import_modules_from_strings
    from mmrotate.registry import DATASETS
    from mmrotate.utils import register_all_modules

    config = Path(config)
    cfg = Config.fromfile(config)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    replay_seed = cfg.randomness.seed if seed is None else seed
    report = _build_sampler_coverage_report(
        dataset=dataset,
        batch_size=cfg.train_dataloader.batch_size,
        batch_sampler_cfg=cfg.train_dataloader.batch_sampler,
        seed=replay_seed,
        epoch=epoch,
        world_size=world_size)
    report.update({
        'config': str(config),
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'ann_file': str(cfg.train_dataloader.dataset.ann_file),
        'config_seed': int(cfg.randomness.seed),
        'seed_overridden': seed is not None,
    })
    publish_json_noreplace(Path(output), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_config(
        config=args.config,
        world_size=args.world_size,
        epoch=args.epoch,
        output=args.output,
        seed=args.seed)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
