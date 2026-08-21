import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Sampler

from mmrotate.registry import DATA_SAMPLERS

from .no_replace import publish_json_noreplace


@DATA_SAMPLERS.register_module()
class DNQueryBudgetBatchSampler(Sampler):
    """Build exact distributed batches under a denoising-query budget.

    Every rank reconstructs the same global plan from the wrapped sampler's
    dataset, seed, and epoch. The plan is then sharded by rank, so a
    non-divisible dataset is covered exactly once without unequal update
    counts or empty rank batches.
    """

    def __init__(
            self,
            sampler: Sampler,
            batch_size: int,
            num_matching_queries: int,
            num_dn_queries: int,
            max_query_area: int,
            update_count_multiple: int = 1,
            audit_path: Optional[str] = None,
            audit_noreplace: bool = False) -> None:
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if num_matching_queries < 1:
            raise ValueError('num_matching_queries must be positive')
        if num_dn_queries < 0:
            raise ValueError('num_dn_queries must be non-negative')
        if max_query_area < 1:
            raise ValueError('max_query_area must be positive')
        if update_count_multiple < 1:
            raise ValueError('update_count_multiple must be positive')

        self.sampler = sampler
        self.batch_size = batch_size
        self.num_matching_queries = num_matching_queries
        self.num_dn_queries = num_dn_queries
        self.max_query_area = max_query_area
        self.update_count_multiple = update_count_multiple
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_noreplace = audit_noreplace
        self.rank = int(sampler.rank)
        self.world_size = int(sampler.world_size)
        self.dataset = sampler.dataset
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f'invalid rank/world size: {self.rank}/{self.world_size}')
        if len(self.dataset) < self.world_size:
            raise ValueError(
                'dataset must contain at least one sample per distributed rank')

        self._gt_counts: Optional[List[int]] = None
        self._cached_epoch: Optional[int] = None
        self._cached_plan: Optional[List[List[List[int]]]] = None
        self._cached_report: Optional[Dict[str, object]] = None

    def _load_gt_counts(self) -> List[int]:
        if self._gt_counts is not None:
            return self._gt_counts
        counts = []
        data_list = getattr(self.dataset, 'data_list', None)
        for index in range(len(self.dataset)):
            if data_list is not None and len(data_list) == len(self.dataset):
                data_info = data_list[index]
            else:
                data_info = self.dataset.get_data_info(index)
            counts.append(len(data_info.get('instances', ())))
        self._gt_counts = counts
        return counts

    def _global_order(self) -> List[int]:
        if not getattr(self.sampler, 'shuffle', True):
            return list(range(len(self.dataset)))
        generator = torch.Generator()
        generator.manual_seed(
            int(self.sampler.seed) + int(getattr(self.sampler, 'epoch', 0)))
        return torch.randperm(
            len(self.dataset), generator=generator).tolist()

    def _query_area(self, indices: Sequence[int]) -> int:
        if not indices:
            return 0
        gt_counts = self._load_gt_counts()
        max_gt = max(gt_counts[index] for index in indices)
        groups = max(1, self.num_dn_queries // max(1, max_gt))
        dn_queries = 2 * max_gt * groups
        return len(indices) * (self.num_matching_queries + dn_queries)**2

    def _allocate(
            self, indices: Sequence[int], update_index: int
    ) -> List[List[int]]:
        gt_counts = self._load_gt_counts()
        ordered = sorted(indices, key=lambda index: (-gt_counts[index], index))
        rank_batches: List[List[int]] = [[] for _ in range(self.world_size)]

        rank_rotation = update_index % self.world_size
        for offset, index in enumerate(ordered[:self.world_size]):
            rank_batches[(rank_rotation + offset) % self.world_size].append(
                index)

        for index in ordered[self.world_size:]:
            candidates = [
                rank for rank, batch in enumerate(rank_batches)
                if len(batch) < self.batch_size
            ]
            if not candidates:
                raise RuntimeError('global batch exceeds per-rank batch size')
            rank = min(
                candidates,
                key=lambda candidate: (
                    self._query_area(rank_batches[candidate] + [index]),
                    len(rank_batches[candidate]),
                    (candidate - rank_rotation) % self.world_size))
            rank_batches[rank].append(index)
        return rank_batches

    def _within_budget(self, rank_batches: Sequence[Sequence[int]]) -> bool:
        return all(
            len(batch) == 1 or self._query_area(batch) <= self.max_query_area
            for batch in rank_batches)

    def _select_update(
            self, order: Sequence[int], position: int, update_index: int
    ) -> Tuple[List[List[int]], int, bool]:
        remaining_size = len(order) - position
        natural_size = min(
            remaining_size, self.batch_size * self.world_size)
        for candidate_size in range(
                natural_size, self.world_size - 1, -1):
            tail_size = remaining_size - candidate_size
            if tail_size and tail_size < self.world_size:
                continue
            rank_batches = self._allocate(
                order[position:position + candidate_size], update_index)
            if self._within_budget(rank_batches):
                return rank_batches, candidate_size, candidate_size < natural_size

        raise RuntimeError(
            'unable to construct a non-empty exact distributed update')

    def _align_update_count(
            self, plan: List[List[List[int]]]) -> int:
        """Split full updates until epoch checkpoints are accumulation-safe."""
        split_count = 0
        while len(plan) % self.update_count_multiple:
            for update_index, rank_batches in enumerate(plan):
                if all(len(batch) >= 2 for batch in rank_batches):
                    first = [batch[::2] for batch in rank_batches]
                    second = [batch[1::2] for batch in rank_batches]
                    plan[update_index:update_index + 1] = [first, second]
                    split_count += 1
                    break
            else:
                raise RuntimeError(
                    'unable to align update count without an empty rank batch')
        return split_count

    def _build_plan(self) -> Tuple[List[List[List[int]]], Dict[str, object]]:
        epoch = int(getattr(self.sampler, 'epoch', 0))
        if self._cached_epoch == epoch and self._cached_plan is not None:
            return self._cached_plan, self._cached_report  # type: ignore

        order = self._global_order()
        plan: List[List[List[int]]] = []
        shrink_update_count = 0
        position = 0
        while position < len(order):
            rank_batches, consumed, shrunk = self._select_update(
                order, position, len(plan))
            plan.append(rank_batches)
            shrink_update_count += int(shrunk)
            position += consumed

        accumulation_split_count = self._align_update_count(plan)
        global_sizes = [
            sum(len(batch) for batch in rank_batches)
            for rank_batches in plan
        ]
        local_sizes = [
            len(batch) for rank_batches in plan for batch in rank_batches
        ]
        query_areas = []
        max_query_area_batch = {}
        for update_index, rank_batches in enumerate(plan):
            for rank, batch in enumerate(rank_batches):
                area = self._query_area(batch)
                query_areas.append(area)
                if (not max_query_area_batch or area >
                        max_query_area_batch['estimated_query_area']):
                    gt_counts = self._load_gt_counts()
                    max_query_area_batch = {
                        'update_index': update_index,
                        'rank': rank,
                        'indices': list(batch),
                        'gt_counts': [gt_counts[index] for index in batch],
                        'estimated_query_area': area,
                    }

        coverage = [
            index for rank_batches in plan for batch in rank_batches
            for index in batch
        ]
        unique = set(coverage)
        checksum_source = ','.join(str(index) for index in sorted(coverage))
        gt_counts = self._load_gt_counts()
        max_gt_index = max(
            range(len(gt_counts)), key=gt_counts.__getitem__)
        report: Dict[str, object] = {
            'epoch': epoch,
            'seed': int(self.sampler.seed),
            'world_size': self.world_size,
            'dataset_size': len(self.dataset),
            'update_count': len(plan),
            'coverage_checksum': hashlib.sha256(
                checksum_source.encode('utf-8')).hexdigest(),
            'duplicate_count': len(coverage) - len(unique),
            'missing_count': len(set(range(len(self.dataset))) - unique),
            'global_batch_size_min': min(global_sizes),
            'global_batch_size_max': max(global_sizes),
            'local_batch_size_min': min(local_sizes),
            'local_batch_size_max': max(local_sizes),
            'shrink_update_count': shrink_update_count,
            'accumulation_split_count': accumulation_split_count,
            'update_count_multiple': self.update_count_multiple,
            'max_gt': gt_counts[max_gt_index],
            'max_gt_sample': {
                'index': max_gt_index,
                'gt_count': gt_counts[max_gt_index],
                'singleton_estimated_query_area': self._query_area(
                    [max_gt_index]),
            },
            'max_estimated_query_area': max(query_areas),
            'max_query_area_budget': self.max_query_area,
            'max_query_area_batch': max_query_area_batch,
        }
        self._cached_epoch = epoch
        self._cached_plan = plan
        self._cached_report = report
        return plan, report

    def _write_audit(self, report: Dict[str, object]) -> None:
        if self.rank != 0 or self.audit_path is None:
            return
        if self.audit_noreplace:
            audit_path = Path(
                str(self.audit_path).format(epoch=int(report['epoch'])))
            publish_json_noreplace(audit_path, report)
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(report, indent=2), encoding='utf-8')

    def __iter__(self) -> Iterator[List[int]]:
        plan, report = self._build_plan()
        self._write_audit(report)
        for rank_batches in plan:
            yield rank_batches[self.rank]

    def __len__(self) -> int:
        plan, _ = self._build_plan()
        return len(plan)
