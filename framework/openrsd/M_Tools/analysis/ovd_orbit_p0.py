"""Pure record contract for the OVD object-orbit P0 exporter.

This module validates full per-carrier score vectors.  It deliberately avoids
model imports, score selection, NMS, and any training or evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


C4_VIEW_IDS = frozenset({'rot000', 'rot090', 'rot180', 'rot270'})
OVD_ORBIT_P0_VIEW_IDS = C4_VIEW_IDS | frozenset({'rot000_a', 'rot000_b'})


class OrbitP0Error(ValueError):
    """Raised when a P0 full-score record violates its sealed contract."""


def _owned_float32(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.shape != shape:
        raise OrbitP0Error(f'{name} must be float32 with shape {shape}')
    if not np.isfinite(array).all():
        raise OrbitP0Error(f'{name} must be finite')
    owned = np.array(array, dtype=np.float32, copy=True)
    owned.setflags(write=False)
    return owned


def _owned_score_vector(value: Any, *, name: str = 'scores') -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.ndim != 1 or array.size == 0:
        raise OrbitP0Error(f'{name} must be a nonempty float32 [C] vector')
    if not np.isfinite(array).all():
        raise OrbitP0Error(f'{name} must be finite')
    owned = np.array(array, dtype=np.float32, copy=True)
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class ScoreCarrier:
    """One complete score vector attached to an exact pre-selection source."""

    view_id: str
    scene_id: str
    source: tuple[int, int]
    box: np.ndarray
    scores: np.ndarray
    calibrated_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.view_id not in OVD_ORBIT_P0_VIEW_IDS:
            raise OrbitP0Error('view_id must belong to the frozen P0 views')
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise OrbitP0Error('scene_id must be a nonempty string')
        if (not isinstance(self.source, tuple) or len(self.source) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                       for item in self.source)):
            raise OrbitP0Error('source must be a nonnegative (level, row) tuple')
        object.__setattr__(
            self, 'box', _owned_float32(self.box, shape=(5,), name='box'))
        scores = _owned_score_vector(self.scores)
        object.__setattr__(self, 'scores', scores)
        if self.calibrated_scores is not None:
            calibrated_scores = _owned_score_vector(
                self.calibrated_scores, name='calibrated_scores')
            if calibrated_scores.shape != scores.shape:
                raise OrbitP0Error(
                    'calibrated_scores must align with scores')
            object.__setattr__(
                self, 'calibrated_scores', calibrated_scores)


def _carrier_key(carrier: ScoreCarrier) -> tuple[str, str, tuple[int, int]]:
    return carrier.scene_id, carrier.view_id, carrier.source


def _same_optional_scores(
        first: np.ndarray | None, second: np.ndarray | None) -> bool:
    return ((first is None and second is None)
            or (first is not None and second is not None
                and np.array_equal(first, second)))


def collapse_carriers(carriers: Iterable[ScoreCarrier]) -> tuple[ScoreCarrier, ...]:
    """Collapse exact duplicate source records and reject divergent duplicates."""
    by_key: dict[tuple[str, str, tuple[int, int]], ScoreCarrier] = {}
    for carrier in carriers:
        if not isinstance(carrier, ScoreCarrier):
            raise OrbitP0Error('carriers must contain ScoreCarrier values')
        key = _carrier_key(carrier)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = carrier
        else:
            differing_fields = []
            if not np.array_equal(previous.box, carrier.box):
                differing_fields.append('box')
            if not np.array_equal(previous.scores, carrier.scores):
                differing_fields.append('scores')
            if not _same_optional_scores(
                    previous.calibrated_scores, carrier.calibrated_scores):
                differing_fields.append('calibrated_scores')
            if differing_fields:
                raise OrbitP0Error(
                    f'conflicting duplicate source carrier {key}: '
                    f'differing fields: {", ".join(differing_fields)}')
    return tuple(by_key[key] for key in sorted(by_key))


def build_manifest(
        carriers: Iterable[ScoreCarrier], *, vocabulary_hash: str, prompt_hash: str,
        observed_vocabulary_hashes: set[str] | None = None) -> dict[str, Any]:
    """Return a deterministic E0 contract manifest for already-captured scores."""
    if not isinstance(vocabulary_hash, str) or not vocabulary_hash:
        raise OrbitP0Error('vocabulary_hash must be a nonempty string')
    if not isinstance(prompt_hash, str) or not prompt_hash:
        raise OrbitP0Error('prompt_hash must be a nonempty string')
    observed = ({vocabulary_hash} if observed_vocabulary_hashes is None
                else set(observed_vocabulary_hashes))
    if observed != {vocabulary_hash}:
        raise OrbitP0Error('observed vocabulary hashes must match vocabulary_hash')

    collapsed = collapse_carriers(carriers)
    if not collapsed:
        raise OrbitP0Error('at least one score carrier is required')
    p0148_scenes = sorted({carrier.scene_id for carrier in collapsed
                           if 'P0148' in carrier.scene_id})
    if p0148_scenes:
        raise OrbitP0Error('P0148 source scene is excluded from P0')
    class_dims = {int(carrier.scores.size) for carrier in collapsed}
    if len(class_dims) != 1:
        raise OrbitP0Error('all score carriers must share one class dimension')

    return {
        'schema_version': 1,
        'status': 'E0_CONTRACT_READY',
        'carrier_count': len(collapsed),
        'scene_count': len({carrier.scene_id for carrier in collapsed}),
        'class_dimension': class_dims.pop(),
        'view_ids': sorted({carrier.view_id for carrier in collapsed}),
        'vocabulary_hash': vocabulary_hash,
        'prompt_hash': prompt_hash,
        'p0148_excluded_scene_count': 0,
    }


def _carrier_record(carrier: ScoreCarrier) -> dict[str, Any]:
    record = {
        'box': [float(value) for value in carrier.box],
        'scene_id': carrier.scene_id,
        'scores': [float(value) for value in carrier.scores],
        'source': [int(value) for value in carrier.source],
        'view_id': carrier.view_id,
    }
    if carrier.calibrated_scores is not None:
        record['calibrated_scores'] = [
            float(value) for value in carrier.calibrated_scores]
    return record


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':')) + '\n').encode('utf-8')


def write_receipt(
        output_dir: Path | str, manifest: dict[str, Any],
        carriers: Iterable[ScoreCarrier]) -> dict[str, Any]:
    """Write deterministic E0 contract artifacts without overwriting a run."""
    collapsed = collapse_carriers(carriers)
    if not collapsed:
        raise OrbitP0Error('at least one score carrier is required')
    if manifest.get('status') != 'E0_CONTRACT_READY':
        raise OrbitP0Error('manifest must have E0_CONTRACT_READY status')
    if manifest.get('carrier_count') != len(collapsed):
        raise OrbitP0Error('manifest carrier_count does not match carriers')

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest_bytes = _json_bytes(manifest)
    carrier_bytes = b''.join(
        _json_bytes(_carrier_record(carrier)) for carrier in collapsed)
    manifest_path = root / 'manifest.json'
    carriers_path = root / 'carriers.jsonl'
    manifest_path.write_bytes(manifest_bytes)
    carriers_path.write_bytes(carrier_bytes)
    digest = hashlib.sha256(manifest_bytes + carrier_bytes).hexdigest()
    receipt = {
        'carrier_count': len(collapsed),
        'class_dimension': manifest['class_dimension'],
        'prompt_hash': manifest['prompt_hash'],
        'sha256': digest,
        'status': manifest['status'],
        'vocabulary_hash': manifest['vocabulary_hash'],
    }
    (root / 'receipt.json').write_bytes(_json_bytes(receipt))
    return receipt


__all__ = ['C4_VIEW_IDS', 'OVD_ORBIT_P0_VIEW_IDS', 'OrbitP0Error',
           'ScoreCarrier', 'build_manifest', 'collapse_carriers',
           'write_receipt']
