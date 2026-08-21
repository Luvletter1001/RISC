#!/usr/bin/env python3
"""CastDet-style dynamic pseudo-label queue with class balance."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional


QUEUE_CLASSES = [
    'tennis-court', 'baseball-diamond', 'soccer-ball-field', 'small-vehicle',
    'large-vehicle', 'harbor', 'bridge', 'storage-tank', 'roundabout',
]


class DynamicPseudoLabelQueue:
    def __init__(self, k_per_class: int = 500, min_conf: float = 0.3):
        self.k = k_per_class
        self.min_conf = min_conf
        self.queues: Dict[str, Deque[dict]] = {c: deque(maxlen=k_per_class) for c in QUEUE_CLASSES}
        self.sv_hard_negative: Deque[dict] = deque(maxlen=k_per_class)

    def add(self, label: dict):
        cls = label.get('class_name', '')
        conf = float(label.get('confidence', 0))
        if conf < self.min_conf:
            return
        if label.get('is_sv_hard_negative'):
            self.sv_hard_negative.append(label)
            return
        if cls not in self.queues:
            return
        q = self.queues[cls]
        if cls == 'small-vehicle' and label.get('unstable_sv'):
            return
        q.append(label)

    def class_counts(self) -> Dict[str, int]:
        return {c: len(q) for c, q in self.queues.items()}

    def sv_fraction(self) -> float:
        total = sum(len(q) for q in self.queues.values()) + len(self.sv_hard_negative)
        if total == 0:
            return 0.0
        return len(self.queues['small-vehicle']) / total

    def is_balanced(self, max_sv_frac: float = 0.35) -> bool:
        return self.sv_fraction() <= max_sv_frac

    def save(self, path: Path):
        data = {c: list(q) for c, q in self.queues.items()}
        data['_sv_hard_negative'] = list(self.sv_hard_negative)
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: Path, k_per_class: int = 500) -> 'DynamicPseudoLabelQueue':
        obj = cls(k_per_class=k_per_class)
        if not path.exists():
            return obj
        data = json.loads(path.read_text(encoding='utf-8'))
        for c in QUEUE_CLASSES:
            obj.queues[c] = deque(data.get(c, []), maxlen=k_per_class)
        obj.sv_hard_negative = deque(data.get('_sv_hard_negative', []), maxlen=k_per_class)
        return obj
