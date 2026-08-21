from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping


def class_distribution_from_histograms(rows: Iterable[Mapping], key: str = "class_histogram") -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in rows:
        hist = row.get(key, {})
        if isinstance(hist, str):
            import json

            hist = json.loads(hist) if hist else {}
        counts.update({str(k): float(v) for k, v in hist.items()})
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def js_kl_divergence(p: Mapping[str, float], q: Mapping[str, float], eps: float = 1.0e-12) -> tuple[float, float]:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0, 0.0
    pv = [float(p.get(key, 0.0)) + eps for key in keys]
    qv = [float(q.get(key, 0.0)) + eps for key in keys]
    ps = sum(pv)
    qs = sum(qv)
    pv = [x / ps for x in pv]
    qv = [x / qs for x in qv]
    mv = [(a + b) / 2.0 for a, b in zip(pv, qv)]
    kl_pq = sum(a * math.log(a / b) for a, b in zip(pv, qv))
    kl_pm = sum(a * math.log(a / m) for a, m in zip(pv, mv))
    kl_qm = sum(b * math.log(b / m) for b, m in zip(qv, mv))
    return float((kl_pm + kl_qm) / 2.0), float(kl_pq)


def mean_float(rows: Iterable[Mapping], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, "")))
        except (TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else float("nan")
