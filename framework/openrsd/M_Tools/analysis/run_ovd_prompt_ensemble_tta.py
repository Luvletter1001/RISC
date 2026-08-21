#!/usr/bin/env python3
"""Summarize or refresh OVD prompt ensemble + rotation TTA repair."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    try:
        return f'{float(value):.4f}'
    except (TypeError, ValueError):
        return 'NA'


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key)), []).append(row)
    return out


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r['ap50']) for r in rows if r.get('status') == 'OK' and r.get('ap50') is not None]
    if not vals:
        return {}
    worst = min((r for r in rows if r.get('ap50') is not None), key=lambda r: float(r['ap50']))
    best = max((r for r in rows if r.get('ap50') is not None), key=lambda r: float(r['ap50']))
    mean = statistics.mean(vals)
    return {
        'mean': mean,
        'worst': float(worst['ap50']),
        'worst_angle': worst.get('target_angle'),
        'best': float(best['ap50']),
        'best_angle': best.get('target_angle'),
        'std': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        'range': max(vals) - min(vals),
        'rsi': min(vals) / mean if mean else 0.0,
    }


def write_md(report: dict[str, Any], md_path: Path, run_ts: str, work_dir: Path) -> None:
    rows = report.get('rows', [])
    by_strategy = group(rows, 'strategy')
    lines = [
        '# F1: OVD4 Prompt Ensemble + Rotation TTA Repair',
        '',
        f'- RUN_TS: `{run_ts}`',
        f'- status: `{report.get("status", "NOT_RUN")}`',
        f'- source_report: `{work_dir / "exp_ovd4/exp_ovd4_results.json"}`',
        f'- best_prompt_family: `{report.get("best_prompt_family")}`',
        f'- source_angles: `{report.get("source_angles", [])}`',
        f'- target_angles: `{report.get("target_angles", [])}`',
        '- level: `existing OVD4 level from source report`',
        '',
        '## Strategy Summary',
        '',
        '| strategy | mean AP50 | worst AP50 | worst angle | best AP50 | best angle | std | range | RSI |',
        '|---|---:|---:|---|---:|---|---:|---:|---:|',
    ]
    summary = {}
    for strategy, srows in sorted(by_strategy.items()):
        st = stats(srows)
        if not st:
            continue
        summary[strategy] = st
        lines.append(
            f'| {strategy} | {fmt(st["mean"])} | {fmt(st["worst"])} | {st["worst_angle"]} | '
            f'{fmt(st["best"])} | {st["best_angle"]} | {fmt(st["std"])} | {fmt(st["range"])} | {fmt(st["rsi"])} |')
    lines.extend(['', '## Angle Rows', '', '| strategy | angle | AP50 | status | predictions |', '|---|---:|---:|---|---|'])
    for row in sorted(rows, key=lambda r: (str(r.get('strategy')), str(r.get('target_angle')))):
        lines.append(f'| {row.get("strategy")} | {row.get("target_angle")} | {fmt(row.get("ap50"))} | {row.get("status")} | `{row.get("predictions", "")}` |')
    a = {r.get('target_angle'): r for r in by_strategy.get('single_prompt_single_view', [])}
    d = {r.get('target_angle'): r for r in by_strategy.get('prompt_ensemble_rotation_tta', [])}
    canonical_damage = None
    if '000' in a and '000' in d and a['000'].get('ap50') is not None and d['000'].get('ap50') is not None:
        canonical_damage = float(d['000']['ap50']) - float(a['000']['ap50'])
    worst_angle = report.get('worst_angle')
    worst_gain = None
    if worst_angle in a and worst_angle in d and a[worst_angle].get('ap50') is not None and d[worst_angle].get('ap50') is not None:
        worst_gain = float(d[worst_angle]['ap50']) - float(a[worst_angle]['ap50'])
    lines.extend([
        '',
        '## Required Questions',
        '',
        f'- prompt ensemble mean effect: `{fmt((summary.get("prompt_ensemble_single_view", {}).get("mean", 0) or 0) - (summary.get("single_prompt_single_view", {}).get("mean", 0) or 0))}`',
        f'- TTA worst-angle gain: `{fmt(worst_gain)}`',
        f'- prompt ensemble + TTA canonical damage: `{fmt(canonical_damage)}`',
        f'- semantic drift rows: `{len(report.get("semantic_drift_rows", []))}`',
        '- fusion head comparison: use OVD3 rows; this F1 source report is alignment-centric unless fusion rows are present.',
    ])
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-work-dir', type=Path, default=Path('/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508'))
    parser.add_argument('--run-ts', required=True)
    parser.add_argument('--out-json', type=Path, required=True)
    parser.add_argument('--out-md', type=Path, required=True)
    args = parser.parse_args()
    report = load_json(args.existing_work_dir / 'exp_ovd4/exp_ovd4_results.json')
    if not report:
        report = {'status': 'NOT_RUN', 'rows': [], 'reason': 'existing OVD4 report missing'}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    write_md(report, args.out_md, args.run_ts, args.existing_work_dir)


if __name__ == '__main__':
    main()
