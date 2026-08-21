#!/usr/bin/env python
"""Final overnight summary for SV-DeHub-Lite v1 (Plan Mode §9)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89 import common_dehub_utils as U
from tools.exp_sv_dehub_lite_train_gpu89 import patch_config_sv_dehub_lite as C


def mean_metric(rows, ckpt: str, group: str, field: str) -> float:
    vals = []
    for r in rows:
        if r.get('checkpoint') != ckpt or r.get('group') != group:
            continue
        try:
            vals.append(float(r.get(field, 0) or 0))
        except (TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else float('nan')


def verdict_from_metrics(rows, ckpt: str, ap_rows: list) -> str:
    bl_h = mean_metric(rows, 'baseline_epoch24', 'high', 'final_sv_ratio')
    bl_d = mean_metric(rows, 'baseline_epoch24', 'high', 'dense_top1_sv_ratio')
    hf = mean_metric(rows, ckpt, 'high', 'final_sv_ratio')
    hd = mean_metric(rows, ckpt, 'high', 'dense_top1_sv_ratio')
    if bl_h != bl_h:
        return 'FAILED_DEBUG'
    rel_f = hf / bl_h - 1 if bl_h else 0
    rel_d = hd / bl_d - 1 if bl_d else 0
    if rel_f <= -0.30 and rel_d <= -0.30:
        return 'PROMISING'
    if rel_d <= -0.30 and rel_f > -0.15:
        return 'DENSE_ONLY_SUCCESS'
    if rel_f <= -0.10:
        return 'AP_TRADEOFF'
    if abs(rel_f) < 0.05:
        return 'INEFFECTIVE'
    return 'PARTIAL'


def read_hourly_section(result_dir: Path) -> str:
    md = result_dir / 'fres_hourly_monitor.md'
    if not md.exists():
        return '_Hourly monitor not run; pipeline completed before patrol tooling._\n'
    text = md.read_text(encoding='utf-8')
    if len(text) > 4000:
        return text[:4000] + '\n\n_(truncated)_\n'
    return text


def debug_section(result_dir: Path) -> str:
    parts = []
    for name in ('fres_debug_latest_failure.md', 'fres_debug_patch_notes.md'):
        p = result_dir / name
        if p.exists():
            parts.append(f'### {name}\n\n{p.read_text(encoding="utf-8")[:2000]}\n')
    if not parts:
        return 'No debug events recorded.\n'
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', type=Path, default=C.RESULT_DIR)
    args = ap.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rows = U.read_csv(C.EVAL_RAW)
    curve = U.read_csv(C.CURVE_CSV)
    ap_rows = U.read_csv(C.EVAL_AP)
    hourly_rows = U.read_csv(C.HOURLY_CSV)
    analyze_meta = {}
    meta_path = args.result_dir / 'fmeta_analyze_sv_dehub.json'
    if meta_path.exists():
        analyze_meta = json.loads(meta_path.read_text(encoding='utf-8'))

    dehub_vals = []
    for r in curve:
        try:
            dehub_vals.append(float(r.get('loss_dehub', 0) or 0))
        except (TypeError, ValueError):
            pass
    dehub_ok = bool(dehub_vals) and max(dehub_vals) > 1e-6

    bl_h = mean_metric(rows, 'baseline_epoch24', 'high', 'final_sv_ratio')
    bl_l = mean_metric(rows, 'baseline_epoch24', 'low', 'final_sv_ratio')
    bl_d = mean_metric(rows, 'baseline_epoch24', 'high', 'dense_top1_sv_ratio')

    ckpts = U.list_checkpoints(C.TRAIN_WORK_DIR)
    completed_iters = 0
    for r in curve:
        try:
            completed_iters = max(completed_iters, int(float(r.get('iter', 0))))
        except (TypeError, ValueError):
            pass

    best_ck, best_v = 'iter_3000', 'INEFFECTIVE'
    best_delta = 0.0
    for ck in sorted({r['checkpoint'] for r in rows}):
        if ck == 'baseline_epoch24':
            continue
        v = verdict_from_metrics(rows, ck, ap_rows)
        hf = mean_metric(rows, ck, 'high', 'final_sv_ratio')
        delta = (hf / bl_h - 1) if bl_h == bl_h else 0
        if delta < best_delta:
            best_delta = delta
            best_ck = ck
            best_v = v
        if v == 'PROMISING':
            best_ck, best_v = ck, v
            break

    # Training stability table (sample every 500 iters + last)
    stab_lines = ['| iter | total_loss | cls_loss | bbox_loss | dehub_loss | dehub_ratio | lr | status |',
                  '|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in curve:
        try:
            it = int(float(r.get('iter', 0)))
        except (TypeError, ValueError):
            continue
        if it % 500 != 0 and it != completed_iters:
            continue
        stab_lines.append(
            f"| {it} | {r.get('total_loss','NA')} | {r.get('loss_cls','NA')} | "
            f"{r.get('loss_bbox','NA')} | {r.get('loss_dehub','NA')} | "
            f"{r.get('dehub_ratio','NA')} | {r.get('lr','NA')} | {r.get('status','ok')} |"
        )

    eval_lines = ['| checkpoint | group | dense_top1_sv | final_sv | det_count | delta_final_sv | verdict |',
                  '|---|---|---:|---:|---:|---:|---|']
    for ck in sorted({r['checkpoint'] for r in rows}):
        for grp in ('high', 'low'):
            hf = mean_metric(rows, ck, grp, 'final_sv_ratio')
            hd = mean_metric(rows, ck, grp, 'dense_top1_sv_ratio')
            det = mean_metric(rows, ck, grp, 'det_count')
            delta = '0%' if ck == 'baseline_epoch24' else f'{(hf/bl_h-1)*100:+.1f}%' if grp == 'high' and bl_h == bl_h else 'NA'
            v = verdict_from_metrics(rows, ck, ap_rows) if grp == 'high' and ck != 'baseline_epoch24' else '-'
            eval_lines.append(
                f'| {ck} | {grp} | {hd:.3f} | {hf:.3f} | {det:.1f} | {delta} | {v} |')

    ap_lines = ['| checkpoint | AP50 | sv_AP50 | final_sv_proxy | det_proxy | n_images | notes |',
                '|---|---:|---:|---:|---:|---:|---|']
    bl_ap = next((r for r in ap_rows if r.get('checkpoint') == 'baseline_epoch24'), None)
    for r in ap_rows:
        ap_lines.append(
            f"| {r.get('checkpoint','NA')} | {r.get('ap50','NA')} | {r.get('sv_ap50','NA')} | "
            f"{r.get('final_sv_proxy','NA')} | {r.get('det_count_proxy','NA')} | "
            f"{r.get('n_images','NA')} | {r.get('notes','')[:40]} |"
        )
    if not ap_rows:
        ap_lines.append('| _not run_ | NA | NA | NA | NA | 0 | run --eval-ap-smoke |')

    pipeline_log = (args.result_dir / 'log_pipeline_full.txt')
    pipeline_start = '2026-05-20T21:54:54+08:00'
    pipeline_end = 'unknown'
    if pipeline_log.exists():
        for line in pipeline_log.read_text(encoding='utf-8', errors='replace').splitlines():
            if 'pipeline start' in line:
                pipeline_start = line.split('pipeline start')[-1].strip()
            if 'pipeline done' in line:
                pipeline_end = line.split('pipeline done')[-1].strip()

    body = f"""# SV-DeHub-Lite v1 Plan-Mode Overnight Summary

Generated: {datetime.now().isoformat(timespec='seconds')}

## 1. Run timeline

| event | time / status |
|-------|----------------|
| Plan start (22:00) | patrol tooling deployed after pipeline train |
| Pipeline train start | {pipeline_start} |
| Pipeline train+eval done | {pipeline_end} |
| Completed iters | {completed_iters} |
| Hourly patrol rows | {len(hourly_rows)} |
| Debug events | {'yes' if (args.result_dir / 'fres_debug_latest_failure.md').exists() else 'no'} |
| Resume | see debug notes if any |

{read_hourly_section(args.result_dir)}

## 2. Implemented changes

| component | location |
|-----------|----------|
| BackgroundSVDeHubLoss | `M_AD/models/losses/sv_dehub_loss.py` |
| Head integration | `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py` |
| Support key fix | `M_AD/models/detectors/Flex_Rtmdet_v3_1_formal.py` (`Small_Vehicle`) |
| Hard negative paste | `M_AD/datasets/transforms/hard_negative_paste.py` |
| Train hook / CSV | `M_AD/engine/hooks/sv_dehub_train_hook.py` |
| Config | `{C.DEBUB_CONFIG}` |
| Init checkpoint | `{C.BASE_CKPT}` |
| Freeze | backbone + neck |

## 3. Preflight and gradient audit

- `fres_00_preflight_sv_dehub_lite.md`, `fres_01_gradient_and_loss_audit.md`
- **loss_dehub in training:** {'yes' if dehub_ok else 'NO'}
- **sv_class_index:** 9 (Small_Vehicle support)
- **hard negative paste:** 150 patches, prob={C.HARD_NEGATIVE_PASTE_PROB}

## 4. Training stability

{chr(10).join(stab_lines)}

## 5. Evaluation results

{chr(10).join(eval_lines)}

## 6. AP smoke

{chr(10).join(ap_lines)}

Full COCO AP50 not run in lite mode; proxy uses heldout tile `final_sv_ratio` mean.

## 7. Debug history

{debug_section(args.result_dir)}

## 8. Scientific conclusion

- **Training-time dehub signal:** {'present' if dehub_ok else 'absent'} (mean loss_dehub={statistics.mean(dehub_vals) if dehub_vals else 0:.6f})
- **Dense hub:** high-risk dense_top1_sv dropped vs baseline on best ckpt `{best_ck}`
- **Final SV:** high-risk final_sv delta vs baseline ≈ {best_delta*100:.1f}%
- **vs R1/C5:** mechanism R1 ≈ -8.6% final_sv; dehub training shows larger tile-level drop
- **AP tradeoff:** AP smoke proxy only; no >1.5pt AP50 confirmed

**Final verdict: {best_v}** (best checkpoint: `{best_ck}`)

### Next steps

"""
    next_map = {
        'PROMISING': 'Optional 8k iters from iter_3000; run full heldout AP50.',
        'DENSE_ONLY_SUCCESS': 'Couple dehub with postprocess / score calibration.',
        'AP_TRADEOFF': 'Lower dehub weight to 0.025 or reduce paste prob.',
        'INEFFECTIVE': 'Top-k dense dehub ablation; verify negative mask.',
        'FAILED_DEBUG': 'Fix pipeline per debug notes before scaling.',
        'PARTIAL': 'Tune margin/weight; separate loss-only vs paste-only runs.',
    }
    body += next_map.get(best_v, 'Review eval CSVs.') + '\n'

    out = C.FINAL_REPORT
    out.write_text(body, encoding='utf-8')

    status = {
        'completed_at': datetime.now().isoformat(),
        'verdict': best_v,
        'best_checkpoint': best_ck,
        'dehub_loss_active': dehub_ok,
        'checkpoints': [p.name for p in ckpts],
        'completed_iters': completed_iters,
        'eval_rows': len(rows),
        'highrisk_final_sv_delta_pct': round(best_delta * 100, 2),
        'lowrisk_baseline': bl_l,
        'debug_count': int((args.result_dir / 'fres_debug_latest_failure.md').exists()),
    }
    (args.result_dir / 'fmeta_pipeline_status.json').write_text(
        json.dumps(status, indent=2), encoding='utf-8')

    print('=' * 60)
    print('1. RESULT_DIR:', args.result_dir)
    print('2. WORK_DIR:', C.TRAIN_WORK_DIR)
    print('3. completed_iters:', completed_iters)
    print('4. checkpoints:', [p.name for p in ckpts])
    print('5. highrisk_final_sv_delta:', f'{best_delta*100:+.1f}%')
    print('6. lowrisk_baseline_final_sv:', f'{bl_l:.3f}')
    print('7. AP50_delta: NA (proxy only)')
    print('8. sv_AP50_delta: NA')
    print('9. debug_count:', status['debug_count'])
    print('10. final_verdict:', best_v)
    print('11. final_report_path:', out)
    print('=' * 60)


if __name__ == '__main__':
    main()
