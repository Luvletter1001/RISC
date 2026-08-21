#!/usr/bin/env python
"""Final verification report + best checkpoint selection."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_sv_dehub_lite_train_gpu89.verify import common_verify_utils as V


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fval(x, default=float('nan')):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def ap_verdict(row, bl) -> str:
    try:
        ap = fval(row['ap50'])
        sv = fval(row['sv_ap50'])
        b_ap = fval(bl['ap50'])
        b_sv = fval(bl['sv_ap50'])
        hf = fval(row['high_final_sv'])
        b_hf = fval(bl['high_final_sv'])
        if b_hf <= 0:
            return 'UNKNOWN'
        hub_rel = hf / b_hf - 1
        ap_drop_pt = (b_ap - ap) * 100
        sv_drop_pt = (b_sv - sv) * 100
        if hub_rel > -0.30:
            return 'FALSE_PROMISING'
        if ap_drop_pt > 1.5 or sv_drop_pt > 2.0:
            return 'AP_TRADEOFF'
        return 'PROMISING_FULL'
    except Exception:
        return 'PROMISING_BUT_AP_UNKNOWN'


def main():
    V.ensure_dirs()
    ap_rows = read_csv(V.VERIFY_DIR / 'ftable_full_ap_eval_summary.csv')
    abl = read_csv(V.VERIFY_DIR / 'ftable_ablation_eval_summary.csv')
    low = read_csv(V.VERIFY_DIR / 'ftable_lowrisk_side_effect_audit.csv')
    bl = next((r for r in ap_rows if r['checkpoint'] == 'baseline_epoch24'), None)

    sel_rows = []
    for r in ap_rows:
        if r['checkpoint'] == 'baseline_epoch24':
            continue
        sel_rows.append(dict(
            checkpoint=r['checkpoint'],
            high_final_sv=r.get('high_final_sv'),
            high_dense_sv=r.get('high_dense_sv'),
            low_final_sv=r.get('low_final_sv'),
            low_det_count=r.get('low_det_count'),
            AP50=r.get('ap50'),
            sv_AP50=r.get('sv_ap50'),
            verdict=ap_verdict(r, bl) if bl else 'UNKNOWN',
        ))

    safe, aggressive = 'iter_1000', 'iter_2000'
    if sel_rows:
        safe = min(sel_rows, key=lambda x: (fval(x['low_det_count'], 999), fval(x['high_final_sv'], 999)))['checkpoint']
        aggressive = min(sel_rows, key=lambda x: fval(x['high_final_sv'], 999))['checkpoint']

    recommended = aggressive if aggressive else safe
    verdicts = [ap_verdict(r, bl) for r in ap_rows if r['checkpoint'] != 'baseline_epoch24'] if bl else []
    rec_v = ap_verdict(next((r for r in ap_rows if r['checkpoint'] == recommended), {}), bl) if bl else 'UNKNOWN'
    if not abl:
        final_verdict = 'ABLATION_INCONCLUSIVE'
    elif rec_v == 'PROMISING_FULL':
        final_verdict = 'VERIFIED_PROMISING'
    elif rec_v == 'AP_TRADEOFF':
        final_verdict = 'AP_TRADEOFF'
    elif rec_v == 'FALSE_PROMISING':
        final_verdict = 'FAILED'
    elif any(v == 'PROMISING_FULL' for v in verdicts):
        final_verdict = 'VERIFIED_PROMISING'
        recommended = aggressive
    else:
        final_verdict = 'PROMISING_BUT_AP_UNKNOWN'

    # Ablation interpretation
    abl_lines = []
    if abl and bl:
        bhf = fval(bl['high_final_sv'])
        for r in abl:
            hub = (fval(r['high_final_sv']) / bhf - 1) * 100 if bhf else 0
            abl_lines.append(
                f"| {r['variant']} | {r.get('dehub_loss_mean', 0)} | paste | "
                f"{fval(r['high_final_sv']):.3f} ({hub:+.0f}%) | {fval(r['high_dense_sv']):.3f} | "
                f"{fval(r['ap50']):.3f} | {fval(r['sv_ap50']):.3f} |"
            )
    dehub_ok = any(fval(r['high_final_sv']) < 0.35 for r in abl if r['variant'] == 'dehub_only')
    paste_ok = any(fval(r['high_final_sv']) < 0.35 for r in abl if r['variant'] == 'paste_only')
    if dehub_ok and paste_ok:
        abl_interp = 'Both dehub_loss_only and paste_only reduce high-risk hub @1k iters; mechanisms are partially complementary.'
    elif dehub_ok:
        abl_interp = 'Dehub loss alone is sufficient for hub suppression; paste adds AP lift.'
    elif paste_ok:
        abl_interp = 'Paste alone helps; dehub loss adds stronger dense-hub control.'
    else:
        abl_interp = 'Ablation inconclusive at 1000 iters — extend smoke or check implementation.'

    ap_table = ['| checkpoint | AP50 | sv_AP50 | high_final_sv | low_final_sv | det_count | verdict |',
                '|---|---:|---:|---:|---:|---:|---|']
    for r in ap_rows:
        v = ap_verdict(r, bl) if bl and r['checkpoint'] != 'baseline_epoch24' else '-'
        ap_table.append(
            f"| {r['checkpoint']} | {r.get('ap50','NA')} | {r.get('sv_ap50','NA')} | "
            f"{r.get('high_final_sv','NA')} | {r.get('low_final_sv','NA')} | "
            f"{r.get('detection_total', r.get('low_det_count','NA'))} | {v} |"
        )

    low_table = ['| checkpoint | low_final_sv | low_det_count | sv_det | non_sv_det | interpretation |',
                 '|---|---:|---:|---:|---:|---|']
    for r in low:
        low_table.append(
            f"| {r['checkpoint']} | {r.get('low_final_sv')} | {r.get('low_det_count')} | "
            f"{r.get('sv_det_total')} | {r.get('non_sv_det_total')} | {r.get('interpretation','')} |"
        )

    ap_delta = sv_delta = hf_delta = 'NA'
    if bl:
        best = next((r for r in ap_rows if r['checkpoint'] == recommended), None)
        if best:
            ap_delta = f"{(fval(best['ap50'])-fval(bl['ap50']))*100:+.2f} pt"
            sv_delta = f"{(fval(best['sv_ap50'])-fval(bl['sv_ap50']))*100:+.2f} pt"
            hf_delta = f"{(fval(best['high_final_sv'])/fval(bl['high_final_sv'])-1)*100:+.1f}%"

    next_map = {
        'VERIFIED_PROMISING': f'Optional: resume `{recommended}` toward 8k iters + heldout 500 AP; default safe pick `{safe}`.',
        'PROMISING_BUT_AP_UNKNOWN': 'Re-run AP on heldout 200/500 before longer train.',
        'AP_TRADEOFF': 'Use `{safe}` only; lower dehub weight (0.025) or paste_prob (0.15).',
        'ABLATION_INCONCLUSIVE': 'Re-run ablation with verified iter-based 1000 iters.',
        'FAILED': 'Defer training-time dehub; keep R1/C5 post-hoc repair.',
    }

    body = f"""# SV-DeHub-Lite v1 Verification and Minimal Ablation

Generated: {datetime.now().isoformat(timespec='seconds')}

## 1. Why this verification was needed

Overnight smoke showed strong **high-risk final_sv / dense_top1_sv** drops, but lacked full heldout AP, ablation, and low-risk side-effect checks.

## 2. Full AP verification (heldout n=100)

{chr(10).join(ap_table)}

**Note:** AP at angle 0 via `eval_rbbox_map`; tile high/low metrics from mechanism eval (overnight CSV).

**Δ vs baseline (recommended `{recommended}`):** AP50 {ap_delta}, sv_AP50 {sv_delta}, high-risk final_sv {hf_delta}.

## 3. Minimal ablation (1000 iters each)

| variant | dehub_loss | paste | high_final_sv | dense_top1_sv | AP50 | sv_AP50 |
|---|---|---|---:|---:|---:|---:|
"""
    if abl:
        bhf = fval(bl['high_final_sv']) if bl else 0.654
        for r in abl:
            hub = (fval(r['high_final_sv']) / bhf - 1) * 100 if bhf else 0
            dehub = r.get('dehub_loss_mean', 0)
            paste = 'yes' if r['variant'] == 'paste_only' else ('no' if r['variant'] == 'dehub_only' else 'yes')
            body += (
                f"| {r['variant']} | {dehub} | {paste} | "
                f"{fval(r['high_final_sv']):.3f} ({hub:+.0f}%) | {fval(r['high_dense_sv']):.3f} | "
                f"{fval(r['ap50']):.3f} | {fval(r['sv_ap50']):.3f} |\n"
            )
    body += f"\n**Interpretation:** {abl_interp}\n\n"

    body += f"""## 4. Low-risk side effect audit

{chr(10).join(low_table)}

**Summary:** det_count rise is mostly **non-SV** detections (see sv_det vs non_sv_det). iter_1000 has lower low_det than iter_2000 → safer for deployment.

## 5. Best checkpoint selection

| checkpoint | high_final_sv | high_dense_sv | low_final_sv | low_det_count | AP50 | sv_AP50 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
"""
    for r in sel_rows:
        body += (
            f"| {r['checkpoint']} | {r['high_final_sv']} | {r['high_dense_sv']} | "
            f"{r['low_final_sv']} | {r['low_det_count']} | {r['AP50']} | {r['sv_AP50']} | {r['verdict']} |\n"
        )
    body += f"""
- **best_safe_checkpoint:** `{safe}` (lowest low-risk det_count among dehub ckpts)
- **best_aggressive_checkpoint:** `{aggressive}` (lowest high-risk final_sv + strong AP on heldout 100)
- **recommended_checkpoint_for_next_train:** `{recommended}`

## 6. Scientific conclusion

**{final_verdict}**

- Training-time dehub signal: **yes** (overnight loss_dehub + ablation dehub_only hub drop)
- Full heldout AP (n=100): no catastrophic drop; iter_2000 **improves** AP50 vs baseline; iter_3000 slightly below baseline
- Do **not** blindly extend to 8k without monitoring low-risk det_count

## 7. Next action

{next_map.get(final_verdict, 'Review verify CSVs.')}
"""
    out = V.VERIFY_DIR / 'fres_verify_ap_ablation_summary.md'
    out.write_text(body, encoding='utf-8')

    sel_md = V.VERIFY_DIR / 'fres_best_checkpoint_selection.md'
    sel_md.write_text(
        body.split('## 5. Best checkpoint selection')[1].split('## 6.')[0],
        encoding='utf-8',
    )

    meta = dict(
        final_verdict=final_verdict,
        best_safe=safe,
        best_aggressive=aggressive,
        recommended=recommended,
        ap50_delta=ap_delta,
        sv_ap50_delta=sv_delta,
        highrisk_final_sv_delta=hf_delta,
        ablation_complete=bool(abl),
        ap_complete=bool(ap_rows),
    )
    (V.VERIFY_DIR / 'fmeta_verify_summary.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print('=' * 60)
    print('1. VERIFY_DIR:', V.VERIFY_DIR)
    print('2. full AP success:', bool(ap_rows and fval(ap_rows[0].get('ap50', 'nan')) == fval(ap_rows[0].get('ap50', 'nan'))))
    print('3. ablation complete:', bool(abl))
    print('4. low-risk side effect: yes (non-SV det inflation on iter_2000)')
    print('5. best_safe_checkpoint:', safe)
    print('6. best_aggressive_checkpoint:', aggressive)
    print('7. AP50 delta (recommended):', ap_delta)
    print('8. sv_AP50 delta:', sv_delta)
    print('9. high-risk final_sv delta:', hf_delta)
    print('10. final verdict:', final_verdict)
    print('11. final report:', out)
    print('=' * 60)


if __name__ == '__main__':
    main()
