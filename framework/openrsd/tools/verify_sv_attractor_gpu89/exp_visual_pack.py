#!/usr/bin/env python
"""Step 7: visual evidence pack."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.verify_sv_attractor_gpu89 import common as C


def run(ctx: C.RunContext) -> dict:
    fig_dir = ctx.work_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fres = ctx.fres_path('fres_07_visual_evidence_pack.md')
    figures = []

    dense_csv = ctx.exp_dir('exp_02_dense') / 'ftable_dense_by_angle.csv'
    if dense_csv.exists():
        rows = C.read_csv(dense_csv)
        angles = [int(r['angle']) for r in rows]
        vals = [float(r['dense_top1_sv_ratio']) for r in rows]
        p = fig_dir / 'angle_vs_dense_sv_ratio.png'
        plt.figure(figsize=(8, 4))
        plt.plot(angles, vals, 'o-')
        plt.xlabel('angle'); plt.ylabel('dense_top1_sv_ratio')
        plt.title('P0148 angle vs dense small-vehicle ratio')
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        figures.append(('P1 dense bias vs angle', p))

    inter_csv = ctx.exp_dir('exp_03_intervention') / 'ftable_embedding_intervention_final.csv'
    if inter_csv.exists():
        rows = C.read_csv(inter_csv)
        ints = sorted(set(r['intervention'] for r in rows))
        means = [C.mean_key([r for r in rows if r['intervention'] == i], 'final_sv_ratio')
                 for i in ints]
        p = fig_dir / 'intervention_final_sv_ratio.png'
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(ints)), means)
        plt.xticks(range(len(ints)), ints, rotation=45, ha='right')
        plt.ylabel('final_sv_ratio'); plt.title('P2 embedding interventions')
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        figures.append(('P2 intervention bar', p))

    geom = ctx.exp_dir('exp_01_mapping') / 'ftable_embedding_geometry.csv'
    if geom.exists():
        rows = [r for r in C.read_csv(geom) if r.get('support_type') == 'visual']
        names = [r['class_name'] for r in rows]
        cos_sv = [float(r['cosine_to_small_vehicle']) for r in rows]
        p = fig_dir / 'cosine_to_small_vehicle.png'
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(names)), cos_sv)
        plt.xticks(range(len(names)), names, rotation=60, ha='right')
        plt.ylabel('cosine'); plt.title('Mapped embedding cosine to small-vehicle')
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        figures.append(('P0/P2 embedding geometry', p))

    crop_dir = ctx.exp_dir('exp_04_bg_gt') / 'crops'
    if crop_dir.is_dir():
        crops = sorted(crop_dir.glob('*.jpg'))[:12]
        if crops:
            p = fig_dir / 'crop_montage_top12.jpg'
            # simple grid montage via cv2 if available
            try:
                import cv2
                thumbs = []
                for cp in crops:
                    im = cv2.imread(str(cp))
                    if im is not None:
                        thumbs.append(cv2.resize(im, (128, 128)))
                if thumbs:
                    row = np.concatenate(thumbs[:6], axis=1)
                    row2 = np.concatenate(thumbs[6:12], axis=1) if len(thumbs) > 6 else row
                    montage = np.vstack([row, row2]) if len(thumbs) > 6 else row
                    cv2.imwrite(str(p), montage)
                    figures.append(('P3 crop montage', p))
            except Exception:
                pass

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Visual Evidence Pack', ctx)
        f.write('## Figures\n\n')
        for title, path in figures:
            f.write(f'### {title}\n\n')
            f.write(f'![{title}]({path})\n\n')
            f.write(f'- path: `{path}`\n')
        if not figures:
            f.write('- status: PARTIAL — no upstream CSV/figures available yet.\n')
        f.write(f'\nfigures_dir: `{fig_dir}`\n')

    ctx.mark_step('visual_pack', 'OK' if figures else 'PARTIAL')
    return dict(status='OK', fres=str(fres), n_figures=len(figures))
