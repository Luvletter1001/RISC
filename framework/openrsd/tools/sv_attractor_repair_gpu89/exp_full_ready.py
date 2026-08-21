#!/usr/bin/env python
"""Write fres_00b_full_ready_check.md — do NOT start full training."""
from __future__ import annotations

import json
from pathlib import Path

from tools.sv_attractor_repair_gpu89 import common as C
from tools.sv_attractor_repair_gpu89 import criteria
from tools.sv_attractor_repair_gpu89 import exp_r2_visual
from tools.sv_attractor_repair_gpu89.eval_ap import run_evaluator_smoke_test, run_subprocess_eval_metric


def run(ctx: C.RepairContext) -> dict:
    fres = ctx.fres('fres_00b_full_ready_check.md')
    splits = C.build_splits(ctx)

    eval_smoke = run_evaluator_smoke_test(ctx, 4)
    subprocess_note = dict(status='SKIPPED', reason='no pkl handy in smoke')
    pkl_candidates = list(ctx.work_dir.glob('**/*.pkl'))
    if pkl_candidates:
        log = ctx.log_dir() / 'openrsd_eval_subprocess.log'
        subprocess_note = run_subprocess_eval_metric(
            pkl_candidates[0], ctx.config, log)

    heldout_csv = ctx.tables_dir() / 'ftable_heldout_smoke_eval.csv'
    heldout_rows = C.read_csv(heldout_csv) if heldout_csv.exists() else []

    r2_visual_impl = exp_r2_visual.is_implemented()
    r2_smoke_ckpt = ctx.progress.get('best_r2_ckpt', '')
    r2_visual_ckpts = ctx.progress.get('r2_visual_ckpts', {})

    same_work_dir = True
    full_cmd = (
        f'CUDA_VISIBLE_DEVICES=8,9 /data/zcy/anaconda3/envs/openrsd/bin/python '
        f'tools/sv_attractor_repair_gpu89/run_sv_attractor_repair_gpu89.py '
        f'--mode full --work-dir {ctx.work_dir} --gpu 8 --exp all'
    )

    with open(fres, 'w') as f:
        C.write_fres_header(f, 'Full-Ready Check (do not auto-start full)', ctx)
        f.write('## 1. Evaluator availability\n\n')
        f.write(f'- `tools/openrsd_eval_metric.py`: present\n')
        f.write(f'- `tools/analysis_tools/eval_metric.py` merge+offline: present\n')
        f.write(f'- DOTAMetric smoke (heldout subset): **{eval_smoke.get("status")}**\n')
        if eval_smoke.get('error'):
            f.write(f'  - failure: `{eval_smoke["error"][:800]}`\n')
        else:
            f.write(f'  - ap50 smoke: `{eval_smoke.get("ap50_overall")}`\n')
        f.write(f'- subprocess openrsd_eval_metric: **{subprocess_note.get("status")}** '
                f'({subprocess_note.get("reason", subprocess_note.get("error", ""))[:200]})\n')

        f.write('\n## 2. AP50 pipeline\n\n')
        ap_ok = eval_smoke.get('status') == 'OK' and heldout_rows
        f.write(f'- heldout `eval_rbbox_map` path: **{"OK" if heldout_rows else "PENDING — run heldout_eval"}**\n')
        if heldout_rows:
            f.write('| method | AP50 | sv_AP50 | det_total | final_sv |\n')
            f.write('|---|---:|---:|---:|---:|\n')
            for r in heldout_rows:
                f.write(f"| {r.get('method')} | {float(r.get('ap50_overall', 'nan')):.4f} | "
                        f"{float(r.get('sv_ap50', 'nan')):.4f} | "
                        f"{float(r.get('detection_total', 'nan')):.1f} | "
                        f"{float(r.get('final_sv_ratio', 'nan')):.4f} |\n")
        f.write('- full outputs: overall AP50, per-class AP50, sv/lv/ship AP50 via `exp_eval` + `eval_ap`\n')
        f.write('- on evaluator failure: error written to log; **no ratio-only substitute for AP**\n')

        f.write('\n## 3. R2 visual prototypes\n\n')
        f.write(f'- R2-smoke (logit-space K8): `{r2_smoke_ckpt}` exists={Path(r2_smoke_ckpt).exists() if r2_smoke_ckpt else False}\n')
        f.write(f'- R2B visual code (`exp_r2_visual.py`): **{"implemented" if r2_visual_impl else "missing"}**\n')
        f.write(f'- R2B checkpoints (full train): `{r2_visual_ckpts or "not trained yet — run full r2_negative + R2B"}`\n')
        f.write('- full: `R2B_visual_proto_K4/K8/K16`, outside-GT mining, activation CSV\n')

        f.write('\n## 4. Train / heldout split\n\n')
        f.write(f'- P0148 excluded prefix: `{splits["p0148_excluded"]}`\n')
        f.write(f'- train_calib: {splits["n_train"]} | heldout_calib: {splits["n_heldout"]}\n')
        p0148_in_train = any(s.startswith(C.P0148_PREFIX) for s in splits['train_calib'])
        p0148_in_held = any(s.startswith(C.P0148_PREFIX) for s in splits['heldout_calib'])
        f.write(f'- P0148 in train_calib: `{p0148_in_train}` (must be False)\n')
        f.write(f'- P0148 in heldout_calib: `{p0148_in_held}` (must be False)\n')

        f.write('\n## 5. Full run work_dir\n\n')
        f.write(f'- smoke work_dir: `{ctx.work_dir}`\n')
        f.write(f'- full reuses same work_dir: **{same_work_dir}** (pass `--work-dir {ctx.work_dir}`)\n')
        f.write(f'- example full command:\n\n```bash\n{full_cmd}\n```\n')

        f.write('\n## 6. Fixed success criteria (full)\n\n')
        for k, v in criteria.FULL_SUCCESS_CRITERIA.items():
            if k != 'notes':
                f.write(f'- {k}: `{v}`\n')
        for note in criteria.FULL_SUCCESS_CRITERIA.get('notes', []):
            f.write(f'- {note}\n')

        f.write('\n## 7. Expected full artifacts\n\n')
        f.write('| artifact | path |\n')
        f.write('|---|---|\n')
        f.write(f'| checkpoints | `{ctx.ckpt_dir()}/R1*.pth, r2*.pth, R2B_visual_proto_K*.pth, r3*.pth` |\n')
        f.write(f'| tables | `{ctx.tables_dir()}/ftable_*.csv` |\n')
        f.write(f'| logs | `{ctx.log_dir()}/*.json, evaluator logs` |\n')
        f.write(f'| reports | `{ctx.result_md_dir}/fres_*.md` |\n')

        f.write('\n## 8. Gate\n\n')
        ap_path_ok = bool(heldout_rows) and all(
            float(r.get('ap50_overall', 'nan')) == float(r.get('ap50_overall', 'nan'))
            for r in heldout_rows if 'error' not in r)
        ready = (
            ap_path_ok
            and r2_visual_impl
            and not p0148_in_train
            and not p0148_in_held
        )
        f.write(f'- DOTAMetric offline smoke: `{eval_smoke.get("status")}` '
                f'(primary AP path: eval_rbbox_map = `{"OK" if ap_path_ok else "FAIL"}`)\n')
        f.write(f'- **full_ready (code + smoke AP path)**: `{ready}` — '
                f'**user must still confirm before launching full.**\n')

    ctx.mark('full_ready', 'OK', ready=ready)
    return dict(status='OK', fres=str(fres), ready=ready)
