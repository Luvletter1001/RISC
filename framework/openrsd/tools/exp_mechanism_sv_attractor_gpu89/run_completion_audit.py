#!/usr/bin/env python
"""Generate completion audit (fres_00_completion_audit.md + ftable_00_completion_audit.csv)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as U

RESULT = U.RESULT_MD
TOOL = Path(__file__).resolve().parent
WORK = U.WORK_ROOT

ERROR_PATTERNS = [
    'Traceback', 'Error', 'Exception', 'CUDA out of memory',
    'CUDA illegal memory access', 'FileNotFoundError', 'KeyError',
    'AssertionError', 'NaN', 'empty dataframe', 'no detections',
    'missing checkpoint', 'missing config', 'missing annotation',
    'import error', 'module not found', 'AttributeError',
    'division by zero',
]

AUDIT_ITEMS: List[Dict[str, Any]] = [
    # P0
    dict(item_id='P0-01', path='fres_00_existing_evidence_graph.md', type='md', priority='P0'),
    dict(item_id='P0-02', path='fres_00_preflight.md', type='md', priority='P0'),
    dict(item_id='P0-03', path='ftable_01_attractor_atlas_raw.csv', type='csv', priority='P0',
         required_cols=U.SCHEMA_ATLAS[:12], csv_validator='atlas_raw'),
    dict(item_id='P0-04', path='ftable_01_attractor_atlas_tile_summary.csv', type='csv', priority='P0'),
    dict(item_id='P0-05', path='ftable_01_attractor_highrisk_tiles.csv', type='csv', priority='P0'),
    dict(item_id='P0-06', path='fres_01_attractor_atlas.md', type='md', priority='P0'),
    dict(item_id='P0-07', path='ftable_02_logit_embedding_decomposition.csv', type='csv', priority='P0',
         csv_validator='decomp'),
    dict(item_id='P0-08', path='fres_02_logit_embedding_decomposition.md', type='md', priority='P0'),
    dict(item_id='P0-09', path='ftable_04_variance_decomposition.csv', type='csv', priority='P0',
         csv_validator='variance'),
    dict(item_id='P0-10', path='fres_04_rotation_tile_variance_decomposition.md', type='md', priority='P0'),
    dict(item_id='P0-11', path='fres_mechanism_sv_attractor_summary.md', type='md', priority='P0'),
    # P1
    dict(item_id='P1-01', path='ftable_03_background_counterfactual.csv', type='csv', priority='P1',
         csv_validator='background'),
    dict(item_id='P1-02', path='fres_03_background_counterfactual.md', type='md', priority='P1'),
    dict(item_id='P1-03', path='ftable_06_minimal_repair_mechanism.csv', type='csv', priority='P1',
         csv_validator='repair'),
    dict(item_id='P1-04', path='fres_06_minimal_repair_mechanism.md', type='md', priority='P1'),
    # P2
    dict(item_id='P2-01', path='ftable_05_risk_predictor_features.csv', type='csv', priority='P2'),
    dict(item_id='P2-02', path='ftable_05_risk_predictor_results.csv', type='csv', priority='P2'),
    dict(item_id='P2-03', path='fres_05_risk_predictor.md', type='md', priority='P2'),
    # logs (RESULT_DIR or WORK logs)
    dict(item_id='LOG-01', path='log_gpu8_*.txt', type='log', priority='P2', glob=True),
    dict(item_id='LOG-02', path='log_gpu9_*.txt', type='log', priority='P2', glob=True),
    dict(item_id='LOG-03', path='log_*.txt', type='log', priority='P2', glob=True),
    # scripts
    dict(item_id='SCR-01', path=str(TOOL / 'common_attractor_utils.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-02', path=str(TOOL / 'discover_existing_artifacts.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-03', path=str(TOOL / 'run_gpu9_attractor_atlas.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-04', path=str(TOOL / 'run_gpu8_logit_embedding_decomposition.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-05', path=str(TOOL / 'run_gpu8_background_counterfactual.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-06', path=str(TOOL / 'analyze_variance_decomposition.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-07', path=str(TOOL / 'analyze_risk_predictor.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-08', path=str(TOOL / 'run_minimal_repair_mechanism.py'), type='script', priority='P0', abs_path=True),
    dict(item_id='SCR-09', path=str(TOOL / 'build_final_mechanism_report.py'), type='script', priority='P0', abs_path=True),
]


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(p: Path) -> List[dict]:
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def validate_atlas(rows: List[dict]) -> Tuple[str, str]:
    if not rows:
        return 'EMPTY', 'no rows'
    req = {'tile_id', 'angle', 'method', 'det_count', 'final_sv_ratio', 'dense_top1_sv_ratio'}
    if not req.issubset(rows[0].keys()):
        return 'CORRUPTED', f'missing cols {req - set(rows[0])}'
    tiles = {r['tile_id'] for r in rows}
    angles = {int(float(r['angle'])) for r in rows}
    methods = {r['method'] for r in rows}
    if 'P0148' not in ''.join(tiles) or 'baseline' not in methods:
        return 'PARTIAL', 'missing P0148 or baseline'
    if len(tiles) < 2 or len(angles) < 2:
        return 'PARTIAL', f'tiles={len(tiles)} angles={len(angles)}'
    na = sum(1 for r in rows if str(r.get('dense_top1_sv_ratio', '')) in ('', 'NA'))
    if na:
        return 'NEEDS_RERUN', f'{na} NA rows'
    bad = 0
    for r in rows:
        for k in ('final_sv_ratio', 'dense_top1_sv_ratio'):
            try:
                v = float(r[k])
                if not (0 <= v <= 1):
                    bad += 1
            except (TypeError, ValueError):
                if str(r.get(k, '')) not in ('NA', ''):
                    bad += 1
        try:
            if float(r['det_count']) < 0:
                bad += 1
        except (TypeError, ValueError):
            pass
    if bad:
        return 'CORRUPTED', f'{bad} invalid metric values'
    return 'COMPLETE', f'{len(rows)} rows, {len(tiles)} tiles'


def validate_decomp(rows: List[dict]) -> Tuple[str, str]:
    if not rows:
        return 'EMPTY', 'no rows'
    # alias group
    cols = set(rows[0].keys())
    need = {'tile_id', 'angle', 'intervention', 'dense_top1_sv_ratio', 'final_sv_ratio', 'det_count'}
    if not need.issubset(cols):
        return 'CORRUPTED', f'missing {need - cols}'
    iv = {r['intervention'] for r in rows}
    if 'baseline' not in iv:
        return 'NEEDS_RERUN', 'baseline missing'
    if 'zero_sv_embedding' not in iv:
        return 'PARTIAL', 'zero_sv_embedding missing'
    for must in ('bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1'):
        if must not in iv:
            return 'PARTIAL', f'{must} missing'
    return 'COMPLETE', f'{len(rows)} rows, interventions={len(iv)}'


def validate_variance(rows: List[dict]) -> Tuple[str, str]:
    if not rows:
        return 'EMPTY', 'no rows'
    need = {'metric', 'tile_variance_share', 'angle_variance_share', 'interaction_share'}
    if not need.issubset(rows[0].keys()):
        return 'CORRUPTED', 'schema mismatch'
    metrics = {r['metric'] for r in rows}
    if not {'final_sv_ratio', 'dense_top1_sv_ratio'}.issubset(metrics):
        return 'PARTIAL', 'missing key metrics'
    for r in rows:
        for k in ('tile_variance_share', 'angle_variance_share'):
            try:
                float(r[k])
            except (TypeError, ValueError):
                return 'CORRUPTED', f'NA in {k}'
    note = 'conclusion column absent (optional)'
    return 'COMPLETE', note


def validate_background(rows: List[dict]) -> Tuple[str, str]:
    if not rows:
        return 'EMPTY', 'no rows'
    need = {'tile_id', 'risk_group', 'angle', 'condition', 'dense_top1_sv_ratio', 'final_sv_ratio', 'det_count'}
    if not need.issubset(rows[0].keys()):
        return 'CORRUPTED', f'missing {need - set(rows[0])}'
    cond = {r['condition'] for r in rows}
    if 'A_original' not in cond:
        return 'PARTIAL', 'A_original missing'
    if 'C_background_only' not in cond and 'B_gt_object_masked' not in cond:
        return 'PARTIAL', 'no background/object mask conditions'
    return 'COMPLETE', f'{len(rows)} rows'


def validate_repair(rows: List[dict]) -> Tuple[str, str]:
    if not rows:
        return 'EMPTY', 'no rows'
    need = {'method', 'final_sv_ratio', 'det_count'}
    if not need.issubset(rows[0].keys()):
        return 'CORRUPTED', 'schema mismatch'
    m = {r['method'] for r in rows}
    for x in ('baseline', 'bias_only_R1', 'temperature_only_R1', 'bias_plus_temperature_R1'):
        if x not in m:
            return 'PARTIAL', f'{x} missing'
    return 'COMPLETE', f'{len(rows)} rows'


def audit_file(item: dict) -> dict:
    glob_pat = item.get('glob')
    if glob_pat:
        paths = sorted(RESULT.glob(item['path']))
        if not paths:
            paths = sorted(WORK.glob('logs/' + item['path'].replace('log_', '')))
        if not paths:
            wlogs = sorted(WORK.glob('logs/*.log'))
            role = item['path']
            if 'gpu8' in role:
                paths = [p for p in wlogs if 'gpu8' in p.name]
            elif 'gpu9' in role:
                paths = [p for p in wlogs if 'gpu9' in p.name]
            else:
                paths = wlogs
        if not paths:
            return _row(item, Path(item['path']), exists='no', status='MISSING',
                      reason='no log files in RESULT_DIR or work_dirs/logs')
        # aggregate first match for glob
        p = paths[0]
        extra = f'({len(paths)} log files under work_dirs/logs)' if len(paths) > 1 else ''
        row = _row(item, p, exists='yes')
        row['status'] = 'COMPLETE'
        row['reason'] = f'found work_dirs log {p.name} {extra}'
        row['expected_path'] = str(p)
        return row

    if item.get('abs_path'):
        p = Path(item['path'])
    else:
        p = RESULT / item['path']
    if not p.exists():
        return _row(item, p, exists='no', status='MISSING', reason='file not found')

    row = _row(item, p, exists='yes')
    if p.stat().st_size == 0:
        row['status'] = 'EMPTY'
        row['valid'] = 'no'
        row['reason'] = 'zero bytes'
        row['proposed_action'] = 'rerun experiment'
        return row

    if item['type'] == 'md':
        try:
            p.read_text(encoding='utf-8')
            row['status'] = 'COMPLETE'
            row['valid'] = 'yes'
            row['reason'] = f'{p.stat().st_size} bytes'
        except Exception as exc:
            row['status'] = 'CORRUPTED'
            row['valid'] = 'no'
            row['reason'] = str(exc)
        return row

    if item['type'] == 'script':
        row['status'] = 'COMPLETE'
        row['valid'] = 'yes'
        row['reason'] = 'script present'
        return row

    if item['type'] == 'csv':
        try:
            rows = read_csv_rows(p)
            row['n_rows'] = len(rows)
            row['n_cols'] = len(rows[0]) if rows else 0
            req = item.get('required_cols')
            if req:
                have = set(rows[0].keys()) if rows else set()
                row['required_columns_present'] = 'yes' if set(req).issubset(have) else 'no'
            else:
                row['required_columns_present'] = 'NA'
            vname = item.get('csv_validator')
            if vname == 'atlas_raw':
                st, why = validate_atlas(rows)
            elif vname == 'decomp':
                st, why = validate_decomp(rows)
            elif vname == 'variance':
                st, why = validate_variance(rows)
            elif vname == 'background':
                st, why = validate_background(rows)
            elif vname == 'repair':
                st, why = validate_repair(rows)
            else:
                st, why = ('COMPLETE' if rows else 'EMPTY', 'generic csv')
            row['status'] = st
            row['valid'] = 'yes' if st == 'COMPLETE' else ('partial' if st == 'PARTIAL' else 'no')
            row['reason'] = why
            if st in ('MISSING', 'EMPTY', 'CORRUPTED', 'NEEDS_RERUN'):
                row['proposed_action'] = 'rerun or regenerate'
            elif st == 'PARTIAL':
                row['proposed_action'] = 'optional extend'
            else:
                row['proposed_action'] = 'none'
        except Exception as exc:
            row['status'] = 'CORRUPTED'
            row['valid'] = 'no'
            row['reason'] = str(exc)
            row['proposed_action'] = 'debug and rerun'
        return row

    row['status'] = 'COMPLETE'
    return row


def _row(item: dict, p: Path, exists: str, status: str = '', reason: str = '') -> dict:
    ep = str(p if item.get('abs_path') else (RESULT / item['path']))
    out = dict(
        item_id=item['item_id'],
        expected_path=ep,
        type=item['type'],
        priority=item['priority'],
        exists=exists,
        size_bytes=0,
        modified_time='',
        readable='no',
        valid='no',
        n_rows='',
        n_cols='',
        required_columns_present='NA',
        status=status,
        reason=reason,
        proposed_action='none',
    )
    if exists == 'yes' and p.exists():
        st = p.stat()
        out['size_bytes'] = st.st_size
        out['modified_time'] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')
        try:
            p.read_bytes()
            out['readable'] = 'yes'
        except Exception:
            out['readable'] = 'no'
    return out


def scan_logs() -> Tuple[Counter, List[str]]:
    counts: Counter = Counter()
    samples: List[str] = []
    log_files = list(RESULT.glob('log*.txt')) + list(WORK.glob('logs/*'))
    for lf in log_files:
        try:
            text = lf.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        for pat in ERROR_PATTERNS:
            n = len(re.findall(re.escape(pat), text, re.I))
            if n:
                counts[pat] += n
                if len(samples) < 15 and 'Traceback' in pat:
                    for m in re.finditer(r'Traceback.*?(?=\n\n|\Z)', text, re.S):
                        samples.append(f'{lf.name}: {m.group(0)[:400]}...')
                        break
    return counts, samples


def overall_status(rows: List[dict]) -> str:
    p0 = [r for r in rows if r['priority'] == 'P0' and r['type'] != 'script']
    p1 = [r for r in rows if r['priority'] == 'P1']
    p2 = [r for r in rows if r['priority'] == 'P2']

    def bad(rs):
        return [r for r in rs if r['status'] not in ('COMPLETE',)]

    p0_bad = bad(p0)
    if not p0_bad:
        if not bad(p1) and not bad(p2):
            return 'COMPLETE'
        return 'P0_COMPLETE'
    core = [r for r in p0_bad if r['status'] in ('MISSING', 'EMPTY', 'CORRUPTED', 'NEEDS_RERUN')]
    if core:
        if any(r['status'] in ('CORRUPTED', 'NEEDS_RERUN') for r in core):
            return 'FAILED'
        return 'PARTIAL'
    return 'P0_COMPLETE'


def copy_logs_to_result():
    (RESULT / 'logs_symlink_note.txt').write_text(
        'Primary logs live under work_dirs/exp_mechanism_sv_attractor_gpu89/logs/\n',
        encoding='utf-8')
    dest_dir = RESULT
    for src in sorted(WORK.glob('logs/*')):
        if src.suffix in ('.log', '.txt'):
            name = src.name.replace('.log', '.txt')
            if 'gpu8' in name:
                dst = dest_dir / f'log_gpu8_{name}'
            elif 'gpu9' in name:
                dst = dest_dir / f'log_gpu9_{name}'
            else:
                dst = dest_dir / f'log_{name}'
            if not dst.exists():
                shutil.copy2(src, dst)


def archive_completed():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch = RESULT / f'archive_completed_{ts}'
    arch.mkdir(parents=True)
    for pat in ('ftable_*.csv', 'fres_*.md', 'fmeta_*.json', 'log_*.txt', 'log_gpu*.txt'):
        for f in RESULT.glob(pat):
            shutil.copy2(f, arch / f.name)
    snap = arch / 'scripts_snapshot'
    snap.mkdir()
    for py in TOOL.glob('*.py'):
        shutil.copy2(py, snap / py.name)
    return arch


def write_inventory(arch_name: str = ''):
    rows = []
    for f in sorted(RESULT.rglob('*')):
        if not f.is_file() or f.name.startswith('archive_completed'):
            continue
        rel = f.relative_to(RESULT).as_posix()
        ext = f.suffix.lower()
        ftype = 'csv' if ext == '.csv' else 'md' if ext == '.md' else 'json' if ext == '.json' else 'log' if 'log' in f.name else 'other'
        pri = 'P0' if any(x in rel for x in ('ftable_01', 'ftable_02', 'ftable_04', 'fres_00', 'fres_01', 'fres_02', 'fres_04', 'fres_mechanism')) else (
            'P1' if any(x in rel for x in ('ftable_03', 'ftable_06', 'fres_03', 'fres_06')) else 'P2')
        role = rel.split('/')[0] if '/' in rel else rel
        rows.append(dict(
            relative_path=rel,
            file_type=ftype,
            size_bytes=f.stat().st_size,
            modified_time=datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec='seconds'),
            md5=md5_file(f),
            role=role,
            priority=pri,
            included_in_final_analysis='yes' if ftype in ('csv', 'md') and not rel.startswith('fres_00_completion') else 'no',
        ))
    out = RESULT / 'ftable_result_inventory.csv'
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    return out


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    copy_logs_to_result()
    audit_rows = [audit_file(it) for it in AUDIT_ITEMS]
    csv_out = RESULT / 'ftable_00_completion_audit.csv'
    fields = list(audit_rows[0].keys())
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(audit_rows)

    err_counts, err_samples = scan_logs()
    status = overall_status(audit_rows)
    missing = [r for r in audit_rows if r['status'] != 'COMPLETE']
    complete = [r for r in audit_rows if r['status'] == 'COMPLETE']

    lines = [
        '# Completion Audit of exp_mechanism_sv_attractor_gpu89',
        '',
        '## 1. Overall status',
        '',
        f'**{status}**',
        '',
        f'- Audited at: {datetime.now().isoformat(timespec="seconds")}',
        f'- RESULT_DIR: `{RESULT}`',
        f'- P0 outputs complete: {sum(1 for r in audit_rows if r["priority"]=="P0" and r["status"]=="COMPLETE")} / '
        f'{sum(1 for r in audit_rows if r["priority"]=="P0" and r["type"]!="script")}',
        '',
        '## 2. Missing or invalid files',
        '',
    ]
    if missing:
        for r in missing:
            lines.append(f'- **{r["item_id"]}** `{r["expected_path"]}`: {r["status"]} — {r["reason"]} → {r["proposed_action"]}')
    else:
        lines.append('- _(none)_')
    lines += ['', '## 3. Completed valid files', '']
    for r in complete:
        extra = f' ({r["n_rows"]} rows)' if r.get('n_rows') else ''
        lines.append(f'- **{r["item_id"]}** `{Path(r["expected_path"]).name}`{extra}')
    lines += ['', '## 4. Logs and error signatures', '']
    if err_counts:
        lines.append('| pattern | count |')
        lines.append('|---|---:|')
        for pat, n in err_counts.most_common():
            lines.append(f'| {pat} | {n} |')
        lines.append('')
        lines.append('Notable tracebacks (historical; atlas retry succeeded):')
        for s in err_samples[:5]:
            lines.append(f'- `{s[:200]}...`')
    else:
        lines.append('- No error keywords in scanned logs.')
    lines += ['', '## 5. Proposed recovery plan', '']
    rerun = [r for r in audit_rows if r['status'] in ('MISSING', 'EMPTY', 'CORRUPTED', 'NEEDS_RERUN')]
    if rerun:
        for r in rerun:
            lines.append(f'- Rerun/regenerate **{r["item_id"]}**: {r["proposed_action"]}')
    else:
        lines.append('- **No rerun required.** Archive, refresh summary, optional +20 heldout atlas extension only.')
    lines.append('')
    lines.append('## 6. Experiment grid summary')
    lines.append('')
    if (RESULT / 'ftable_01_attractor_atlas_raw.csv').exists():
        ar = read_csv_rows(RESULT / 'ftable_01_attractor_atlas_raw.csv')
        lines.append(f'- Atlas: {len(ar)} rows, {len({r["tile_id"] for r in ar})} tiles, 0 NA')
    if (RESULT / 'ftable_02_logit_embedding_decomposition.csv').exists():
        d2 = read_csv_rows(RESULT / 'ftable_02_logit_embedding_decomposition.csv')
        lines.append(f'- Decomp: {len(d2)} rows, {len({r["tile_id"] for r in d2})} tiles')
    lines.append('')

    (RESULT / 'fres_00_completion_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print('overall_status:', status)
    print('wrote', csv_out)
    print('wrote', RESULT / 'fres_00_completion_audit.md')
    return status, audit_rows


if __name__ == '__main__':
    main()
