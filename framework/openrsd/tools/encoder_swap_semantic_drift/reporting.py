#!/usr/bin/env python3
"""Markdown report helpers for encoder swap experiments."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import shutil

from tools.encoder_swap_semantic_drift.common import ExpResult, write_csv

# Canonical result-md filenames (user spec); also written under exp_XX_* subdirs.
FRES_ALIASES = {
    'exp_00': 'fres_00_encoder_inventory.md',
    'exp_01': 'fres_01_text_embedding_geometry.md',
    'exp_02': 'fres_02_image_support_embedding_geometry.md',
    'exp_03': 'fres_03_text_encoder_swap_detector.md',
    'exp_04': 'fres_04_image_encoder_swap_visual_support.md',
    'exp_05': 'fres_05_cross_modal_pairing_matrix.md',
    'exp_06': 'fres_06_encoder_swap_dense_heatmap_figures.md',
    'exp_07': 'fres_07_heldout_safety_audit.md',
    'exp_08': 'fres_08_encoder_intervention_causal_test.md',
}


def rows_to_md_table(rows: List[dict], max_rows: int = 40) -> str:
    if not rows:
        return '_no rows_'
    cols = list(rows[0].keys())
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for r in rows[:max_rows]:
        lines.append('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    if len(rows) > max_rows:
        lines.append(f'\n_({len(rows) - max_rows} more rows in CSV)_')
    return '\n'.join(lines)


def write_fres_report(
        path: Path,
        meta: dict,
        sections: Dict[str, str],
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {meta.get('title', 'Experiment')}",
        '',
        f"Generated: {meta.get('generated', datetime.now().isoformat(timespec='seconds'))}",
        f"Status: {meta.get('status', 'UNKNOWN')}",
        f"Workdir: `{meta.get('workdir', '')}`",
        f"Command: `{meta.get('command', '')}`",
        f"GPU: `{meta.get('gpu', '')}`",
        f"Start time: {meta.get('start_time', '')}",
        f"End time: {meta.get('end_time', '')}",
        f"Runtime: {meta.get('runtime_sec', '')} sec",
        f"Return code: {meta.get('return_code', 0)}",
        '',
        '> **Scope:** This experiment swaps **text/image prompt encoders** (support embeddings), '
        '**not** the detector visual backbone unless explicitly marked BLOCKED.',
        '',
    ]
    for key in (
        'why', 'setup', 'outputs', 'table', 'sanity', 'interpretation',
        'next_guess', 'next_decision',
    ):
        if key in sections and sections[key]:
            title = {
                'why': '1. Why this experiment',
                'setup': '2. Setup',
                'outputs': '3. Output files',
                'table': '4. Main table',
                'sanity': '5. Sanity checks',
                'interpretation': '6. Interpretation',
                'next_guess': '7. Next guess',
                'next_decision': '8. Next experiment decision',
            }[key]
            lines += [f'## {title}', '', sections[key], '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def finalize_exp(
        ctx,
        result: ExpResult,
        gpu: str,
        started: float,
        command: str,
        why: str,
        setup: str,
        sanity: str,
        interpretation: str,
        guesses: List[str],
        decision: str,
) -> Path:
    import time

    sub = ctx.exp_result(result.work_subdir)
    fres = sub / f"fres_{result.exp_id.replace('exp_', '')}.md"
    csv_paths = {k: v for k, v in result.outputs.items() if v.endswith('.csv')}
    table = rows_to_md_table(result.rows)
    outputs = '\n'.join(f'- `{k}`: `{v}`' for k, v in result.outputs.items())
    write_fres_report(
        fres,
        dict(
            title=result.work_subdir.replace('_', ' ').title(),
            generated=datetime.now().isoformat(timespec='seconds'),
            status=result.status,
            workdir=str(ctx.work_dir / result.work_subdir),
            command=command,
            gpu=gpu,
            start_time=datetime.fromtimestamp(started).isoformat(timespec='seconds'),
            end_time=datetime.now().isoformat(timespec='seconds'),
            runtime_sec=round(time.time() - started, 1),
            return_code=0 if result.status in ('COMPLETE', 'DONE', 'PARTIAL') else 1,
        ),
        dict(
            why=why,
            setup=setup,
            outputs=outputs or '_none_',
            table=table,
            sanity=sanity,
            interpretation=interpretation or result.error,
            next_guess='\n'.join(f'{i+1}. {g}' for i, g in enumerate(guesses)),
            next_decision=decision,
        ),
    )
    for name, path in csv_paths.items():
        if result.rows and Path(path).exists() is False:
            write_csv(Path(path), result.rows)
    alias = FRES_ALIASES.get(result.exp_id)
    if alias:
        dest = ctx.result_md_dir / alias
        shutil.copy2(fres, dest)
    return fres


def update_live_status(ctx, state: dict):
    lines = [
        '# Encoder swap live status',
        '',
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        '',
        f"- start_time: {state.get('start_time')}",
        f"- target_stop_time: {state.get('target_stop_time')}",
        f"- selected_gpus: {state.get('selected_gpus')}",
        f"- running_experiment: {state.get('running_experiment')}",
        f"- mode: {state.get('mode')} (smoke={state.get('smoke')})",
        '',
        '## Progress',
        f"- completed: {len(state.get('completed_experiments', []))}",
        f"- failed: {len(state.get('failed_experiments', []))}",
        f"- blocked: {len(state.get('blocked_experiments', []))}",
        '',
        '### Completed',
        '\n'.join(f'- {e}' for e in state.get('completed_experiments', [])) or '- none',
        '',
        '### Failed',
        '\n'.join(f'- {e}' for e in state.get('failed_experiments', [])) or '- none',
        '',
        f"**next_action:** {state.get('next_action', '')}",
        f"**last_error:** {state.get('last_error', '')[:500]}",
        '',
        f"**latest_hypothesis:** {state.get('latest_hypothesis_update', '')}",
    ]
    ctx.live_status_path.write_text('\n'.join(lines), encoding='utf-8')
