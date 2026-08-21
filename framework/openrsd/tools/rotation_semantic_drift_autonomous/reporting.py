#!/usr/bin/env python
"""Generate fres_*.md and update hypothesis board."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.rotation_semantic_drift_autonomous.common_autonomous import (
    AutonomousCtx,
    HYPOTHESES,
    read_csv,
    write_csv,
)


def write_fres_report(
    path: Path,
    meta: Dict[str, Any],
    sections: Dict[str, str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {meta.get('experiment_name', 'Experiment')}",
        '',
        f"Generated: {meta.get('generated', datetime.now().isoformat())}",
        f"Status: {meta.get('status', 'UNKNOWN')}",
        f"Workdir: {meta.get('workdir', 'NA')}",
        f"Command: {meta.get('command', 'NA')}",
        f"GPU: {meta.get('gpu', 'NA')}",
        f"Start time: {meta.get('start_time', 'NA')}",
        f"End time: {meta.get('end_time', 'NA')}",
        f"Runtime: {meta.get('runtime_sec', 'NA')} s",
        f"Return code: {meta.get('return_code', 'NA')}",
        '',
    ]
    for title, body in sections.items():
        lines.extend([f'## {title}', '', body.strip(), ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def default_sections(
    why: str,
    setup: str,
    outputs: str,
    table: str,
    sanity: str,
    interpretation: str,
    hypothesis_table: str,
    next_guess: str,
    next_decision: str,
) -> Dict[str, str]:
    return {
        '1. Why this experiment': why,
        '2. Experimental setup': setup,
        '3. Outputs': outputs,
        '4. Main table': table,
        '5. Sanity checks': sanity,
        '6. Result interpretation': interpretation,
        '7. Hypothesis update': hypothesis_table,
        '8. Next guess': next_guess,
        '9. Next experiment decision': next_decision,
    }


def update_hypothesis_board(
    ctx: AutonomousCtx,
    updates: Dict[str, Dict[str, str]],
):
    """updates: {H1: {verdict, evidence, next_needed}}"""
    rows = read_csv(ctx.hypothesis_board_path)
    if not rows:
        from tools.rotation_semantic_drift_autonomous.common_autonomous import init_hypothesis_board
        init_hypothesis_board(ctx)
        rows = read_csv(ctx.hypothesis_board_path)
    by_id = {r['hypothesis_id']: r for r in rows}
    for hid, upd in updates.items():
        if hid not in by_id:
            desc = next((d for h, d in HYPOTHESES if h == hid), '')
            by_id[hid] = dict(hypothesis_id=hid, description=desc, verdict='INCONCLUSIVE',
                              evidence='', last_updated='', next_needed='')
        by_id[hid].update(upd)
        by_id[hid]['last_updated'] = datetime.now().isoformat()
    write_csv(ctx.hypothesis_board_path, list(by_id.values()),
              ['hypothesis_id', 'description', 'verdict', 'evidence', 'last_updated', 'next_needed'])


def update_live_status(ctx: AutonomousCtx, state: dict, event: str, detail: str = ''):
    lines = [
        '# Live status — autonomous rotation semantic drift',
        '',
        f"Updated: {datetime.now().isoformat()}",
        f"Event: **{event}**",
        '',
        f"Target stop: {state.get('target_stop_time', 'NA')}",
        f"Elapsed hours: {state.get('elapsed_hours', 0):.2f}",
        f"Running: {state.get('running_experiment', 'none')}",
        f"Completed: {', '.join(state.get('completed_experiments', [])) or 'none'}",
        f"Failed: {', '.join(state.get('failed_experiments', [])) or 'none'}",
        f"Blocked: {', '.join(state.get('blocked_experiments', [])) or 'none'}",
        f"GPUs: {state.get('selected_gpus', [])}",
        f"Next action: {state.get('latest_best_next_action', 'NA')}",
        '',
        '## Detail',
        detail or '(none)',
        '',
        '## Last error',
        str(state.get('last_error', 'none')),
    ]
    ctx.live_status_path.write_text('\n'.join(lines), encoding='utf-8')


def rows_to_markdown_table(rows: List[dict], max_rows: int = 30) -> str:
    if not rows:
        return '_no rows_'
    keys = list(rows[0].keys())[:12]
    hdr = '| ' + ' | '.join(keys) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(keys)) + ' |'
    body = []
    for r in rows[:max_rows]:
        body.append('| ' + ' | '.join(str(r.get(k, ''))[:40] for k in keys) + ' |')
    return '\n'.join([hdr, sep] + body)
