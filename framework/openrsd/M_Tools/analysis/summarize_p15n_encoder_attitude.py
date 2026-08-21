from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _read_scalars(path: Path) -> tuple[tuple[int, float, float] | None,
                                       tuple[int, float, float] | None]:
    rows: list[tuple[int, float, float]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'dota/mAP' not in data:
                continue
            rows.append((
                int(data.get('step', -1)),
                float(data['dota/mAP']),
                float(data.get('dota/AP50', data['dota/mAP'])),
            ))
    if not rows:
        return None, None
    return max(rows, key=lambda item: item[1]), rows[-1]


def _parse_ts(line: str, prefix: str) -> datetime | None:
    if not line.startswith(prefix):
        return None
    value = line[len(prefix):len(prefix) + 19]
    try:
        return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None


def _log_status(exp_dir: Path, name: str) -> tuple[str, str, str, str]:
    log = exp_dir / f'{name}_train.log'
    if not log.exists():
        return 'running_or_missing_log', '-', '-', '-'
    start_dt = None
    done_dt = None
    done_line = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        parsed_start = _parse_ts(line, 'START ')
        if parsed_start is not None:
            start_dt = parsed_start
        parsed_done = _parse_ts(line, 'DONE ')
        if parsed_done is not None:
            done_dt = parsed_done
            done_line = line
    status = 'running'
    if done_line is not None:
        status = done_line
    duration_min = '-'
    if start_dt is not None:
        end_dt = done_dt if done_dt is not None else datetime.now()
        duration_min = f'{(end_dt - start_dt).total_seconds() / 60.0:.1f}'
    start = start_dt.strftime('%H:%M:%S') if start_dt is not None else '-'
    done = done_dt.strftime('%H:%M:%S') if done_dt is not None else '-'
    return status, start, done, duration_min


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Summarize P15N encoder attitude experiments.')
    parser.add_argument(
        '--root',
        default='work_dirs/p15n_encoder_attitude_hrrsd_20260624',
        help='Experiment root containing per-run work dirs and train logs.')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'missing root: {root}')

    print('\t'.join(
        ['name', 'status', 'start', 'done', 'duration_min',
         'best_epoch', 'best_mAP', 'best_AP50', 'last_epoch', 'last_mAP',
         'last_AP50']))
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        status, start, done, duration_min = _log_status(root, exp_dir.name)
        scalar_paths = sorted(exp_dir.glob('*/vis_data/scalars.json'))
        if not scalar_paths:
            print('\t'.join([
                exp_dir.name,
                status,
                start, done, duration_min,
                '-', '-', '-', '-', '-', '-',
            ]))
            continue
        best, last = _read_scalars(scalar_paths[-1])
        if best is None or last is None:
            continue
        print('\t'.join([
            exp_dir.name,
            status,
            start,
            done,
            duration_min,
            str(best[0]),
            f'{best[1]:.4f}',
            f'{best[2]:.4f}',
            str(last[0]),
            f'{last[1]:.4f}',
            f'{last[2]:.4f}',
        ]))


if __name__ == '__main__':
    main()
