#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor")
RUNS = [
    {
        "name": "full_openvocab_s2_12angle",
        "session": "rsa_full_openvocab_s2",
        "csv": "metrics/open_vocab_benchmark_rows.csv",
        "expected_rows": 2500 * 12,
        "log": "logs/full_openvocab_s2_12angle.log",
    },
    {
        "name": "full_causal_intervention_s3_12angle",
        "session": "rsa_full_causal_s3",
        "csv": "metrics/open_vocab_causal_rows.csv",
        "expected_rows": 500 * 12 * 6,
        "log": "logs/full_causal_intervention_s3_12angle.log",
    },
    {
        "name": "full_context_counterfactual_s3_12angle",
        "session": "rsa_full_context_s3",
        "csv": "metrics/context_counterfactual_rows.csv",
        "expected_rows": 500 * 12 * 2,
        "log": "logs/full_context_counterfactual_s3_12angle.log",
    },
    {
        "name": "full_dehub_safety_s3_12angle",
        "session": "rsa_full_dehub_s3",
        "csv": "metrics/dehub_safety_rows.csv",
        "expected_rows": 500 * 12 * 2,
        "log": "logs/full_dehub_safety_s3_12angle.log",
    },
]


def _count_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _row_key(row: dict) -> tuple[str, str, int, str, str]:
    return (
        row.get("checkpoint_name", ""),
        row.get("tile_id", ""),
        int(float(row.get("angle") or 0)),
        row.get("condition", ""),
        row.get("intervention", ""),
    )


def _rows_files(run_dir: Path, rel_csv: str) -> list[Path]:
    files = [run_dir / rel_csv]
    shard_root = run_dir / "shards"
    if shard_root.exists():
        files.extend(sorted(path / rel_csv for path in shard_root.glob("shard_*") if (path / rel_csv).exists()))
    return files


def _count_unique_rows(paths: list[Path]) -> tuple[int, int, int]:
    keys = set()
    raw_rows = 0
    existing_files = 0
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        existing_files += 1
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows += 1
                try:
                    keys.add(_row_key(row))
                except Exception:
                    continue
    return len(keys), raw_rows, existing_files


def _manifest_status(path: Path) -> str:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return "RUNNING_OR_NOT_WRITTEN"
    try:
        return str(json.loads(manifest.read_text()).get("status", ""))
    except Exception as exc:
        return f"MANIFEST_READ_ERROR:{type(exc).__name__}"


def _log_files(root: Path, run: dict) -> list[Path]:
    files = [root / run["log"]]
    files.extend(sorted((root / "logs").glob(f"{run['name']}_shard_*.log")))
    return files


def _last_done_line(log_paths: list[Path]) -> str:
    newest_mtime = -1.0
    newest_signal = ""
    for log_path in log_paths:
        if not log_path.exists():
            continue
        last = ""
        with log_path.open(errors="ignore") as f:
            for line in f:
                if (
                    line.startswith("DONE ")
                    or line.startswith("FAILED ")
                    or line.startswith("status=")
                    or line.startswith("SKIP_DONE ")
                ):
                    last = line.strip()
        if last:
            mtime = log_path.stat().st_mtime
            if mtime >= newest_mtime:
                newest_mtime = mtime
                newest_signal = f"{log_path.name}: {last}"
    return newest_signal


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "session",
        "status",
        "rows",
        "raw_rows_including_duplicates",
        "row_files",
        "expected_rows",
        "progress_pct",
        "rows_csv",
        "manifest",
        "log",
        "last_log_signal",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for run in RUNS:
        run_dir = root / run["name"]
        rows_csv = run_dir / run["csv"]
        row_files = _rows_files(run_dir, run["csv"])
        expected = int(run["expected_rows"])
        count, raw_rows, row_file_count = _count_unique_rows(row_files)
        if row_file_count == 0:
            count = _count_data_rows(rows_csv)
            raw_rows = count
        pct = (count / expected * 100.0) if expected else 0.0
        rows.append(
            {
                "name": run["name"],
                "session": run["session"],
                "status": _manifest_status(run_dir),
                "rows": count,
                "raw_rows_including_duplicates": raw_rows,
                "row_files": row_file_count,
                "expected_rows": expected,
                "progress_pct": f"{pct:.2f}",
                "rows_csv": str(rows_csv),
                "manifest": str(run_dir / "manifest.json"),
                "log": str(root / run["log"]),
                "last_log_signal": _last_done_line(_log_files(root, run)),
            }
        )
    if args.write:
        _write_csv(root / "full_openrsd_status.csv", rows)
        (root / "full_openrsd_status.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True))
    print("| run | status | rows | expected | progress | last signal |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for row in rows:
        print(
            f"| {row['name']} | {row['status']} | {row['rows']} | {row['expected_rows']} | "
            f"{row['progress_pct']}% | `{row['last_log_signal']}` |"
        )


if __name__ == "__main__":
    main()
