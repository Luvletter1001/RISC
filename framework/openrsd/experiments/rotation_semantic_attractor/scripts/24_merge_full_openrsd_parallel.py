#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ROOT = Path("/data/zcy/OpenRSD_runs/rotation_semantic_attractor")
RUNS = [
    {
        "name": "full_openvocab_s2_12angle",
        "rows": "metrics/open_vocab_benchmark_rows.csv",
        "rows_merged": "metrics/open_vocab_benchmark_rows_merged.csv",
        "norms": "metrics/open_vocab_benchmark_embedding_norms.csv",
        "norms_merged": "metrics/open_vocab_benchmark_embedding_norms_merged.csv",
        "assets": "indexes/open_vocab_benchmark_transient_asset_index.tsv",
        "assets_merged": "indexes/open_vocab_benchmark_transient_asset_index_merged.tsv",
    },
    {
        "name": "full_causal_intervention_s3_12angle",
        "rows": "metrics/open_vocab_causal_rows.csv",
        "rows_merged": "metrics/open_vocab_causal_rows_merged.csv",
        "norms": "metrics/open_vocab_causal_embedding_norms.csv",
        "norms_merged": "metrics/open_vocab_causal_embedding_norms_merged.csv",
        "assets": "indexes/open_vocab_causal_transient_asset_index.tsv",
        "assets_merged": "indexes/open_vocab_causal_transient_asset_index_merged.tsv",
    },
    {
        "name": "full_dehub_safety_s3_12angle",
        "rows": "metrics/dehub_safety_rows.csv",
        "rows_merged": "metrics/dehub_safety_rows_merged.csv",
        "norms": "metrics/dehub_safety_embedding_norms.csv",
        "norms_merged": "metrics/dehub_safety_embedding_norms_merged.csv",
        "assets": "indexes/dehub_safety_transient_asset_index.tsv",
        "assets_merged": "indexes/dehub_safety_transient_asset_index_merged.tsv",
    },
]


def _csv_paths(run_dir: Path, rel_path: str) -> list[Path]:
    paths = [run_dir / rel_path]
    shard_root = run_dir / "shards"
    if shard_root.exists():
        paths.extend(sorted(path / rel_path for path in shard_root.glob("shard_*") if (path / rel_path).exists()))
    return paths


def _row_key(row: dict) -> tuple[str, str, int, str, str]:
    return (
        row.get("checkpoint_name", ""),
        row.get("tile_id", ""),
        int(float(row.get("angle") or 0)),
        row.get("condition", ""),
        row.get("intervention", ""),
    )


def _merge_csv(paths: list[Path], out_path: Path, *, key_fields: bool) -> tuple[int, int]:
    fields: list[str] | None = None
    rows: list[dict] = []
    seen = set()
    raw = 0
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = list(reader.fieldnames or [])
            for row in reader:
                raw += 1
                if key_fields:
                    key = _row_key(row)
                else:
                    key = tuple(row.get(field, "") for field in fields or [])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    if fields is None:
        return 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    print("| run | rows_unique | rows_raw | norms_unique | assets_unique |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for run in RUNS:
        run_dir = root / run["name"]
        rows_unique, rows_raw = _merge_csv(_csv_paths(run_dir, run["rows"]), run_dir / run["rows_merged"], key_fields=True)
        norms_unique, _ = _merge_csv(_csv_paths(run_dir, run["norms"]), run_dir / run["norms_merged"], key_fields=False)
        assets_unique, _ = _merge_csv(_csv_paths(run_dir, run["assets"]), run_dir / run["assets_merged"], key_fields=False)
        print(f"| {run['name']} | {rows_unique} | {rows_raw} | {norms_unique} | {assets_unique} |")


if __name__ == "__main__":
    main()
