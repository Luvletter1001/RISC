#!/usr/bin/env python3
"""Build a deterministic real-image one-batch dataset for FOCUS P1A smoke."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from focus_p1a_realbatch_common import (
    ensure_exp_tree,
    md_table,
    read_csv,
    resolve,
    select_realbatch_images,
    valid_focus_targets,
    write_json,
    write_manifest,
)


DEFAULT_EXP = Path("resultmd/exp_focus_ovd_p1a_realbatch_unblock_20260609")


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# FOCUS P1A Realbatch Smoke Dataset",
        "",
        f"- status: `{payload['status']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- anti_target_count: `{payload['anti_target_count']}`",
        f"- preserve_target_count: `{payload['preserve_target_count']}`",
        f"- coordinate_frame: `{payload['coordinate_frame']}`",
        "",
        "## Selected Images",
        "",
    ]
    lines.extend(md_table(payload["selected_images"], [
        "tile_id",
        "angle",
        "anti_target_count",
        "preserve_target_count",
        "image_path",
        "annotation_path",
        "annotation_object_count",
    ]))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spatial-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXP / "configs")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    spatial_targets = resolve(repo_root, args.spatial_targets)
    output_dir = resolve(repo_root, args.output_dir)
    exp_dir = output_dir.parent
    ensure_exp_tree(exp_dir)
    write_manifest(exp_dir, repo_root, {"spatial_targets": str(spatial_targets)})

    rows = valid_focus_targets(read_csv(spatial_targets))
    selected_images = select_realbatch_images(rows)
    role_counter = Counter()
    for item in selected_images:
        role_counter["anti_negative"] += int(item.get("anti_target_count", 0))
        role_counter["preserve_positive"] += int(item.get("preserve_target_count", 0))
    ok = (
        len(selected_images) > 0
        and role_counter["anti_negative"] > 0
        and role_counter["preserve_positive"] > 0
        and all(item.get("image_exists") for item in selected_images)
        and all(item.get("annotation_exists") for item in selected_images))
    payload = {
        "status": "PASS_REALBATCH_SMOKE_DATASET_READY" if ok else "BLOCKED_NO_REALBATCH_TARGET_PAIR",
        "spatial_targets": str(spatial_targets),
        "batch_size": len(selected_images),
        "selected_images": selected_images,
        "selected_tile_id": ";".join(item.get("tile_id", "") for item in selected_images),
        "selected_angle": ";".join(str(item.get("angle", "")) for item in selected_images),
        "anti_target_count": int(role_counter["anti_negative"]),
        "preserve_target_count": int(role_counter["preserve_positive"]),
        "image_paths": [item.get("image_path", "") for item in selected_images],
        "annotation_paths": [item.get("annotation_path", "") for item in selected_images],
        "coordinate_frame": "rotated_angle_sweep",
        "real_batch_source": "dataloader_real_batch",
    }
    write_json(output_dir / "p1a_realbatch_smoke_dataset.json", payload)
    write_markdown(output_dir / "p1a_realbatch_smoke_dataset_summary.md", payload)
    print(json.dumps({
        "status": payload["status"],
        "batch_size": payload["batch_size"],
        "anti": payload["anti_target_count"],
        "preserve": payload["preserve_target_count"],
    }, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
