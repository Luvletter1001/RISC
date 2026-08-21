#!/usr/bin/env python3
"""Merge G2 live predictions into 12x12 OVD TTA outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from strict_gpu_runner_v3 import write_json, write_text


ANGLES12 = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--head", default="alignment")
    parser.add_argument("--prompt-family", default="F3_orientation_aware")
    parser.add_argument("--source-angles", default=",".join(ANGLES12))
    parser.add_argument("--target-angles", default=",".join(ANGLES12))
    parser.add_argument("--score-thr", default="0.001,0.01,0.05")
    parser.add_argument("--nms-iou", default="0.1,0.3,0.5")
    parser.add_argument("--score-rule", default="max,mean")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/data/zcy/anaconda3/envs/openrsd/bin/python"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_angles = [f"{int(x):03d}" for x in args.source_angles.split(",") if x]
    tgt_angles = [f"{int(x):03d}" for x in args.target_angles.split(",") if x]
    score_thrs = [float(x) for x in args.score_thr.split(",") if x]
    nms_ious = [float(x) for x in args.nms_iou.split(",") if x]
    score_rules = [x for x in args.score_rule.split(",") if x]
    py = args.python_bin if args.python_bin.exists() else Path(sys.executable)
    out_root = args.work_dir / "G3_live_12x12_tta" / args.head / args.prompt_family
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    source_pkls = []
    for angle in src_angles:
        p = args.work_dir / "G2_live_inference" / args.head / args.prompt_family / f"angle_{angle}" / "predictions.pkl"
        source_pkls.append(p)
    all_sources_exist = all(p.exists() for p in source_pkls)
    for target in tgt_angles:
        for score_thr in score_thrs:
            for nms_iou in nms_ious:
                for rule in score_rules:
                    out = out_root / f"target_{target}" / f"score_{score_thr}_nms_{nms_iou}_{rule}" / "merged_predictions.pkl"
                    cmd = [str(py), "M_Tools/analysis/rotation_tta_merge.py"]
                    for angle, p in zip(src_angles, source_pkls):
                        cmd += ["--prediction", str(p), "--angle", str(int(angle))]
                    cmd += ["--target-angle", str(int(target)), "--score-thr", str(score_thr), "--nms-iou", str(nms_iou), "--score-rule", rule, "--out", str(out)]
                    stdout = out.with_suffix(".stdout.log")
                    stderr = out.with_suffix(".stderr.log")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if not all_sources_exist:
                        rc = 2
                        stderr.write_text("missing G2 source predictions\n", encoding="utf-8")
                    else:
                        with stdout.open("w", encoding="utf-8") as so, stderr.open("w", encoding="utf-8") as se:
                            proc = subprocess.run(["rtk", "env", f"PYTHONPATH={args.repo_root}:{args.repo_root / 'tools'}", "PYTHONNOUSERSITE=1", "MPLCONFIGDIR=/tmp/mplconfig", "CUDA_VISIBLE_DEVICES=4,5", *cmd], cwd=str(args.repo_root), stdout=so, stderr=se, text=True, check=False)
                            rc = proc.returncode
                    rows.append({"target_angle": target, "score_thr": score_thr, "nms_iou": nms_iou, "score_rule": rule, "status": "OK" if rc == 0 and out.exists() else "FAILED", "merged_pkl": str(out), "stdout": str(stdout), "stderr": str(stderr)})
    ok = [r for r in rows if r["status"] == "OK"]
    status = "DONE_LIVE_GPU" if all_sources_exist and len(ok) == len(rows) and len(src_angles) == 12 and len(tgt_angles) == 12 else ("PARTIAL_LIVE_GPU" if ok else "FAILED")
    payload = {"status": status, "source_pkls": [str(p) for p in source_pkls], "source_pkls_exist": all_sources_exist, "rows": rows}
    write_json(args.out_json, payload)
    lines = ["# G3 Live Full 12x12 OVD TTA", "", f"- status: `{status}`", f"- complete_12x12: `{len(src_angles) == 12 and len(tgt_angles) == 12 and all_sources_exist}`", "", "## Source Pkls", ""]
    for p in source_pkls:
        lines.append(f"- `{p}` exists={p.exists()} mtime={p.stat().st_mtime if p.exists() else 'NA'}")
    lines += ["", "## Merge Rows", "", "| target | score_thr | nms_iou | score_rule | status | merged_pkl |", "|---:|---:|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['target_angle']} | {r['score_thr']} | {r['nms_iou']} | {r['score_rule']} | {r['status']} | `{r['merged_pkl']}` |")
    write_text(args.out_md, "\n".join(lines))


if __name__ == "__main__":
    main()

