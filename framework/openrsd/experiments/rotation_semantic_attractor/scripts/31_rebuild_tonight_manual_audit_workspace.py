#!/usr/bin/env python3
"""Rebuild the verified expanded manual-labeling workspace.

This is deliberately limited to audit artifacts. It does not rerun full
inference, train models, or create human labels.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/data1/zcy/OpenRSD")
AUDIT = REPO / "experiments/rotation_semantic_attractor/reports/visual_summary/audit"
CROP_DIR = REPO / "experiments/rotation_semantic_attractor/reports/visual_summary/crops_verified_expanded"
TEMPLATE = AUDIT / "human_sv_crop_audit_template_VERIFIED_EXPANDED.csv"
LABELS = AUDIT / "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv"
WORK = AUDIT / "tonight_manual_audit_20260601"
QUARANTINE = AUDIT / "OLD_CROP_PACK_QUARANTINED.txt"
SCRIPT19 = REPO / "experiments/rotation_semantic_attractor/scripts/19_audit_unannotated_sv_contamination.py"

REQUIRED_COUNTS = {
    "valid_unmatched_sv": 300,
    "strict_object_flip": 100,
    "true_sv_positive_control": 100,
    "degenerate_large_sv_box": 150,
    "padding_artifact": 100,
}
SUBSETS = {
    "valid_unmatched_sv_300_for_corrected_fsv.csv": ("valid_unmatched_sv", 300, None),
    "true_sv_positive_control_100_for_qc.csv": ("true_sv_positive_control", 100, None),
    "degenerate_large_sv_box_60_spotcheck.csv": ("degenerate_large_sv_box", 60, 20260601),
    "padding_artifact_30_spotcheck.csv": ("padding_artifact", 30, 20260601),
    "strict_object_flip_30_qualitative_candidates.csv": ("strict_object_flip", 30, 20260601),
}
HTML_NAMES = {
    "valid_unmatched_sv_300_for_corrected_fsv.csv": "valid_unmatched_sv_300_gallery.html",
    "true_sv_positive_control_100_for_qc.csv": "true_sv_positive_control_100_gallery.html",
    "degenerate_large_sv_box_60_spotcheck.csv": "degenerate_large_sv_box_60_gallery.html",
    "padding_artifact_30_spotcheck.csv": "padding_artifact_30_gallery.html",
    "strict_object_flip_30_qualitative_candidates.csv": "strict_object_flip_30_gallery.html",
}
PURPOSE = {
    "valid_unmatched_sv": "只有这一组进入 corrected-FSV。判断 unmatched small-vehicle 预测是真实漏标车辆，还是真正误检。",
    "true_sv_positive_control": "正控组，用于检查人工标注者能否识别已标注真实 small-vehicle。",
    "degenerate_large_sv_box": "不进入 corrected-FSV。只审计异常大框 small-vehicle 预测这一独立 failure mode。",
    "padding_artifact": "不进入 corrected-FSV。只检查 valid-mask / padding 过滤边界。",
    "strict_object_flip": "不进入 corrected-FSV。用于论文 qualitative case 预筛。",
}
DISPLAY = {
    "true_vehicle_annot_missing": "真实小车，GT 未标注",
    "true_vehicle_annot_present_but_missed_match": "真实小车，GT 有标但自动匹配漏了",
    "non_vehicle_background": "不是车：背景/纹理误检",
    "non_vehicle_object_conflict": "不是小车：其他物体冲突",
    "ambiguous": "看不清/不确定",
    "invalid_visualization": "图像或框疑似错位，不能判",
    "true_vehicle_annot_present": "真实小车，GT 已标注",
    "actual_large_sv_degenerate_prediction": "确认为异常大框 small-vehicle 预测",
    "padding_or_boundary_related": "与 padding / 边界有关",
    "large_context_texture": "大面积上下文纹理误检",
    "object_conflict": "其他物体冲突",
    "uncertain": "不确定",
    "valid_padding_artifact": "确认为 padding / mask 伪影",
    "not_padding_artifact": "不是 padding 伪影",
    "good_qualitative_case": "适合作为论文定性图",
    "not_clear": "不够清楚",
    "wrong_or_ambiguous": "错误或歧义太大",
}
LABEL_OPTIONS = {
    "valid_unmatched_sv": [
        "true_vehicle_annot_missing",
        "true_vehicle_annot_present_but_missed_match",
        "non_vehicle_background",
        "non_vehicle_object_conflict",
        "ambiguous",
        "invalid_visualization",
    ],
    "true_sv_positive_control": [
        "true_vehicle_annot_present",
        "true_vehicle_annot_present_but_missed_match",
        "ambiguous",
        "invalid_visualization",
        "non_vehicle_background",
        "non_vehicle_object_conflict",
    ],
    "degenerate_large_sv_box": [
        "actual_large_sv_degenerate_prediction",
        "padding_or_boundary_related",
        "large_context_texture",
        "object_conflict",
        "uncertain",
    ],
    "padding_artifact": ["valid_padding_artifact", "not_padding_artifact", "uncertain"],
    "strict_object_flip": ["good_qualitative_case", "not_clear", "wrong_or_ambiguous"],
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backup(path: Path, stamp: str) -> str:
    if not path.exists():
        return ""
    dst = path.with_name(path.name + f".bak_rebuild_{stamp}")
    shutil.copy2(path, dst)
    return str(dst)


def ensure_label_csv(stamp: str) -> tuple[list[str], list[dict[str, str]], str]:
    if not LABELS.exists():
        shutil.copy2(TEMPLATE, LABELS)
        action = "copied_template"
    else:
        action = "kept_existing"
    label_headers, label_rows = read_csv(LABELS)
    changed = False
    for col in ["human_label", "human_confidence", "human_notes"]:
        if col not in label_headers:
            label_headers.append(col)
            changed = True
    if changed:
        backup(LABELS, stamp)
        write_csv(LABELS, label_headers, label_rows)
    return label_headers, label_rows, action


def merge_subset_labels_into_master(label_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row.get("crop_id", ""): dict(row) for row in label_rows}
    for subset_name in SUBSETS:
        path = WORK / subset_name
        if not path.exists():
            continue
        _, rows = read_csv(path)
        for row in rows:
            cid = row.get("crop_id", "")
            if cid in by_id and row.get("human_label", "").strip():
                by_id[cid]["human_label"] = row.get("human_label", "").strip()
                by_id[cid]["human_confidence"] = row.get("human_confidence", "").strip()
                by_id[cid]["human_notes"] = row.get("human_notes", "").strip()
    return list(by_id.values())


def preflight(rows: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    checks: list[dict[str, str]] = []
    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("repo_root_exists", REPO.exists(), str(REPO))
    add("template_exists", TEMPLATE.exists(), str(TEMPLATE))
    add("label_csv_exists", LABELS.exists(), str(LABELS))
    add("crop_dir_exists", CROP_DIR.exists(), str(CROP_DIR))
    add("old_crop_pack_quarantine_marker_exists", QUARANTINE.exists(), str(QUARANTINE))
    add("corrected_fsv_script_exists", SCRIPT19.exists(), str(SCRIPT19))
    counts = Counter(row.get("audit_category", "") for row in rows)
    add("row_count_750", len(rows) == 750, f"rows={len(rows)}")
    add("audit_category_distribution", all(counts.get(k, 0) == v for k, v in REQUIRED_COUNTS.items()), json.dumps(dict(counts), ensure_ascii=False))
    missing = []
    for row in rows:
        for col in ["image_path_full_tile", "image_path_zoom", "image_path_gt_context", "metadata_json"]:
            if not Path(row.get(col, "")).exists():
                missing.append(f"{row.get('crop_id')}:{col}")
                break
    add("all_image_and_metadata_paths_exist", not missing, f"missing={len(missing)}")
    valid = [r for r in rows if r.get("audit_category") == "valid_unmatched_sv"]
    add("valid_unmatched_valid_for_human_true", all(r.get("valid_for_human_audit", "").lower() == "true" for r in valid), f"total={len(valid)}")
    add("valid_unmatched_coordinate_frame_consistent_true", all(r.get("coordinate_frame_consistent", "").lower() == "true" for r in valid), f"total={len(valid)}")
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "BLOCKED"
    return status, checks


def render_gallery(path: Path, rows: list[dict[str, str]], category: str) -> None:
    options = "".join(
        f"<li><b>{html.escape(DISPLAY.get(label, label))}</b><br><code>{html.escape(label)}</code></li>"
        for label in LABEL_OPTIONS[category]
    )
    cards = []
    for i, row in enumerate(rows, 1):
        meta = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(row.get(k, ''))}</td></tr>"
            for k in ["crop_id", "model_name", "tile_id", "angle", "score", "pred_class", "raw_box_type", "raw_box_angle", "raw_box_area", "best_gt_class", "best_gt_iou", "valid_mask_ratio_inside_box", "padding_overlap_ratio", "human_label", "human_confidence"]
        )
        figs = "".join(
            f"<figure><img src=\"file://{html.escape(row[col])}\"><figcaption>{caption}</figcaption></figure>"
            for col, caption in [
                ("image_path_full_tile", "整图叠加视图"),
                ("image_path_zoom", "预测框局部放大"),
                ("image_path_gt_context", "GT 上下文视图"),
            ]
        )
        cards.append(f"<section><h2>{i}. {html.escape(row.get('crop_id', ''))}</h2><div class='imgs'>{figs}</div><table>{meta}</table></section>")
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(category)} gallery</title>
<style>
body{{font-family:Arial,sans-serif;margin:20px;background:#f8fafc;color:#172033}}
.notice{{background:#fff7ed;border:1px solid #fed7aa;padding:10px;border-radius:6px}}
.imgs{{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px}}
section{{background:white;border:1px solid #d8dee4;border-radius:6px;margin:0 0 18px;padding:12px}}
img{{width:100%;height:auto;border:1px solid #cbd5e1;background:#888}}
figcaption{{font-size:13px;color:#475569}}
table{{border-collapse:collapse;margin-top:8px}}th,td{{border:1px solid #d8dee4;padding:4px 7px;text-align:left;font-size:13px}}
code{{font-size:12px}}
@media(max-width:900px){{.imgs{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(category)}</h1>
<p class="notice">{html.escape(PURPOSE[category])}</p>
<p>此静态 HTML 只用于查看；点击保存标注请运行服务器脚本 <code>29_serve_tonight_manual_audit_app.py</code>。后台 CSV 仍保存英文枚举值。</p>
<h2>允许标签</h2><ul>{options}</ul>
{''.join(cards)}
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def write_docs() -> None:
    readme = """# Verified Expanded Human Labeling README

## 标注顺序

Priority 1: `valid_unmatched_sv_300_for_corrected_fsv.csv`

Priority 2: `true_sv_positive_control_100_for_qc.csv`

Priority 3: `degenerate_large_sv_box_60_spotcheck.csv`

Priority 4: `padding_artifact_30_spotcheck.csv` 和 `strict_object_flip_30_qualitative_candidates.csv`

## 规则

- 不要标旧 200 crop。
- corrected-FSV 只使用 `valid_unmatched_sv` 且 `valid_for_human_audit=true` 的行。
- 不要把 degenerate / padding / strict flip / true control 纳入 corrected-FSV。
- 看不清就标 `ambiguous`。
- 如果图或框仍然疑似错位，标 `invalid_visualization`。
- 默认置信度是 `high`；只有确实不确定时改成 `medium` 或 `low`。
- 点击式标注请运行 `29_serve_tonight_manual_audit_app.py`，CSV 后台仍使用英文枚举。
"""
    runbook = f"""# Tonight Manual Audit Runbook

## 启动点击式标注界面

```bash
cd {REPO}
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/mplconfig /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/29_serve_tonight_manual_audit_app.py --host 0.0.0.0 --port 8766
```

然后在浏览器打开服务器地址的 `http://<server-ip>:8766/`。

## 第一优先级

打开 `valid_unmatched_sv` 子集，完成 300 张。

对应 CSV:

`{WORK / 'valid_unmatched_sv_300_for_corrected_fsv.csv'}`

## 校验

```bash
cd {REPO}
rtk /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/26_validate_tonight_manual_labels.py --label-csv {LABELS} --output-dir {WORK} --manual-audit-dir {WORK} --allow-partial
```

## corrected-FSV

```bash
cd {REPO}
rtk /data/zcy/anaconda3/envs/openrsd/bin/python experiments/rotation_semantic_attractor/scripts/27_run_corrected_fsv_after_manual_labels.py --repo-root {REPO} --label-csv {LABELS} --output-dir {WORK / 'corrected_fsv'} --seed 20260601
```
"""
    memo = """# Paper Thesis Memo — 2026-06-01 Night

## Core thesis

Remote-sensing detectors exhibit rotation-conditioned small-vehicle prediction burden. Closed-set detectors show cross-architecture false-SV burden, while OpenRSD/open-vocab models expose an embedding-mediated semantic attractor.

## Tonight manual audit update

- valid_unmatched_sv labeled:
- annotation-missing rate:
- corrected-FSV:
- corrected-FSV CI:
- true-SV positive control verdict:
- degenerate large box verdict:
- good strict flip qualitative cases:

## Cannot write

- all unmatched SV are hallucinations
- all no-SV risk-group images contain no real vehicles
- DeHub is safe
- DeHub improves AP
- context alone proves vehicle hallucination
"""
    (WORK / "README_HUMAN_LABELING.md").write_text(readme, encoding="utf-8")
    (WORK / "TONIGHT_8H_RUNBOOK.md").write_text(runbook, encoding="utf-8")
    (WORK / "paper_thesis_memo_20260601_night.md").write_text(memo, encoding="utf-8")


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force:
        raise SystemExit("Use --force to rebuild the manual audit workspace.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WORK.mkdir(parents=True, exist_ok=True)
    label_headers, label_rows, label_action = ensure_label_csv(stamp)
    label_rows = merge_subset_labels_into_master(label_rows)
    backup_path = backup(LABELS, stamp)
    write_csv(LABELS, label_headers, label_rows)

    status, checks = preflight(label_rows)
    write_json(WORK / "preflight_tonight_audit.json", {"status": status, "checks": checks})
    md = ["# Preflight Tonight Manual Audit", "", f"- status: `{status}`", "", "| check | status | detail |", "|---|---|---|"]
    md.extend(f"| {r['check']} | {r['status']} | {r['detail']} |" for r in checks)
    (WORK / "preflight_tonight_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if status != "PASS":
        write_json(WORK / "workspace_generation_manifest.json", {"status": "BLOCKED", "preflight": checks})
        print(json.dumps({"status": "BLOCKED", "preflight": str(WORK / "preflight_tonight_audit.md")}, ensure_ascii=False))
        return 2

    by_category: dict[str, list[dict[str, str]]] = {}
    for row in label_rows:
        by_category.setdefault(row.get("audit_category", ""), []).append(row)

    subset_paths: dict[str, str] = {}
    html_paths: dict[str, str] = {}
    subset_counts: dict[str, int] = {}
    for filename, (category, n, seed) in SUBSETS.items():
        rows = list(by_category.get(category, []))
        if seed is not None:
            rng = random.Random(seed)
            rows = sorted(rows, key=lambda r: r.get("crop_id", ""))
            rows = rng.sample(rows, n)
            rows = sorted(rows, key=lambda r: r.get("crop_id", ""))
        if len(rows) != n:
            raise RuntimeError(f"{filename}: expected {n}, got {len(rows)}")
        out = WORK / filename
        write_csv(out, label_headers, rows)
        subset_paths[filename] = str(out)
        subset_counts[filename] = len(rows)
        html_out = WORK / "html" / HTML_NAMES[filename]
        render_gallery(html_out, rows, category)
        html_paths[HTML_NAMES[filename]] = str(html_out)

    write_docs()

    validator = REPO / "experiments/rotation_semantic_attractor/scripts/26_validate_tonight_manual_labels.py"
    corrected = REPO / "experiments/rotation_semantic_attractor/scripts/27_run_corrected_fsv_after_manual_labels.py"
    summary = REPO / "experiments/rotation_semantic_attractor/scripts/28_summarize_tonight_manual_audit.py"
    validation_run = run_cmd([
        sys.executable, str(validator),
        "--label-csv", str(LABELS),
        "--output-dir", str(WORK),
        "--manual-audit-dir", str(WORK),
        "--allow-partial",
    ])
    corrected_run = run_cmd([
        sys.executable, str(corrected),
        "--repo-root", str(REPO),
        "--label-csv", str(LABELS),
        "--output-dir", str(WORK / "corrected_fsv"),
        "--seed", "20260601",
    ])
    summary_run = run_cmd([
        sys.executable, str(summary),
        "--label-csv", str(LABELS),
        "--corrected-fsv-dir", str(WORK / "corrected_fsv"),
        "--output-dir", str(WORK),
    ])

    validation_status = {}
    if (WORK / "label_validation_report.json").exists():
        validation_status = json.loads((WORK / "label_validation_report.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "DONE",
        "label_action": label_action,
        "backup_path": backup_path,
        "label_csv": str(LABELS),
        "subset_paths": subset_paths,
        "html_paths": html_paths,
        "readme": str(WORK / "README_HUMAN_LABELING.md"),
        "runbook": str(WORK / "TONIGHT_8H_RUNBOOK.md"),
        "memo": str(WORK / "paper_thesis_memo_20260601_night.md"),
        "subset_counts": subset_counts,
        "validation_status": validation_status.get("status", ""),
        "valid_unmatched_labeled": validation_status.get("valid_unmatched_labeled", ""),
        "true_sv_positive_control_labeled": validation_status.get("true_sv_positive_control_labeled", ""),
        "validation_run": validation_run,
        "corrected_run": corrected_run,
        "summary_run": summary_run,
    }
    write_json(WORK / "workspace_generation_manifest.json", manifest)
    print(json.dumps({
        "status": "DONE",
        "validation_status": manifest["validation_status"],
        "valid_unmatched_labeled": manifest["valid_unmatched_labeled"],
        "label_csv": str(LABELS),
        "work_dir": str(WORK),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
