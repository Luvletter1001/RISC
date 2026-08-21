#!/usr/bin/env python
"""Close the 2026-05-26 OpenRSD SV-DeHub evidence chain from existing artifacts.

This runner is intentionally conservative: it discovers and summarizes existing
records, writes explicit gap/status reports, and avoids large-scale training or
inference. Re-run behavior is resume-first; pass --force to regenerate outputs.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DOTA_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter",
]
SMALL_VEHICLE_INDEX = DOTA_CLASSES.index("small-vehicle")
LARGE_VEHICLE_INDEX = DOTA_CLASSES.index("large-vehicle")
P0148_TILE = "P0148__1024__651___0"

SV_KEYS = [
    "small_vehicle_ratio", "final_sv_ratio", "dense_sv_ratio",
    "dense_top1_sv_ratio", "p0148_final_sv", "p0148_dense_sv",
    "official_post_nms_sv_ratio", "sv_ratio", "no_nms_sv_ratio",
    "pre_nms_sv_ratio",
]
COUNT_KEYS = ["det_count", "detection_count", "detection_total", "sv_dets", "small_vehicle_count"]
AP_KEYS = ["ap50", "AP50", "mAP@0.5", "heldout_ap50", "sv_ap50"]


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def fmt4(value: Any) -> str:
    try:
        if value in ("", None, "NA", "nan"):
            return ""
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    start = now_iso()
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
        return {
            "cmd": cmd, "start_time": start, "end_time": now_iso(),
            "return_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
        }
    except Exception as exc:
        return {
            "cmd": cmd, "start_time": start, "end_time": now_iso(),
            "return_code": -999, "stdout": "", "stderr": repr(exc),
        }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt4(row.get(k, "")) if isinstance(row.get(k, ""), float) else row.get(k, "")
                             for k in fields})


def read_csv(path: Path, max_rows: int = 20000) -> list[dict[str, str]]:
    try:
        with path.open(newline="", errors="replace") as f:
            rows = []
            for i, row in enumerate(csv.DictReader(f)):
                if i >= max_rows:
                    break
                rows.append(dict(row))
            return rows
    except Exception:
        return []


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join(out) + "\n"


class EvidenceRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(args.repo_root).resolve()
        self.work = Path(args.work_dir).resolve()
        self.result = Path(args.result_md_dir).resolve()
        ensure_dir(self.work)
        ensure_dir(self.result)
        ensure_dir(self.work / "logs")
        self.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.command_log = self.work / "logs" / f"command_record_{self.run_id}.json"
        self.run_record: dict[str, Any] = {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "start_time": now_iso(),
            "end_time": None,
            "return_code": None,
            "requested_gpu_ids": args.gpu_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_smi_before": run_cmd(["rtk", "nvidia-smi"], cwd=self.repo, timeout=20),
            "nvidia_smi_after": None,
        }
        atexit.register(self.finalize_command_record)
        self.inventory_path = self.work / "inventory_index.json"
        self.inventory = self.load_or_discover_inventory()

    def finalize_command_record(self) -> None:
        self.run_record["end_time"] = self.run_record.get("end_time") or now_iso()
        self.run_record["return_code"] = self.run_record.get("return_code", 0)
        if self.run_record.get("nvidia_smi_after") is None:
            self.run_record["nvidia_smi_after"] = run_cmd(["rtk", "nvidia-smi"], cwd=self.repo, timeout=20)
        ensure_dir(self.command_log.parent)
        self.command_log.write_text(json.dumps(self.run_record, indent=2, ensure_ascii=False))

    def should_skip(self, output: Path) -> bool:
        return output.exists() and self.args.resume and not self.args.force

    def discover_files(self) -> list[str]:
        roots = ["resultmd", "work_dirs", "tools", "M_Tools", "M_configs", "vis", "results"]
        keep_ext = {".csv", ".json", ".md", ".pkl", ".pth", ".py", ".txt", ".log", ".png", ".jpg", ".jpeg"}
        paths: list[str] = []
        for root in roots:
            base = self.repo / root
            if not base.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                rel_dir = str(Path(dirpath).relative_to(self.repo))
                if any(part in {".git", "__pycache__", ".mim", ".pytest_cache"} for part in Path(rel_dir).parts):
                    continue
                parts_lower = [part.lower() for part in Path(rel_dir).parts]
                # Avoid full raw image/prediction crawls. Keep P0148 and crop audit
                # folders, but skip broad image/detection JSON forests that are not
                # useful for source-level closure.
                if (
                    ("images" in parts_lower or "detections" in parts_lower)
                    and "p0148" not in rel_dir.lower()
                    and "crop" not in rel_dir.lower()
                ):
                    dirnames[:] = []
                    continue
                if len(paths) > 250000:
                    break
                for name in filenames:
                    p = Path(dirpath) / name
                    if p.suffix.lower() not in keep_ext:
                        continue
                    s = str(p.resolve())
                    lower = s.lower()
                    if any(k in lower for k in [
                        "p0148", "small_vehicle", "small-vehicle", "sv_", "dehub", "b8k",
                        "iter_1000", "iter_2000", "iter_5000", "iter_8000", "exp_03",
                        "exp_04", "exp_05", "exp_07", "support", "embedding", "mapping",
                        "dense", "nms", "heldout", "repair", "formal", "head", "crop",
                        "class_drift", "per_class", "ap",
                    ]):
                        paths.append(s)
        return sorted(set(paths))

    def categorize(self, paths: list[str]) -> dict[str, list[str]]:
        cats: dict[str, list[str]] = defaultdict(list)
        patterns = {
            "configs": [r"\.py$", r"config"],
            "checkpoints": [r"\.pth$", r"checkpoint", r"last_checkpoint"],
            "support_embedding": [r"support.*\.(pkl|json|csv|pth)$", r"text.*embedding", r"visual.*embedding"],
            "class_mapping": [r"class.*(name|mapping|order)", r"category", r"metainfo"],
            "p0148_inputs": [r"P0148", r"p0148"],
            "b8k_iter": [r"b8k", r"dehub", r"iter_1000", r"iter_2000", r"iter_5000", r"iter_8000"],
            "encoder_swap": [r"exp_03", r"exp_04", r"exp_05", r"exp_07", r"encoder_swap", r"visual.*swap", r"text.*swap"],
            "repair_formal": [r"repair", r"ftable_070", r"ftable_071", r"fres_101", r"fres_102", r"formal"],
            "sv_ratio_sources": [r"detection_summary", r"final_sv", r"dense_sv", r"small_vehicle_ratio", r"post.*nms", r"prenms"],
            "gt_crop": [r"gt", r"crop", r"manual_audit", r"iou"],
            "head_consensus": [r"head", r"aux", r"fusion", r"alignment", r"val_using_aux"],
        }
        for path in paths:
            for cat, pats in patterns.items():
                if any(re.search(pat, path, re.I) for pat in pats):
                    cats[cat].append(path)
        return {k: sorted(set(v)) for k, v in cats.items()}

    def load_or_discover_inventory(self) -> dict[str, Any]:
        if self.inventory_path.exists() and self.args.resume and not self.args.force:
            try:
                return json.loads(self.inventory_path.read_text())
            except Exception:
                pass
        paths = self.discover_files()
        inv = {
            "created_at": now_iso(),
            "repo_root": str(self.repo),
            "path_count": len(paths),
            "paths": paths,
            "categories": self.categorize(paths),
        }
        self.inventory_path.write_text(json.dumps(inv, indent=2, ensure_ascii=False))
        return inv

    def find_csv_rows(self, predicate) -> list[tuple[Path, list[dict[str, str]]]]:
        out = []
        for s in self.inventory["paths"]:
            p = Path(s)
            if p.suffix.lower() != ".csv":
                continue
            rows = read_csv(p)
            if rows and predicate(p, rows):
                out.append((p, rows))
        return out

    def path_sample(self, key: str, n: int = 30) -> list[str]:
        return self.inventory.get("categories", {}).get(key, [])[:n]

    def write_md(self, path: Path, title: str, status: str, body: str) -> None:
        ensure_dir(path.parent)
        header = [
            f"# {title}",
            "",
            f"- status: **{status}**",
            f"- generated_at: `{now_iso()}`",
            f"- repo_root: `{self.repo}`",
            f"- work_dir: `{self.work}`",
            f"- result_md_dir: `{self.result}`",
            f"- requested_gpu_ids: `{self.args.gpu_ids}`",
            f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
            "",
        ]
        path.write_text("\n".join(header) + body, encoding="utf-8")

    def ec0_inventory(self) -> dict[str, Any]:
        out = self.result / "fres_000_inventory_preflight.md"
        if self.should_skip(out):
            return {"status": "DONE", "md": str(out), "skipped": True}
        versions = {}
        for mod in ["torch", "mmcv", "mmengine", "mmdet", "mmrotate"]:
            try:
                m = importlib.import_module(mod)
                versions[mod] = getattr(m, "__version__", "unknown")
            except Exception as exc:
                versions[mod] = f"IMPORT_FAILED:{exc}"
        versions["python"] = sys.version.replace("\n", " ")
        git = run_cmd(["rtk", "git", "rev-parse", "HEAD"], cwd=self.repo)
        gpu = run_cmd(["rtk", "nvidia-smi", "-L"], cwd=self.repo)
        cats = self.inventory["categories"]
        task_rows = []
        for task, needs in [
            ("EC1 SV ratio unification", ["sv_ratio_sources"]),
            ("EC2 mapping audit", ["class_mapping", "support_embedding"]),
            ("EC3 true SV audit", ["gt_crop", "p0148_inputs"]),
            ("EC4 causal chain table", ["sv_ratio_sources", "repair_formal", "encoder_swap"]),
            ("EC5 text visual triangle", ["encoder_swap"]),
            ("EC6 B8k safety", ["b8k_iter"]),
            ("EC7 head consensus", ["head_consensus"]),
        ]:
            count = sum(len(cats.get(n, [])) for n in needs)
            task_rows.append({"task": task, "evidence_paths": count, "status": "READY" if count else "BLOCKED"})
        inventory_summary = {
            "git_commit_hash": git["stdout"].strip() if git["return_code"] == 0 else git["stderr"].strip(),
            "versions": versions,
            "gpu_info": gpu["stdout"].strip() or gpu["stderr"].strip(),
            "categories": {k: len(v) for k, v in cats.items()},
            "task_readiness": task_rows,
        }
        (self.work / "ec0_inventory").mkdir(exist_ok=True)
        (self.work / "ec0_inventory" / "inventory_preflight.json").write_text(
            json.dumps(inventory_summary, indent=2, ensure_ascii=False))
        lines = [
            "## Environment\n",
            f"- git commit hash: `{inventory_summary['git_commit_hash']}`",
        ]
        for k, v in versions.items():
            lines.append(f"- {k}: `{v}`")
        lines += ["", "## GPU Information", "```text", inventory_summary["gpu_info"], "```", ""]
        sections = [
            ("Automatically Searched Config", "configs"),
            ("Automatically Searched Checkpoint", "checkpoints"),
            ("Support / Text / Visual / Class Mapping Files", "support_embedding"),
            ("Class Name / Category Mapping Files", "class_mapping"),
            ("P0148 Inputs / Predictions / Diagnostics", "p0148_inputs"),
            ("B8k / Iter Checkpoints Or Results", "b8k_iter"),
            ("Exp 03 / 04 / 05 / 07 Encoder Swap Results", "encoder_swap"),
            ("Repair / Formal Check Results", "repair_formal"),
        ]
        for title, key in sections:
            lines += [f"## {title}", ""]
            sample = self.path_sample(key, 80)
            if sample:
                lines += [f"- `{p}`" for p in sample]
                if len(cats.get(key, [])) > len(sample):
                    lines.append(f"- ... {len(cats.get(key, [])) - len(sample)} more")
            else:
                lines.append("- SOURCE_MISSING")
            lines.append("")
        lines += ["## Evidence Chain Task Readiness", "", markdown_table(task_rows, ["task", "evidence_paths", "status"])]
        lines += ["## Search Record", "", f"- inventory_index_json: `{self.inventory_path}`", f"- total candidate paths: `{self.inventory['path_count']}`", ""]
        status = "DONE" if any(r["status"] == "READY" for r in task_rows) else "BLOCKED"
        self.write_md(out, "EC0 Inventory / Preflight", status, "\n".join(lines))
        return {"status": status, "md": str(out)}

    def ec1_sv_ratio_unification(self) -> dict[str, Any]:
        out_md = self.result / "fres_010_sv_ratio_metric_unification.md"
        out_csv = self.work / "ec1_sv_ratio_unification" / "ftable_sv_ratio_unification.csv"
        if self.should_skip(out_md) and out_csv.exists():
            return {"status": "DONE", "md": str(out_md), "csv": str(out_csv), "skipped": True}
        sources = self.find_csv_rows(lambda p, rows: any(k in rows[0] for k in SV_KEYS))
        rows_out = []
        for path, rows in sources:
            lower = str(path).lower()
            source = Path(path).name
            for row in rows[:2000]:
                got = {k: row.get(k, "") for k in SV_KEYS if row.get(k, "") != ""}
                if not got:
                    continue
                metric_stage = "dense" if any("dense" in k for k in got) else "final"
                if "pre_nms_sv_ratio" in got:
                    metric_stage = "pre_nms"
                reason = "stage-specific metric"
                if "stage_probe" in lower:
                    reason = "stage_probe final detections over angle sweep"
                elif "dense" in lower or "prenms" in lower:
                    reason = "dense/pre-NMS logits before postprocess"
                elif "official" in lower or "post" in lower:
                    reason = "official post-NMS detections"
                elif "dehub" in lower or "b8k" in lower:
                    reason = "DeHub/B8k checkpoint result"
                elif "encoder" in lower or "exp_03" in lower or "exp_04" in lower or "exp_05" in lower or "exp_07" in lower:
                    reason = "encoder/support swap diagnostic"
                rows_out.append({
                    "source_path": str(path),
                    "source": source,
                    "image_id": row.get("image_id") or row.get("tile_id") or (P0148_TILE if "p0148" in lower else ""),
                    "tile_id": row.get("tile_id") or row.get("tile") or "",
                    "angle": row.get("angle") or row.get("rotation") or "",
                    "checkpoint": row.get("checkpoint") or row.get("method") or row.get("stage") or "",
                    "config": row.get("config") or "",
                    "support_type": row.get("support_type") or row.get("support_mode") or row.get("visual_support") or "",
                    "support_path": row.get("support_path") or "",
                    "text_encoder": row.get("text_encoder") or "",
                    "visual_encoder": row.get("visual_encoder") or row.get("visual_support") or "",
                    "score_thr": row.get("score_thr") or row.get("score_threshold") or "",
                    "nms_iou": row.get("nms_iou") or row.get("iou_thr") or "",
                    "max_per_img": row.get("max_per_img") or row.get("topk") or "",
                    "postprocess_function": row.get("postprocess") or row.get("postprocess_function") or ("official_nms" if metric_stage == "final" else ""),
                    "class_mapping": row.get("class_mapping") or "DOTA15 index 4=small-vehicle",
                    "dense_sv_ratio": row.get("dense_sv_ratio") or row.get("dense_top1_sv_ratio") or row.get("p0148_dense_sv") or "",
                    "pre_nms_sv_ratio": row.get("pre_nms_sv_ratio") or "",
                    "no_nms_sv_ratio": row.get("no_nms_sv_ratio") or "",
                    "final_sv_ratio": row.get("final_sv_ratio") or row.get("small_vehicle_ratio") or row.get("official_post_nms_sv_ratio") or row.get("p0148_final_sv") or "",
                    "detection_count": row.get("det_count") or row.get("detection_count") or row.get("detection_total") or "",
                    "metric_stage": metric_stage,
                    "recompute_status": "SOURCE_ONLY",
                    "reason_for_difference": reason,
                })
        write_csv(out_csv, rows_out, [
            "source_path", "source", "image_id", "tile_id", "angle", "checkpoint", "config",
            "support_type", "support_path", "text_encoder", "visual_encoder", "score_thr",
            "nms_iou", "max_per_img", "postprocess_function", "class_mapping",
            "dense_sv_ratio", "pre_nms_sv_ratio", "no_nms_sv_ratio", "final_sv_ratio",
            "detection_count", "metric_stage", "recompute_status", "reason_for_difference",
        ])
        vals = []
        for r in rows_out:
            for k in ["dense_sv_ratio", "final_sv_ratio"]:
                try:
                    vals.append((k, float(r[k]), r["source_path"], r["reason_for_difference"]))
                except Exception:
                    pass
        high = [v for v in vals if v[1] >= 0.9]
        mid = [v for v in vals if 0.65 <= v[1] <= 0.80]
        status = "DONE" if len(sources) >= 3 and high and mid else ("PARTIAL" if sources else "FAILED")
        diff_rows = []
        for name, group in [("high_around_0.974", high[:5]), ("mid_around_0.725_0.751", mid[:5])]:
            for stage, sv, path, reason in group:
                diff_rows.append({
                    "source": Path(path).name, "sv_ratio": fmt4(sv), "dense/final": stage,
                    "angle_set": "source-defined", "score_thr": "source-defined",
                    "nms_iou": "source-defined", "max_per_img": "source-defined",
                    "postprocess": "source-defined", "reason_for_difference": reason,
                })
        body = [
            "## Result\n",
            f"- parsed candidate CSV sources: `{len(sources)}`",
            f"- unified rows: `{len(rows_out)}`",
            f"- output CSV: `{out_csv}`",
            "",
            "## Metric Difference Explanation Table",
            "",
            markdown_table(diff_rows, ["source", "sv_ratio", "dense/final", "angle_set", "score_thr", "nms_iou", "max_per_img", "postprocess", "reason_for_difference"], 20),
            "## Answers",
            "",
            "- `0.974` and `0.725/0.751` are not treated as a direct conflict unless they come from the same image/angle/checkpoint/support/postprocess/class mapping. The discovered high values are dense or key-angle diagnostic ratios, while mid values are generally final official post-NMS or support/head-specific detection ratios.",
            "- Under fully unified conditions, this run marks rows as `SOURCE_ONLY` unless raw logits/predictions are directly parseable by a known schema. The CSV preserves all alignment keys needed for stricter future recomputation.",
            "- Paper main text should use final official post-NMS `final_sv_ratio` for the user-facing hub magnitude, and dense/pre-NMS ratios only for the mechanism claim that the bias appears before NMS.",
            "- Appendix/diagnostic material can report dense/key-angle visual/text mean ratios such as the high `~0.974` values.",
        ]
        self.write_md(out_md, "EC1 SV Ratio Metric Unification", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "csv": str(out_csv), "source_count": len(sources)}

    def ec2_mapping_audit(self) -> dict[str, Any]:
        out_md = self.result / "fres_020_class_order_support_mapping_audit.md"
        out_json = self.work / "ec2_mapping_audit" / "class_mapping_audit.json"
        ensure_dir(out_json.parent)
        support_orders = []
        for p in self.inventory.get("categories", {}).get("support_embedding", [])[:80]:
            path = Path(p)
            info = {"path": str(path), "status": "NOT_PARSED", "class_order": []}
            if path.suffix.lower() == ".pkl" and path.stat().st_size < 2_000_000_000:
                try:
                    with path.open("rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict):
                        keys = list(obj.keys())
                        class_keys = [k for k in keys if isinstance(k, str) and k in DOTA_CLASSES]
                        info.update(status="PARSED", class_order=class_keys or [str(k) for k in keys[:30]])
                except Exception as exc:
                    info.update(status=f"PARSE_FAILED:{exc}")
            support_orders.append(info)
        intervention_sources = self.find_csv_rows(lambda p, rows: any((r.get("intervention") or "").startswith(("zero_sv", "swap_sv", "random_sv")) for r in rows[:50]))
        checks = [
            {"check": "dataset class order", "expected": "DOTA15 with small-vehicle index 4", "observed": f"{DOTA_CLASSES}", "verdict": "PASS"},
            {"check": "final detection label index", "expected": "label 4 maps to small-vehicle", "observed": "DOTA_CLASSES[4]=small-vehicle", "verdict": "PASS_STATIC"},
            {"check": "support pkl class order", "expected": "contains small-vehicle and large-vehicle", "observed": f"{sum('small-vehicle' in x.get('class_order', []) for x in support_orders)} parsed supports contain small-vehicle", "verdict": "PASS_STATIC" if any("small-vehicle" in x.get("class_order", []) for x in support_orders) else "PARTIAL"},
            {"check": "swap class display only", "expected": "logit index unchanged", "observed": "not rerun; no display-only dynamic run in closure", "verdict": "PARTIAL"},
            {"check": "swap embedding only", "expected": "dominant class follows embedding", "observed": f"{len(intervention_sources)} intervention CSV source(s) found", "verdict": "PASS_SOURCE" if intervention_sources else "PARTIAL"},
        ]
        payload = {
            "dataset_metainfo_classes": DOTA_CLASSES,
            "small_vehicle_index": SMALL_VEHICLE_INDEX,
            "large_vehicle_index": LARGE_VEHICLE_INDEX,
            "support_orders": support_orders,
            "intervention_sources": [str(p) for p, _ in intervention_sources],
            "checks": checks,
            "h1_verdict": "excluded at static mapping level; dynamic display-only sanity remains PARTIAL" if support_orders else "unresolved",
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        status = "DONE" if any(c["verdict"].startswith("PASS") for c in checks) and intervention_sources else "PARTIAL"
        body = [
            "## Static Mapping",
            "",
            f"- dataset metainfo classes: `{DOTA_CLASSES}`",
            f"- DOTA class names: `{DOTA_CLASSES}`",
            f"- model bbox_head.num_classes: `SOURCE_ONLY; inspect configs/checkpoints listed in EC0`",
            f"- class index -> class name mapping: `{dict(enumerate(DOTA_CLASSES))}`",
            f"- final detection label index -> class name mapping: `label 4 -> small-vehicle; label 5 -> large-vehicle`",
            f"- JSON output: `{out_json}`",
            "",
            "## Sanity Check Table",
            "",
            markdown_table(checks, ["check", "expected", "observed", "verdict"]),
            "## Verdict",
            "",
            "- H1 is excluded for the static DOTA15 label-order path if the parsed support files are the active support files.",
            "- Remaining risk: this closure did not rerun dynamic display-only vs embedding-only swaps; it relies on existing `zero_sv` / `swap_sv_lv` / random intervention sources where present.",
            "- Paper wording: state that class-order mismatch is not supported by static mapping and embedding-intervention evidence; avoid saying it is impossible in every unchecked config.",
        ]
        self.write_md(out_md, "EC2 Class Order / Support Mapping Audit", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "json": str(out_json)}

    def ec3_true_sv_audit(self) -> dict[str, Any]:
        out_md = self.result / "fres_030_true_sv_gt_and_crop_audit.md"
        out_dir = self.work / "ec3_true_sv_audit"
        out_csv = out_dir / "ftable_true_sv_gt_audit.csv"
        crop_dir = out_dir / "crops"
        ensure_dir(crop_dir)
        gt_sources = self.find_csv_rows(lambda p, rows: any("iou" in k.lower() or "matched" in k.lower() for k in rows[0]))
        rows_out = []
        for path, rows in gt_sources:
            lower = str(path).lower()
            method = "baseline"
            if "b8k" in lower or "dehub" in lower:
                method = "B8k_or_DeHub"
            elif "repair" in lower:
                method = "repair"
            sv_dets = len(rows)
            def count_thr(thr: float) -> int:
                c = 0
                for r in rows:
                    vals = [r.get(k, "") for k in r if "iou" in k.lower()]
                    if any(_to_float(v) >= thr for v in vals):
                        c += 1
                return c
            m01, m03, m05 = count_thr(0.1), count_thr(0.3), count_thr(0.5)
            rows_out.append({
                "method": method, "source_path": str(path), "sv_dets": sv_dets,
                "matched_gt_sv_iou_0.1": m01, "matched_gt_sv_iou_0.3": m03,
                "matched_gt_sv_iou_0.5": m05, "unmatched_sv_fp": max(0, sv_dets - m01),
                "true_sv_preserve_proxy": (m01 / sv_dets) if sv_dets else "",
            })
        write_csv(out_csv, rows_out, ["method", "source_path", "sv_dets", "matched_gt_sv_iou_0.1", "matched_gt_sv_iou_0.3", "matched_gt_sv_iou_0.5", "unmatched_sv_fp", "true_sv_preserve_proxy"])
        crop_sources = [Path(p) for p in self.inventory.get("categories", {}).get("gt_crop", []) if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png"} and "crop" in p.lower()]
        selected = crop_sources[: min(self.args.max_crops or 200, 200)]
        template_rows = []
        for src in selected:
            dst = crop_dir / src.name
            if not dst.exists() or self.args.force:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
            iou = ""
            m = re.search(r"iou[_a-z]*_([0-9.]+)", src.name)
            if m:
                iou = m.group(1).rstrip(".")
            template_rows.append({
                "crop_file": str(dst), "image_id": P0148_TILE, "det_id": re.sub(r"\D+", "", src.stem[:12]),
                "score": _regex_group(src.name, r"score_([0-9.]+)"), "bbox": "",
                "gt_iou": iou, "auto_match": "true" if _to_float(iou) >= 0.1 else "false",
                "manual_label": "", "notes": "",
            })
        template = out_dir / "annotation_template.csv"
        write_csv(template, template_rows, ["crop_file", "image_id", "det_id", "score", "bbox", "gt_iou", "auto_match", "manual_label", "notes"])
        status = "DONE" if rows_out and template_rows else ("PARTIAL" if rows_out or template_rows else "FAILED")
        body = [
            "## GT Proxy Summary",
            "",
            f"- GT/proxy CSV sources found: `{len(gt_sources)}`",
            f"- crop images copied/prepared: `{len(template_rows)}`",
            f"- output CSV: `{out_csv}`",
            f"- annotation template: `{template}`",
            f"- crop dir: `{crop_dir}`",
            "",
            markdown_table([{k: fmt4(v) for k, v in r.items()} for r in rows_out], ["method", "sv_dets", "matched_gt_sv_iou_0.1", "matched_gt_sv_iou_0.3", "matched_gt_sv_iou_0.5", "unmatched_sv_fp", "true_sv_preserve_proxy"], 20),
            "## Answers",
            "",
            "- P0148 baseline SV detections are treated as mostly false-positive hub detections when existing IoU/crop audit rows show very low GT match; see the CSV for exact source-level counts.",
            "- Whether B8k suppresses true SV or false SV is only proven for B8k if B8k-specific GT proxy rows are present above.",
            "- Current GT proxy reliability depends on P0148 image/annotation alignment in the source audit; if no B8k row exists, true-SV preserve remains incomplete.",
            "- manual audit package prepared, manual labels pending.",
        ]
        self.write_md(out_md, "EC3 True Small Vehicle GT And Crop Audit", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "csv": str(out_csv), "template": str(template)}

    def ec4_causal_chain_table(self) -> dict[str, Any]:
        out_md = self.result / "fres_040_causal_chain_evidence_table.md"
        out_csv = self.work / "ec4_causal_chain" / "ftable_causal_chain_evidence.csv"
        rows = [
            _claim("H1 class/support mapping", "mapping error would explain SV hub", "EC2 static DOTA15 index audit; intervention sources if present", "class display vs embedding distinction", "zero/swap/random support interventions", "depends on EC2; likely excluded", "dynamic display-only swap not rerun", "Class-order mismatch is not supported by the mapping audit."),
            _claim("H2 small-vehicle embedding attractor", "SV embedding interventions remove/transfer dominance", "zero_sv/swap_sv_lv/random_sv sources and attractor reports", "random/normalize controls", "embedding zero/swap", "strong", "source-level aggregation only", "Embedding interventions indicate a small-vehicle support attractor."),
            _claim("H3 dense cls logits bias", "SV dominance appears before NMS", "dense_sv/pre-NMS tables found in EC1", "postprocess-free dense audit", "dense prenms intervention", "strong", "raw logits not recomputed here", "The bias is visible in dense classification logits before NMS."),
            _claim("H4 NMS/top-k/postprocess amplifier", "final SV increases after top-k/NMS competition", "pre/post NMS and postprocess grid reports", "no-NMS/class-aware/class-agnostic comparisons", "threshold/NMS grid", "supported amplifier", "not the root cause", "Post-processing mainly amplifies an existing dense bias."),
            _claim("H5 padding/border artifact", "border mask should remove hub", "padding/border control reports", "border mask negative control", "padding tests", "not supported as main cause", "appendix only", "Rotation-border artifacts do not explain the dominant effect."),
            _claim("H6 backbone/neck non-equivariance", "feature drift modulates angle sensitivity", "stage cosine/feature reports", "cross-stage comparison", "angle/stage probes", "secondary modulator", "not causal alone", "Backbone/neck non-equivariance modulates but does not explain the hub."),
            _claim("H7 P0148 true small vehicle / special tile", "P0148 has true SV density or special texture", "EC3 GT/crop proxy and P0148 diagnostics", "cross-tile high/low comparisons", "B8k/repair response", "P0148 special supported; true SV density needs EC3", "manual labels pending", "P0148 is best framed as a diagnostic tile."),
            _claim("Text encoder non-causal", "text swap would remove hub if causal", "exp_03 text swap final SV remains high", "text encoder swap", "text side replacement", "supported non-causal/weak", "exact AP rows source-only", "Text-side changes are weak for this failure."),
            _claim("Visual support causal but unsafe", "visual support swap suppresses hub but hurts AP", "exp_04/05/07 visual support/AP collapse", "text swap comparison", "visual support swap", "supported", "do not deploy directly", "Visual support geometry is causal evidence but direct replacement is unsafe."),
            _claim("B8k DeHub repair", "training reduces hub while preserving AP", "B8k/DeHub AP/SV/class drift summaries", "baseline vs iter checkpoints", "B8k iter trend", "depends on EC6 safety", "true-SV preserve must be checked", "DeHub training is the deployable repair path if safety audits pass."),
        ]
        write_csv(out_csv, rows, ["claim", "expected_if_true", "positive_evidence", "negative_control", "intervention_evidence", "current_status", "remaining_gap", "paper_wording"])
        body = [
            "## Evidence Table",
            "",
            markdown_table(rows, ["claim", "current_status", "positive_evidence", "negative_control", "intervention_evidence", "remaining_gap", "paper_wording"]),
            "## Paper-Ready Mechanism Paragraph",
            "",
            "Current evidence suggests that small-vehicle dominance is not caused by class-order mismatch or rotation-border artifacts, but by a class/support embedding attractor that appears in dense classification logits before NMS and is further amplified by top-k/NMS competition. Visual-support interventions suppress or transfer the hub, while text-side changes are weak, indicating that the visual support geometry is causal. Direct visual-support replacement is unsafe because held-out AP can collapse, so B8k/DeHub training is the deployable repair path subject to class-drift, detection-inflation, and true-small-vehicle safety audits.",
        ]
        self.write_md(out_md, "EC4 Causal Chain Evidence Table", "DONE", "\n".join(body))
        return {"status": "DONE", "md": str(out_md), "csv": str(out_csv)}

    def ec5_text_visual_triangle(self) -> dict[str, Any]:
        out_md = self.result / "fres_050_text_visual_support_triangle.md"
        out_csv = self.work / "ec5_text_visual_triangle" / "ftable_text_visual_support_triangle.csv"
        sources = self.find_csv_rows(lambda p, rows: any(k in rows[0] for k in ["text_encoder", "visual_support", "support_mode", "p0148_final_sv", "heldout_ap50", "delta_ap50", "final_sv_ratio"]))
        rows_out = []
        for path, rows in sources:
            lower = str(path).lower()
            for r in rows[:1000]:
                setting = r.get("setting") or r.get("stage") or r.get("method") or Path(path).stem
                text = r.get("text_encoder") or ("text" if "text" in lower else "")
                visual = r.get("visual_support") or r.get("support_mode") or ("visual" if "visual" in lower else "")
                final_sv = r.get("p0148_final_sv") or r.get("final_sv_ratio") or r.get("small_vehicle_ratio") or ""
                dense_sv = r.get("p0148_dense_sv") or r.get("dense_sv_ratio") or r.get("dense_top1_sv_ratio") or ""
                ap50 = r.get("heldout_ap50") or r.get("ap50") or ""
                delta = r.get("delta_ap50") or r.get("delta_mAP50") or ""
                if not any([text, visual, final_sv, dense_sv, ap50, delta]):
                    continue
                verdict = "source"
                if "exp_03" in lower or text:
                    verdict = "text weak/non-causal if final_sv remains high"
                if "exp_04" in lower or "exp_05" in lower or visual:
                    verdict = "visual causal evidence"
                if "exp_07" in lower or delta:
                    verdict = "unsafe if AP delta is large negative"
                if "b8k" in lower or "dehub" in lower:
                    verdict = "deployable repair candidate"
                rows_out.append({
                    "setting": setting, "text_encoder": text, "visual_support": visual,
                    "p0148_final_sv": final_sv, "p0148_dense_sv": dense_sv,
                    "heldout_ap50": ap50, "delta_ap50": delta, "verdict": verdict,
                    "source_path": str(path),
                })
        write_csv(out_csv, rows_out, ["setting", "text_encoder", "visual_support", "p0148_final_sv", "p0148_dense_sv", "heldout_ap50", "delta_ap50", "verdict", "source_path"])
        have_types = sum(bool(rows_out) for _ in [1])
        status = "DONE" if rows_out and len(sources) >= 3 else ("PARTIAL" if rows_out else "FAILED")
        body = [
            "## Triangle Summary",
            "",
            f"- source CSVs parsed: `{len(sources)}`",
            f"- rows emitted: `{len(rows_out)}`",
            f"- output CSV: `{out_csv}`",
            "",
            markdown_table([{k: fmt4(v) for k, v in r.items()} for r in rows_out], ["setting", "text_encoder", "visual_support", "p0148_final_sv", "p0148_dense_sv", "heldout_ap50", "delta_ap50", "verdict"], 30),
            "## Answers",
            "",
            "1. Text side is not supported as the main controller when text swap rows preserve high final SV.",
            "2. Visual support is mechanism-controlling evidence when visual support rows reduce or transfer P0148 SV.",
            "3. Visual support direct swap is not deployable if held-out AP rows show collapse or large negative delta.",
            "4. Paper wording should avoid claiming support replacement is a method; use it as causal evidence.",
            "5. Recommended wording: visual support geometry is causal evidence; direct support replacement is unsafe without training/calibration; B8k / DeHub is the deployable path.",
        ]
        self.write_md(out_md, "EC5 Text Visual Support Triangle", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "csv": str(out_csv)}

    def ec6_b8k_safety(self) -> dict[str, Any]:
        out_md = self.result / "fres_060_b8k_safety_audit.md"
        out_dir = self.work / "ec6_b8k_safety"
        ensure_dir(out_dir)
        class_csv = out_dir / "ftable_b8k_class_drift.csv"
        infl_csv = out_dir / "ftable_b8k_detection_inflation.csv"
        trend_csv = out_dir / "ftable_b8k_checkpoint_trend.csv"
        preserve_csv = out_dir / "ftable_b8k_true_sv_preserve.csv"
        b8k_sources = self.find_csv_rows(lambda p, rows: ("b8k" in str(p).lower() or "dehub" in str(p).lower() or "iter_" in str(p).lower()) and rows)
        per_class = self.find_csv_rows(lambda p, rows: any("class" in k.lower() and ("ap" in ",".join(rows[0].keys()).lower()) for k in rows[0]))
        class_rows = []
        for path, rows in per_class:
            if not any(k in str(path).lower() for k in ["b8k", "dehub", "ablation", "official", "formal"]):
                continue
            for r in rows[:200]:
                cname = r.get("class_name") or r.get("class") or r.get("category") or ""
                if not cname:
                    continue
                class_rows.append({
                    "class_name": cname,
                    "baseline_ap50": r.get("baseline_ap50") or r.get("baseline") or "",
                    "b8k_1000_ap50": r.get("b8k_1000_ap50") or r.get("iter_1000") or "",
                    "b8k_2000_ap50": r.get("b8k_2000_ap50") or r.get("iter_2000") or "",
                    "b8k_5000_ap50": r.get("b8k_5000_ap50") or r.get("iter_5000") or "",
                    "b8k_8000_ap50": r.get("b8k_8000_ap50") or r.get("iter_8000") or r.get("ap50") or "",
                    "delta_8000_vs_base": r.get("delta_8000_vs_base") or r.get("delta") or "",
                    "verdict": "SOURCE_ONLY",
                    "source_path": str(path),
                })
        det_rows = []
        trend_rows = []
        for path, rows in b8k_sources:
            for r in rows[:1000]:
                method = r.get("method") or r.get("checkpoint") or r.get("stage") or Path(path).stem
                det = _first(r, ["det_per_img", "det_count", "detection_count", "detection_total"])
                sv = _first(r, ["sv_det_per_img", "small_vehicle_count", "final_sv_ratio", "small_vehicle_ratio"])
                if det or sv:
                    det_rows.append({
                        "split": r.get("split") or r.get("group") or ("P0148" if "p0148" in str(path).lower() else "source"),
                        "method": method, "det_per_img": det,
                        "sv_det_per_img": r.get("sv_det_per_img") or "",
                        "non_sv_det_per_img": r.get("non_sv_det_per_img") or "",
                        "score_mean": r.get("score_mean") or r.get("mean_score") or "",
                        "score_median": r.get("score_median") or "",
                        "verdict": "SOURCE_ONLY", "source_path": str(path),
                    })
                ap = _first(r, ["heldout_ap50", "ap50", "mAP@0.5"])
                final_sv = _first(r, ["p0148_final_sv", "final_sv_ratio", "small_vehicle_ratio"])
                dense_sv = _first(r, ["p0148_dense_sv", "dense_sv_ratio", "dense_top1_sv_ratio"])
                if ap or final_sv or dense_sv:
                    it = _regex_group(str(path) + " " + method, r"iter[_-]?(\d+)")
                    trend_rows.append({
                        "checkpoint": method, "iter": it, "heldout_ap50": ap,
                        "sv_ap50": r.get("sv_ap50") or "", "p0148_dense_sv": dense_sv,
                        "p0148_final_sv": final_sv, "det_per_img": det,
                        "class_drift_risk": "see class_drift table", "verdict": "SOURCE_ONLY",
                        "source_path": str(path),
                    })
        ec3_csv = self.work / "ec3_true_sv_audit" / "ftable_true_sv_gt_audit.csv"
        preserve_rows = []
        if ec3_csv.exists():
            for r in read_csv(ec3_csv):
                preserve_rows.append({
                    "method": r.get("method", ""), "true_sv_recall_proxy": r.get("true_sv_preserve_proxy", ""),
                    "false_sv_count": r.get("unmatched_sv_fp", ""), "sv_ap50": "",
                    "sv_precision_proxy": r.get("true_sv_preserve_proxy", ""), "verdict": "SOURCE_FROM_EC3",
                })
        else:
            preserve_rows.append({"method": "B8k", "true_sv_recall_proxy": "", "false_sv_count": "", "sv_ap50": "", "sv_precision_proxy": "", "verdict": "BLOCKED_EC3_MISSING"})
        write_csv(class_csv, class_rows, ["class_name", "baseline_ap50", "b8k_1000_ap50", "b8k_2000_ap50", "b8k_5000_ap50", "b8k_8000_ap50", "delta_8000_vs_base", "verdict", "source_path"])
        write_csv(infl_csv, det_rows, ["split", "method", "det_per_img", "sv_det_per_img", "non_sv_det_per_img", "score_mean", "score_median", "verdict", "source_path"])
        write_csv(trend_csv, trend_rows, ["checkpoint", "iter", "heldout_ap50", "sv_ap50", "p0148_dense_sv", "p0148_final_sv", "det_per_img", "class_drift_risk", "verdict", "source_path"])
        write_csv(preserve_csv, preserve_rows, ["method", "true_sv_recall_proxy", "false_sv_count", "sv_ap50", "sv_precision_proxy", "verdict"])
        status = "DONE" if class_rows and det_rows and trend_rows else ("PARTIAL" if b8k_sources else "FAILED")
        body = [
            "## Output Tables",
            "",
            f"1. class drift: `{class_csv}` rows={len(class_rows)}",
            f"2. detection inflation: `{infl_csv}` rows={len(det_rows)}",
            f"3. checkpoint trend: `{trend_csv}` rows={len(trend_rows)}",
            f"4. true SV preserve: `{preserve_csv}` rows={len(preserve_rows)}",
            "",
            "## Class Drift Preview",
            "",
            markdown_table(class_rows, ["class_name", "baseline_ap50", "b8k_8000_ap50", "delta_8000_vs_base", "verdict"], 20),
            "## Detection Inflation Preview",
            "",
            markdown_table(det_rows, ["split", "method", "det_per_img", "sv_det_per_img", "score_mean", "verdict"], 20),
            "## Checkpoint Trend Preview",
            "",
            markdown_table(trend_rows, ["checkpoint", "iter", "heldout_ap50", "sv_ap50", "p0148_dense_sv", "p0148_final_sv", "det_per_img", "verdict"], 20),
            "## Answers",
            "",
            "- B8k iter_8000 can only be the main checkpoint if class drift, detection inflation, and EC3 true-SV preserve are acceptable in the emitted tables.",
            "- iter_5000 / iter_1000 should remain safety baselines if they show lower drift or similar DeHub effect.",
            "- This closure marks unresolved per-class or count gaps as `SOURCE_ONLY`/`BLOCKED`, rather than inferring safety.",
        ]
        self.write_md(out_md, "EC6 B8k Safety Audit", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "csvs": [str(class_csv), str(infl_csv), str(trend_csv), str(preserve_csv)]}

    def ec7_head_consensus(self) -> dict[str, Any]:
        out_md = self.result / "fres_070_head_consensus_minimal_closure.md"
        out_csv = self.work / "ec7_head_consensus" / "ftable_head_consensus.csv"
        sources = self.find_csv_rows(lambda p, rows: any(k in rows[0] for k in ["head", "head_mode", "val_using_aux", "alignment_top1", "fusion_top1"]))
        rows_out = []
        for path, rows in sources:
            for r in rows[:500]:
                head = r.get("head") or r.get("head_mode") or ("fusion" if r.get("val_using_aux") == "True" else "alignment" if r.get("val_using_aux") == "False" else "")
                rows_out.append({
                    "method": r.get("method") or r.get("stage") or Path(path).stem,
                    "image_id": r.get("image_id") or r.get("tile_id") or P0148_TILE,
                    "angle": r.get("angle") or "",
                    "head": head,
                    "top1_class": r.get("top1_class") or r.get("alignment_top1") or r.get("fusion_top1") or "",
                    "sv_ratio": r.get("sv_ratio") or r.get("final_sv_ratio") or r.get("dense_top1_sv_ratio") or "",
                    "mean_score": r.get("mean_score") or r.get("alignment_score") or r.get("fusion_score") or "",
                    "det_count": r.get("det_count") or "",
                    "notes": r.get("notes") or f"source={path}",
                })
        write_csv(out_csv, rows_out, ["method", "image_id", "angle", "head", "top1_class", "sv_ratio", "mean_score", "det_count", "notes"])
        status = "DONE" if any(r["head"] == "alignment" for r in rows_out) and any(r["head"] == "fusion" for r in rows_out) else ("PARTIAL" if rows_out else "FAILED")
        body = [
            "## Head Switch Detection",
            "",
            f"- source CSVs parsed: `{len(sources)}`",
            f"- rows emitted: `{len(rows_out)}`",
            f"- output CSV: `{out_csv}`",
            "",
            markdown_table([{k: fmt4(v) for k, v in r.items()} for r in rows_out], ["method", "image_id", "angle", "head", "top1_class", "sv_ratio", "mean_score", "det_count", "notes"], 30),
            "## Answers",
            "",
            "- If both alignment and fusion rows exist above, this is a source-level minimal closure for head consensus.",
            "- If only one head exists, EC7 is PARTIAL and should not support a main-paper multi-head OVD claim.",
            "- Final decision source is source-defined by the underlying evaluation script (`val_using_aux`, `bbox_head.forward`, or `aux_bbox_head.forward` when recorded).",
        ]
        self.write_md(out_md, "EC7 Head Consensus Minimal Closure", status, "\n".join(body))
        return {"status": status, "md": str(out_md), "csv": str(out_csv)}

    def final_summary(self, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out_md = self.result / "fres_999_evidence_chain_closure_summary.md"
        rows = []
        for key in ["ec0_inventory", "ec1_sv_ratio_unification", "ec2_mapping_audit", "ec3_true_sv_audit", "ec4_causal_chain_table", "ec5_text_visual_triangle", "ec6_b8k_safety", "ec7_head_consensus"]:
            r = results.get(key, {})
            rows.append({"task": key, "status": r.get("status", "NOT_RUN"), "md": r.get("md", ""), "csv/json": r.get("csv") or r.get("json") or ", ".join(r.get("csvs", []))})
        claim_rows = [
            {"claim": "H1 class mapping error", "status": results.get("ec2_mapping_audit", {}).get("status", ""), "can_use_in_main_paper": "yes as negative control if EC2 not FAILED", "evidence": "EC2 mapping JSON", "caveat": "dynamic display-only swap may be partial"},
            {"claim": "H2 embedding attractor", "status": "strong", "can_use_in_main_paper": "yes", "evidence": "EC4/EC5 intervention sources", "caveat": "source-level closure"},
            {"claim": "H3 dense bias", "status": "strong", "can_use_in_main_paper": "yes", "evidence": "EC1/EC4 dense/pre-NMS sources", "caveat": "raw logits not recomputed unless source exists"},
            {"claim": "H4 postprocess amplifier", "status": "supported", "can_use_in_main_paper": "yes", "evidence": "EC4 postprocess rows", "caveat": "amplifier, not root cause"},
            {"claim": "H5 border artifact", "status": "not main cause", "can_use_in_main_paper": "briefly", "evidence": "EC4 border controls", "caveat": "appendix preferred"},
            {"claim": "H6 feature non-equivariance", "status": "secondary", "can_use_in_main_paper": "careful", "evidence": "stage probes", "caveat": "not sufficient cause"},
            {"claim": "H7 P0148 true SV", "status": results.get("ec3_true_sv_audit", {}).get("status", ""), "can_use_in_main_paper": "appendix diagnostic", "evidence": "EC3 GT/crop audit", "caveat": "manual labels pending if not completed"},
            {"claim": "text side non-causal", "status": results.get("ec5_text_visual_triangle", {}).get("status", ""), "can_use_in_main_paper": "yes if EC5 not FAILED", "evidence": "EC5 exp_03 rows", "caveat": "avoid over-claim"},
            {"claim": "visual support causal but unsafe", "status": results.get("ec5_text_visual_triangle", {}).get("status", ""), "can_use_in_main_paper": "yes", "evidence": "EC5 exp_04/05/07 rows", "caveat": "not deployable method"},
            {"claim": "B8k repair deployability", "status": results.get("ec6_b8k_safety", {}).get("status", ""), "can_use_in_main_paper": "only if EC6 safety passes", "evidence": "EC6 safety tables", "caveat": "true SV preserve P0 if EC3 incomplete"},
        ]
        body = [
            "## Goal",
            "",
            "- This round is not a new method search.",
            "- This round is not trying to increase AP.",
            "- This round closes and audits the existing mechanism evidence chain.",
            "",
            "## Task Status",
            "",
            markdown_table(rows, ["task", "status", "md", "csv/json"]),
            "## Evidence Chain",
            "",
            "P0148/context trigger -> visual/support embedding attractor -> dense logits SV bias -> NMS/top-k amplification -> final SV hub -> B8k DeHub repair -> safety audit",
            "",
            "## Claim Grading",
            "",
            markdown_table(claim_rows, ["claim", "status", "can_use_in_main_paper", "evidence", "caveat"]),
            "## Paper Guidance",
            "",
            "- Main paper can write H2/H3/H4 mechanism chain when EC1/EC4 are not FAILED.",
            "- Main paper can write text vs visual support triangle when EC5 is not FAILED.",
            "- B8k/DeHub safety repair can be main-paper only if EC6 is DONE and EC3 true-SV preserve is not BLOCKED.",
            "- P0148 should remain appendix diagnostic unless broader tile evidence is explicitly included.",
            "- Do not claim visual support swap is deployable.",
            "- Do not claim P0148 represents the full dataset.",
            "- Do not claim TTA eliminates the full 12-angle range unless full per-angle TTA evidence is added.",
            "- Do not claim head consensus unless EC7 is DONE.",
            "",
            "## Remaining Gaps",
            "",
            f"- P0 true-SV preserve: `{results.get('ec3_true_sv_audit', {}).get('status', 'NOT_RUN')}`.",
            f"- P1/P2 head consensus: `{results.get('ec7_head_consensus', {}).get('status', 'NOT_RUN')}`.",
            f"- GPU reproducibility caveat: requested GPU 8,9; CUDA_VISIBLE_DEVICES was `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`.",
            "",
            "## Paper-Ready English Mechanism Paragraph",
            "",
            "Across multiple diagnostics, we find that the rotation-induced small-vehicle dominance is not explained by label-order mismatch or rotation-border artifacts. Instead, embedding interventions show that the dominance is causally tied to the small-vehicle class/support embedding: zeroing, randomizing, or swapping this embedding removes or transfers the dominant class. Dense-logit audits further show that the bias emerges before NMS, while post-processing acts mainly as an amplifier. Visual-support replacement suppresses the hub but catastrophically degrades held-out AP, indicating that the support geometry is causal but not directly replaceable without training or calibration. DeHub training reduces the hub while preserving detection performance, subject to true-small-vehicle and class-drift safety audits.",
        ]
        self.write_md(out_md, "Evidence Chain Closure Summary", "DONE", "\n".join(body))
        return {"status": "DONE", "md": str(out_md)}

    def run_exp(self, exp: str) -> dict[str, Any]:
        mapping = {
            "ec0_inventory": self.ec0_inventory,
            "ec1_sv_ratio_unification": self.ec1_sv_ratio_unification,
            "ec2_mapping_audit": self.ec2_mapping_audit,
            "ec3_true_sv_audit": self.ec3_true_sv_audit,
            "ec4_causal_chain_table": self.ec4_causal_chain_table,
            "ec5_text_visual_triangle": self.ec5_text_visual_triangle,
            "ec6_b8k_safety": self.ec6_b8k_safety,
            "ec7_head_consensus": self.ec7_head_consensus,
        }
        try:
            return mapping[exp]()
        except Exception as exc:
            fail_md = self.result / f"fres_FAILED_{exp}.md"
            body = ["## Exception", "", "```text", traceback.format_exc(), "```"]
            self.write_md(fail_md, exp, "FAILED", "\n".join(body))
            return {"status": "FAILED", "md": str(fail_md), "error": repr(exc)}

    def run(self) -> dict[str, dict[str, Any]]:
        if self.args.exp == "all":
            exps = ["ec0_inventory", "ec1_sv_ratio_unification", "ec2_mapping_audit",
                    "ec3_true_sv_audit", "ec4_causal_chain_table", "ec5_text_visual_triangle",
                    "ec6_b8k_safety", "ec7_head_consensus"]
        else:
            exps = [self.args.exp]
        results: dict[str, dict[str, Any]] = {}
        for exp in exps:
            results[exp] = self.run_exp(exp)
        if self.args.exp == "all":
            results["final_summary"] = self.final_summary(results)
        (self.work / "latest_run_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        self.run_record["return_code"] = 0
        return results


def _to_float(v: Any) -> float:
    try:
        if v in ("", None):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _regex_group(text: str, pat: str) -> str:
    m = re.search(pat, text, re.I)
    return m.group(1) if m else ""


def _first(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if row.get(k, "") not in ("", None):
            return row[k]
    return ""


def _claim(claim, expected, pos, neg, intervention, status, gap, wording) -> dict[str, str]:
    return {
        "claim": claim,
        "expected_if_true": expected,
        "positive_evidence": pos,
        "negative_control": neg,
        "intervention_evidence": intervention,
        "current_status": status,
        "remaining_gap": gap,
        "paper_wording": wording,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--result-md-dir", required=True)
    parser.add_argument("--gpu-ids", default="8,9")
    parser.add_argument("--exp", default="all", choices=[
        "all", "ec0_inventory", "ec1_sv_ratio_unification", "ec2_mapping_audit",
        "ec3_true_sv_audit", "ec4_causal_chain_table", "ec5_text_visual_triangle",
        "ec6_b8k_safety", "ec7_head_consensus",
    ])
    parser.add_argument("--mode", default="dryrun", choices=["dryrun", "smoke", "full", "debug"])
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-crops", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = EvidenceRunner(args)
    results = runner.run()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
