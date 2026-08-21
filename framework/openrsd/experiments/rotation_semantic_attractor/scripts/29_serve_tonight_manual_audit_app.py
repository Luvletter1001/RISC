#!/usr/bin/env python3
"""Serve a click-to-label UI for the verified expanded manual audit workspace."""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


SUBSETS = {
    "valid_unmatched_sv": {
        "csv": "valid_unmatched_sv_300_for_corrected_fsv.csv",
        "title": "未匹配小车 300 张：用于 corrected-FSV",
        "purpose": "只有这一组会进入 corrected-FSV。请判断红框里的 small-vehicle 预测到底是真实但未标注的小车，还是背景/其他物体造成的真正误检。",
        "labels": [
            "true_vehicle_annot_missing",
            "true_vehicle_annot_present_but_missed_match",
            "non_vehicle_background",
            "non_vehicle_object_conflict",
            "ambiguous",
            "invalid_visualization",
        ],
    },
    "true_sv_positive_control": {
        "csv": "true_sv_positive_control_100_for_qc.csv",
        "title": "真实小车正控 100 张：用于人工标注 QC",
        "purpose": "这一组是正控，用来检查人工标注者能否识别已标注的真实小车。",
        "labels": [
            "true_vehicle_annot_present",
            "true_vehicle_annot_present_but_missed_match",
            "ambiguous",
            "invalid_visualization",
            "non_vehicle_background",
            "non_vehicle_object_conflict",
        ],
    },
    "degenerate_large_sv_box": {
        "csv": "degenerate_large_sv_box_60_spotcheck.csv",
        "title": "退化大框小车 60 张：单独 failure mode 抽查",
        "purpose": "这一组不用于 corrected-FSV。它只用于审计另一类失败模式：模型把 small-vehicle 预测成异常大框。",
        "labels": [
            "actual_large_sv_degenerate_prediction",
            "padding_or_boundary_related",
            "large_context_texture",
            "object_conflict",
            "uncertain",
        ],
    },
    "padding_artifact": {
        "csv": "padding_artifact_30_spotcheck.csv",
        "title": "padding / 边界伪影 30 张：过滤边界抽查",
        "purpose": "这一组不用于 corrected-FSV。它用于检查 valid-mask / padding 过滤是否合理。",
        "labels": [
            "valid_padding_artifact",
            "not_padding_artifact",
            "uncertain",
        ],
    },
    "strict_object_flip": {
        "csv": "strict_object_flip_30_qualitative_candidates.csv",
        "title": "严格物体翻转 30 张：论文定性图候选",
        "purpose": "这一组用于预筛论文 qualitative case，不用于 corrected-FSV。",
        "labels": [
            "good_qualitative_case",
            "not_clear",
            "wrong_or_ambiguous",
        ],
    },
}

CONFIDENCE = ["high", "medium", "low"]
LABEL_DISPLAY = {
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
CONFIDENCE_DISPLAY = {
    "high": "高置信",
    "medium": "中置信",
    "low": "低置信",
}
META_DISPLAY = {
    "crop_id": "样本 ID",
    "audit_category": "审计类别",
    "model_name": "模型",
    "tile_id": "图块 ID",
    "angle": "旋转角度",
    "score": "预测分数",
    "pred_class": "预测类别",
    "raw_box_type": "原始框类型",
    "raw_box_angle": "原始框角度",
    "raw_box_area": "原始框面积",
    "valid_mask_ratio_inside_box": "框内有效区域比例",
    "padding_overlap_ratio": "padding 重叠比例",
    "best_gt_class": "最接近 GT 类别",
    "best_gt_iou": "最接近 GT IoU",
    "coordinate_frame_consistent": "坐标系一致",
    "valid_for_human_audit": "可用于人工审计",
}
META_COLS = [
    "crop_id",
    "audit_category",
    "model_name",
    "tile_id",
    "angle",
    "score",
    "pred_class",
    "raw_box_type",
    "raw_box_angle",
    "raw_box_area",
    "valid_mask_ratio_inside_box",
    "padding_overlap_ratio",
    "best_gt_class",
    "best_gt_iou",
    "coordinate_frame_consistent",
    "valid_for_human_audit",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({h: row.get(h, "") for h in headers})
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def ensure_label_headers(headers: list[str]) -> list[str]:
    out = list(headers)
    for col in ["human_label", "human_confidence", "human_notes"]:
        if col not in out:
            out.append(col)
    return out


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


class AuditState:
    def __init__(self, repo_root: Path, work_dir: Path, label_csv: Path):
        self.repo_root = repo_root
        self.work_dir = work_dir
        self.label_csv = label_csv
        self.crop_root = repo_root / "experiments/rotation_semantic_attractor/reports/visual_summary/crops_verified_expanded"
        self.lock = threading.Lock()

    def subset_path(self, subset: str) -> Path:
        return self.work_dir / SUBSETS[subset]["csv"]

    def load_subset(self, subset: str) -> tuple[list[str], list[dict[str, str]]]:
        headers, rows = read_csv(self.subset_path(subset))
        return ensure_label_headers(headers), rows

    def load_master(self) -> tuple[list[str], list[dict[str, str]]]:
        headers, rows = read_csv(self.label_csv)
        return ensure_label_headers(headers), rows

    def summary(self) -> dict[str, Any]:
        data = {}
        for subset in SUBSETS:
            _, rows = self.load_subset(subset)
            labeled = sum(1 for row in rows if row.get("human_label", "").strip())
            data[subset] = {
                "total": len(rows),
                "labeled": labeled,
                "remaining": len(rows) - labeled,
            }
        return data

    def save_label(self, subset: str, crop_id: str, human_label: str, human_confidence: str, human_notes: str) -> dict[str, Any]:
        if subset not in SUBSETS:
            raise ValueError("unknown subset")
        if human_label and human_label not in SUBSETS[subset]["labels"]:
            raise ValueError(f"invalid label for {subset}: {human_label}")
        if human_confidence and human_confidence not in CONFIDENCE:
            raise ValueError(f"invalid confidence: {human_confidence}")
        with self.lock:
            subset_headers, subset_rows = self.load_subset(subset)
            matched = False
            for row in subset_rows:
                if row.get("crop_id") == crop_id:
                    row["human_label"] = human_label
                    row["human_confidence"] = human_confidence
                    row["human_notes"] = human_notes
                    matched = True
                    break
            if not matched:
                raise ValueError(f"crop_id not found in subset: {crop_id}")
            write_csv_atomic(self.subset_path(subset), subset_headers, subset_rows)

            master_headers, master_rows = self.load_master()
            for row in master_rows:
                if row.get("crop_id") == crop_id:
                    row["human_label"] = human_label
                    row["human_confidence"] = human_confidence
                    row["human_notes"] = human_notes
                    break
            else:
                raise ValueError(f"crop_id not found in master label csv: {crop_id}")
            write_csv_atomic(self.label_csv, master_headers, master_rows)
        return {"ok": True, "summary": self.summary()}

    def clear_label(self, subset: str, crop_id: str) -> dict[str, Any]:
        return self.save_label(subset, crop_id, "", "", "")


def image_url(path: str) -> str:
    return "/file?path=" + quote(path)


def display_label(label: str) -> str:
    text = LABEL_DISPLAY.get(label, label)
    return f"{text} ({label})"


def display_confidence(confidence: str) -> str:
    text = CONFIDENCE_DISPLAY.get(confidence, confidence)
    return f"{text} ({confidence})"


def render_index(state: AuditState) -> str:
    summary = state.summary()
    cards = []
    for key, spec in SUBSETS.items():
        s = summary[key]
        cards.append(
            f"<a class='card' href='/subset/{esc(key)}'>"
            f"<strong>{esc(spec['title'])}</strong>"
            f"<span>已标注 {s['labeled']} / {s['total']}，剩余 {s['remaining']}</span>"
            f"</a>"
        )
    return page_shell(
        "今晚人工审计",
        "<h1>今晚人工审计</h1>"
        "<p>选择一个子集，查看三张图，然后点击中文分类和置信度。保存会立刻写入服务器上的 CSV，后台仍使用英文枚举值。</p>"
        "<p class='warn'>不要使用旧 200 张 crop。corrected-FSV 只使用“未匹配小车 300 张”。</p>"
        "<div class='cards'>" + "\n".join(cards) + "</div>"
        "<p><button onclick='runValidation()'>运行标签校验</button> <span id='validation'></span></p>",
    )


def render_subset(state: AuditState, subset: str, index: int) -> str:
    if subset not in SUBSETS:
        return page_shell("Unknown subset", "<h1>Unknown subset</h1>")
    spec = SUBSETS[subset]
    _, rows = state.load_subset(subset)
    if not rows:
        return page_shell(spec["title"], f"<h1>{esc(spec['title'])}</h1><p>No rows.</p>")
    index = max(0, min(index, len(rows) - 1))
    row = rows[index]
    labeled = sum(1 for r in rows if r.get("human_label", "").strip())
    next_unlabeled = next((i for i, r in enumerate(rows) if not r.get("human_label", "").strip()), index)
    nav = [
        f"<a href='/'>首页</a>",
        f"<a href='/subset/{esc(subset)}?i={max(0, index - 1)}'>上一张</a>",
        f"<a href='/subset/{esc(subset)}?i={min(len(rows) - 1, index + 1)}'>下一张</a>",
        f"<a href='/subset/{esc(subset)}?i={next_unlabeled}'>下一张未标注</a>",
    ]
    labels = " ".join(
        f"<button class='label' title='{esc(label)}' data-label='{esc(label)}' onclick='setLabel(this.dataset.label)'>{esc(display_label(label))}</button>"
        for label in spec["labels"]
    )
    conf = " ".join(
        f"<button class='conf' title='{esc(c)}' data-confidence='{esc(c)}' onclick='setConfidence(this.dataset.confidence)'>{esc(display_confidence(c))}</button>"
        for c in CONFIDENCE
    )
    meta_rows = "\n".join(f"<tr><th>{esc(META_DISPLAY.get(col, col))}</th><td>{esc(row.get(col, ''))}</td></tr>" for col in META_COLS)
    current_confidence = (row.get("human_confidence", "") if row.get("human_label", "").strip() else "") or "high"
    body = f"""
<nav>{" | ".join(nav)}</nav>
<h1>{esc(spec["title"])}</h1>
<p>{esc(spec["purpose"])}</p>
<p><strong>进度：</strong>已标注 {labeled} / {len(rows)}。<strong>当前：</strong>{index + 1} / {len(rows)}，crop_id={esc(row.get("crop_id", ""))}</p>
<div class="grid">
  <figure><img src="{image_url(row.get("image_path_full_tile", ""))}"><figcaption>整图叠加视图</figcaption></figure>
  <figure><img src="{image_url(row.get("image_path_zoom", ""))}"><figcaption>预测框局部放大</figcaption></figure>
  <figure><img src="{image_url(row.get("image_path_gt_context", ""))}"><figcaption>GT 上下文视图</figcaption></figure>
</div>
<section class="controls">
  <h2>人工分类</h2>
  <div>{labels}</div>
  <h2>置信度</h2>
  <div>{conf}</div>
  <h2>备注</h2>
  <textarea id="notes" rows="3">{esc(row.get("human_notes", ""))}</textarea>
  <p>
    <button onclick="saveCurrent()">保存当前选择</button>
    <button onclick="clearCurrent()">清空当前标签</button>
    <label><input type="checkbox" id="autoNext" checked> 点击分类后自动跳到下一张</label>
  </p>
  <p id="status"></p>
</section>
<table class="meta">{meta_rows}</table>
<script>
const subset = {json.dumps(subset)};
const cropId = {json.dumps(row.get("crop_id", ""))};
let currentLabel = {json.dumps(row.get("human_label", ""))};
let currentConfidence = {json.dumps(current_confidence)};
function refreshButtons() {{
  document.querySelectorAll('.label').forEach(b => b.classList.toggle('selected', b.dataset.label === currentLabel));
  document.querySelectorAll('.conf').forEach(b => b.classList.toggle('selected', b.dataset.confidence === currentConfidence));
}}
function setLabel(label) {{
  currentLabel = label;
  refreshButtons();
  saveCurrent();
}}
function setConfidence(confidence) {{
  currentConfidence = confidence;
  refreshButtons();
}}
async function saveCurrent() {{
  const payload = {{subset, crop_id: cropId, human_label: currentLabel, human_confidence: currentConfidence, human_notes: document.getElementById('notes').value}};
  const res = await fetch('/api/label', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
  const data = await res.json();
  document.getElementById('status').textContent = data.ok ? '已保存：' + currentLabel + ' / ' + currentConfidence : '错误：' + data.error;
  if (data.ok && document.getElementById('autoNext').checked && currentLabel) {{
    setTimeout(() => window.location.href = '/subset/{esc(subset)}?i={min(len(rows) - 1, index + 1)}', 250);
  }}
}}
async function clearCurrent() {{
  currentLabel = '';
  currentConfidence = '';
  document.getElementById('notes').value = '';
  refreshButtons();
  const res = await fetch('/api/clear', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{subset, crop_id: cropId}})}});
  const data = await res.json();
  document.getElementById('status').textContent = data.ok ? '已清空' : '错误：' + data.error;
}}
refreshButtons();
</script>
"""
    return page_shell(spec["title"], body)


def page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#172033;line-height:1.4}}
a{{color:#0f5ea8}} nav{{margin-bottom:12px}} .warn{{color:#9a3412;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{display:block;background:white;border:1px solid #d8dee4;border-radius:6px;padding:14px;text-decoration:none;color:#172033}}
.card span{{display:block;margin-top:8px;color:#475569}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px;align-items:start}}
figure{{margin:0;background:white;border:1px solid #d8dee4;padding:8px}} img{{width:100%;height:auto;display:block}}
figcaption{{font-size:13px;color:#475569;margin-top:4px}}
.controls{{background:white;border:1px solid #d8dee4;border-radius:6px;padding:12px;margin:14px 0}}
button{{margin:4px;padding:7px 10px;border:1px solid #94a3b8;border-radius:5px;background:#fff;cursor:pointer}}
button.selected{{background:#14532d;color:white;border-color:#14532d}}
button.label{{font-family:monospace}} textarea{{width:100%;box-sizing:border-box}}
.meta{{border-collapse:collapse;background:white;margin-top:12px}} .meta th,.meta td{{border:1px solid #d8dee4;padding:4px 7px;text-align:left;font-size:13px}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
<script>
async function runValidation(){{
  const el = document.getElementById('validation');
  if (el) el.textContent = '校验中...';
  const res = await fetch('/api/validate', {{method:'POST'}});
  const data = await res.json();
  if (el) el.textContent = data.ok ? JSON.stringify(data.result) : '错误：' + data.error;
}}
</script>
</head><body>{body}</body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: AuditState

    def send_text(self, text: str, status: int = 200, content_type: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: Any, status: int = 200) -> None:
        self.send_text(json.dumps(data, ensure_ascii=False), status=status, content_type="application/json; charset=utf-8")

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(render_index(self.state))
            return
        if parsed.path.startswith("/subset/"):
            subset = unquote(parsed.path.split("/", 2)[2])
            qs = parse_qs(parsed.query)
            index = int(qs.get("i", ["0"])[0])
            self.send_text(render_subset(self.state, subset, index))
            return
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            path = Path(unquote(qs.get("path", [""])[0]))
            if not path.exists() or not str(path).startswith(str(self.state.crop_root)):
                self.send_text("not found", status=404, content_type="text/plain; charset=utf-8")
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_text("not found", status=404, content_type="text/plain; charset=utf-8")

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            path = Path(unquote(qs.get("path", [""])[0]))
            if not path.exists() or not str(path).startswith(str(self.state.crop_root)):
                self.send_response(404)
                self.end_headers()
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            return
        self.send_response(200 if parsed.path in {"/"} or parsed.path.startswith("/subset/") else 404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/label":
                payload = self.read_json()
                result = self.state.save_label(
                    payload.get("subset", ""),
                    payload.get("crop_id", ""),
                    payload.get("human_label", ""),
                    payload.get("human_confidence", ""),
                    payload.get("human_notes", ""),
                )
                self.send_json(result)
                return
            if self.path == "/api/clear":
                payload = self.read_json()
                self.send_json(self.state.clear_label(payload.get("subset", ""), payload.get("crop_id", "")))
                return
            if self.path == "/api/validate":
                script = self.state.repo_root / "experiments/rotation_semantic_attractor/scripts/26_validate_tonight_manual_labels.py"
                cmd = [
                    sys.executable,
                    str(script),
                    "--label-csv",
                    str(self.state.label_csv),
                    "--output-dir",
                    str(self.state.work_dir),
                    "--manual-audit-dir",
                    str(self.state.work_dir),
                    "--allow-partial",
                ]
                proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                result = json.loads(proc.stdout.strip() or "{}") if proc.stdout.strip().startswith("{") else {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                }
                self.send_json({"ok": True, "result": result})
                return
            self.send_json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/data1/zcy/OpenRSD")
    parser.add_argument("--work-dir", default="/data1/zcy/OpenRSD/experiments/rotation_semantic_attractor/reports/visual_summary/audit/tonight_manual_audit_20260601")
    parser.add_argument("--label-csv", default="/data1/zcy/OpenRSD/experiments/rotation_semantic_attractor/reports/visual_summary/audit/human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    Handler.state = AuditState(Path(args.repo_root), Path(args.work_dir), Path(args.label_csv))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving clickable audit UI at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
