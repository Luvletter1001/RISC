#!/usr/bin/env python3
"""Overnight maintenance watchdog for rotation_sv_repair.

The watchdog is intentionally conservative about existing artifacts:

* never deletes outputs;
* never overwrites baseline lock files;
* backs up every file it may rewrite;
* records every round, exception, auto-fix, and rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PY = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
ERROR_PATTERNS = [
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "NaN",
    "KeyError",
    "AssertionError",
    "FAILED",
    "empty result",
    "no predictions",
    "missing gt",
    "missing tile",
    "process killed",
    "segmentation fault",
    "illegal memory access",
]

REQUIRED_REPORTS = {
    "baseline": ("fres_010_baseline_reconfirm.md", "ftable_010_baseline_reconfirm.csv"),
    "postprocess": ("fres_020_postprocess_repair.md", "ftable_020_postprocess_repair.csv"),
    "embedding": ("fres_030_embedding_calibration.md", "ftable_030_embedding_calibration.csv"),
    "prompt": ("fres_031_prompt_dynamic_vocabulary.md", "ftable_031_prompt_dynamic_vocabulary.csv"),
    "adapter": ("fres_040_rotation_adapter.md", "ftable_040_rotation_adapter.csv"),
    "orbit": ("fres_050_orbit_distillation.md", "ftable_050_orbit_distillation.csv"),
    "ovd_teacher_student": ("fres_060_ovd_teacher_student.md", "ftable_060_ovd_teacher_student.csv"),
    "official_ap": ("fres_070_official_ap_eval.md", "ftable_070_official_ap_eval.csv"),
    "true_sv": ("fres_071_true_sv_preservation.md", "ftable_071_true_sv_preservation.csv"),
    "oracle": ("fres_080_oracle_upper_bound.md", "ftable_080_oracle_upper_bound.csv"),
    "stats": ("fres_091_statistical_confidence.md", "ftable_091_statistical_confidence.csv"),
    "audit": ("fres_092_human_audit_package.md", ""),
    "ablation": ("fres_093_ablation_matrix.md", "ftable_093_ablation_matrix.csv"),
    "queue": ("fres_099_after_baseline_full_queue.md", ""),
    "verdict": ("fres_090_final_verdict.md", ""),
}

P0_TABLES = [
    "ftable_020_postprocess_repair.csv",
    "ftable_030_embedding_calibration.csv",
    "ftable_031_prompt_dynamic_vocabulary.csv",
    "ftable_040_rotation_adapter.csv",
    "ftable_050_orbit_distillation.csv",
    "ftable_060_ovd_teacher_student.csv",
    "ftable_070_official_ap_eval.csv",
    "ftable_071_true_sv_preservation.csv",
    "ftable_080_oracle_upper_bound.csv",
    "ftable_091_statistical_confidence.csv",
    "ftable_093_ablation_matrix.csv",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tail(text: str, n: int = 200) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def scalar(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def status_from_md(text: str) -> str:
    m = re.search(r"\*\*status:\*\*\s*`?([A-Z0-9_/-]+)`?", text)
    if m:
        return m.group(1)
    for s in ("DONE", "PARTIAL", "FAILED", "BLOCKED", "PROMISING", "BAD"):
        if s in text:
            return s
    return "MISSING_STATUS"


@dataclass
class Watchdog:
    repo_root: Path
    work_dir: Path
    result_dir: Path
    monitor_dir: Path
    gpu_ids: str
    interval_minutes: float
    max_hours: float
    mode: str
    resume: bool
    auto_fix: bool
    auto_rerun: bool
    no_delete: bool
    respect_baseline_lock: bool
    round_index: int = 0
    started_at: str = field(default_factory=now)
    errors_found: list[dict[str, Any]] = field(default_factory=list)
    errors_fixed: list[dict[str, Any]] = field(default_factory=list)
    reruns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tables_dir(self) -> Path:
        return self.result_dir / "tables"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def baseline_lock_dir(self) -> Path:
        return self.work_dir / "baseline_lock"

    def env(self, gpus: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "MPLCONFIGDIR": "/tmp/mplconfig",
                "PYTHONPATH": f"{self.repo_root}:{self.repo_root / 'tools'}",
                "CUDA_VISIBLE_DEVICES": gpus or self.gpu_ids,
                "NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
            }
        )
        return env

    def ensure_dirs(self) -> None:
        for path in (self.monitor_dir, self.logs_dir, self.tables_dir):
            path.mkdir(parents=True, exist_ok=True)

    def backup_file(self, path: Path) -> str:
        if not path.exists():
            return ""
        backup_dir = self.monitor_dir / f"backup_before_overnight_{stamp()}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        dst = backup_dir / path.name
        shutil.copy2(path, dst)
        return str(dst)

    def run_cmd(self, cmd: list[str], log: Path, timeout: int = 7200, gpus: str | None = None) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write(f"# started {now()}\n")
            f.write("$ " + " ".join(cmd) + "\n\n")
            f.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.repo_root),
                    env=self.env(gpus),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                f.write(f"\n# exit {proc.returncode} at {now()}\n")
                return int(proc.returncode)
            except subprocess.TimeoutExpired:
                f.write(f"\n# TIMEOUT after {timeout}s at {now()}\n")
                return 124
            except Exception:
                f.write("\n# EXCEPTION\n")
                f.write(traceback.format_exc())
                return 125

    def suite_cmd(self, exp: str, mode: str = "full", gpus: str | None = None, debug_small: bool = False) -> list[str]:
        cmd = [
            str(PY),
            "M_Tools/rotation_sv_repair/run_rotation_sv_repair_suite.py",
            "--repo-root",
            str(self.repo_root),
            "--work-dir",
            str(self.work_dir),
            "--result-dir",
            str(self.result_dir),
            "--gpu-ids",
            gpus or self.gpu_ids,
            "--mode",
            mode,
            "--exp",
            exp,
            "--resume",
            "--force",
        ]
        if exp in {"adapter", "orbit", "ovd_teacher_student"}:
            cmd.extend(["--train-adapter", "--distill"])
        if debug_small:
            cmd.extend(["--debug-small", "--max-images", "32"])
        return cmd

    def query_gpus(self) -> list[dict[str, Any]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, timeout=20)
        except Exception as exc:
            return [{"error": str(exc)}]
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_used_mb": parts[2],
                        "memory_total_mb": parts[3],
                        "utilization_gpu_pct": parts[4],
                    }
                )
        return rows

    def pick_gpus(self) -> str:
        rows = self.query_gpus()
        parsed = {}
        for row in rows:
            try:
                parsed[int(row["index"])] = (int(row["memory_used_mb"]), int(row["utilization_gpu_pct"]))
            except Exception:
                pass
        for pair in ((8, 9), (6, 7), (4, 5)):
            if all(i in parsed and parsed[i][0] < 1000 and parsed[i][1] < 40 for i in pair):
                return f"{pair[0]},{pair[1]}"
        for idx in (8, 9, 6, 7, 4, 5):
            if idx in parsed and parsed[idx][0] < 1000:
                return str(idx)
        return self.gpu_ids

    def process_status(self) -> dict[str, Any]:
        cmds = {
            "repair_processes": ["pgrep", "-af", "rotation_sv_repair|run_rotation_sv_repair_suite|tools/train.py|tools/test.py"],
            "python_processes": ["pgrep", "-af", "python"],
            "tmux": ["tmux", "list-sessions"],
            "zombies": ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        }
        out: dict[str, Any] = {}
        for key, cmd in cmds.items():
            try:
                text = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
            except subprocess.CalledProcessError as exc:
                text = exc.output
            except Exception as exc:
                text = str(exc)
            if key == "zombies":
                text = "\n".join([ln for ln in text.splitlines() if " Z" in ln or "defunct" in ln.lower()])
            out[key] = text.strip()
        stale = []
        for log in sorted(self.logs_dir.glob("*.log"))[-100:]:
            try:
                age = time.time() - log.stat().st_mtime
                if age > 7200 and log.name.startswith(("after_baseline_full", "overnight_rerun")):
                    stale.append({"log": str(log), "age_sec": int(age), "size": log.stat().st_size})
            except Exception:
                pass
        out["stale_logs"] = stale
        return out

    def disk_status(self) -> dict[str, Any]:
        try:
            usage = shutil.disk_usage(self.result_dir)
            free_gb = round(usage.free / (1024**3), 2)
        except Exception:
            free_gb = -1
        writable = False
        probe = self.monitor_dir / ".write_probe"
        try:
            probe.write_text(now(), encoding="utf-8")
            writable = True
            if not self.no_delete:
                probe.unlink(missing_ok=True)
        except Exception:
            writable = False
        return {"free_gb": free_gb, "result_dir_writable": writable, "low_space": free_gb >= 0 and free_gb < 20}

    def check_required_files(self) -> list[dict[str, Any]]:
        issues = []
        for module, (md_name, csv_name) in REQUIRED_REPORTS.items():
            md = self.result_dir / md_name
            if not md.exists() or md.stat().st_size == 0:
                issues.append({"module": module, "type": "missing_or_empty_md", "path": str(md)})
            else:
                st = status_from_md(read_text(md))
                if st == "MISSING_STATUS":
                    issues.append({"module": module, "type": "md_missing_status", "path": str(md)})
            if csv_name:
                csv_path = self.tables_dir / csv_name
                rows = read_csv_rows(csv_path)
                if not csv_path.exists() or csv_path.stat().st_size == 0:
                    issues.append({"module": module, "type": "missing_or_empty_csv", "path": str(csv_path)})
                elif len(rows) == 0:
                    issues.append({"module": module, "type": "csv_header_only_or_unreadable", "path": str(csv_path)})
                elif len(rows) < 2 and module not in {"oracle"}:
                    issues.append({"module": module, "type": "small_csv", "rows": len(rows), "path": str(csv_path)})
        return issues

    def baseline_integrity(self) -> dict[str, Any]:
        csv_path = self.tables_dir / "ftable_010_baseline_reconfirm.csv"
        lock_csv = self.baseline_lock_dir / "ftable_010_baseline_reconfirm.csv"
        rows = read_csv_rows(csv_path)
        lock_rows = read_csv_rows(lock_csv)
        p0148_ok = [
            r for r in rows
            if "P0148" in r.get("tile_id", "") and r.get("status") == "OK"
        ]
        status_counter: dict[str, int] = {}
        for row in rows:
            status_counter[row.get("status", "")] = status_counter.get(row.get("status", ""), 0) + 1
        ok = bool(rows) and len(rows) == 452 and len(p0148_ok) == 12 and bool(lock_rows)
        return {
            "ok": ok,
            "baseline_rows": len(rows),
            "lock_rows": len(lock_rows),
            "p0148_ok_angles": len(p0148_ok),
            "status_counter": status_counter,
            "baseline_csv": str(csv_path),
            "lock_csv": str(lock_csv),
        }

    def parse_queue_status(self) -> dict[str, Any]:
        path = self.work_dir / "after_baseline_full_queue_status.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": str(path)}
        bad = [r for r in data if int(r.get("full_exit", 1)) != 0]
        return {"ok": not bad, "path": str(path), "rows": len(data), "nonzero": bad}

    def evidence_audit(self) -> dict[str, Any]:
        audit: dict[str, Any] = {"tables": [], "proxy_tables": [], "missing_tables": [], "best_method_checks": []}
        for table in P0_TABLES:
            path = self.tables_dir / table
            rows = read_csv_rows(path)
            text = read_text(path)
            item = {"table": table, "path": str(path), "rows": len(rows), "exists": path.exists()}
            data_sources = sorted({r.get("data_source", "") for r in rows if r.get("data_source", "")})
            statuses = sorted({r.get("status", "") for r in rows if r.get("status", "")})
            verdicts = sorted({r.get("verdict", "") for r in rows if r.get("verdict", "")})
            item.update({"data_sources": data_sources, "statuses": statuses, "verdicts": verdicts})
            if not path.exists() or not rows:
                audit["missing_tables"].append(item)
            if "PROXY" in text or "PARTIAL_PROXY" in text or "PROXY_NOT_OFFICIAL" in text:
                audit["proxy_tables"].append(item)
            audit["tables"].append(item)
        verdict = read_text(self.result_dir / "fres_090_final_verdict.md")
        status_match = re.search(r"\*\*status:\*\*\s*`?([A-Z0-9_/-]+)`?", verdict)
        declared_status = status_match.group(1) if status_match else ""
        good_claim = declared_status == "GOOD" or bool(re.search(r"\|[^\n]*\|[^\n]*\|\s*GOOD\s*\|", verdict))
        if good_claim and ("PROXY" in verdict or "LOW_N" in verdict):
            audit["best_method_checks"].append(
                {
                    "type": "VERDICT_INCONSISTENT",
                    "message": "Final verdict declares GOOD while proxy/LOW_N evidence remains.",
                }
            )
        if "PROMISING_PROXY_ONLY" in verdict or "proxy" in verdict.lower():
            audit["best_method_checks"].append(
                {"type": "PROXY_EVIDENCE", "message": "Best method remains proxy-supported, not paper-main-table ready."}
            )
        return audit

    def scan_logs(self) -> list[dict[str, Any]]:
        hits = []
        logs = sorted(self.logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)[-80:]
        for log in logs:
            text = read_text(log)
            for pat in ERROR_PATTERNS:
                if pat.lower() in text.lower():
                    hits.append(
                        {
                            "log": str(log),
                            "pattern": pat,
                            "tail": tail(text, 40),
                        }
                    )
                    break
        return hits

    def rebuild_md_from_csv(self, module: str, md: Path, csv_path: Path, reason: str) -> None:
        rows = read_csv_rows(csv_path)
        backup = self.backup_file(md)
        with open(md, "w", encoding="utf-8") as f:
            f.write(f"# Rebuilt {module} Report\n\n")
            f.write("- **status:** `PARTIAL`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write("- **repair_action:** `REBUILT_FROM_CSV`\n")
            f.write(f"- **reason:** `{reason}`\n")
            f.write(f"- **csv:** `{csv_path}`\n")
            f.write(f"- **rows:** {len(rows)}\n")
            if backup:
                f.write(f"- **backup:** `{backup}`\n")
            f.write("\nThis report was rebuilt by overnight_watchdog from the available CSV table.\n")
        self.errors_fixed.append({"module": module, "action": "REBUILT_FROM_CSV", "md": str(md), "csv": str(csv_path)})

    def maybe_rerun_exp(self, module: str, reason: str, round_no: int, mode: str = "full") -> dict[str, Any]:
        if not self.auto_rerun:
            return {"module": module, "action": "rerun_skipped_disabled", "reason": reason}
        if module == "baseline" and self.respect_baseline_lock:
            return {"module": module, "action": "rerun_skipped_baseline_lock", "reason": reason}
        gpus = self.pick_gpus()
        log = self.logs_dir / f"overnight_rerun_round_{round_no:03d}_{module}_{mode}.log"
        for name in REQUIRED_REPORTS.get(module, ("", "")):
            if name:
                p = self.result_dir / name if name.endswith(".md") else self.tables_dir / name
                if p.exists():
                    self.backup_file(p)
        rc = self.run_cmd(self.suite_cmd(module, mode=mode, gpus=gpus), log, timeout=7200, gpus=gpus)
        row = {"module": module, "mode": mode, "exit": rc, "reason": reason, "log": str(log), "gpus": gpus}
        self.reruns.append(row)
        if rc != 0 and mode == "full":
            dbg_log = self.logs_dir / f"overnight_rerun_round_{round_no:03d}_{module}_debug.log"
            dbg_gpu = gpus.split(",")[0]
            dbg = self.run_cmd(
                self.suite_cmd(module, mode="debug", gpus=dbg_gpu, debug_small=True),
                dbg_log,
                timeout=1800,
                gpus=dbg_gpu,
            )
            row["debug_exit"] = dbg
            row["debug_log"] = str(dbg_log)
        return row

    def auto_fix_issues(self, issues: list[dict[str, Any]], evidence: dict[str, Any], round_no: int) -> list[dict[str, Any]]:
        actions = []
        if not self.auto_fix:
            return actions
        for issue in issues:
            module = issue["module"]
            if issue["type"] == "missing_or_empty_md":
                md_name, csv_name = REQUIRED_REPORTS[module]
                csv_path = self.tables_dir / csv_name if csv_name else Path("")
                if csv_name and csv_path.exists() and len(read_csv_rows(csv_path)) > 0:
                    self.rebuild_md_from_csv(module, self.result_dir / md_name, csv_path, issue["type"])
                    actions.append({"module": module, "action": "REBUILT_FROM_CSV"})
                elif module not in {"queue", "verdict"}:
                    actions.append(self.maybe_rerun_exp(module, issue["type"], round_no))
            elif issue["type"] in {"missing_or_empty_csv", "csv_header_only_or_unreadable", "small_csv"}:
                if module not in {"baseline"}:
                    actions.append(self.maybe_rerun_exp(module, issue["type"], round_no))
        for table in evidence.get("missing_tables", []):
            module = self.module_from_table(table["table"])
            if module and module != "baseline":
                actions.append(self.maybe_rerun_exp(module, "missing_p0_table", round_no))
        return actions

    def module_from_table(self, table: str) -> str:
        for module, (_, csv_name) in REQUIRED_REPORTS.items():
            if csv_name == table:
                return module
        return ""

    def write_round_report(
        self,
        round_no: int,
        start: str,
        end: str,
        proc: dict[str, Any],
        gpu: list[dict[str, Any]],
        disk: dict[str, Any],
        file_issues: list[dict[str, Any]],
        baseline: dict[str, Any],
        queue: dict[str, Any],
        evidence: dict[str, Any],
        log_hits: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        exception: str = "",
    ) -> None:
        path = self.monitor_dir / f"fres_watchdog_round_{round_no:03d}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Watchdog Round {round_no:03d}\n\n")
            f.write("WATCHDOG_ROUND_START\n\n")
            f.write(f"- **start:** {start}\n")
            f.write(f"- **end:** {end}\n")
            f.write(f"- **mode:** `{self.mode}`\n")
            f.write(f"- **gpu_ids:** `{self.gpu_ids}`\n")
            f.write(f"- **status:** `{'WATCHDOG_ROUND_EXCEPTION' if exception else 'WATCHDOG_ROUND_DONE'}`\n")
            if exception:
                f.write(f"\n```text\n{exception[-4000:]}\n```\n")
            f.write("\n## Process Check\n\n")
            f.write(f"- stale_logs: {len(proc.get('stale_logs', []))}\n")
            f.write(f"- tmux:\n```text\n{str(proc.get('tmux', ''))[:3000]}\n```\n")
            f.write("\n## GPU Check\n\n```json\n")
            f.write(json.dumps(gpu, indent=2, ensure_ascii=False))
            f.write("\n```\n\n## Disk Check\n\n```json\n")
            f.write(json.dumps(disk, indent=2, ensure_ascii=False))
            f.write("\n```\n\n## File Integrity Issues\n\n")
            if file_issues:
                for issue in file_issues:
                    f.write(f"- `{issue.get('type')}` {issue.get('module')} `{issue.get('path', '')}` rows={issue.get('rows', '')}\n")
            else:
                f.write("- CHECKED_OK\n")
            f.write("\n## Baseline Lock Integrity\n\n```json\n")
            f.write(json.dumps(baseline, indent=2, ensure_ascii=False))
            f.write("\n```\n\n## Queue Status\n\n```json\n")
            f.write(json.dumps(queue, indent=2, ensure_ascii=False))
            f.write("\n```\n\n## Evidence Strength\n\n")
            f.write(f"- p0_tables: {len(evidence.get('tables', []))}\n")
            f.write(f"- proxy_tables: {len(evidence.get('proxy_tables', []))}\n")
            f.write(f"- missing_tables: {len(evidence.get('missing_tables', []))}\n")
            for item in evidence.get("best_method_checks", []):
                f.write(f"- `{item['type']}`: {item['message']}\n")
            f.write("\n## Log Error Scan\n\n")
            f.write(f"- hits: {len(log_hits)}\n")
            for hit in log_hits[:12]:
                f.write(f"- `{hit['pattern']}` in `{hit['log']}`\n")
            f.write("\n## Auto Actions\n\n")
            if actions:
                for action in actions:
                    f.write(f"- `{action.get('action', 'rerun')}` {action.get('module')} exit={action.get('exit', '')} log=`{action.get('log', '')}`\n")
            else:
                f.write("- no action needed or action deferred\n")
            f.write("\nWATCHDOG_ROUND_DONE\n")
        if log_hits:
            self.write_error_scan(round_no, log_hits)

    def write_error_scan(self, round_no: int, hits: list[dict[str, Any]]) -> None:
        path = self.monitor_dir / f"error_scan_round_{round_no:03d}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Error Scan Round {round_no:03d}\n\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- **hits:** {len(hits)}\n\n")
            for hit in hits:
                f.write(f"## {hit['pattern']} in `{hit['log']}`\n\n")
                f.write("```text\n")
                f.write(hit["tail"][-8000:])
                f.write("\n```\n\n")

    def write_p0_reports(self, baseline: dict[str, Any], evidence: dict[str, Any]) -> None:
        self.write_p0_baseline(baseline)
        self.write_p0_verdict(evidence)
        self.write_p0_official_ap(evidence)
        self.write_p0_true_sv(evidence)

    def write_p0_baseline(self, baseline: dict[str, Any]) -> None:
        path = self.monitor_dir / "fres_p0_baseline_lock_integrity.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# P0 Baseline Lock Integrity\n\n")
            f.write(f"- **status:** `{'DONE' if baseline.get('ok') else 'FAILED'}`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- baseline rows: `{baseline.get('baseline_rows')}`\n")
            f.write(f"- lock rows: `{baseline.get('lock_rows')}`\n")
            f.write(f"- P0148 OK angles: `{baseline.get('p0148_ok_angles')}`\n")
            f.write("\n```json\n")
            f.write(json.dumps(baseline, indent=2, ensure_ascii=False))
            f.write("\n```\n")

    def write_p0_verdict(self, evidence: dict[str, Any]) -> None:
        path = self.monitor_dir / "fres_p0_final_verdict_consistency_audit.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# P0 Final Verdict Consistency Audit\n\n")
            proxy_count = len(evidence.get("proxy_tables", []))
            missing_count = len(evidence.get("missing_tables", []))
            checks = evidence.get("best_method_checks", [])
            status = "PARTIAL" if proxy_count or missing_count or checks else "DONE"
            f.write(f"- **status:** `{status}`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- proxy tables: `{proxy_count}`\n")
            f.write(f"- missing tables: `{missing_count}`\n")
            f.write("\n| table | rows | data_sources | statuses | verdicts |\n")
            f.write("|:---|---:|:---|:---|:---|\n")
            for item in evidence.get("tables", []):
                f.write(
                    f"| {item['table']} | {item['rows']} | `{item.get('data_sources')}` | "
                    f"`{item.get('statuses')}` | `{item.get('verdicts')}` |\n"
                )
            if checks:
                f.write("\n## Issues\n\n")
                for check in checks:
                    f.write(f"- `{check['type']}`: {check['message']}\n")

    def write_p0_official_ap(self, evidence: dict[str, Any]) -> None:
        rows = read_csv_rows(self.tables_dir / "ftable_070_official_ap_eval.csv")
        statuses = sorted({r.get("status", "") for r in rows})
        proxy = any("PROXY" in (r.get("status", "") + r.get("verdict", "") + r.get("data_source", "")) for r in rows)
        path = self.monitor_dir / "fres_p0_official_ap_completion.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# P0 Official AP Completion\n\n")
            f.write(f"- **status:** `{'PARTIAL_PROXY' if proxy else 'DONE' if rows else 'FAILED_AP_EVAL'}`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- rows: `{len(rows)}`\n")
            f.write(f"- statuses: `{statuses}`\n")
            f.write("- conclusion: official AP remains proxy unless the table status is not PROXY_AP.\n")

    def write_p0_true_sv(self, evidence: dict[str, Any]) -> None:
        rows = read_csv_rows(self.tables_dir / "ftable_071_true_sv_preservation.csv")
        required = {"true_sv_recall_before", "true_sv_recall_after", "true_sv_preserve_rate", "hard_negative_suppression_rate"}
        missing_fields = sorted(required - set(rows[0].keys() if rows else []))
        proxy = any("PROXY" in str(r) for r in rows)
        path = self.monitor_dir / "fres_p0_true_sv_preservation_completion.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# P0 True SV Preservation Completion\n\n")
            f.write(f"- **status:** `{'PARTIAL_PROXY' if proxy or missing_fields else 'DONE' if rows else 'FAILED'}`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- rows: `{len(rows)}`\n")
            f.write(f"- missing_fields: `{missing_fields}`\n")
            f.write(f"- proxy: `{proxy}`\n")

    def write_p1_reports(self) -> None:
        self.write_simple_deepening("fres_p1_ovd_teacher_student_deepening.md", "ftable_060_ovd_teacher_student.csv", "OVD Teacher-Student Deepening")
        self.write_simple_deepening("fres_p1_adapter_training_maintenance.md", "ftable_040_rotation_adapter.csv", "Adapter Training Maintenance")
        self.write_simple_deepening("fres_p1_prompt_deepening.md", "ftable_031_prompt_dynamic_vocabulary.csv", "Prompt Deepening")
        self.write_simple_deepening("fres_p1_oracle_completion.md", "ftable_080_oracle_upper_bound.csv", "Oracle Completion")

    def write_simple_deepening(self, name: str, table: str, title: str) -> None:
        rows = read_csv_rows(self.tables_dir / table)
        proxy = any("PROXY" in str(r) or "PARTIAL" in str(r) for r in rows)
        path = self.monitor_dir / name
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# P1 {title}\n\n")
            f.write(f"- **status:** `{'PARTIAL_PROXY' if proxy else 'DONE' if rows else 'FAILED'}`\n")
            f.write(f"- **generated:** {now()}\n")
            f.write(f"- source_table: `{self.tables_dir / table}`\n")
            f.write(f"- rows: `{len(rows)}`\n")
            if proxy:
                f.write("- remaining_gap: table contains proxy/PARTIAL evidence; full scientific support needs real evaluator or hook-derived evidence.\n")
            if rows:
                f.write("\n| method | verdict | sv_ratio_drop | dense_sv_drop | true_sv_preserve | heldout_drift |\n")
                f.write("|:---|:---|---:|---:|---:|---:|\n")
                for r in rows[:12]:
                    f.write(
                        f"| {r.get('method') or r.get('oracle_type') or r.get('prompt_family')} | {r.get('verdict', '')} | "
                        f"{r.get('sv_ratio_drop', '')} | {r.get('dense_sv_drop', '')} | "
                        f"{r.get('true_sv_preserve') or r.get('true_sv_preserve_rate') or r.get('true_sv_preserve_upper_bound', '')} | "
                        f"{r.get('heldout_drift') or r.get('final_test_drift', '')} |\n"
                    )

    def method_summary(self) -> list[dict[str, Any]]:
        sources = [
            ("postprocess", self.tables_dir / "ftable_020_postprocess_repair.csv"),
            ("embedding", self.tables_dir / "ftable_030_embedding_calibration.csv"),
            ("prompt", self.tables_dir / "ftable_031_prompt_dynamic_vocabulary.csv"),
            ("adapter", self.tables_dir / "ftable_040_rotation_adapter.csv"),
            ("orbit", self.tables_dir / "ftable_050_orbit_distillation.csv"),
            ("ovd_teacher_student", self.tables_dir / "ftable_060_ovd_teacher_student.csv"),
            ("ablation", self.tables_dir / "ftable_093_ablation_matrix.csv"),
        ]
        out = []
        for family, path in sources:
            rows = read_csv_rows(path)
            if not rows:
                continue
            def score(row: dict[str, str]) -> tuple[float, float, float]:
                return (
                    scalar(row, "sv_ratio_drop"),
                    scalar(row, "dense_sv_drop"),
                    scalar(row, "true_sv_preserve") or scalar(row, "true_sv_preserve_rate"),
                )
            best = sorted(rows, key=score, reverse=True)[0]
            verdict = best.get("verdict", "")
            proxy = "PROXY" in str(best) or "PARTIAL" in str(best)
            final_test_ok = "no" if proxy else "yes"
            out.append(
                {
                    "method": best.get("method") or best.get("prompt_family") or family,
                    "family": family,
                    "sv_fp_drop": best.get("sv_ratio_drop", best.get("hub_sv_suppression", "")),
                    "dense_sv_drop": best.get("dense_sv_drop", ""),
                    "court_recovery": max(
                        scalar(best, "tennis_flip_drop"),
                        scalar(best, "tennis_flip"),
                        scalar(best, "court_recovery_upper_bound"),
                    ),
                    "true_sv_preserve": best.get("true_sv_preserve", best.get("true_sv_preserve_rate", "")),
                    "mAP50_delta": best.get("mAP50_delta", best.get("heldout_ap_proxy_delta", "")),
                    "final_test_ok": final_test_ok,
                    "verdict": "PROMISING" if proxy and "BAD" not in verdict and "RISK" not in verdict else verdict,
                }
            )
        return out

    def write_final_report(self, completed: bool = False) -> None:
        path = self.monitor_dir / "fres_overnight_maintenance_final_report.md"
        timeline = self.monitor_dir / "watchdog_timeline.csv"
        methods = self.method_summary()
        enough = "PARTIAL"
        reason = "best methods remain proxy/PARTIAL in AP, true-SV, or statistical evidence tables."
        if any(r.get("final_test_ok") == "yes" and r.get("verdict") == "GOOD" for r in methods):
            enough = "YES"
            reason = "at least one method has non-proxy GOOD evidence."
        elif not methods:
            enough = "NO"
            reason = "method tables are missing or unreadable."
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Overnight Maintenance Final Report\n\n")
            f.write(f"- **status:** `{'DONE' if completed else 'RUNNING'}`\n")
            f.write(f"- **started_at:** {self.started_at}\n")
            f.write(f"- **updated_at:** {now()}\n")
            f.write(f"- **timeline:** `{timeline}`\n")
            f.write(f"- **errors_found:** {len(self.errors_found)}\n")
            f.write(f"- **errors_fixed:** {len(self.errors_fixed)}\n")
            f.write(f"- **reruns_launched:** {len(self.reruns)}\n")
            f.write("\n## Module Summary\n\n")
            f.write("| module | before_status | after_status | action_taken | evidence_level | remaining_risk |\n")
            f.write("|---|---|---|---|---|---|\n")
            for module, (md_name, csv_name) in REQUIRED_REPORTS.items():
                md = self.result_dir / md_name
                st = status_from_md(read_text(md)) if md.exists() else "MISSING"
                rows = read_csv_rows(self.tables_dir / csv_name) if csv_name else []
                proxy = "PROXY" in read_text(md) or any("PROXY" in str(r) for r in rows)
                f.write(
                    f"| {module} | checked | {st} | watchdog_audit | "
                    f"{'proxy' if proxy else 'table-backed' if rows or md.exists() else 'missing'} | "
                    f"{'needs real evaluator/hook evidence' if proxy else 'low' if md.exists() else 'missing artifact'} |\n"
                )
            f.write("\n## Method Summary\n\n")
            f.write("| method | sv_fp_drop | dense_sv_drop | court_recovery | true_sv_preserve | mAP50_delta | final_test_ok | verdict |\n")
            f.write("|---|---:|---:|---:|---:|---:|---|---|\n")
            for r in methods:
                f.write(
                    f"| {r['method']} | {r['sv_fp_drop']} | {r['dense_sv_drop']} | {r['court_recovery']} | "
                    f"{r['true_sv_preserve']} | {r['mAP50_delta']} | {r['final_test_ok']} | {r['verdict']} |\n"
                )
            f.write("\n## Morning Reading List\n\n")
            for rel in [
                "fres_p0_final_verdict_consistency_audit.md",
                "fres_p0_official_ap_completion.md",
                "fres_p0_true_sv_preservation_completion.md",
                "fres_watchdog_round_001.md",
                "../fres_090_final_verdict.md",
            ]:
                f.write(f"- `{self.monitor_dir / rel}`\n")
            f.write("\n## Paper Main Table Readiness\n\n")
            f.write(f"当前结果是否足以写论文主方法表？ **{enough}**\n\n")
            f.write(f"Reason: {reason}\n\n")
            f.write("## Next Prompt Suggestion\n\n")
            f.write("Ask Codex to replace proxy official AP / true-SV / hook-consensus evidence with real cached evaluator outputs before drafting the main paper table.\n")

    def update_final_verdict(self, evidence: dict[str, Any]) -> None:
        path = self.result_dir / "fres_090_final_verdict.md"
        if not path.exists():
            return
        backup = self.result_dir / f"fres_090_final_verdict.before_overnight_{stamp()}.md"
        shutil.copy2(path, backup)
        text = read_text(path)
        text = re.sub(r"- \*\*status:\*\* `GOOD`", "- **status:** `PARTIAL`", text)
        if "## Overnight maintenance and evidence audit" in text:
            text = text.split("## Overnight maintenance and evidence audit")[0].rstrip() + "\n\n"
        proxy_tables = [i["table"] for i in evidence.get("proxy_tables", [])]
        missing_tables = [i["table"] for i in evidence.get("missing_tables", [])]
        section = [
            "## Overnight maintenance and evidence audit",
            "",
            f"- **updated_at:** {now()}",
            f"- **watchdog_report:** `{self.monitor_dir / 'fres_overnight_maintenance_final_report.md'}`",
            f"- **verdict_backup:** `{backup}`",
            f"- **verified_p0_tables:** `{len(evidence.get('tables', []))}`",
            f"- **proxy_tables:** `{proxy_tables}`",
            f"- **missing_tables:** `{missing_tables}`",
            "- **automatic_repair:** watchdog rebuilt/rechecked artifacts where possible without modifying baseline lock.",
            "- **paper_main_table_status:** `PARTIAL` because official AP / true-SV / statistical evidence still contains proxy signals.",
            "- **appendix_ready:** postprocess, embedding, prompt, orbit, OVD, adapter proxy tables are appendix-ready with explicit limitations.",
            "- **tomorrow_priority:** replace proxy official AP and true-SV preservation with evaluator-derived tables.",
            "",
        ]
        path.write_text(text.rstrip() + "\n\n" + "\n".join(section), encoding="utf-8")

    def update_state(self, round_no: int, start: str, end: str, gpu: list[dict[str, Any]], disk: dict[str, Any], next_actions: list[str]) -> None:
        state = {
            "current_round": round_no,
            "started_at": self.started_at,
            "last_round_start": start,
            "last_round_end": end,
            "errors_found": len(self.errors_found),
            "errors_fixed": len(self.errors_fixed),
            "reruns_launched": len(self.reruns),
            "gpu_status": gpu,
            "disk_status": disk,
            "next_actions": next_actions,
        }
        (self.monitor_dir / "watchdog_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def append_timeline(self, row: dict[str, Any]) -> None:
        path = self.monitor_dir / "watchdog_timeline.csv"
        exists = path.exists()
        fields = [
            "round",
            "start",
            "end",
            "file_issues",
            "log_hits",
            "baseline_ok",
            "queue_ok",
            "proxy_tables",
            "actions",
            "exception",
        ]
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)

    def run_round(self, round_no: int) -> None:
        start = now()
        exception = ""
        proc: dict[str, Any] = {}
        gpu: list[dict[str, Any]] = []
        disk: dict[str, Any] = {}
        file_issues: list[dict[str, Any]] = []
        baseline: dict[str, Any] = {}
        queue: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        log_hits: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        try:
            proc = self.process_status()
            gpu = self.query_gpus()
            disk = self.disk_status()
            file_issues = self.check_required_files()
            baseline = self.baseline_integrity()
            queue = self.parse_queue_status()
            evidence = self.evidence_audit()
            log_hits = self.scan_logs()
            self.errors_found.extend(file_issues)
            self.errors_found.extend(log_hits)
            if disk.get("low_space"):
                actions.append({"action": "stop_new_training_low_disk", "free_gb": disk.get("free_gb")})
            else:
                actions.extend(self.auto_fix_issues(file_issues, evidence, round_no))
            self.write_p0_reports(baseline, evidence)
            self.write_p1_reports()
            self.update_final_verdict(evidence)
        except Exception:
            exception = traceback.format_exc()
        end = now()
        self.write_round_report(round_no, start, end, proc, gpu, disk, file_issues, baseline, queue, evidence, log_hits, actions, exception)
        self.append_timeline(
            {
                "round": round_no,
                "start": start,
                "end": end,
                "file_issues": len(file_issues),
                "log_hits": len(log_hits),
                "baseline_ok": baseline.get("ok", False),
                "queue_ok": queue.get("ok", False),
                "proxy_tables": len(evidence.get("proxy_tables", [])) if evidence else "",
                "actions": len(actions),
                "exception": "yes" if exception else "no",
            }
        )
        next_actions = []
        if file_issues:
            next_actions.append("continue artifact repair/rerun")
        if evidence.get("proxy_tables"):
            next_actions.append("proxy evidence remains; audit and targeted rerun only")
        if not next_actions:
            next_actions.append("continue periodic monitoring")
        self.update_state(round_no, start, end, gpu, disk, next_actions)
        self.write_final_report(completed=False)

    def loop(self) -> int:
        self.ensure_dirs()
        deadline = datetime.now() + timedelta(hours=self.max_hours)
        round_no = self.round_index + 1
        ran_once = False
        while (not ran_once) or datetime.now() <= deadline:
            try:
                self.run_round(round_no)
            except Exception:
                exc = traceback.format_exc()
                (self.monitor_dir / f"fres_watchdog_round_{round_no:03d}.md").write_text(
                    f"# Watchdog Round {round_no:03d}\n\nWATCHDOG_ROUND_EXCEPTION\n\n```text\n{exc}\n```\n",
                    encoding="utf-8",
                )
            ran_once = True
            round_no += 1
            if self.max_hours <= 0:
                break
            sleep_s = max(1, int(self.interval_minutes * 60))
            if datetime.now() + timedelta(seconds=sleep_s) > deadline:
                break
            time.sleep(sleep_s)
        self.write_final_report(completed=True)
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overnight watchdog for rotation_sv_repair")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--monitor-dir", required=True)
    p.add_argument("--gpu-ids", default="8,9")
    p.add_argument("--interval-minutes", type=float, default=30)
    p.add_argument("--max-hours", type=float, default=12)
    p.add_argument("--mode", choices=["maintain", "aggressive", "audit-only"], default="maintain")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--auto-fix", action="store_true")
    p.add_argument("--auto-rerun", action="store_true")
    p.add_argument("--no-delete", action="store_true")
    p.add_argument("--respect-baseline-lock", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if "openrsd" not in sys.executable and PY.exists() and os.environ.get("ROTATION_SV_WATCHDOG_REEXEC") != "1":
            os.environ["ROTATION_SV_WATCHDOG_REEXEC"] = "1"
            os.execv(str(PY), [str(PY)] + sys.argv)
        wd = Watchdog(
            repo_root=Path(args.repo_root).resolve(),
            work_dir=Path(args.work_dir).resolve(),
            result_dir=Path(args.result_dir).resolve(),
            monitor_dir=Path(args.monitor_dir).resolve(),
            gpu_ids=args.gpu_ids,
            interval_minutes=args.interval_minutes,
            max_hours=args.max_hours,
            mode=args.mode,
            resume=args.resume,
            auto_fix=args.auto_fix and args.mode != "audit-only",
            auto_rerun=args.auto_rerun and args.mode != "audit-only",
            no_delete=args.no_delete,
            respect_baseline_lock=args.respect_baseline_lock,
        )
        if args.resume:
            state_path = wd.monitor_dir / "watchdog_state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                wd.round_index = int(state.get("current_round", 0))
                wd.started_at = state.get("started_at", wd.started_at)
            except Exception:
                pass
        return wd.loop()
    except Exception:
        try:
            monitor = Path(args.monitor_dir)
            monitor.mkdir(parents=True, exist_ok=True)
            (monitor / "watchdog_top_level_exception.log").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
