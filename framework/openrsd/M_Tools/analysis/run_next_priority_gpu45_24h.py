#!/usr/bin/env python3
"""Next-priority OpenRSD GPU4/5 24h scheduler."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gpu45_task_runner import (  # noqa: E402
    GPU45,
    ProgressWriter,
    append_jsonl,
    command_string,
    gpu45_query,
    now,
    result_to_dict,
    run_child,
    run_quiet,
    run_ts,
    shell_join,
    write_json,
    write_text,
)


ANGLES12 = "000,030,060,090,120,150,180,210,240,270,300,330"
SCRIPT_DIR = Path("M_Tools/analysis")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def default_work_dir(repo_root: Path) -> tuple[str, Path]:
    ts = run_ts()
    return ts, repo_root / "work_dirs" / f"openrsd_next_priority_gpu45_{ts}"


def md_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "P0": args.result_md_dir / f"{args.run_ts}_P0_preflight_gpu45.md",
        "P1": args.result_md_dir / f"{args.run_ts}_P1_ovd4_full12_prompt_tta_gpu45.md",
        "P2": args.result_md_dir / f"{args.run_ts}_P2_ovd5_real_hook_adapter_gpu45.md",
        "P3": args.result_md_dir / f"{args.run_ts}_P3_zero_ap_drilldown_gpu45.md",
        "P4": args.result_md_dir / f"{args.run_ts}_P4_geometry_per_bin_ap_rerun_gpu45.md",
        "P5": args.result_md_dir / f"{args.run_ts}_P5_train_split_repair_and_training_probe_gpu45.md",
        "summary": args.result_md_dir / f"{args.run_ts}_summary_next_priority_gpu45_24h.md",
    }


def json_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {key: args.work_dir / f"{key}.json" for key in ["P0", "P1", "P2", "P3", "P4", "P5", "summary", "batchsize"]}


def wrapper_cmd(args: argparse.Namespace, script: str, mode: str, task_key: str, extra: list[str] | None = None) -> list[str]:
    paths = md_paths(args)
    jpaths = json_paths(args)
    cmd = [
        str(args.python_bin), str(SCRIPT_DIR / script),
        "--repo-root", str(args.repo_root),
        "--result-md-dir", str(args.result_md_dir),
        "--work-dir", str(args.work_dir),
        "--run-ts", args.run_ts,
        "--mode", mode,
        "--out-md", str(paths[task_key]),
        "--out-json", str(jpaths[task_key]),
    ]
    if script in {"run_ovd4_full12_prompt_tta.py", "run_ovd5_real_hook_adapter.py", "probe_dota1_train_split_and_training.py"}:
        cmd.extend(["--weights-dir", str(args.weights_dir), "--gpu-ids", args.gpu_ids])
    if script == "run_ovd4_full12_prompt_tta.py":
        cmd.extend([
            "--batch-size", str(args.batch_size or 1),
            "--num-workers", str(args.num_workers),
            "--max-images-for-smoke", str(args.max_images_for_smoke),
        ])
        if args.max_images_full is not None:
            cmd.extend(["--max-images-full", str(args.max_images_full)])
        if args.fallback_on_failure:
            cmd.append("--fallback-on-failure")
    if script == "run_ovd5_real_hook_adapter.py":
        p2_csv = args.work_dir / "P2_hooks/real_hook_diagnostic.csv"
        cmd.extend([
            "--max-images-for-smoke", str(args.max_images_for_smoke),
            "--max-images-full", str(args.max_images_full or args.max_images_for_smoke),
            "--max-live-images", str(args.max_images_full or args.max_images_for_smoke),
            "--out-csv", str(p2_csv),
        ])
    if script == "probe_dota1_train_split_and_training.py":
        cmd.extend(["--batch-size", str(args.batch_size or 1), "--max-iters", str(args.max_iters_for_smoke)])
    if extra:
        cmd.extend(extra)
    return cmd


def planned_commands(args: argparse.Namespace) -> dict[str, str]:
    planned = {
        "P1": wrapper_cmd(args, "run_ovd4_full12_prompt_tta.py", "full", "P1"),
        "P2": wrapper_cmd(args, "run_ovd5_real_hook_adapter.py", "full", "P2"),
        "P3": wrapper_cmd(args, "audit_zero_ap_drilldown.py", "full", "P3"),
        "P4": wrapper_cmd(args, "rerun_geometry_per_bin_ap_clean.py", "full", "P4"),
        "P5": wrapper_cmd(args, "probe_dota1_train_split_and_training.py", "full", "P5"),
    }
    return {key: command_string(cmd, args.repo_root, args.gpu_ids) for key, cmd in planned.items()}


def version_probe(args: argparse.Namespace) -> dict[str, Any]:
    code = (
        "import json, sys\n"
        "d={'python': sys.version.split()[0]}\n"
        "mods=['torch','mmcv','mmengine','mmdet','mmrotate']\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod=__import__(m); d[m]=getattr(mod,'__version__','UNKNOWN')\n"
        "    except Exception as e:\n"
        "        d[m]='NOT_AVAILABLE:'+repr(e)[:120]\n"
        "try:\n"
        "    import torch; d['cuda_available']=torch.cuda.is_available(); d['cuda']=torch.version.cuda\n"
        "except Exception: pass\n"
        "print(json.dumps(d, ensure_ascii=False))\n"
    )
    rc, out, err = run_quiet(["rtk", "env", f"PYTHONPATH={args.repo_root}:{args.repo_root / 'tools'}", str(args.python_bin), "-c", code], cwd=args.repo_root, timeout=60)
    if rc == 0:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    return {"error": err or out, "return_code": rc}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    git_rc, git_hash, _ = run_quiet(["rtk", "git", "rev-parse", "HEAD"], cwd=args.repo_root, timeout=20)
    git_rc2, git_status, _ = run_quiet(["rtk", "git", "status", "--short"], cwd=args.repo_root, timeout=20)
    angle_root = args.repo_root / "data/DOTA1_1024_500/angle_sweep_val/realistic"
    train_candidates = [
        args.repo_root / "data/DOTA1_1024_500/train",
        args.repo_root / "data/DOTA1_1024_500/trainval",
        args.repo_root / "data/DOTA1_1024_500/trainval_split",
        args.repo_root / "data/DOTA1_1024_500/trainval1024",
        args.repo_root / "data/DOTA1_1024_500/split_ss_train",
        args.repo_root / "data/DOTA1_1024_500/ss_train",
    ]
    config = args.repo_root / "M_configs/Step3_A12_SelfTrain/A12_flex_rtm_v3_1_DOTA2only_ss_train.py"
    checkpoint = args.weights_dir / "MMR_AD_A12_flex_rtm_v3_1_DOTA2only_ss_train_textcls00/epoch_12.pth"
    support = args.repo_root / "data/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl"
    required_scripts = [
        "run_next_priority_gpu45_24h.py",
        "gpu45_task_runner.py",
        "audit_zero_ap_drilldown.py",
        "run_ovd4_full12_prompt_tta.py",
        "run_ovd5_real_hook_adapter.py",
        "rerun_geometry_per_bin_ap_clean.py",
        "probe_dota1_train_split_and_training.py",
    ]
    checks = {
        "repo_root_exists": args.repo_root.exists(),
        "result_md_dir_writable": os.access(args.result_md_dir, os.W_OK),
        "weights_dir_exists": args.weights_dir.exists(),
        "work_dir_exists": args.work_dir.exists(),
        "config_exists": config.exists(),
        "checkpoint_exists": checkpoint.exists(),
        "ovd2_old_exists": (args.repo_root / "work_dirs/openrsd_ovd_rotation_20260508/exp_ovd2").exists(),
        "ovd4_old_exists": (args.repo_root / "work_dirs/openrsd_ovd_rotation_20260508/exp_ovd4").exists(),
        "support_pkl_exists": support.exists(),
        "dota_angle_sweep": {a: (angle_root / f"angle_{a}").exists() for a in ANGLES12.split(",")},
        "train_candidates": {str(p): p.exists() for p in train_candidates},
        "scripts": {name: (THIS_DIR / name).exists() for name in required_scripts},
    }
    gpu = gpu45_query()
    critical_ok = checks["repo_root_exists"] and checks["weights_dir_exists"] and checks["config_exists"] and checks["checkpoint_exists"]
    status = "DONE" if critical_ok else "FAILED"
    payload = {
        "status": status,
        "generated_at": now(),
        "run_ts": args.run_ts,
        "work_dir": str(args.work_dir),
        "git_commit": git_hash.strip() if git_rc == 0 else "NA",
        "git_status_short": git_status.strip() if git_rc2 == 0 else "NA",
        "versions": version_probe(args),
        "gpu45": gpu,
        "checks": checks,
        "planned_commands": planned_commands(args),
    }
    write_json(json_paths(args)["P0"], payload)
    lines = [
        "# P0 Preflight GPU45",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- status: `{status}`",
        f"- RUN_TS: `{args.run_ts}`",
        f"- work_dir: `{args.work_dir}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- config: `{config}` exists `{config.exists()}`",
        f"- checkpoint: `{checkpoint}` exists `{checkpoint.exists()}`",
        f"- support_pkl: `{support}` exists `{support.exists()}`",
        "",
        "## Environment Versions",
        "",
        "| package | version |",
        "|---|---|",
    ]
    for key, value in payload["versions"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## GPU4 / GPU5", "", "| gpu | name | mem used | mem total | util | processes |", "|---:|---|---:|---:|---:|---:|"])
    for idx in ("4", "5"):
        item = gpu.get(idx, {})
        lines.append(f"| {idx} | {item.get('name', 'NA')} | {item.get('memory_used_mb', 'NA')} | {item.get('memory_total_mb', 'NA')} | {item.get('utilization_gpu_pct', 'NA')} | {len(item.get('processes', []))} |")
    lines.extend(["", "## Checks", "", "| item | value |", "|---|---|"])
    for key, value in checks.items():
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)}` |")
    lines.extend(["", "## Planned Full Commands", ""])
    for key, cmd in payload["planned_commands"].items():
        lines.append(f"- {key}: `{cmd}`")
    write_text(md_paths(args)["P0"], "\n".join(lines))
    return payload


def run_wrapper(args: argparse.Namespace, progress: ProgressWriter, task_key: str, script: str, mode: str, gpu_ids: str = GPU45) -> dict[str, Any]:
    cmd = wrapper_cmd(args, script, mode, task_key)
    progress.set(mode, task_key, 0.0, command_string(cmd, args.repo_root, gpu_ids), next_task="")
    result = run_child(
        task_name=f"{task_key}_{mode}",
        argv=cmd,
        repo_root=args.repo_root,
        log_dir=args.work_dir / f"{task_key}_logs" / mode,
        gpu_ids=gpu_ids,
    )
    payload = result_to_dict(result)
    append_jsonl(args.work_dir / "commands.jsonl", payload)
    if result.return_code:
        append_jsonl(args.work_dir / "failures.jsonl", payload)
        progress.fail(f"{task_key}_{mode}: rc={result.return_code} {result.failure_kind}")
    task_json = json_paths(args)[task_key]
    report = load_json(task_json)
    if not report:
        report = {"status": "FAILED" if result.return_code else "DONE", "child_result": payload}
        write_json(task_json, report)
    progress.complete(f"{task_key}_{mode}")
    return report


def batchsize_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    selected = args.batch_size or 1
    for bs in [int(x) for x in args.batch_size_candidates.split(",") if x.strip()]:
        code = (
            "import json, time, torch\n"
            "ok=torch.cuda.is_available()\n"
            "info={'cuda_available':ok,'device_count':torch.cuda.device_count() if ok else 0}\n"
            "if ok:\n"
            "    torch.cuda.synchronize(); t=time.time(); x=torch.empty((256,256),device='cuda'); y=x+1; torch.cuda.synchronize(); info['elapsed']=time.time()-t; info['mem_allocated']=torch.cuda.memory_allocated()\n"
            "print(json.dumps(info))\n"
        )
        result = run_child(
            task_name=f"P0_batchsize_bs{bs}",
            argv=[str(args.python_bin), "-c", code],
            repo_root=args.repo_root,
            log_dir=args.work_dir / "P0_batchsize" / f"bs{bs}",
            gpu_ids=args.gpu_ids,
            timeout_sec=120,
        )
        payload = result_to_dict(result)
        append_jsonl(args.work_dir / "commands.jsonl", payload)
        status = "OK" if result.return_code == 0 else "FAILED"
        if status == "OK":
            selected = bs
        rows.append({"batch_size": bs, "status": status, "failure_kind": result.failure_kind, "stdout": result.stdout_path, "stderr": result.stderr_path})
    report = {
        "status": "DONE" if any(r["status"] == "OK" for r in rows) else "FAILED",
        "rows": rows,
        "selected_batch_size": selected,
        "note": "This is a CUDA availability probe; model-specific batch pressure is still recorded in each task log.",
    }
    write_json(json_paths(args)["batchsize"], report)
    return report


def run_smoke(args: argparse.Namespace, progress: ProgressWriter) -> dict[str, Any]:
    reports = {}
    reports["P1"] = run_wrapper(args, progress, "P1", "run_ovd4_full12_prompt_tta.py", "smoke", "4")
    reports["P2"] = run_wrapper(args, progress, "P2", "run_ovd5_real_hook_adapter.py", "smoke", "4")
    reports["P3"] = run_wrapper(args, progress, "P3", "audit_zero_ap_drilldown.py", "smoke", "4")
    reports["P4"] = run_wrapper(args, progress, "P4", "rerun_geometry_per_bin_ap_clean.py", "smoke", "4")
    reports["P5"] = run_wrapper(args, progress, "P5", "probe_dota1_train_split_and_training.py", "smoke", "4")
    ok = reports["P1"].get("status") in {"DONE", "PARTIAL"} and reports["P2"].get("status") in {"DONE", "PARTIAL"}
    payload = {"status": "DONE" if ok else "PARTIAL", "reports": {k: v.get("status") for k, v in reports.items()}}
    write_json(args.work_dir / "smoke_summary.json", payload)
    return payload


def run_full_tasks(args: argparse.Namespace, progress: ProgressWriter) -> dict[str, Any]:
    reports = {}
    task_queue = [
        {"task": "P1", "script": "run_ovd4_full12_prompt_tta.py", "gpu": args.gpu_ids},
        {"task": "P2", "script": "run_ovd5_real_hook_adapter.py", "gpu": args.gpu_ids},
        {"task": "P3", "script": "audit_zero_ap_drilldown.py", "gpu": "4"},
        {"task": "P4", "script": "rerun_geometry_per_bin_ap_clean.py", "gpu": "4"},
        {"task": "P5", "script": "probe_dota1_train_split_and_training.py", "gpu": args.gpu_ids},
    ]
    write_json(args.work_dir / "task_queue.json", task_queue)
    for item in task_queue:
        reports[item["task"]] = run_wrapper(args, progress, item["task"], item["script"], "full", item["gpu"])
        if reports[item["task"]].get("status") == "FAILED" and not args.fallback_on_failure:
            break
    return reports


def write_summary(args: argparse.Namespace, pre: dict[str, Any], smoke: dict[str, Any] | None, batch: dict[str, Any] | None, reports: dict[str, Any] | None, progress: ProgressWriter | None, status: str) -> None:
    reports = reports or {}
    paths = md_paths(args)
    elapsed = (time.time() - progress.start_time) if progress else 0.0
    payload = {
        "status": status,
        "run_ts": args.run_ts,
        "work_dir": str(args.work_dir),
        "git_commit": pre.get("git_commit"),
        "gpu45": pre.get("gpu45"),
        "preflight": pre.get("status"),
        "smoke": smoke,
        "batchsize": batch,
        "task_status": {key: load_json(json_paths(args)[key]).get("status", "NOT_RUN") for key in ["P1", "P2", "P3", "P4", "P5"]},
        "md_paths": {k: str(v) for k, v in paths.items()},
        "elapsed_hours": elapsed / 3600.0,
        "ran_over_24h": elapsed >= args.time_budget_hours * 3600,
        "commands_jsonl": str(args.work_dir / "commands.jsonl"),
        "failures_jsonl": str(args.work_dir / "failures.jsonl"),
        "gpu_status_jsonl": str(args.work_dir / "gpu_status.jsonl"),
    }
    write_json(json_paths(args)["summary"], payload)
    lines = [
        "# Summary Next Priority GPU45 24h",
        "",
        f"- RUN_TS: `{args.run_ts}`",
        f"- status: `{status}`",
        f"- work_dir: `{args.work_dir}`",
        f"- git_commit: `{pre.get('git_commit', 'NA')}`",
        f"- elapsed_hours: `{elapsed / 3600.0:.2f}`",
        f"- ran_over_24h: `{elapsed >= args.time_budget_hours * 3600}`",
        f"- commands_jsonl: `{args.work_dir / 'commands.jsonl'}`",
        f"- failures_jsonl: `{args.work_dir / 'failures.jsonl'}`",
        f"- gpu_status_jsonl: `{args.work_dir / 'gpu_status.jsonl'}`",
        "",
        "## Preflight",
        "",
        f"- dryrun/preflight: `{pre.get('status')}`",
        f"- smoke: `{smoke.get('status') if smoke else 'NOT_RUN'}`",
        f"- selected batch size: `{(batch or {}).get('selected_batch_size', args.batch_size or 1)}`",
        "",
        "## Task Status",
        "",
        "| task | status | md |",
        "|---|---|---|",
    ]
    for key in ["P1", "P2", "P3", "P4", "P5"]:
        task_status = payload["task_status"].get(key, "NOT_RUN")
        lines.append(f"| {key} | {task_status} | `{paths[key]}` |")
    lines.extend([
        "",
        "## GPU4/GPU5",
        "",
        "| gpu | name | mem used | mem total | util |",
        "|---:|---|---:|---:|---:|",
    ])
    for idx in ("4", "5"):
        item = (pre.get("gpu45") or {}).get(idx, {})
        lines.append(f"| {idx} | {item.get('name', 'NA')} | {item.get('memory_used_mb', 'NA')} | {item.get('memory_total_mb', 'NA')} | {item.get('utilization_gpu_pct', 'NA')} |")
    lines.extend([
        "",
        "## Key Results",
        "",
        f"- P1: `{payload['task_status'].get('P1', 'NOT_RUN')}`. See `{paths['P1']}`.",
        f"- P2: `{payload['task_status'].get('P2', 'NOT_RUN')}`. See `{paths['P2']}`.",
        f"- P3: `{payload['task_status'].get('P3', 'NOT_RUN')}`. See `{paths['P3']}`.",
        f"- P4: `{payload['task_status'].get('P4', 'NOT_RUN')}`. See `{paths['P4']}`.",
        f"- P5: `{payload['task_status'].get('P5', 'NOT_RUN')}`. See `{paths['P5']}`.",
        "",
        "## Paper-Ready / Not Ready",
        "",
        "- Paper-ready claims require DONE or clearly scoped PARTIAL task md evidence.",
        "- Claims involving Definition B joint-bin AP remain not ready unless P4 reports real joint-bin rows.",
        "- Training-effectiveness claims remain not ready unless P5 produces a checkpoint plus angle-wise eval.",
    ])
    write_text(paths["summary"], "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(args.work_dir, args.run_ts, args.time_budget_hours)
    progress.total = 10
    progress.start()
    try:
        progress.set("P0", "dryrun preflight", 0.0, next_task="smoke")
        pre = preflight(args)
        progress.complete("P0_dryrun", next_task="smoke")
        if args.mode == "dryrun":
            write_summary(args, pre, None, None, None, progress, pre.get("status", "FAILED"))
            return pre
        if pre.get("status") == "FAILED":
            write_summary(args, pre, None, None, None, progress, "FAILED")
            return pre
        progress.set("P0", "smoke", 0.2, next_task="batchsize")
        smoke = run_smoke(args, progress)
        progress.complete("P0_smoke", next_task="batchsize")
        if args.mode == "smoke":
            write_summary(args, pre, smoke, None, None, progress, smoke.get("status", "PARTIAL"))
            return smoke
        progress.set("P0", "batchsize probe", 0.4, next_task="P1")
        batch = batchsize_probe(args)
        args.batch_size = int(batch.get("selected_batch_size") or args.batch_size or 1)
        progress.complete("P0_batchsize", next_task="P1")
        reports = run_full_tasks(args, progress)
        progress.complete("primary_queue", next_task="extended_until_24h")
        deadline = progress.start_time + args.time_budget_hours * 3600.0
        while time.time() < deadline:
            remaining = deadline - time.time()
            progress.set("extended", "queue exhausted; heartbeat hold", 1.0, command="sleep", next_task="summary")
            time.sleep(min(300.0, max(1.0, remaining)))
        progress.complete("extended_24h_hold", next_task="summary")
        write_summary(args, pre, smoke, batch, reports, progress, "DONE")
        return {"status": "DONE", "reports": {k: v.get("status") for k, v in reports.items()}}
    finally:
        progress.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-priority OpenRSD GPU45 24h scheduler")
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--gpu-ids", default=GPU45)
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--time-budget-hours", type=float, default=24.0)
    parser.add_argument("--keep-gpus-busy", action="store_true")
    parser.add_argument("--fallback-on-failure", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-size-candidates", default="1,2,4,8,16,24,32")
    parser.add_argument("--target-vram-ratio", type=float, default=0.88)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images-for-smoke", type=int, default=16)
    parser.add_argument("--max-images-full", type=int, default=None)
    parser.add_argument("--max-iters-for-smoke", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    default_python = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--python-bin", type=Path, default=default_python if default_python.exists() else Path(sys.executable))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    if args.work_dir is None:
        args.run_ts, args.work_dir = default_work_dir(args.repo_root)
    else:
        args.work_dir = args.work_dir.resolve()
        suffix = args.work_dir.name.rsplit("_", 2)
        args.run_ts = "_".join(suffix[-2:]) if len(suffix) >= 2 and suffix[-2].isdigit() else run_ts()
    if args.gpu_ids != GPU45 and args.gpu_ids != "4":
        raise SystemExit("This scheduler only allows physical GPUs 4,5 or single GPU 4 for smoke.")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
