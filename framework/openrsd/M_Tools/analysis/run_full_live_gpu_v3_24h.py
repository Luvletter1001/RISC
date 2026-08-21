#!/usr/bin/env python3
"""Full-live GPU v3 scheduler for OpenRSD.

This runner is deliberately strict about GPU evidence.  It creates a fresh
work directory, writes progress files immediately, runs autopsy/preflight,
launches live CUDA inference/hook/training jobs, and records any fallback.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from strict_gpu_runner_v3 import (
    ProgressWriter,
    StrictGpuRunner,
    append_jsonl,
    query_gpu_rows,
    read_text,
    write_json,
    write_text,
)


ANGLES12 = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]
ANGLES_A = "000,030,060,090,120,150"
ANGLES_B = "180,210,240,270,300,330"


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pybin() -> Path:
    p = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
    return p if p.exists() else Path(sys.executable)


def parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def make_context(args: argparse.Namespace) -> dict[str, Any]:
    run_ts = args.run_ts or os.environ.get("RUN_TS") or now_ts()
    work_dir = args.work_dir or (args.repo_root / "work_dirs" / f"full_live_gpu_v3_{run_ts}")
    work_dir.mkdir(parents=True, exist_ok=True)
    args.result_md_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "G0": args.result_md_dir / f"{run_ts}_G0_failure_autopsy_and_gpu_preflight.md",
        "G1": args.result_md_dir / f"{run_ts}_G1_train_fix_and_full_dota1_finetune.md",
        "G2": args.result_md_dir / f"{run_ts}_G2_live_full12_openrsd_inference.md",
        "G3": args.result_md_dir / f"{run_ts}_G3_live_full12x12_ovd_tta.md",
        "G4": args.result_md_dir / f"{run_ts}_G4_large_scale_live_hook_diagnostic.md",
        "G5": args.result_md_dir / f"{run_ts}_G5_zero_ap_gpu_reactivation.md",
        "SUM": args.result_md_dir / f"{run_ts}_summary_full_live_gpu_v3.md",
    }
    for name, path in paths.items():
        if not path.exists():
            write_text(path, f"# {name}\n\n- status: `NOT_RUN`\n- RUN_TS: `{run_ts}`\n")
    task_queue = [
        "G1 training smoke",
        "G2 live inference alignment F3 all12",
        "G4 live hook alignment/fusion 512",
        "G2 live inference fusion F3 all12",
        "G3 live 12x12 TTA",
        "G5 prompt sweep live inference",
        "G1 full training if smoke succeeds",
        "G4 extended 2048",
        "G2 additional prompt families F1/F0",
        "G6 joint-bin AP background",
    ]
    write_json(work_dir / "task_queue.json", {"run_ts": run_ts, "queue": task_queue})
    for f in ["commands.jsonl", "failures.jsonl", "gpu_status.jsonl"]:
        (work_dir / f).touch(exist_ok=True)
    return {"run_ts": run_ts, "work_dir": work_dir, "md": paths}


def result_dict(result: Any) -> dict[str, Any]:
    return asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)


def run_preflight(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter) -> dict[str, Any]:
    progress.update(current_task="G0", current_subtask="failure autopsy and GPU preflight", next_task="smoke live inference")
    out_json = ctx["work_dir"] / "G0_failure_autopsy_and_gpu_preflight.json"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/failure_autopsy_last_run_v3.py",
        "--repo-root", str(args.repo_root),
        "--result-md-dir", str(args.result_md_dir),
        "--weights-dir", str(args.weights_dir),
        "--work-dir", str(ctx["work_dir"]),
        "--run-ts", ctx["run_ts"],
        "--gpu-ids", args.gpu_ids,
        "--out-md", str(ctx["md"]["G0"]),
        "--out-json", str(out_json),
    ]
    res = runner.run("G0_preflight", cmd, ctx["work_dir"] / "G0_logs", gpu_ids=args.gpu_ids, monitor_gpu=False)
    progress.completed += 1
    progress.update(current_task="G0", current_subtask="done")
    return {"result": result_dict(res), "json": str(out_json)}


def run_config_fix(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, max_iters: int, batch_size: int) -> dict[str, Any]:
    out_json = ctx["work_dir"] / "G1_train_configs" / f"fix_configs_{max_iters}.json"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/fix_train_config_from_last_error_v3.py",
        "--repo-root", str(args.repo_root),
        "--work-dir", str(ctx["work_dir"]),
        "--max-iters", str(max_iters),
        "--batch-size", str(batch_size),
        "--out-json", str(out_json),
    ]
    res = runner.run(f"G1_config_fix_{max_iters}", cmd, ctx["work_dir"] / "G1_logs" / f"config_fix_{max_iters}", gpu_ids=args.gpu_ids, monitor_gpu=False)
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    return {"runner": result_dict(res), "payload": payload}


def run_training_smoke(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter, max_iters: int = 5) -> dict[str, Any]:
    progress.update(current_task="G1", current_subtask=f"training smoke {max_iters} iter", current_angle="baseline_short_ft", batch_size=1, next_task="G2 live inference")
    cfg_info = run_config_fix(args, ctx, runner, max_iters=max_iters, batch_size=1)
    cfg = Path(cfg_info.get("payload", {}).get("configs", {}).get("baseline_short_ft", ""))
    train_work = ctx["work_dir"] / "G1_training" / f"baseline_short_ft_smoke_{max_iters}"
    cmd = [str(pybin()), "tools/train.py", str(cfg), "--work-dir", str(train_work)]
    res = runner.run("G1_train_smoke_baseline", cmd, ctx["work_dir"] / "G1_logs" / "train_smoke_baseline", gpu_ids=args.gpu_ids, batch_size=1, timeout_sec=3600)
    entered_loop = "Epoch(" in read_text(Path(res.stdout_path)) or "loss" in read_text(Path(res.stdout_path)) or (train_work / "epoch_1.pth").exists()
    status = "DONE_LIVE_GPU" if res.return_code == 0 and entered_loop and max(res.peak_mem_mb.values() or [0]) >= 2000 else "FAILED_TRAIN_BUT_FALLBACK_STARTED"
    payload = {"status": status, "config": str(cfg), "train_work": str(train_work), "entered_train_loop": entered_loop, "result": result_dict(res)}
    write_json(ctx["work_dir"] / "G1_train_smoke.json", payload)
    lines = [
        "# G1 Train Fix And Full DOTA1 Finetune",
        "",
        f"- status: `{status}`",
        f"- train_root: `{args.repo_root / 'data/DOTA1_1024_500/ss_train'}`",
        f"- config: `{cfg}`",
        f"- entered_train_loop: `{entered_loop}`",
        f"- peak_vram: `{res.peak_mem_mb}`",
        f"- stdout: `{res.stdout_path}`",
        f"- stderr: `{res.stderr_path}`",
        f"- fallback: `G2/G4 continue regardless of train status`",
    ]
    write_text(ctx["md"]["G1"], "\n".join(lines))
    progress.completed += 1
    if status.startswith("FAILED"):
        progress.last_failure = f"G1 train smoke rc={res.return_code} {res.failure_kind}"
    progress.update(current_task="G1", current_subtask=status)
    return payload


def run_inference_task(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, name: str, gpu_ids: str, angles: str, heads: str, prompts: str, max_images: int, batch_size: int) -> dict[str, Any]:
    out_json = ctx["work_dir"] / "G2_json" / f"{name}.json"
    out_md = ctx["work_dir"] / "G2_json" / f"{name}.md"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/live_full12_openrsd_inference_v3.py",
        "--repo-root", str(args.repo_root),
        "--result-md-dir", str(args.result_md_dir),
        "--weights-dir", str(args.weights_dir),
        "--work-dir", str(ctx["work_dir"]),
        "--run-ts", ctx["run_ts"],
        "--gpu-ids", gpu_ids,
        "--angles", angles,
        "--prompt-families", prompts,
        "--heads", heads,
        "--batch-size", str(batch_size),
        "--num-workers", str(args.num_workers),
        "--max-images", str(max_images),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]
    res = runner.run(name, cmd, ctx["work_dir"] / "G2_logs" / name, gpu_ids=gpu_ids, batch_size=batch_size, timeout_sec=None)
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    return {"name": name, "result": result_dict(res), "payload": payload}


def write_g2_aggregate(ctx: dict[str, Any], parts: list[dict[str, Any]], status_override: str | None = None) -> None:
    rows: list[dict[str, Any]] = []
    for part in parts:
        rows.extend(part.get("payload", {}).get("rows", []))
    live_rows = [r for r in rows if r.get("status") == "OK" and max((r.get("peak_mem_mb") or {"0": 0}).values() or [0]) >= 2000 and any((r.get("process_seen") or {}).values())]
    angles = sorted({r.get("angle") for r in rows if r.get("angle")})
    status = status_override or ("DONE_LIVE_GPU" if len(angles) == 12 and live_rows else ("PARTIAL_LIVE_GPU" if live_rows else "GPU_NOT_USED"))
    lines = ["# G2 Live Full-12 OpenRSD Inference", "", f"- status: `{status}`", f"- angles_seen: `{angles}`", "", "| name | status | rc | peak_mem | json |", "|---|---|---:|---|---|"]
    for part in parts:
        res = part.get("result", {})
        lines.append(f"| {part.get('name')} | {part.get('payload', {}).get('status')} | {res.get('return_code')} | `{res.get('peak_mem_mb')}` | `{ctx['work_dir'] / 'G2_json' / (part.get('name') + '.json')}` |")
    lines += ["", "## Prediction Files", ""]
    for r in rows:
        lines.append(f"- `{r.get('formal_predictions') or r.get('predictions')}` mtime={r.get('formal_predictions_mtime')} peak={r.get('peak_mem_mb')} process={r.get('process_seen')}")
    write_text(ctx["md"]["G2"], "\n".join(lines))
    write_json(ctx["work_dir"] / "G2_aggregate.json", {"status": status, "parts": parts, "rows": rows})


def run_hook_task(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, name: str, gpu_ids: str, heads: str, angles: str, max_live_images: int) -> dict[str, Any]:
    out_json = ctx["work_dir"] / "G4_json" / f"{name}.json"
    out_md = ctx["work_dir"] / "G4_json" / f"{name}.md"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/large_scale_live_hooks_v3.py",
        "--repo-root", str(args.repo_root),
        "--work-dir", str(ctx["work_dir"] / name),
        "--run-ts", ctx["run_ts"],
        "--gpu-ids", gpu_ids,
        "--angles", angles,
        "--heads", heads,
        "--max-live-images", str(max_live_images),
        "--out-md", str(out_md),
        "--out-json", str(out_json),
    ]
    res = runner.run(name, cmd, ctx["work_dir"] / "G4_logs" / name, gpu_ids=gpu_ids, batch_size=1, timeout_sec=None)
    payload = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    return {"name": name, "result": result_dict(res), "payload": payload}


def write_g4_aggregate(ctx: dict[str, Any], parts: list[dict[str, Any]]) -> None:
    rows = []
    live = []
    for p in parts:
        rows.append(p)
        res = p.get("result", {})
        if max((res.get("peak_mem_mb") or {"0": 0}).values() or [0]) >= 2000 and any((res.get("process_seen") or {}).values()):
            live.append(p)
    status = "DONE_LIVE_GPU" if live else "GPU_NOT_USED"
    lines = ["# G4 Large-Scale Live Hook Diagnostic", "", f"- status: `{status}`", "", "| name | status | row_count | peak_mem | process_seen | json |", "|---|---|---:|---|---|---|"]
    for p in parts:
        res = p.get("result", {})
        payload = p.get("payload", {})
        lines.append(f"| {p.get('name')} | {payload.get('status')} | {payload.get('row_count')} | `{res.get('peak_mem_mb')}` | `{res.get('process_seen')}` | `{ctx['work_dir'] / 'G4_json' / (p.get('name') + '.json')}` |")
    write_text(ctx["md"]["G4"], "\n".join(lines))
    write_json(ctx["work_dir"] / "G4_aggregate.json", {"status": status, "parts": rows})


def run_g3(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter) -> dict[str, Any]:
    progress.update(current_task="G3", current_subtask="12x12 TTA merge using G2 pkl", next_task="G5 zero AP reactivation")
    out_json = ctx["work_dir"] / "G3_live_12x12_tta.json"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/live_full12x12_ovd_tta_v3.py",
        "--repo-root", str(args.repo_root),
        "--work-dir", str(ctx["work_dir"]),
        "--run-ts", ctx["run_ts"],
        "--head", "alignment",
        "--prompt-family", "F3_orientation_aware",
        "--out-md", str(ctx["md"]["G3"]),
        "--out-json", str(out_json),
    ]
    res = runner.run("G3_live_12x12_tta", cmd, ctx["work_dir"] / "G3_logs", gpu_ids=args.gpu_ids, batch_size=None, timeout_sec=None)
    progress.completed += 1
    progress.update(current_task="G3", current_subtask="done")
    return {"result": result_dict(res), "payload": json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}}


def run_g5(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter, batch_size: int) -> dict[str, Any]:
    progress.update(current_task="G5", current_subtask="zero AP prompt sweep live inference", next_task="summary or extended")
    out_json = ctx["work_dir"] / "G5_zero_ap_gpu_reactivation.json"
    cmd = [
        str(pybin()),
        "M_Tools/analysis/zero_ap_gpu_reactivation_v3.py",
        "--repo-root", str(args.repo_root),
        "--weights-dir", str(args.weights_dir),
        "--result-md-dir", str(args.result_md_dir),
        "--work-dir", str(ctx["work_dir"]),
        "--run-ts", ctx["run_ts"],
        "--gpu-ids", args.gpu_ids,
        "--batch-size", str(batch_size),
        "--max-images", "256",
        "--out-md", str(ctx["md"]["G5"]),
        "--out-json", str(out_json),
    ]
    res = runner.run("G5_zero_ap_reactivation", cmd, ctx["work_dir"] / "G5_logs", gpu_ids=args.gpu_ids, batch_size=batch_size, timeout_sec=None)
    progress.completed += 1
    progress.update(current_task="G5", current_subtask="done")
    return {"result": result_dict(res), "payload": json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}}


def write_summary(args: argparse.Namespace, ctx: dict[str, Any], reports: dict[str, Any], progress: ProgressWriter) -> None:
    rows = query_gpu_rows(args.gpu_ids)
    lines = [
        "# Summary Full Live GPU v3",
        "",
        f"- RUN_TS: `{ctx['run_ts']}`",
        f"- work_dir: `{ctx['work_dir']}`",
        f"- real_gpu_ids: `{args.gpu_ids}`",
        f"- progress_md: `{ctx['work_dir'] / 'progress.md'}`",
        f"- current_task: `{progress.current_task}`",
        "",
        "## Task Status",
        "",
        "| task | status | artifact |",
        "|---|---|---|",
    ]
    for key in ["G0", "G1", "G2", "G3", "G4", "G5"]:
        status = "UNKNOWN"
        if key.lower() in reports:
            payload = reports[key.lower()]
            status = payload.get("payload", {}).get("status") or payload.get("status") or payload.get("payload", {}).get("status", "UNKNOWN")
        lines.append(f"| {key} | {status} | `{ctx['md'].get(key)}` |")
    lines.extend(["", "## Current GPU", "", "```text", json.dumps(rows, indent=2), "```", "", "## Reports JSON", ""])
    for key, value in reports.items():
        lines.append(f"- {key}: `{str(value)[:500]}`")
    write_text(ctx["md"]["SUM"], "\n".join(lines))


def run_smoke(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    progress.update(current_task="smoke", current_subtask="live inference 64 images", current_angle="angle_000 F3 alignment", batch_size=args.batch_size or 4, next_task="hook smoke")
    reports["g2_smoke"] = run_inference_task(args, ctx, runner, "smoke_live_inference", "4", "000", "alignment", "F3_orientation_aware", 64, args.batch_size or 4)
    progress.completed += 1
    progress.update(current_task="smoke", current_subtask="live hook 16 images", next_task="training smoke")
    reports["g4_smoke"] = run_hook_task(args, ctx, runner, "smoke_live_hook", "4", "alignment", "000", 16)
    progress.completed += 1
    reports["g1_smoke"] = run_training_smoke(args, ctx, runner, progress, max_iters=5)
    write_json(ctx["work_dir"] / "smoke_summary.json", reports)
    return reports


def run_batch_probe(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter) -> dict[str, Any]:
    candidates = parse_csv(args.batch_size_candidates, int)
    rows = []
    selected = 1
    for bs in candidates:
        progress.update(current_task="batch_probe", current_subtask=f"inference bs={bs}", batch_size=bs, current_angle="angle_000,090", next_task="full queue")
        name = f"batch_probe_bs{bs}"
        part = run_inference_task(args, ctx, runner, name, args.gpu_ids, "000,090", "alignment", "F3_orientation_aware", 32, bs)
        res = part["result"]
        peak = max((res.get("peak_mem_mb") or {"0": 0}).values() or [0])
        ok = res.get("return_code") == 0 and peak >= 2000
        rows.append({"batch_size": bs, "ok": ok, "peak_mem_mb": res.get("peak_mem_mb"), "return_code": res.get("return_code")})
        if ok:
            selected = bs
        if res.get("failure_kind") in {"CUDA_OOM", "OOM"}:
            break
    payload = {"status": "DONE", "rows": rows, "selected_batch_size": selected}
    write_json(ctx["work_dir"] / "batch_probe.json", payload)
    progress.completed += 1
    progress.update(current_task="batch_probe", current_subtask=f"selected bs={selected}", batch_size=selected)
    return payload


def run_full(args: argparse.Namespace, ctx: dict[str, Any], runner: StrictGpuRunner, progress: ProgressWriter) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    reports["g0"] = run_preflight(args, ctx, runner, progress)
    reports["smoke"] = run_smoke(args, ctx, runner, progress)
    batch = run_batch_probe(args, ctx, runner, progress)
    reports["batch"] = batch
    selected_bs = int(batch.get("selected_batch_size") or args.batch_size or 1)

    reports["g1"] = reports["smoke"].get("g1_smoke", {})

    progress.update(current_task="G2", current_subtask="live full-12 inference alignment F3", batch_size=selected_bs, next_task="G4 live hook")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(run_inference_task, args, ctx, runner, "G2_alignment_F3_gpu4_a", "4", ANGLES_A, "alignment", "F3_orientation_aware", 0, selected_bs),
            ex.submit(run_inference_task, args, ctx, runner, "G2_alignment_F3_gpu5_b", "5", ANGLES_B, "alignment", "F3_orientation_aware", 0, selected_bs),
        ]
        g2_parts = [f.result() for f in futs]
    progress.completed += 1
    write_g2_aggregate(ctx, g2_parts)
    reports["g2"] = {"payload": json.loads((ctx["work_dir"] / "G2_aggregate.json").read_text(encoding="utf-8"))}

    progress.update(current_task="G4", current_subtask="live hooks 512 images/angle", batch_size=1, next_task="G2 fusion")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(run_hook_task, args, ctx, runner, "G4_alignment_gpu4", "4", "alignment", ",".join(ANGLES12), 512),
            ex.submit(run_hook_task, args, ctx, runner, "G4_fusion_gpu5", "5", "fusion", ",".join(ANGLES12), 512),
        ]
        g4_parts = [f.result() for f in futs]
    progress.completed += 1
    write_g4_aggregate(ctx, g4_parts)
    reports["g4"] = {"payload": json.loads((ctx["work_dir"] / "G4_aggregate.json").read_text(encoding="utf-8"))}

    progress.update(current_task="G2", current_subtask="live fusion F3 all12", batch_size=selected_bs, next_task="G3 TTA")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [
            ex.submit(run_inference_task, args, ctx, runner, "G2_fusion_F3_gpu4_a", "4", ANGLES_A, "fusion", "F3_orientation_aware", 0, selected_bs),
            ex.submit(run_inference_task, args, ctx, runner, "G2_fusion_F3_gpu5_b", "5", ANGLES_B, "fusion", "F3_orientation_aware", 0, selected_bs),
        ]
        g2_fusion = [f.result() for f in futs]
    g2_all = g2_parts + g2_fusion
    write_g2_aggregate(ctx, g2_all)
    reports["g2"] = {"payload": json.loads((ctx["work_dir"] / "G2_aggregate.json").read_text(encoding="utf-8"))}

    reports["g3"] = run_g3(args, ctx, runner, progress)
    reports["g5"] = run_g5(args, ctx, runner, progress, selected_bs)

    deadline = time.time() + max(0.0, args.time_budget_hours * 3600 - (time.time() - progress.start_time))
    extended_done = False
    while time.time() < deadline and not extended_done:
        progress.update(current_task="extended", current_subtask="additional F1/F0 prompt live inference", batch_size=selected_bs, next_task="extended hooks")
        ext = run_inference_task(args, ctx, runner, "G2_extended_F1_F0_angle000_090", args.gpu_ids, "000,090", "alignment", "F1_aerial_context,F0_raw_class", 0, selected_bs)
        append_jsonl(ctx["work_dir"] / "extended_queue.jsonl", {"task": "G2_extended_F1_F0_angle000_090", "result": ext})
        extended_done = True
    write_summary(args, ctx, reports, progress)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--run-ts", default=None)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--time-budget-hours", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-candidates", default="1,2,4,8,16,24,32,48,64")
    parser.add_argument("--target-vram-ratio", type=float, default=0.85)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images-for-smoke", type=int, default=64)
    parser.add_argument("--force-live", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--forbid-old-pkl-as-result", action="store_true")
    parser.add_argument("--keep-gpus-busy", action="store_true")
    parser.add_argument("--fallback-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    ctx = make_context(args)
    runner = StrictGpuRunner(args.repo_root, ctx["work_dir"], args.gpu_ids)
    progress = ProgressWriter(ctx["work_dir"], args.gpu_ids, total_tasks=10)
    progress.start()
    reports: dict[str, Any] = {}
    try:
        if args.mode == "dryrun":
            reports["g0"] = run_preflight(args, ctx, runner, progress)
        elif args.mode == "smoke":
            reports["g0"] = run_preflight(args, ctx, runner, progress)
            reports["smoke"] = run_smoke(args, ctx, runner, progress)
            reports["batch"] = run_batch_probe(args, ctx, runner, progress)
        else:
            reports = run_full(args, ctx, runner, progress)
    except Exception as exc:  # noqa: BLE001
        progress.last_failure = repr(exc)
        append_jsonl(ctx["work_dir"] / "failures.jsonl", {"time": datetime.now().isoformat(), "fatal": repr(exc)})
        write_summary(args, ctx, reports, progress)
        raise
    finally:
        write_summary(args, ctx, reports, progress)
        progress.stop()
    print("FULL_LIVE_GPU_V3_RUNNER_DONE")
    print(f"RUN_TS={ctx['run_ts']}")
    print(f"work_dir={ctx['work_dir']}")
    print(f"progress_md={ctx['work_dir'] / 'progress.md'}")


if __name__ == "__main__":
    main()

