#!/usr/bin/env python3
"""GPU45 wrapper for real OpenRSD feature/text/logit hooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gpu45_task_runner import append_jsonl, result_to_dict, run_child, shell_join, write_json, write_text  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def find_config(existing_work_dir: Path) -> Path | None:
    for pattern in [
        "exp_ovd3/*/alignment/angle_000/eval_config.py",
        "exp_ovd2/*/alignment/angle_000/eval_config.py",
        "exp_ovd1/*/alignment/angle_000/eval_config.py",
    ]:
        matches = sorted(existing_work_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def dump_named_modules(args: argparse.Namespace, cfg_path: Path) -> dict[str, Any]:
    out_path = args.work_dir / "P2_hooks/model_named_modules.txt"
    err_path = args.work_dir / "P2_hooks/model_named_modules_error.txt"
    script = args.work_dir / "P2_hooks/dump_named_modules_inline.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"""
import json, os, sys, traceback
from pathlib import Path
repo = Path({str(args.repo_root)!r})
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'tools'))
try:
    from openrsd_env import preload_installed_mmengine
    preload_installed_mmengine()
except Exception:
    pass
from mmdet.utils import register_all_modules as register_mmdet
from mmrotate.utils import register_all_modules as register_mmrotate
register_mmdet(init_default_scope=False)
register_mmrotate(init_default_scope=False)
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
cfg = Config.fromfile({str(cfg_path)!r})
cfg.work_dir = {str(args.work_dir / 'P2_hooks/module_dump_work')!r}
cfg.test_dataloader.batch_size = 1
cfg.test_dataloader.num_workers = 0
cfg.test_dataloader.persistent_workers = False
if 'custom_hooks' in cfg:
    cfg.custom_hooks = [h for h in cfg.custom_hooks if h.get('type') != 'EMAHook']
runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)
lines = []
for name, module in runner.model.named_modules():
    lines.append(f"{{name}}\\t{{module.__class__.__name__}}")
Path({str(out_path)!r}).parent.mkdir(parents=True, exist_ok=True)
Path({str(out_path)!r}).write_text("\\n".join(lines) + "\\n", encoding='utf-8')
""",
        encoding="utf-8",
    )
    result = run_child(
        task_name="P2_dump_named_modules",
        argv=[str(args.python_bin), str(script)],
        repo_root=args.repo_root,
        log_dir=args.work_dir / "P2_logs/P2_dump_named_modules",
        gpu_ids=args.gpu_ids,
    )
    if result.return_code and not out_path.exists():
        err_path.write_text(result.stderr_tail + "\n" + result.stdout_tail, encoding="utf-8")
    payload = result_to_dict(result)
    append_jsonl(args.work_dir / "commands.jsonl", payload)
    if result.return_code:
        append_jsonl(args.work_dir / "failures.jsonl", payload)
    return {"path": str(out_path), "exists": out_path.exists(), "result": payload}


def summarize(args: argparse.Namespace, hook_summary: dict[str, Any], child_results: list[dict[str, Any]], dryrun_commands: list[str], named_modules: dict[str, Any]) -> dict[str, Any]:
    status = hook_summary.get("status") or ("DRYRUN" if args.mode == "dryrun" else "FAILED")
    if any(r.get("return_code") for r in child_results):
        status = "PARTIAL" if hook_summary.get("row_count", 0) else "FAILED"
    payload = {
        "status": status,
        "hook_summary": hook_summary,
        "named_modules": named_modules,
        "child_results": child_results,
        "dryrun_commands": dryrun_commands,
        "csv": str(args.out_csv),
        "json": str(args.out_json),
    }
    write_json(args.work_dir / "P2_hooks/P2_wrapper_summary.json", payload)
    lines = [
        "# P2 OVD5 Real Hook Adapter",
        "",
        f"- generated_at: `{args.run_ts}`",
        f"- status: `{status}`",
        f"- work_dir: `{args.work_dir}`",
        f"- named_modules: `{named_modules.get('path', '')}` exists `{named_modules.get('exists', False)}`",
        f"- csv: `{args.out_csv}`",
        f"- json: `{args.out_json}`",
        f"- successful_hooks: `{hook_summary.get('successful_hooks', [])}`",
        f"- failed_or_missing_hooks: `{hook_summary.get('failed_or_missing_hooks', [])}`",
        f"- row_count: `{hook_summary.get('row_count', 0)}`",
        "",
        "## Hook Status",
        "",
        "| module_or_layer | status |",
        "|---|---|",
    ]
    ok = set(hook_summary.get("successful_hooks", []))
    missing = set(hook_summary.get("failed_or_missing_hooks", []))
    for layer in sorted(ok | missing):
        lines.append(f"| {layer} | {'HOOK_OK' if layer in ok else 'NOT_AVAILABLE'} |")
    lines.extend([
        "",
        "## Answers",
        "",
        f"- raw alignment logits captured: `{'alignment_logits' in ok}`",
        f"- raw fusion logits captured: `{'fusion_logits' in ok}`",
        "- feature cosine / logit drift conclusions are computed in the CSV when canonical and rotated hook vectors are both available.",
        "- Hook failures are non-fatal and listed in the JSON errors array.",
        "",
        "## Commands",
        "",
    ])
    for cmd in dryrun_commands:
        lines.append(f"- `{cmd}`")
    for result in child_results:
        lines.append(f"- `{result.get('command')}`")
    if hook_summary.get("errors"):
        lines.extend(["", "## Errors", ""])
        for err in hook_summary.get("errors", [])[:50]:
            lines.append(f"- `{err}`")
    write_text(args.out_md, "\n".join(lines))
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = args.existing_work_dir
    cfg = find_config(existing)
    angles = "000,090" if args.mode == "smoke" else "000,030,060,090,120,150,240"
    max_live = args.max_images_for_smoke if args.mode == "smoke" else args.max_live_images
    cmd = [
        str(args.python_bin),
        "M_Tools/analysis/run_ovd_feature_text_logit_hooks.py",
        "--repo-root", str(args.repo_root),
        "--existing-work-dir", str(existing),
        "--prompt-family", "F3_orientation_aware",
        "--angles", angles,
        "--heads", "alignment,fusion",
        "--max-images", str(args.max_images_for_smoke if args.mode == "smoke" else args.max_images_full),
        "--max-live-images", str(max_live),
        "--live-work-dir", str(args.work_dir / "P2_hooks/live"),
        "--live-hooks",
        "--include-prediction-proxy",
        "--out-csv", str(args.out_csv),
        "--out-json", str(args.out_json),
    ]
    dryrun_commands = [shell_join(cmd)]
    child_results: list[dict[str, Any]] = []
    named_modules = {"path": str(args.work_dir / "P2_hooks/model_named_modules.txt"), "exists": False, "note": "dryrun"}
    if args.mode == "dryrun":
        hook_summary = {"status": "DRYRUN", "successful_hooks": [], "failed_or_missing_hooks": [], "row_count": 0}
        return summarize(args, hook_summary, child_results, dryrun_commands, named_modules)
    if cfg:
        named_modules = dump_named_modules(args, cfg)
    else:
        named_modules = {"path": str(args.work_dir / "P2_hooks/model_named_modules.txt"), "exists": False, "note": "no eval_config found"}
    result = run_child(task_name="P2_real_hook_adapter", argv=cmd, repo_root=args.repo_root, log_dir=args.work_dir / "P2_logs/P2_real_hook_adapter", gpu_ids=args.gpu_ids)
    payload = result_to_dict(result)
    child_results.append(payload)
    append_jsonl(args.work_dir / "commands.jsonl", payload)
    if result.return_code:
        append_jsonl(args.work_dir / "failures.jsonl", payload)
    hook_summary = load_json(args.out_json)
    return summarize(args, hook_summary, child_results, dryrun_commands, named_modules)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--existing-work-dir", type=Path, default=Path("/data1/zcy/OpenRSD/work_dirs/openrsd_ovd_rotation_20260508"))
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--mode", choices=["dryrun", "smoke", "full", "debug"], default="dryrun")
    parser.add_argument("--max-images-for-smoke", type=int, default=16)
    parser.add_argument("--max-images-full", type=int, default=512)
    parser.add_argument("--max-live-images", type=int, default=512)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    default_python = Path("/data/zcy/anaconda3/envs/openrsd/bin/python")
    parser.add_argument("--python-bin", type=Path, default=default_python if default_python.exists() else Path(sys.executable))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_md_dir = args.result_md_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
