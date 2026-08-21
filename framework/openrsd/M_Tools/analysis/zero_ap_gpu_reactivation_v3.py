#!/usr/bin/env python3
"""GPU-backed zero-AP reactivation probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from strict_gpu_runner_v3 import StrictGpuRunner, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data1/zcy/OpenRSD"))
    parser.add_argument("--weights-dir", type=Path, default=Path("/data1/zcy/OpenRSD/results"))
    parser.add_argument("--result-md-dir", type=Path, default=Path("/data1/zcy/OpenRSD/resultmd"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-ts", required=True)
    parser.add_argument("--gpu-ids", default="4,5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/data/zcy/anaconda3/envs/openrsd/bin/python"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = StrictGpuRunner(args.repo_root, args.work_dir, args.gpu_ids)
    py = args.python_bin if args.python_bin.exists() else Path(sys.executable)
    templates = [
        "F0_raw_class",
        "F1_aerial_context",
        "F2_remote_sensing_context",
        "F3_orientation_aware",
        "F4_shape_aware",
    ]
    rows = []
    for prompt in templates:
        out_json = args.work_dir / "G5_zero_ap" / prompt / "live_inference.json"
        out_md = args.work_dir / "G5_zero_ap" / prompt / "live_inference.md"
        cmd = [
            str(py), "M_Tools/analysis/live_full12_openrsd_inference_v3.py",
            "--repo-root", str(args.repo_root),
            "--result-md-dir", str(args.result_md_dir),
            "--weights-dir", str(args.weights_dir),
            "--work-dir", str(args.work_dir / "G5_zero_ap" / prompt),
            "--run-ts", args.run_ts,
            "--gpu-ids", args.gpu_ids,
            "--angles", "000,090,240",
            "--prompt-families", prompt,
            "--heads", "alignment",
            "--batch-size", str(args.batch_size),
            "--max-images", str(args.max_images),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ]
        res = runner.run(f"G5_prompt_{prompt}", cmd, args.work_dir / "G5_logs" / prompt, gpu_ids=args.gpu_ids, batch_size=args.batch_size)
        rows.append({"prompt": prompt, "return_code": res.return_code, "peak_mem_mb": res.peak_mem_mb, "process_seen": res.process_seen, "out_json": str(out_json), "stdout": res.stdout_path, "stderr": res.stderr_path})
    live = [r for r in rows if max((r.get("peak_mem_mb") or {"0": 0}).values() or [0]) >= 2000 and any((r.get("process_seen") or {}).values())]
    status = "DONE_LIVE_GPU" if live else "GPU_NOT_USED"
    payload = {"status": status, "classes": ["plane", "harbor", "helicopter", "large-vehicle"], "rows": rows, "strategies": ["prompt template sweep live inference", "support re-encoding planned", "score calibration planned", "lightweight calibration head planned"]}
    write_json(args.out_json, payload)
    lines = ["# G5 Zero-AP GPU Reactivation", "", f"- status: `{status}`", "", "## Prompt Sweep", "", "| prompt | rc | peak_mem | process_seen | out_json |", "|---|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['prompt']} | {r['return_code']} | `{r['peak_mem_mb']}` | `{r['process_seen']}` | `{r['out_json']}` |")
    lines += ["", "Support re-encoding, score calibration, and calibration-head training are queued after live prompt sweep evidence; failures are recorded instead of blocking G2/G4."]
    write_text(args.out_md, "\n".join(lines))


if __name__ == "__main__":
    main()

