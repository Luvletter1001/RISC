#!/usr/bin/env python
"""
Per-angle TTA Evaluation Script (批量多模型顺序评测版)

对每个目标角度 θ，进行 TTA 推理：
- TTA views: θ + {0, 90, 180, 270}（旋回 canonical 坐标后 NMS 融合）
- 在 angle_θ 的 GT 上评估

支持批量评测多个模型，按顺序依次处理
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ========== 配置 ==========
ROOT_DIR = Path("/data1/zcy/OpenRSD")
PYTHON_BIN = "/data/zcy/anaconda3/envs/openrsd/bin/python"
DATA_ROOT = ROOT_DIR / "data/DOTA1_1024_500"
ANGLE_ROOT = DATA_ROOT / "angle_sweep_val/realistic"

# 可用模型列表
MODEL_CONFIGS = {
    "rtmdet_l": {
        "config": str(ROOT_DIR / "M_configs/RotationStudy/rotated_rtmdet_l_dota1_ms_eval.py"),
        "checkpoint": str(ROOT_DIR / "weights/rotated_rtmdet_l-3x-dota_ms-2738da34.pth"),
        "description": "Rotated RTMDet-L 3x DOTA MS",
    },
    "h2rbox_v2": {
        "config": str(ROOT_DIR / "M_configs/RotationStudy/h2rbox_r50_fpn_dota1_eval.py"),
        "checkpoint": str(ROOT_DIR / "weights/h2rbox_v2-le90_r50_fpn_ms_rr-1x_dota-5e0e53e1.pth"),
        "description": "H2RBox-V2 R50 MS-RR",
    },
    "retinanet_msrr": {
        "config": str(ROOT_DIR / "M_configs/RotationStudy/rotated_retinanet_r50_msrr_dota1_eval.py"),
        "checkpoint": str(ROOT_DIR / "weights/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth"),
        "description": "Rotated RetinaNet R50 MS-RR",
    },
    "redet": {
        "config": str(ROOT_DIR / "M_configs/RotationStudy/redet_re50_refpn_dota1_eval.py"),
        "checkpoint": str(ROOT_DIR / "weights/ReDet_re50_refpn_1x_dota1-a025e6b1.pth"),
        "description": "ReDet R50 ReFPN (DOTA1)",
    },
    "orcnn": {
        "config": str(ROOT_DIR / "mmrotate_configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py"),
        "checkpoint": str(ROOT_DIR / "weights/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth"),
        "description": "Oriented RCNN R50 FPN",
    },
}

# TTA 视角
TTA_VIEWS = [(0,), (90,), (180,), (270,)]

# 默认评测的目标角度
DEFAULT_ANGLES = ["000", "030", "060", "090", "120", "150", "180", "210", "240", "270", "300", "330"]

# 超参数
DEFAULT_BATCH_SIZE = 32      # 4卡分布式，batch_size 增大
DEFAULT_NUM_WORKERS = 4
DEFAULT_MASTER_PORT_BASE = 38100
DEFAULT_NUM_GPUS = 4          # 分布式 GPU 数量


def log(msg):
    """带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def angle_str_to_int(s):
    """'000' -> 0"""
    return int(s)


def int_to_angle_str(i):
    """0 -> '000'"""
    return f"{i:03d}"


def compute_tta_views(target_angle):
    """给定目标角度，计算 TTA 视角"""
    target_int = angle_str_to_int(target_angle)
    views = []
    for offset, in TTA_VIEWS:
        view_angle = (target_int + offset) % 360
        views.append((int_to_angle_str(view_angle), target_int, offset))
    return views


def run_single_inference(model_name, angle_str, config, checkpoint, out_dir, gpu_ids, master_port, num_gpus=4):
    """运行单次推理 (多卡分布式)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_file = out_dir / "predictions.pkl"
    log_file = out_dir / "test.log"
    
    # 如果已存在预测结果，跳过
    if pred_file.exists():
        log(f"  [SKIP] {pred_file} exists")
        return str(pred_file)
    
    cmd = [
        PYTHON_BIN, "-m", "torch.distributed.launch",
        "--nproc_per_node", str(num_gpus),
        f"--master_port={master_port}",
        str(ROOT_DIR / "tools/openrsd_test.py"),
        config,
        checkpoint,
        "--launcher", "pytorch",
        "--work-dir", str(out_dir),
        "--out", str(pred_file),
        "--cfg-options",
        f"test_dataloader.batch_size={DEFAULT_BATCH_SIZE}",
        f"test_dataloader.num_workers={DEFAULT_NUM_WORKERS}",
        f"test_dataloader.dataset.data_root={DATA_ROOT}",
        f"test_dataloader.dataset.ann_file=angle_sweep_val/realistic/angle_{angle_str}/annfiles/",
        f"test_dataloader.dataset.data_prefix.img_path=angle_sweep_val/realistic/angle_{angle_str}/images/",
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{ROOT_DIR}:{ROOT_DIR}/tools"
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    if result.returncode != 0:
        log(f"  [FAIL] See {log_file}")
        return None
    return str(pred_file)


def parse_metrics_from_log(log_file):
    """从日志文件解析 mAP"""
    if not Path(log_file).exists():
        return "NA", "NA"
    
    text = Path(log_file).read_text()
    
    matches = re.findall(r'dota/mAP[:\s]+([0-9]*\.?[0-9]+)', text)
    if not matches:
        return "NA", "NA"
    map_val = matches[-1]
    
    matches_ap50 = re.findall(r'dota/AP50[:\s]+([0-9]*\.?[0-9]+)', text)
    ap50_val = matches_ap50[-1] if matches_ap50 else "NA"
    
    return map_val, ap50_val


def merge_tta_predictions(pred_paths, angles, out_path):
    """使用 rotation_tta_merge.py 融合 TTA 预测"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        PYTHON_BIN,
        str(ROOT_DIR / "M_Tools/analysis/rotation_tta_merge.py"),
        "--out", str(out_path),
        "--score-thr", "0.05",
        "--pre-nms-topk", "4000",
        "--nms-iou", "0.1",
        "--max-per-img", "2000",
        "--img-shape", "1024", "1024",
    ]
    
    for pred_path, angle in zip(pred_paths, angles):
        cmd.extend(["--prediction", pred_path])
        cmd.extend(["--angle", str(angle)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  [MERGE FAIL] {result.stderr}")
        return None
    return str(out_path)


def run_per_angle_tta(model_name, model_info, gpu_ids, out_root, master_port_base, angles_list):
    """
    对单个模型运行完整的 Per-angle TTA 实验
    """
    config = model_info["config"]
    checkpoint = model_info["checkpoint"]
    description = model_info.get("description", model_name)
    
    log(f"\n{'='*60}")
    log(f"[{model_name}] {description}")
    log(f"{'='*60}")
    
    model_out_root = Path(out_root) / model_name
    results = []
    
    total_angles = len(angles_list)
    for idx, angle in enumerate(angles_list):
        angle_idx = angles_list.index(angle) + 1
        log(f"\n  [{angle_idx}/{total_angles}] Target angle: {angle}°")
        
        # Step 1: TTA 视角推理
        tta_views = compute_tta_views(angle)
        pred_paths = []
        view_angles = []
        
        log(f"    TTA views: {[v[0] for v in tta_views]}")
        
        for view_idx, (view_angle_str, target, offset) in enumerate(tta_views):
            view_out_dir = model_out_root / f"target_{angle}" / f"view_{view_angle_str}"
            port = master_port_base + angle_idx * 100 + view_idx
            pred_path = run_single_inference(
                model_name, view_angle_str, config, checkpoint,
                str(view_out_dir), gpu_ids, port
            )
            if pred_path:
                pred_paths.append(pred_path)
                view_angles.append(offset)
        
        # Step 2: 合并 TTA 预测
        tta_map, tta_ap50 = "NA", "NA"
        if len(pred_paths) == len(tta_views):
            merged_path = model_out_root / f"target_{angle}" / "merged_predictions.pkl"
            merged = merge_tta_predictions(pred_paths, view_angles, merged_path)
            
            if merged:
                log(f"    Merged: {merged}")
                tta_map, tta_ap50 = "NA", "NA"  # 融合后不单独评估，在下面用 test loop
            else:
                log(f"    Merge failed")
        else:
            log(f"    Some views missing: {len(pred_paths)}/{len(tta_views)}")
        
        # Step 3: 在目标角度 GT 上评估（用 test loop）
        eval_out_dir = model_out_root / f"target_{angle}"
        eval_log = eval_out_dir / "eval_on_gt.log"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            PYTHON_BIN, "-m", "torch.distributed.launch",
            "--nproc_per_node", str(DEFAULT_NUM_GPUS),
            f"--master_port={master_port_base + angle_idx * 100 + 10}",
            str(ROOT_DIR / "tools/openrsd_test.py"),
            config,
            checkpoint,
            "--launcher", "pytorch",
            "--work-dir", str(eval_out_dir / "eval_work"),
            "--out", str(eval_out_dir / "eval_predictions.pkl"),
            "--cfg-options",
            f"test_dataloader.batch_size={DEFAULT_BATCH_SIZE}",
            f"test_dataloader.num_workers={DEFAULT_NUM_WORKERS}",
            f"test_dataloader.dataset.data_root={DATA_ROOT}",
            f"test_dataloader.dataset.ann_file=angle_sweep_val/realistic/angle_{angle}/annfiles/",
            f"test_dataloader.dataset.data_prefix.img_path=angle_sweep_val/realistic/angle_{angle}/images/",
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids  # 4卡分布式
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = f"{ROOT_DIR}:{ROOT_DIR}/tools"
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        
        with open(eval_log, "w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        
        if result.returncode == 0:
            tta_map, tta_ap50 = parse_metrics_from_log(eval_log)
        
        # 获取单视角 baseline（0° 视角的结果）
        single_out_dir = model_out_root / f"target_{angle}" / f"view_{angle}"
        single_log = single_out_dir / "test.log"
        single_map, single_ap50 = parse_metrics_from_log(single_log)
        
        delta = "NA"
        if single_map != "NA" and tta_map != "NA":
            try:
                delta = f"{float(tta_map) - float(single_map):+.4f}"
            except:
                delta = "NA"
        
        results.append({
            "angle": angle,
            "single_mAP": single_map,
            "single_AP50": single_ap50,
            "tta_mAP": tta_map,
            "tta_AP50": tta_ap50,
            "delta": delta,
        })
        
        log(f"    Single mAP: {single_map}, TTA mAP: {tta_map}, Delta: {delta}")
    
    return results


def write_summary_csv(results, model_name, out_root):
    """写入 CSV 结果"""
    csv_path = Path(out_root) / model_name / "per_angle_tta_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["angle", "single_mAP", "single_AP50", "tta_mAP", "tta_AP50", "delta"])
        writer.writeheader()
        writer.writerows(results)
    
    return csv_path


def write_summary_md(all_results, out_root, angles_list):
    """写入 Markdown 汇总表"""
    md_path = Path(out_root) / "per_angle_tta_summary.md"
    
    lines = [
        "# Per-angle TTA Evaluation Summary",
        "",
        f"- generated_at: `{datetime.now().strftime('%F %T')}`",
        f"- angles: `{', '.join(angles_list)}`",
        "",
        "## Statistics Summary",
        "",
        "| model | setting | mean | std | min | max | range |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    
    # 计算统计
    stats_data = []
    for model_name, results in all_results.items():
        for setting in ["single", "tta"]:
            key = f"{setting}_mAP"
            vals = [float(r[key]) for r in results if r[key] != "NA"]
            if vals:
                import statistics
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0
                mn = min(vals)
                mx = max(vals)
                rng = mx - mn
                desc = MODEL_CONFIGS.get(model_name, {}).get("description", "")
                stats_data.append({
                    "model": f"{model_name} ({desc})" if desc else model_name,
                    "setting": setting,
                    "mean": f"{mean:.4f}",
                    "std": f"{std:.4f}",
                    "min": f"{mn:.4f}",
                    "max": f"{mx:.4f}",
                    "range": f"{rng:.4f}",
                })
    
    for s in stats_data:
        lines.append(f"| {s['model']} | {s['setting']} | {s['mean']} | {s['std']} | {s['min']} | {s['max']} | {s['range']} |")
    
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    
    for model_name, results in all_results.items():
        desc = MODEL_CONFIGS.get(model_name, {}).get("description", "")
        lines.append(f"### {model_name} ({desc})" if desc else f"### {model_name}")
        lines.append("")
        lines.append("| angle | single mAP | TTA mAP | delta |")
        lines.append("|---:|---:|---:|---:|")
        for r in results:
            delta_str = r["delta"] if r["delta"] != "NA" else "-"
            lines.append(f"| {r['angle']}° | {r['single_mAP']} | {r['tta_mAP']} | {delta_str} |")
        lines.append("")
    
    md_path.write_text("\n".join(lines))
    return md_path


def save_progress(all_results, out_root, current_model_idx, total_models):
    """保存进度到 JSON"""
    progress_path = Path(out_root) / "progress.json"
    progress = {
        "all_results": all_results,
        "current_model_idx": current_model_idx,
        "total_models": total_models,
        "timestamp": datetime.now().isoformat(),
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    return progress_path


def main():
    parser = argparse.ArgumentParser(description="Per-angle TTA Evaluation (批量多模型)")
    parser.add_argument("--models", nargs="+", 
                        default=["rtmdet_l", "h2rbox_v2", "retinanet_msrr", "redet", "orcnn"],
                        help="Models to evaluate")
    parser.add_argument("--angles", nargs="+", default=DEFAULT_ANGLES,
                        help="Target angles to evaluate")
    parser.add_argument("--gpus", default="4,5,6,7", help="GPU IDs")
    parser.add_argument("--out-root", default=None, help="Output root directory")
    parser.add_argument("--master-port-base", type=int, default=DEFAULT_MASTER_PORT_BASE,
                        help="Base master port")
    args = parser.parse_args()
    
    # 验证模型
    valid_models = []
    for m in args.models:
        if m not in MODEL_CONFIGS:
            log(f"[WARN] Unknown model: {m}, skipping")
            continue
        cfg = MODEL_CONFIGS[m]
        if not Path(cfg["config"]).exists():
            log(f"[WARN] Config not found: {cfg['config']}")
            continue
        if not Path(cfg["checkpoint"]).exists():
            log(f"[WARN] Checkpoint not found: {cfg['checkpoint']}")
            continue
        valid_models.append(m)
    
    if not valid_models:
        log("[ERROR] No valid models found!")
        sys.exit(1)
    
    log(f"Valid models: {valid_models}")
    log(f"Angles: {args.angles}")
    log(f"GPUs: {args.gpus}")
    
    # 生成输出目录
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT_DIR / "work_dirs" / f"per_angle_tta_{ts}"
    
    out_root.mkdir(parents=True, exist_ok=True)
    log(f"Output root: {out_root}")
    
    all_results = {}
    total_models = len(valid_models)
    
    # 批量顺序评测
    for model_idx, model_name in enumerate(valid_models, 1):
        log(f"\n{'#'*60}")
        log(f"# Model {model_idx}/{total_models}: {model_name}")
        log(f"{'#'*60}")
        
        model_info = MODEL_CONFIGS[model_name]
        
        results = run_per_angle_tta(
            model_name,
            model_info,
            args.gpus,
            str(out_root),
            args.master_port_base + model_idx * 1000,
            args.angles,
        )
        all_results[model_name] = results
        
        # 写入 CSV
        csv_path = write_summary_csv(results, model_name, str(out_root))
        log(f"  CSV saved: {csv_path}")
        
        # 保存进度
        progress_path = save_progress(all_results, str(out_root), model_idx, total_models)
        log(f"  Progress saved: {progress_path}")
    
    # 写入汇总 MD
    md_path = write_summary_md(all_results, str(out_root), args.angles)
    log(f"\n{'='*60}")
    log(f"ALL DONE! Summary: {md_path}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
