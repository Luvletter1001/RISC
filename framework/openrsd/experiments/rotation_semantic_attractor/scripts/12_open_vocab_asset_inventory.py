#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT
from experiments.rotation_semantic_attractor.src.model_adapters.open_vocab_asset_inventory import (
    find_runnable_open_vocab_pairs,
    inventory_summary,
    render_asset_request_markdown,
    render_inventory_markdown,
    scan_open_vocab_assets,
)
from experiments.rotation_semantic_attractor.src.utils.io import write_json


def default_roots() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "experiments/rotation_semantic_attractor/configs",
        PROJECT_ROOT / "M_configs/Step2_A10_Large_Pretrain_Stage3",
        PROJECT_ROOT / "M_configs/Step3_A12_SelfTrain",
        PROJECT_ROOT / "M_configs/Other/A13_InContext",
        PROJECT_ROOT / "M_configs/experiments",
        PROJECT_ROOT / "results/MMR_AD_A10_flex_rtm_v3_1_formal",
        PROJECT_ROOT / "results/MMR_AD_A08_e_rtm_v2_base_recheck",
        PROJECT_ROOT / "data/DOTA1_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl",
        PROJECT_ROOT / "data/Neg_supports_v2.pkl",
        PROJECT_ROOT / "data/normalized_class_dict.pkl",
        PROJECT_ROOT / "data/7_25_pca_meta_DINOv2_256.pkl",
        PROJECT_ROOT / "tools/rotation_overnight_gpu89",
        PROJECT_ROOT / "tools/exp_next_plan_sv_dehub_step23_overnight",
        PROJECT_ROOT / "tools/exp_sv_dehub_lite_train_gpu89",
        PROJECT_ROOT / "mmdet_configs/grounding_dino",
        Path("/mnt/data/OpenRSD"),
    ]
    return [path for path in candidates if path.exists()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="Additional root to scan.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments/rotation_semantic_attractor/outputs/open_vocab_assets"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "experiments/rotation_semantic_attractor/reports"))
    args = parser.parse_args()

    roots = default_roots() + [Path(path) for path in args.root]
    assets = scan_open_vocab_assets(roots)
    pairs = find_runnable_open_vocab_pairs(assets)
    summary = inventory_summary(assets, pairs)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    write_json(
        output_dir / "open_vocab_asset_inventory.json",
        {
            "roots": [str(path) for path in roots],
            "summary": summary,
            "runnable_pairs": pairs,
            "assets": assets,
        },
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "open_vocab_asset_inventory.md").write_text(render_inventory_markdown(summary, assets, pairs))
    (report_dir / "open_vocab_asset_request.md").write_text(render_asset_request_markdown(summary))

    print(f"status={summary['status']}")
    print(f"num_assets={summary['num_assets']}")
    print(f"num_runnable_pairs={summary['num_runnable_pairs']}")
    print(f"inventory_json={output_dir / 'open_vocab_asset_inventory.json'}")
    print(f"inventory_md={report_dir / 'open_vocab_asset_inventory.md'}")
    print(f"asset_request_md={report_dir / 'open_vocab_asset_request.md'}")


if __name__ == "__main__":
    main()
