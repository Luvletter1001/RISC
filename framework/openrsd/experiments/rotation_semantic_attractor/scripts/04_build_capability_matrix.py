#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.model_adapters.capabilities import build_capability_profile
from experiments.rotation_semantic_attractor.src.utils.io import write_csv, write_json


def _markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Capability Matrix",
        "",
        "NOT_APPLICABLE means the model architecture does not expose that concept. UNSUPPORTED_BY_CURRENT_CODE means it is theoretically meaningful but not implemented in the current adapter.",
        "",
        "| model_name | family | architecture | final | pre_nms | dense | query | query_status | query_reason | text_prompt | text_embedding | classifier_channel | real_context | dehub_compare | asset_status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {model_family} | {architecture_type} | {supports_final_predictions} | {supports_pre_nms_predictions} | {supports_dense_logits} | {supports_query_logits} | {query_logits_status} | {query_logits_reason} | {supports_text_prompts} | {supports_text_embedding_intervention} | {supports_classifier_channel_intervention} | {supports_real_context_counterfactual} | {supports_dehub_checkpoint_comparison} | {asset_status} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    args = parser.parse_args()
    registry = read_jsonish(args.model_registry)["models"]
    rows = [build_capability_profile(model_key, cfg) for model_key, cfg in registry.items()]
    output_dir = Path(args.output_dir) if args.output_dir else exp_path("outputs")
    csv_path = output_dir / "capability_matrix.csv"
    json_path = output_dir / "capability_matrix.json"
    report_path = exp_path("reports", "capability_matrix.md")
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    _markdown(report_path, rows)
    print(f"capability_csv={csv_path}")
    print(f"capability_report={report_path}")


if __name__ == "__main__":
    main()
