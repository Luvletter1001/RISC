#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, read_jsonish
from experiments.rotation_semantic_attractor.src.model_adapters.capabilities import build_capability_profile
from experiments.rotation_semantic_attractor.src.utils.io import read_json, write_csv
from experiments.rotation_semantic_attractor.src.utils.status import CLAIM_NONE, ExperimentStatus, status_metadata


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    registry = read_jsonish(manifest["args"]["model_registry"])["models"] if manifest.get("args", {}).get("model_registry") else {}
    rows = []
    for model in manifest.get("selected_models", []):
        capability = build_capability_profile(model, registry.get(model, {}))
        if capability["model_family"] == "closed_set":
            meta = status_metadata(
                ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE,
                "open-vocabulary model was not selected in this smoke run",
                claim_level=CLAIM_NONE,
            )
        elif capability["asset_status"] == ExperimentStatus.NOT_AVAILABLE_ASSET:
            meta = status_metadata(
                ExperimentStatus.NOT_AVAILABLE_ASSET,
                "no available open-vocabulary model asset found",
                claim_level=CLAIM_NONE,
            )
        elif capability["supports_text_embedding_intervention"]:
            meta = status_metadata(
                ExperimentStatus.DONE_SMOKE,
                "embedding-level intervention hook is available for smoke",
                claim_level="open_vocab_embedding_mechanism",
                is_scientific_result=False,
                include_in_main_table=False,
            )
        elif capability["supports_text_prompts"]:
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "prompt path may be available, but embedding tensor modification is not implemented",
                claim_level=CLAIM_NONE,
            )
        else:
            meta = status_metadata(
                ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
                "open-vocabulary embedding intervention hook is not implemented for this adapter",
                claim_level=CLAIM_NONE,
            )
        rows.append(
            {
                "model_name": model,
                "model_family": capability["model_family"],
                "tile_id": "",
                "angle": "",
                "intervention_type": "embedding_intervention",
                "intervention_target": "small-vehicle",
                "selected_in_this_run": True,
                "intervention_is_prompt_only": False,
                "intervention_is_embedding_level": capability["supports_text_embedding_intervention"],
                "intervention_is_visual_support_level": capability["supports_visual_support_intervention"],
                "is_causal_intervention": capability["supports_text_embedding_intervention"],
                "is_deployable_repair_candidate": False,
                "final_fr_sv": "",
                "final_fsv": "",
                "dense_or_query_sv": "",
                "ap_proxy_if_available": "",
                "det_per_img": "",
                **meta,
                "unsupported_reason": meta["status_reason"],
            }
        )
    out = Path(args.output_dir or run_dir / "metrics") / "open_vocab_intervention.csv"
    write_csv(out, rows)
    print(f"open_vocab_intervention={out}")


if __name__ == "__main__":
    main()
