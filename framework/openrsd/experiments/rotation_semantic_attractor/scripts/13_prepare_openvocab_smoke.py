#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import PROJECT_ROOT, add_common_args, exp_path, read_jsonish
from experiments.rotation_semantic_attractor.src.model_adapters.capabilities import build_capability_profile
from experiments.rotation_semantic_attractor.src.utils.env import collect_env
from experiments.rotation_semantic_attractor.src.utils.io import write_csv, write_json
from experiments.rotation_semantic_attractor.src.utils.status import CLAIM_NONE, ExperimentStatus, status_metadata


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--model-registry", default=str(exp_path("configs", "model_registry.yaml")))
    args = parser.parse_args()
    run_dir = Path(args.output_dir or exp_path("outputs", "smoke", "openvocab"))
    metrics_dir = run_dir / "metrics"
    registry = read_jsonish(args.model_registry)["models"]
    profiles = {key: build_capability_profile(key, cfg) for key, cfg in registry.items()}
    candidates = [
        key
        for key, profile in profiles.items()
        if profile["model_family"] in {"open_vocab", "generic_ovd", "openrsd"}
        and profile["asset_status"] == ExperimentStatus.DONE_SMOKE
    ]
    if candidates:
        selected = candidates[:1]
        meta = status_metadata(
            ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
            "open-vocabulary smoke asset exists, but this scaffold has no runnable open-vocabulary adapter path yet",
            claim_level=CLAIM_NONE,
        )
    else:
        selected = []
        meta = status_metadata(
            ExperimentStatus.NOT_AVAILABLE_ASSET,
            "no available open-vocabulary model asset found",
            claim_level=CLAIM_NONE,
        )
    manifest = {
        "command": "13_prepare_openvocab_smoke.py",
        "environment": collect_env(PROJECT_ROOT),
        "selected_models": selected,
        "status": meta["status"],
        "status_reason": meta["status_reason"],
        "failures": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    rows = [
        {
            "model_name": selected[0] if selected else "open_vocab_candidate",
            "model_family": "generic_ovd" if selected else "unknown",
            "tile_id": "",
            "angle": "",
            "intervention_type": "embedding_intervention",
            "intervention_target": "small-vehicle",
            "selected_in_this_run": bool(selected),
            "intervention_is_prompt_only": False,
            "intervention_is_embedding_level": False,
            "intervention_is_visual_support_level": False,
            "is_causal_intervention": False,
            "is_deployable_repair_candidate": False,
            "final_fr_sv": "",
            "final_fsv": "",
            "dense_or_query_sv": "",
            "ap_proxy_if_available": "",
            "det_per_img": "",
            **meta,
            "unsupported_reason": meta["status_reason"],
        }
    ]
    write_csv(metrics_dir / "open_vocab_intervention.csv", rows)
    print(f"openvocab_run_dir={run_dir}")
    print(f"openvocab_status={meta['status']}")


if __name__ == "__main__":
    main()

