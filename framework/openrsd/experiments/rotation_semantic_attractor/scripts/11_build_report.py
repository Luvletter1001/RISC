#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, exp_path
from experiments.rotation_semantic_attractor.scripts import build_report_helpers as helpers
from experiments.rotation_semantic_attractor.src.utils.io import read_csv, read_json
from experiments.rotation_semantic_attractor.src.utils.status import ExperimentStatus


def _read_csv(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _table(rows: list[dict], limit: int = 12) -> str:
    if not rows:
        return "No rows.\n"
    fields = []
    seen = set()
    for row in rows:
        for field in row.keys():
            if field not in seen:
                fields.append(field)
                seen.add(field)
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > limit:
        lines.append(f"\nShowing {limit}/{len(rows)} rows.")
    return "\n".join(lines) + "\n"


def _status_rows(*groups: list[dict]) -> list[dict]:
    rows = []
    for group in groups:
        for row in group:
            if row.get("status"):
                rows.append(row)
    return rows


def _auxiliary_status_rows(rows: list[dict], fields: list[str]) -> list[dict]:
    status_rows = []
    for row in rows:
        source = row.get("model_name") or row.get("smoke") or row.get("stage") or ""
        for field in fields:
            status = row.get(field)
            if status:
                reason = row.get(field.replace("_status", "_reason"), "")
                status_rows.append(
                    {
                        "source": source,
                        "status_field": field,
                        "status": status,
                        "status_reason": reason,
                    }
                )
    return status_rows


def _verdict(counts: dict, has_real_prediction: bool) -> str:
    if counts["failure_count"] > 0:
        return "FAIL"
    if not has_real_prediction:
        return "PASS_SCHEMA_ONLY"
    if counts["unsupported_count"] > 0:
        return "PASS_WITH_UNSUPPORTED_HOOKS"
    if counts["not_applicable_count"] > 0 or counts["not_selected_count"] > 0:
        return "PASS_WITH_NOT_APPLICABLE"
    return "PASS"


def _matrix_verdict(matrix_run_dir: Path | None) -> str:
    if not matrix_run_dir:
        return ""
    path = matrix_run_dir / "smoke_verdict.txt"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        if line.startswith("verdict="):
            return line.split("=", 1)[1].strip()
    return ""


def _appendix(rows: list[dict]) -> str:
    appendix_statuses = {
        ExperimentStatus.NOT_APPLICABLE,
        ExperimentStatus.NOT_SELECTED_IN_THIS_SMOKE,
        ExperimentStatus.UNSUPPORTED_BY_CURRENT_CODE,
        ExperimentStatus.NOT_AVAILABLE_ASSET,
        ExperimentStatus.NOT_RUN,
        ExperimentStatus.FAILED,
    }
    appendix_rows = [row for row in rows if row.get("status") in appendix_statuses]
    return _table(appendix_rows, limit=40)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--openvocab-run-dir", default="")
    parser.add_argument("--matrix-run-dir", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    openvocab_run_dir = Path(args.openvocab_run_dir) if args.openvocab_run_dir else None
    matrix_run_dir = Path(args.matrix_run_dir) if args.matrix_run_dir else None
    metrics_dir = run_dir / "metrics"
    out = Path(args.output or exp_path("reports", "rotation_semantic_attractor_report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    capability_rows = _read_csv(exp_path("outputs", "capability_matrix.csv"))
    false_hub_summary = _read_csv(metrics_dir / "false_hub_summary_by_model.csv")
    stage_rows = _read_csv(metrics_dir / "stage_decomposition_summary.csv")
    open_vocab_rows = _read_csv(metrics_dir / "open_vocab_intervention.csv")
    if openvocab_run_dir:
        open_vocab_rows.extend(_read_csv(openvocab_run_dir / "metrics" / "open_vocab_intervention.csv"))
    closedset_rows = _read_csv(metrics_dir / "closedset_channel_intervention.csv")
    context_rows = _read_csv(metrics_dir / "context_counterfactual.csv")
    dehub_rows = _read_csv(metrics_dir / "dehub_safety.csv")
    matrix_rows = _read_csv(matrix_run_dir / "smoke_matrix_summary.csv") if matrix_run_dir else []

    auxiliary_rows = _auxiliary_status_rows(
        capability_rows,
        [
            "asset_status",
            "query_logits_status",
        ],
    ) + _auxiliary_status_rows(
        stage_rows,
        [
            "dense_logits_status",
            "query_logits_status",
            "pre_nms_status",
            "post_nms_status",
        ],
    )
    all_status_rows = _status_rows(
        false_hub_summary,
        stage_rows,
        open_vocab_rows,
        closedset_rows,
        context_rows,
        dehub_rows,
        matrix_rows,
        auxiliary_rows,
    )
    counts = helpers.status_summary_counts(all_status_rows)
    has_real_prediction = bool(false_hub_summary)
    verdict = _matrix_verdict(matrix_run_dir) or _verdict(counts, has_real_prediction)

    scientific_rows = helpers.scientific_rows(false_hub_summary + stage_rows + dehub_rows, include_smoke=False)
    smoke_scientific_rows = helpers.scientific_rows(false_hub_summary + stage_rows, include_smoke=True)
    proxy_rows = [row for row in closedset_rows + context_rows + dehub_rows if row.get("status") in {ExperimentStatus.SMOKE_PROXY, ExperimentStatus.SCHEMA_ONLY}]

    sections = [
        "# Rotation Semantic Attractor Smoke Matrix Report",
        "",
        "## Executive Status Summary",
        f"- verdict: `{verdict}`",
        "- status counts include primary row statuses plus capability/stage status fields.",
        f"- failure_count: `{counts['failure_count']}`",
        f"- not_applicable_count: `{counts['not_applicable_count']}`",
        f"- not_selected_count: `{counts['not_selected_count']}`",
        f"- unsupported_count: `{counts['unsupported_count']}`",
        f"- proxy_count: `{counts['proxy_count']}`",
        f"- requested GPUs: `{manifest.get('environment', {}).get('cuda_visible_devices', '')}`",
        f"- device used: `{manifest.get('device', '')}`",
        "",
        "## Capability Matrix",
        _table(capability_rows, limit=20),
        "## Smoke Execution Matrix",
        _table(matrix_rows or [
            {
                "smoke": "closedset",
                "run_dir": str(run_dir),
                "selected_models": manifest.get("selected_models", []),
                "status": ExperimentStatus.DONE_SMOKE if false_hub_summary else ExperimentStatus.FAILED,
                "status_reason": "closed-set smoke final prediction and false-hub taxonomy completed" if false_hub_summary else "closed-set smoke outputs missing",
            }
        ]),
        "## Actual Scientific Result Tables",
        "No `DONE_FULL` scientific rows are present in this smoke report. Smoke measurements below are engineering smoke outputs, not full scientific conclusions.",
        _table(scientific_rows),
        "## Smoke Scientific Measurements",
        _table(smoke_scientific_rows),
        "## Proxy / Schema Validation",
        "These rows validate code paths and schemas only; they do not support scientific claims.",
        _table(proxy_rows, limit=20),
        helpers.render_stage_decomposition_section(stage_rows),
        "## Intervention Section",
        "### Open-vocabulary prompt / embedding / visual-support intervention",
        _table(open_vocab_rows, limit=20),
        "### Closed-set classifier-channel intervention",
        _table(closedset_rows, limit=20),
        "## DeHub Safety",
        "Smoke proxy rows are excluded from DeHub repair claims.",
        _table(dehub_rows, limit=20),
        "## NOT_APPLICABLE / NOT_SELECTED / UNSUPPORTED / FAILED Appendix",
        _appendix(all_status_rows),
        "## Reproduction Commands",
        "```bash",
        "CUDA_VISIBLE_DEVICES=6,7,8,9 PYTHONNOUSERSITE=1 bash experiments/rotation_semantic_attractor/scripts/run_smoke_closedset.sh",
        "bash experiments/rotation_semantic_attractor/scripts/run_smoke_openvocab.sh || true",
        "CUDA_VISIBLE_DEVICES=6,7,8,9 PYTHONNOUSERSITE=1 bash experiments/rotation_semantic_attractor/scripts/run_smoke_matrix.sh",
        "```",
    ]
    out.write_text("\n".join(sections) + "\n")
    print(f"report={out}")


if __name__ == "__main__":
    main()
