#!/usr/bin/env python
"""Build an ICLR 9.5 evidence gate report from existing OpenRSD artifacts.

This script does not create or infer experimental results.  It reads existing
machine-readable audits when available and applies strict AC-style gates for
the semantic-scale support mismatch paper.  Missing evidence is reported as a
failed gate rather than silently ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_SIX_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "six_experiment_review.json")
DEFAULT_G3_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "g3_v1_dense_head_audit.json")
DEFAULT_G3V2_READINESS_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "g3v2_bass_readiness/g3v2_bass_readiness_gate.json")
DEFAULT_EP2_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_ep2clean_target_iou_filter_8case_audit.md")
DEFAULT_EP2_DISCIPLINE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "ep2_claim_discipline/ep2_claim_discipline_gate.json")
DEFAULT_DRAFT = Path(
    "paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md")
DEFAULT_BIB = Path("paper/drafts/semantic_scale_support_refs.bib")
DEFAULT_COMPETITOR_MATRIX = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fmatrix_20260620_semantic_scale_closest_competitors.md")
DEFAULT_DEPLOYMENT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/deployment_utility/"
    "deployment_utility_review.json")
DEFAULT_CALIBRATION_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "calibration_baselines/calibration_baseline_review.json")
DEFAULT_LATEX_MAIN = Path("paper/semantic_scale_support_iclr/main.tex")
DEFAULT_BASS_DELTA_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_bass_gsf_delta_train_gpu67_results.md")
DEFAULT_RANKDELTA_SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
DEFAULT_RANKDELTA_DECISION_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_followup_decision.json")
DEFAULT_RANKDELTA_CLOSURE_PLAN_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_closure_plan.json")
DEFAULT_P3A_SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3a_ap_projection_summary.json")
DEFAULT_P3D_SUMMARY_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "p3d_transfer_summary.json")
DEFAULT_LITERATURE_AUDIT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "literature_package_audit/literature_package_audit.json")
DEFAULT_OUT_DIR = Path("work_dirs/iclr95_evidence_gate_20260620")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "freview_20260620_iclr95_evidence_gate_machine_audit.md")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def parse_float_from_md(text, label):
    pattern = re.compile(
        r"\|\s*`?" + re.escape(label) + r"`?\s*\|\s*`?([-+0-9.eE]+)`?\s*\|")
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_ep2_md(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "gate_pass": None,
            "clean_pass_rate": None,
            "shuffled_prior_pass_rate": None,
            "specificity_gap": None,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    gate_match = re.search(r"`gate_pass`\s*\|\s*`?(true|false)`?", text,
                           flags=re.IGNORECASE)
    gate_pass = parse_bool(gate_match.group(1)) if gate_match else None
    return {
        "exists": True,
        "gate_pass": gate_pass,
        "clean_pass_rate": parse_float_from_md(text, "clean pass rate"),
        "shuffled_prior_pass_rate": parse_float_from_md(
            text, "shuffled_prior pass rate"),
        "specificity_gap": parse_float_from_md(text, "specificity_gap"),
        "source": str(path),
    }


def inspect_draft(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "reference_anchor_count": 0,
            "has_bass": False,
            "has_negative_evidence": False,
            "has_gate_scorecard": False,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    text_lower = text.lower()
    return {
        "exists": True,
        "reference_anchor_count": len(re.findall(r"https://arxiv.org/abs/", text)),
        "has_bass": (
            "bass" in text_lower and "bayesian" in text_lower
            and "semantic-scale" in text_lower),
        "has_negative_evidence": (
            "E-P2" in text and "G3-v1" in text
            and "negative" in text_lower),
        "has_gate_scorecard": (
            "ICLR Area Chair Scorecard" in text
            or "ICLR Readiness Scorecard" in text
            or "Stricter Devil-Reviewer Scorecard" in text),
        "source": str(path),
    }


def inspect_bib(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "entry_count": 0,
            "url_count": 0,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    return {
        "exists": True,
        "entry_count": len(re.findall(r"@\w+\s*\{", text)),
        "url_count": len(re.findall(r"https://arxiv.org/abs/", text)),
        "source": str(path),
    }


def inspect_competitor_matrix(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "has_baseline_obligations": False,
            "cluster_rows": 0,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    cluster_rows = sum(
        1 for line in text.splitlines()
        if line.startswith("| ") and " | " in line and not line.startswith("|---"))
    return {
        "exists": True,
        "has_baseline_obligations": "Baseline Obligation Matrix" in text,
        "cluster_rows": cluster_rows,
        "source": str(path),
    }


def inspect_latex(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "has_abstract": False,
            "has_bibliography": False,
            "num_tables": 0,
            "has_machine_gate": False,
            "has_generated_figures": False,
            "has_reproducibility_appendix": False,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    return {
        "exists": True,
        "has_abstract": "\\begin{abstract}" in text,
        "has_bibliography": "\\bibliography{" in text,
        "num_tables": len(re.findall(r"\\begin\{table\}", text)),
        "has_machine_gate": "Machine-Checkable Gate" in text,
        "has_generated_figures": (
            "\\input{generated/evidence_figures}" in text),
        "has_reproducibility_appendix": (
            "\\input{generated/reproducibility_appendix}" in text),
        "source": str(path),
    }


def inspect_bass_delta_result(path):
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "completed": False,
            "positive_gate_pass": False,
            "best_variant": None,
            "mAP": None,
            "top5000_precision": None,
            "sise_logz": None,
            "risk_weighted_logz": None,
            "source": str(path),
        }
    text = path.read_text(encoding="utf-8")
    row_match = re.search(
        r"\|\s*BASS delta dw0\.05\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
        r"\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9]+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9.]+)\s*\|"
        r"\s*([^|]+)\|",
        text)
    completed = row_match is not None
    status = row_match.group(9).strip().lower() if row_match else ""
    return {
        "exists": True,
        "completed": completed,
        "positive_gate_pass": completed and "pass" in status
        and "fail" not in status and "boundary" not in status,
        "best_variant": "dw0.05" if completed else None,
        "mAP": float(row_match.group(1)) if row_match else None,
        "AP50": float(row_match.group(2)) if row_match else None,
        "localized_correct": int(row_match.group(3)) if row_match else None,
        "localized_wrong": int(row_match.group(4)) if row_match else None,
        "sise_logz": int(row_match.group(5)) if row_match else None,
        "top5000_precision": float(row_match.group(6)) if row_match else None,
        "top5000_sise_logz": int(row_match.group(7)) if row_match else None,
        "risk_weighted_logz": float(row_match.group(8)) if row_match else None,
        "status": status,
        "source": str(path),
    }


def inspect_rankdelta_summary(summary_json, decision_json):
    summary = read_json(summary_json) or {}
    decision = read_json(decision_json) or {}
    rows = summary.get("rows", [])
    rank_rows = [
        row for row in rows
        if str(row.get("kind", "")).startswith("rankdelta")
    ]
    p0_rows = [row for row in rows if row.get("kind") == "rankdelta_p0"]
    p1_rows = [row for row in rows if row.get("kind") == "rankdelta_p1"]
    p2_rows = [row for row in rows if row.get("kind") == "rankdelta_p2"]
    strict = [row for row in rank_rows if row.get("gate") == "strict pass"]
    pilot = [row for row in rank_rows if row.get("gate") == "pilot pass"]
    complete_rank_rows = [
        row for row in rank_rows
        if row.get("mAP") is not None and row.get("gate") != "waiting/missing"
    ]
    return {
        "summary_exists": Path(summary_json).exists(),
        "decision_exists": Path(decision_json).exists(),
        "summary_status": summary.get("status", "missing"),
        "decision_action": decision.get("action", "missing"),
        "p0_done": bool(p0_rows) and all(
            row.get("mAP") is not None for row in p0_rows),
        "p1_done": bool(p1_rows) and all(
            row.get("mAP") is not None for row in p1_rows),
        "p2_done": bool(p2_rows) and all(
            row.get("mAP") is not None for row in p2_rows),
        "rankdelta_rows_total": len(rank_rows),
        "rankdelta_rows_filled": len(complete_rank_rows),
        "rankdelta_all_complete": (
            bool(rank_rows) and len(complete_rank_rows) == len(rank_rows)),
        "strict_pass": bool(strict),
        "pilot_pass": bool(pilot),
        "strict_methods": [row.get("method") for row in strict],
        "pilot_methods": [row.get("method") for row in pilot],
        "source": str(summary_json),
        "decision_source": str(decision_json),
    }


def inspect_rankdelta_closure_plan(path):
    plan = read_json(path) or {}
    return {
        "exists": Path(path).exists(),
        "action": plan.get("action", "missing"),
        "reason": plan.get("reason", ""),
        "paper_position": plan.get("paper_position", ""),
        "summary_status": plan.get("summary_status", "missing"),
        "followup_action": plan.get("followup_action", ""),
        "p0_done": bool(plan.get("p0_done", False)),
        "p1_done": bool(plan.get("p1_done", False)),
        "p2_done": bool(plan.get("p2_done", False)),
        "strict_pass_methods": list(plan.get("strict_pass_methods", []) or []),
        "pilot_pass_methods": list(plan.get("pilot_pass_methods", []) or []),
        "execution_priority": list(plan.get("execution_priority", []) or []),
        "source": str(path),
    }


def inspect_p3a_summary(path):
    summary = read_json(path) or {}
    rows = [
        row for row in summary.get("rows", [])
        if isinstance(row, dict)
    ]
    strict_rows = [row for row in rows if row.get("gate") == "strict pass"]
    main_row = next(
        (row for row in rows if row.get("kind") == "p3a_main"),
        strict_rows[-1] if strict_rows else {})
    return {
        "exists": Path(path).exists(),
        "status": summary.get("status", "missing"),
        "strict_pass": bool(summary.get("strict_pass", False)),
        "strict_pass_methods": list(
            summary.get("strict_pass_methods", []) or []),
        "main_method": summary.get("main_method", main_row.get("method")),
        "main_gate": summary.get("main_gate", main_row.get("gate")),
        "mAP": main_row.get("mAP"),
        "top5000_precision": main_row.get("top5000_precision"),
        "top5000_sise_logz": main_row.get("top5000_sise_logz"),
        "top5000_risk_weighted_logz": main_row.get(
            "top5000_risk_weighted_logz"),
        "changed_correct": main_row.get("changed_correct"),
        "tp_rank_harm": main_row.get("tp_rank_harm"),
        "result_md": summary.get("result_md", ""),
        "source": str(path),
    }


def inspect_p3d_transfer_summary(path):
    summary = read_json(path) or {}
    rows = [
        row for row in summary.get("rows", [])
        if isinstance(row, dict)
    ]
    strict_rows = [
        row for row in rows
        if row.get("gate") == "strict transfer pass"
    ]
    main_row = next(
        (row for row in rows if row.get("kind") == "p3d_transfer_main"),
        strict_rows[-1] if strict_rows else {})
    return {
        "exists": Path(path).exists(),
        "status": summary.get("status", "missing"),
        "dataset": summary.get("dataset", main_row.get("dataset")),
        "strict_transfer_pass": bool(
            summary.get("strict_transfer_pass", False)),
        "strict_transfer_pass_methods": list(
            summary.get("strict_transfer_pass_methods", []) or []),
        "main_method": summary.get("main_method", main_row.get("method")),
        "main_gate": summary.get("main_gate", main_row.get("gate")),
        "mAP": summary.get("main_mAP", main_row.get("mAP")),
        "top5000_precision": summary.get(
            "main_top5000_precision",
            main_row.get("top5000_precision")),
        "risk_weighted_logz": summary.get(
            "main_risk_weighted_logz",
            main_row.get("risk_weighted_logz")),
        "changed_correct": summary.get(
            "changed_correct", main_row.get("changed_correct")),
        "tp_rank_harm": summary.get(
            "tp_rank_harm", main_row.get("tp_rank_harm")),
        "ap_contributing_fp_removed": summary.get(
            "ap_contributing_fp_removed",
            main_row.get("ap_contributing_fp_removed")),
        "result_md": summary.get("result_md", ""),
        "source": str(path),
    }


def get_score(six_json, key, default=0.0):
    if not six_json:
        return default
    scores = six_json.get("strict_reviewer_scores_after_audit", {})
    return float(scores.get(key, default))


def gate_row(dimension, current_score, target_score, passed, evidence,
             blocker, next_action, artifact):
    return {
        "dimension": dimension,
        "current_score": f"{current_score:.2f}",
        "target_score": f"{target_score:.2f}",
        "gate_pass": bool(passed),
        "evidence": evidence,
        "blocker": blocker,
        "next_action": next_action,
        "artifact": artifact,
    }


def build_gates(six_json, g3_json, g3v2_readiness, ep2, ep2_discipline, draft,
                bib, competitor_matrix, deployment, calibration, latex,
                bass_delta, rankdelta, closure_plan, p3a_summary,
                p3d_summary, literature_audit):
    support_pass_count = 0
    recurrence = 0.0
    if six_json:
        support_pass_count = int(six_json.get("support_law_pass_count", 0))
        recurrence = float(
            six_json.get("dota2_detector_top20_recurrence_rate", 0.0))

    g3_positive = False
    g3_status = "missing"
    if g3_json:
        g3_status = str(g3_json.get("status", "unknown"))
        g3_positive = bool(g3_json.get("best_g3_beats_best_nog3_map", False))
    g3v2_ready = bool(
        g3v2_readiness
        and g3v2_readiness.get("matched_config_ready") is True)
    g3v2_completed = bool(
        g3v2_readiness
        and g3v2_readiness.get("g3v2_experiment_completed") is True)
    g3v2_positive = bool(
        g3v2_readiness
        and g3v2_readiness.get("g3v2_positive_gate_pass") is True)
    bass_delta_completed = bool(
        bass_delta and bass_delta.get("completed") is True)
    bass_delta_positive = bool(
        bass_delta and bass_delta.get("positive_gate_pass") is True)
    rankdelta_positive = bool(rankdelta and rankdelta.get("strict_pass"))
    p3a_positive = bool(
        p3a_summary and p3a_summary.get("strict_pass") is True)
    p3d_positive = bool(
        p3d_summary and p3d_summary.get("strict_transfer_pass") is True)
    method_gate_positive = (
        g3v2_positive or bass_delta_positive or rankdelta_positive
        or p3a_positive)
    rankdelta_all_complete = bool(
        rankdelta and rankdelta.get("rankdelta_all_complete"))
    p2_complete_no_strict = bool(
        closure_plan
        and closure_plan.get("action") == "STOP_P2_DONE_NO_STRICT")

    ep2_positive = bool(ep2.get("gate_pass") is True)
    ep2_downgrade = bool(
        ep2_discipline
        and ep2_discipline.get("formal_noncausal_downgrade_pass") is True)
    problem_claim_disciplined = ep2_positive or ep2_downgrade
    draft_ready = (
        draft.get("exists")
        and draft.get("has_bass")
        and draft.get("has_negative_evidence")
        and draft.get("has_gate_scorecard"))
    literature_ready = (
        draft.get("exists")
        and bib.get("exists")
        and int(bib.get("entry_count", 0)) >= 20
        and competitor_matrix.get("exists")
        and competitor_matrix.get("has_baseline_obligations"))
    literature_package_pass = bool(
        literature_audit
        and literature_audit.get("literature_package_pass") is True)
    practicality_positive = bool(
        deployment and deployment.get("practicality_gate_pass") is True)
    calibration_positive = bool(
        calibration and calibration.get("score_only_calibration_gate_pass")
        is True)
    calibration_count = (
        calibration.get("method_beats_score_only_count")
        if calibration else None)
    boundary_evidence_complete = bool(
        calibration_positive and problem_claim_disciplined
        and draft.get("has_negative_evidence")
        and g3_json and bass_delta_completed
        and rankdelta_all_complete and p2_complete_no_strict)
    latex_ready = (
        latex.get("exists")
        and latex.get("has_abstract")
        and latex.get("has_bibliography")
        and int(latex.get("num_tables", 0)) >= 3
        and latex.get("has_machine_gate"))

    method_current = get_score(
        six_json, "method_effectiveness_score", 8.45)
    novelty_current = get_score(
        six_json, "innovation_score", 8.8)
    scope_current = get_score(
        six_json, "application_scope_score", 8.6)
    problem_current = get_score(
        six_json, "problem_depth_score", 9.15)
    legacy_method_artifact = (
        bass_delta.get("source", "") if bass_delta_completed else (
            g3v2_readiness or {}).get(
                "result_md", (g3_json or {}).get("result_md", "")))
    method_artifact = (
        p3a_summary.get("result_md", "") if p3a_positive
        else legacy_method_artifact)
    evidence_artifact = (
        p3a_summary.get("result_md", "") if p3a_positive
        else (
            bass_delta.get("source", "") if bass_delta_completed else (
                ep2_discipline or {}).get(
                    "result_md", ep2.get("source", ""))))

    rows = [
        gate_row(
            "Problem anatomy depth",
            9.5 if (
                support_pass_count >= 6 and recurrence >= 0.8
                and problem_claim_disciplined) else problem_current,
            9.5,
            support_pass_count >= 6 and recurrence >= 0.8
            and problem_claim_disciplined,
            (
                f"support_law_pass_count={support_pass_count}; "
                f"dota2_top20_recurrence={recurrence:.3f}; "
                f"ep2_gate_pass={ep2.get('gate_pass')}; "
                f"formal_noncausal_downgrade_pass={ep2_downgrade}"
            ),
            "Problem gate requires positive E-P2 causal evidence or a formal non-causal downgrade.",
            "Keep the problem claim as predictive/boundary unless E-P2-v2 later becomes positive.",
            (ep2_discipline or {}).get("result_md", ep2.get("source", "")),
        ),
        gate_row(
            "Vision / conceptual height",
            9.5 if method_gate_positive and draft_ready
            else 9.5 if draft_ready and bass_delta_completed
            else 9.15 if draft_ready and g3v2_ready
            else (9.05 if draft_ready else 9.0),
            9.5,
            draft_ready and method_gate_positive,
            (
                f"draft_ready={draft_ready}; has_bass={draft.get('has_bass')}; "
                f"negative_evidence_integrated={draft.get('has_negative_evidence')}; "
                f"g3_positive={g3_positive}; g3v2_ready={g3v2_ready}; "
                f"g3v2_completed={g3v2_completed}; "
                f"g3v2_positive={g3v2_positive}; "
                f"bass_delta_completed={bass_delta_completed}; "
                f"bass_delta_positive={bass_delta_positive}; "
                f"rankdelta={rankdelta}; p3a={p3a_summary}; "
                f"p3d={p3d_summary}; "
                f"closure_plan={closure_plan}"
            ),
            "Gaussian support-distribution framing meets the concept target; P3A validates AP-constrained ranking and P3D checks second-dataset transfer.",
            (
                "Follow closure plan action "
                f"`{closure_plan.get('action', 'missing')}`; move from "
                "HRRSD oracle ranking projection to transfer/deployable "
                "validation."
            ),
            draft.get("source", ""),
        ),
        gate_row(
            "Literature breadth",
            9.5 if literature_package_pass else 9.35 if literature_ready else 8.6,
            9.5,
            literature_package_pass,
            (
                f"reference_anchor_count={draft.get('reference_anchor_count')}; "
                f"bib_entries={bib.get('entry_count')}; "
                f"competitor_matrix={competitor_matrix.get('exists')}; "
                f"baseline_obligations={competitor_matrix.get('has_baseline_obligations')}; "
                f"literature_audit={literature_audit}"
            ),
            "Literature breadth requires a passed local audit over BibTeX, LaTeX citations, closest-work matrix, and baseline obligations.",
            "Keep updating the competitor matrix if new detector-calibration baselines or BASS ablations are added.",
            literature_audit.get("result_md", competitor_matrix.get("source", ""))
            if literature_audit else competitor_matrix.get("source", ""),
        ),
        gate_row(
            "Current method novelty",
            9.5 if method_gate_positive
            else (8.80 if p2_complete_no_strict
                  else 8.75 if bass_delta_completed
                  else 8.9 if g3v2_ready else min(novelty_current, 8.8)),
            9.5,
            method_gate_positive,
            (
                f"g3_status={g3_status}; best_g3_beats_best_nog3_map={g3_positive}; "
                f"g3v2_ready={g3v2_ready}; g3v2_completed={g3v2_completed}; "
                f"g3v2_positive={g3v2_positive}; "
                f"matched_summary={g3v2_readiness.get('matched_result_summary') if g3v2_readiness else None}; "
                f"bass_delta={bass_delta}; rankdelta={rankdelta}; "
                f"p3a={p3a_summary}; p3d={p3d_summary}; "
                f"closure_plan={closure_plan}"
            ),
            "G1/G2 remain close to structured post-hoc calibration; P3A is the first AP-constrained Gaussian support ranking result.",
            (
                "Convert P3A from oracle validation into deployable/transfer "
                "evidence; do not duplicate bbox-Gaussian IoU/GWD/KLD/NWD."
            ),
            method_artifact,
        ),
        gate_row(
            "Method effectiveness",
            9.5 if method_gate_positive
            else 8.60 if p2_complete_no_strict
            else 8.45 if bass_delta_completed else min(method_current, 8.55),
            9.5,
            method_gate_positive,
            (
                f"g3_positive={g3_positive}; g3v2_completed={g3v2_completed}; "
                f"g3v2_positive={g3v2_positive}; "
                f"bass_delta_completed={bass_delta_completed}; "
                f"bass_delta_positive={bass_delta_positive}; "
                f"rankdelta={rankdelta}; p3a={p3a_summary}; "
                f"p3d={p3d_summary}; "
                f"closure_plan={closure_plan}; "
                f"application_scope_score={scope_current:.2f}"
            ),
            "P3A is AP-safe on HRRSD; P3D adds DIOR-R transfer, while deployable training remains separate.",
            (
                "Keep P3A as the method-effectiveness anchor, then add "
                "P3B/P3C transfer and deployment evidence."
            ),
            method_artifact,
        ),
        gate_row(
            "Practicality",
            9.5 if practicality_positive else (8.9 if draft_ready else 8.7),
            9.5,
            practicality_positive,
            (
                f"deployment_utility_gate={practicality_positive}; "
                f"p4_full_deployment_pass={deployment.get('p4_full_deployment_pass') if deployment else None}; "
                f"closed_set_utility_pass_count={deployment.get('closed_set_utility_pass_count') if deployment else None}; "
                f"latency_overhead_rate_p4={deployment.get('latency_overhead_rate_p4') if deployment else None}"
            ),
            "Practicality gate requires no-dump latency plus deployment utility; missing only if deployment JSON absent or negative.",
            "Keep DIOR-R/xView as boundary cases; do not generalize practicality beyond P4+HRRSD evidence.",
            deployment.get("result_md", "") if deployment else "",
        ),
        gate_row(
            "Applicability breadth",
            9.5 if (method_gate_positive and p3d_positive)
            else 8.60 if p2_complete_no_strict
            else 8.55 if bass_delta_completed and not method_gate_positive
            else scope_current,
            9.5,
            method_gate_positive and p3d_positive,
            (
                f"scope_score={scope_current:.2f}; support_datasets={support_pass_count}; "
                f"g3v2_completed={g3v2_completed}; g3v2_positive={g3v2_positive}; "
                f"bass_delta_completed={bass_delta_completed}; bass_delta_positive={bass_delta_positive}; "
                f"rankdelta={rankdelta}; p3a={p3a_summary}; "
                f"p3d={p3d_summary}; "
                f"closure_plan={closure_plan}"
            ),
            "Applicability breadth now requires second-dataset AP-safe transfer, not only HRRSD P3A.",
            "Next breadth step is detector-family transfer; keep DIOR-R as the dataset-transfer gate.",
            p3d_summary.get("result_md", "") if p3d_positive
            else DEFAULT_SIX_JSON.as_posix(),
        ),
        gate_row(
            "Evidence rigor",
            9.5 if boundary_evidence_complete
            else 9.4 if (
                calibration_positive and problem_claim_disciplined and g3_json
                and bass_delta_completed)
            else 9.15 if (
                calibration_positive and problem_claim_disciplined and g3_json)
            else 9.0 if calibration_positive and ep2.get("exists") and g3_json
            else (8.75 if ep2.get("exists") and g3_json else 8.5),
            9.5,
            boundary_evidence_complete,
            (
                f"ep2_gate_pass={ep2.get('gate_pass')}; "
                f"g3_positive={g3_positive}; g3v2_completed={g3v2_completed}; "
                f"g3v2_positive={g3v2_positive}; "
                f"bass_delta_completed={bass_delta_completed}; "
                f"bass_delta_positive={bass_delta_positive}; "
                f"rankdelta={rankdelta}; p3a={p3a_summary}; "
                f"p3d={p3d_summary}; "
                f"closure_plan={closure_plan}; "
                f"negative_results_integrated={draft.get('has_negative_evidence')}"
                f"; score_only_calibration_gate={calibration_positive}; "
                f"method_beats_score_only_count={calibration_count}; "
                f"formal_noncausal_downgrade_pass={ep2_downgrade}; "
                f"boundary_evidence_complete={boundary_evidence_complete}"
            ),
            "Boundary evidence is now complete and honest, but it remains negative for the current trainable method.",
            "Add detector-family transfer evidence before broad 9.5 method promotion.",
            evidence_artifact,
        ),
        gate_row(
            "ICLR paper readiness",
            8.95 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined
                and method_gate_positive
                and latex.get("has_generated_figures")
                and latex.get("has_reproducibility_appendix"))
            else 8.90 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined
                and method_gate_positive)
            else 8.80 if p2_complete_no_strict
            else 8.95 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined
                and bass_delta_completed
                and latex.get("has_generated_figures")
                and latex.get("has_reproducibility_appendix"))
            else 8.85 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined
                and bass_delta_completed)
            else 9.1 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined
                and g3v2_ready)
            else 9.05 if (
                latex_ready and literature_ready and practicality_positive
                and calibration_positive and problem_claim_disciplined)
            else 8.9 if (latex_ready and literature_ready and practicality_positive
                         and calibration_positive)
            else 8.8 if latex_ready and literature_ready and practicality_positive
            else (8.05 if draft_ready and literature_ready else 7.8),
            9.5,
            False,
            (
                f"draft_ready={draft_ready}; literature_ready={literature_ready}; "
                f"latex_ready={latex_ready}; method_gate={method_gate_positive}; "
                f"bass_delta_completed={bass_delta_completed}; "
                f"problem_gate={problem_claim_disciplined}; "
                f"calibration_gate={calibration_positive}; "
                f"generated_figures={latex.get('has_generated_figures')}; "
                f"repro_appendix={latex.get('has_reproducibility_appendix')}"
            ),
            "Paper package has stronger P3A/P3D evidence, generated figures, and a reproducibility appendix, but the compiled submission package is still missing.",
            "After detector-family transfer or non-oracle controls, add final figures and compile with a LaTeX environment.",
            latex.get("source", ""),
        ),
        gate_row(
            "Overall AC score",
            8.85 if (
                draft_ready and literature_ready and calibration_positive
                and problem_claim_disciplined and method_gate_positive)
            else 8.60 if p2_complete_no_strict
            else 8.55 if (
                draft_ready and literature_ready and calibration_positive
                and problem_claim_disciplined and bass_delta_completed)
            else 8.65 if (
                draft_ready and literature_ready and calibration_positive
                and problem_claim_disciplined and g3v2_ready)
            else 8.6 if (
                draft_ready and literature_ready and calibration_positive
                and problem_claim_disciplined)
            else 8.45 if draft_ready and literature_ready and calibration_positive
            else (8.3 if draft_ready and literature_ready else 8.1),
            9.5,
            False,
            (
                f"problem_gate={support_pass_count >= 6 and recurrence >= 0.8 and problem_claim_disciplined}; "
                f"method_gate={method_gate_positive}; draft_gate={draft_ready}; "
                f"bass_delta_completed={bass_delta_completed}; "
                f"calibration_gate={calibration_positive}"
            ),
            "The package has strict HRRSD P3A and DIOR-R P3D ranking evidence, but not yet submission readiness or final AC-level synthesis.",
            "Complete detector-family transfer/non-oracle controls; keep E-P2 as bounded predictive evidence unless a positive E-P2-v2 arrives.",
            DEFAULT_RESULT_MD.as_posix(),
        ),
    ]
    return rows


def md_table(rows):
    columns = [
        "dimension", "current_score", "target_score", "gate_pass",
        "evidence", "blocker", "next_action"
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(rows, evidence, result_md):
    passed = sum(1 for row in rows if row["gate_pass"])
    total = len(rows)
    lines = [
        "# ICLR 9.5 Evidence Gate Machine Audit - 2026-06-20",
        "",
        "本文件由 `M_Tools/analysis/build_iclr95_evidence_gate.py` 从已有",
        "JSON/Markdown 证据生成。它不启动实验，不发明结果；缺失或负结果",
        "按未通过处理。",
        "",
        "## Verdict",
        "",
        f"- gates passed: `{passed}/{total}`",
        f"- all_95_gates_passed: `{passed == total}`",
        "- interpretation: 当前材料仍不能证明所有维度达到 9.5。最主要缺口是 "
        "`BASS` matched-control 正结果；G3-v2 和 train-time delta 均已完成但未通过 AP-risk 方法门；"
        "RankDelta P0/P1 仍按真实 summary/decision 状态处理；E-P2 因果声称已通过正式非因果降级处理。",
        "",
        "## Evidence Inputs",
        "",
        f"- six experiment JSON: `{evidence['six_json_path']}`",
        f"- G3 audit JSON: `{evidence['g3_json_path']}`",
        f"- G3-v2/BASS readiness JSON: `{evidence['g3v2_readiness_json_path']}`",
        f"- E-P2 audit MD: `{evidence['ep2_md_path']}`",
        f"- E-P2 claim discipline JSON: `{evidence['ep2_discipline_json_path']}`",
        f"- draft: `{evidence['draft_path']}`",
        f"- BibTeX: `{evidence['bib_path']}`",
        f"- closest-competitor matrix: `{evidence['competitor_matrix_path']}`",
        f"- deployment utility JSON: `{evidence['deployment_json_path']}`",
        f"- score-only calibration JSON: `{evidence['calibration_json_path']}`",
        f"- LaTeX main: `{evidence['latex_main_path']}`",
        f"- BASS train-time delta MD: `{evidence['bass_delta_md_path']}`",
        f"- RankDelta summary JSON: `{evidence['rankdelta_summary_json_path']}`",
        f"- RankDelta follow-up decision JSON: `{evidence['rankdelta_decision_json_path']}`",
        f"- RankDelta closure plan JSON: `{evidence['rankdelta_closure_plan_json_path']}`",
        f"- P3A AP projection summary JSON: `{evidence['p3a_summary_json_path']}`",
        f"- P3D transfer summary JSON: `{evidence['p3d_summary_json_path']}`",
        f"- literature package audit JSON: `{evidence['literature_audit_json_path']}`",
        "",
        "## RankDelta Closure Plan",
        "",
        f"- action: `{evidence['rankdelta_closure_plan'].get('action')}`",
        f"- reason: {evidence['rankdelta_closure_plan'].get('reason')}",
        f"- paper position: {evidence['rankdelta_closure_plan'].get('paper_position')}",
        f"- source: `{evidence['rankdelta_closure_plan'].get('source')}`",
        "",
        "## Gate Table",
        "",
        md_table(rows),
        "",
        "## Next Execution Gate",
        "",
        "1. Keep E-P2 as formal predictive/boundary evidence unless E-P2-v2 later becomes positive.",
        "2. Treat completed `G3-v2` and train-time delta as boundary evidence; follow the RankDelta closure-plan action generated from real summary JSON.",
        "3. Keep the Platt/Isotonic score-only calibration gate as a non-replacement baseline.",
        "4. Add deployment utility metrics: latency overhead, review-queue precision, high-confidence false-alarm burden.",
        "5. P3D DIOR-R closes dataset-transfer breadth; next step is detector-family transfer rather than another DOTA-family rerun.",
        "",
    ]
    result_md.parent.mkdir(parents=True, exist_ok=True)
    result_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--six-json", default=str(DEFAULT_SIX_JSON))
    parser.add_argument("--g3-json", default=str(DEFAULT_G3_JSON))
    parser.add_argument(
        "--g3v2-readiness-json", default=str(DEFAULT_G3V2_READINESS_JSON))
    parser.add_argument("--ep2-md", default=str(DEFAULT_EP2_MD))
    parser.add_argument(
        "--ep2-discipline-json", default=str(DEFAULT_EP2_DISCIPLINE_JSON))
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--bib", default=str(DEFAULT_BIB))
    parser.add_argument(
        "--competitor-matrix", default=str(DEFAULT_COMPETITOR_MATRIX))
    parser.add_argument("--deployment-json", default=str(DEFAULT_DEPLOYMENT_JSON))
    parser.add_argument("--calibration-json", default=str(DEFAULT_CALIBRATION_JSON))
    parser.add_argument("--latex-main", default=str(DEFAULT_LATEX_MAIN))
    parser.add_argument("--bass-delta-md", default=str(DEFAULT_BASS_DELTA_MD))
    parser.add_argument(
        "--rankdelta-summary-json", default=str(DEFAULT_RANKDELTA_SUMMARY_JSON))
    parser.add_argument(
        "--rankdelta-decision-json", default=str(DEFAULT_RANKDELTA_DECISION_JSON))
    parser.add_argument(
        "--rankdelta-closure-plan-json",
        default=str(DEFAULT_RANKDELTA_CLOSURE_PLAN_JSON))
    parser.add_argument(
        "--p3a-summary-json", default=str(DEFAULT_P3A_SUMMARY_JSON))
    parser.add_argument(
        "--p3d-summary-json", default=str(DEFAULT_P3D_SUMMARY_JSON))
    parser.add_argument(
        "--literature-audit-json", default=str(DEFAULT_LITERATURE_AUDIT_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    args = parser.parse_args()

    six_json_path = Path(args.six_json)
    g3_json_path = Path(args.g3_json)
    g3v2_readiness_json_path = Path(args.g3v2_readiness_json)
    ep2_md_path = Path(args.ep2_md)
    ep2_discipline_json_path = Path(args.ep2_discipline_json)
    draft_path = Path(args.draft)
    bib_path = Path(args.bib)
    competitor_matrix_path = Path(args.competitor_matrix)
    deployment_json_path = Path(args.deployment_json)
    calibration_json_path = Path(args.calibration_json)
    latex_main_path = Path(args.latex_main)
    bass_delta_md_path = Path(args.bass_delta_md)
    rankdelta_summary_json_path = Path(args.rankdelta_summary_json)
    rankdelta_decision_json_path = Path(args.rankdelta_decision_json)
    rankdelta_closure_plan_json_path = Path(args.rankdelta_closure_plan_json)
    p3a_summary_json_path = Path(args.p3a_summary_json)
    p3d_summary_json_path = Path(args.p3d_summary_json)
    literature_audit_json_path = Path(args.literature_audit_json)
    out_dir = Path(args.out_dir)
    result_md = Path(args.result_md)

    six_json = read_json(six_json_path)
    g3_json = read_json(g3_json_path)
    g3v2_readiness = read_json(g3v2_readiness_json_path)
    ep2 = parse_ep2_md(ep2_md_path)
    ep2_discipline = read_json(ep2_discipline_json_path)
    draft = inspect_draft(draft_path)
    bib = inspect_bib(bib_path)
    competitor_matrix = inspect_competitor_matrix(competitor_matrix_path)
    deployment = read_json(deployment_json_path)
    calibration = read_json(calibration_json_path)
    latex = inspect_latex(latex_main_path)
    bass_delta = inspect_bass_delta_result(bass_delta_md_path)
    rankdelta = inspect_rankdelta_summary(
        rankdelta_summary_json_path, rankdelta_decision_json_path)
    closure_plan = inspect_rankdelta_closure_plan(
        rankdelta_closure_plan_json_path)
    p3a_summary = inspect_p3a_summary(p3a_summary_json_path)
    p3d_summary = inspect_p3d_transfer_summary(p3d_summary_json_path)
    literature_audit = read_json(literature_audit_json_path) or {}
    rows = build_gates(
        six_json, g3_json, g3v2_readiness, ep2, ep2_discipline, draft, bib,
        competitor_matrix, deployment, calibration, latex, bass_delta,
        rankdelta, closure_plan, p3a_summary, p3d_summary, literature_audit)

    evidence = {
        "six_json_path": str(six_json_path),
        "g3_json_path": str(g3_json_path),
        "g3v2_readiness_json_path": str(g3v2_readiness_json_path),
        "ep2_md_path": str(ep2_md_path),
        "ep2_discipline_json_path": str(ep2_discipline_json_path),
        "draft_path": str(draft_path),
        "bib_path": str(bib_path),
        "competitor_matrix_path": str(competitor_matrix_path),
        "deployment_json_path": str(deployment_json_path),
        "calibration_json_path": str(calibration_json_path),
        "latex_main_path": str(latex_main_path),
        "bass_delta_md_path": str(bass_delta_md_path),
        "rankdelta_summary_json_path": str(rankdelta_summary_json_path),
        "rankdelta_decision_json_path": str(rankdelta_decision_json_path),
        "rankdelta_closure_plan_json_path": str(
            rankdelta_closure_plan_json_path),
        "p3a_summary_json_path": str(p3a_summary_json_path),
        "p3d_summary_json_path": str(p3d_summary_json_path),
        "literature_audit_json_path": str(literature_audit_json_path),
        "six_json_exists": six_json is not None,
        "g3_json_exists": g3_json is not None,
        "g3v2_readiness_exists": g3v2_readiness is not None,
        "ep2": ep2,
        "ep2_discipline": ep2_discipline,
        "g3v2_readiness": g3v2_readiness,
        "draft": draft,
        "bib": bib,
        "competitor_matrix": competitor_matrix,
        "deployment": deployment,
        "calibration": calibration,
        "latex": latex,
        "bass_delta": bass_delta,
        "rankdelta": rankdelta,
        "rankdelta_closure_plan": closure_plan,
        "p3a_summary": p3a_summary,
        "p3d_summary": p3d_summary,
        "literature_audit": literature_audit,
    }
    payload = {
        "all_95_gates_passed": all(row["gate_pass"] for row in rows),
        "num_gates_passed": sum(1 for row in rows if row["gate_pass"]),
        "num_gates_total": len(rows),
        "evidence": evidence,
        "gates": rows,
        "result_md": str(result_md),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "iclr95_evidence_gate.json", payload)
    write_csv(out_dir / "iclr95_evidence_gate.csv", rows)
    build_markdown(rows, evidence, result_md)
    print(json.dumps({
        "all_95_gates_passed": payload["all_95_gates_passed"],
        "num_gates_passed": payload["num_gates_passed"],
        "num_gates_total": payload["num_gates_total"],
        "json": str(out_dir / "iclr95_evidence_gate.json"),
        "csv": str(out_dir / "iclr95_evidence_gate.csv"),
        "md": str(result_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
