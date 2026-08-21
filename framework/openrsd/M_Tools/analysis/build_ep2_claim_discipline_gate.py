#!/usr/bin/env python
"""Audit whether E-P2 is honestly downgraded from causal to predictive claim.

This script is intentionally not an experiment.  It verifies a paper-discipline
route: if the strict E-P2 path intervention is negative, the paper may still
have a strong problem-anatomy claim only if the manuscript explicitly reframes
the claim as predictive/boundary evidence and avoids unqualified causal claims.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_EP2_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_ep2clean_target_iou_filter_8case_audit.md")
DEFAULT_DRAFT = Path(
    "paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md")
DEFAULT_LATEX_MAIN = Path("paper/semantic_scale_support_iclr/main.tex")
DEFAULT_OUT_DIR = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "ep2_claim_discipline")
DEFAULT_RESULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_ep2_claim_discipline_gate.md")

GUARD_WORDS = (
    "not", "no ", "failed", "fail", "negative", "insufficient", "missing",
    "cannot", "does not", "do not", "remain", "open", "boundary",
    "downgrade", "requires", "require", "depend", "pending", "without",
    "rather than", "not yet",
)

REQUIRED_DRAFT_PHRASES = (
    "predictive",
    "boundary",
    "does not pass",
    "not a causal proof",
    "diagnostic",
)

REQUIRED_LATEX_PHRASES = (
    "predictive",
    "boundary",
    "does not pass",
    "diagnostic",
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
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
    return {
        "exists": True,
        "gate_pass": parse_bool(gate_match.group(1)) if gate_match else None,
        "clean_pass_rate": parse_float_from_md(text, "clean pass rate"),
        "shuffled_prior_pass_rate": parse_float_from_md(
            text, "shuffled_prior pass rate"),
        "specificity_gap": parse_float_from_md(text, "specificity_gap"),
        "source": str(path),
    }


def read_text(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def find_missing_phrases(text, phrases):
    low = text.lower()
    return [phrase for phrase in phrases if phrase.lower() not in low]


def unqualified_causal_mentions(text):
    low = text.lower()
    bad = []
    for match in re.finditer(r"\bcausal(?:ly|ity)?\b", low):
        start = max(0, match.start() - 120)
        end = min(len(low), match.end() + 80)
        window = low[start:end]
        guarded = any(word in window for word in GUARD_WORDS)
        if not guarded:
            bad.append({
                "offset": match.start(),
                "context": text[start:end].replace("\n", " "),
            })
    return bad


def inspect_text(name, path, required_phrases):
    text = read_text(path)
    missing = find_missing_phrases(text, required_phrases)
    bad_causal = unqualified_causal_mentions(text)
    return {
        "name": name,
        "path": str(path),
        "exists": bool(text),
        "missing_required_phrases": missing,
        "unqualified_causal_mentions": bad_causal,
        "has_predictive_framing": "predictive" in text.lower(),
        "has_boundary_framing": "boundary" in text.lower(),
        "has_negative_ep2": (
            "e-p2" in text.lower()
            and ("does not pass" in text.lower()
                 or "not a causal proof" in text.lower()
                 or "fail" in text.lower())),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def build_markdown(path, payload):
    ep2 = payload["ep2"]
    draft = payload["draft"]
    latex = payload["latex"]
    lines = [
        "# E-P2 Claim Discipline Gate - 2026-06-20",
        "",
        "## 结论",
        "",
        f"- `ep2_claim_discipline_gate_pass`: `{str(payload['ep2_claim_discipline_gate_pass']).lower()}`",
        f"- `formal_noncausal_downgrade_pass`: `{str(payload['formal_noncausal_downgrade_pass']).lower()}`",
        f"- `ep2_gate_pass`: `{ep2.get('gate_pass')}`",
        f"- `unqualified_causal_mentions_total`: `{payload['unqualified_causal_mentions_total']}`",
        "",
        "这个 gate 不把 E-P2 负结果改写成正因果证据。它只检查论文是否已经",
        "把中心主张严格降级为 predictive / boundary / diagnostic evidence，",
        "并避免未限定的 causal overclaim。",
        "",
        "## E-P2 Evidence",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| `clean_pass_rate` | `{ep2.get('clean_pass_rate')}` |",
        f"| `shuffled_prior_pass_rate` | `{ep2.get('shuffled_prior_pass_rate')}` |",
        f"| `specificity_gap` | `{ep2.get('specificity_gap')}` |",
        "",
        "## Manuscript Checks",
        "",
        "| artifact | exists | missing_required_phrases | unqualified_causal_mentions |",
        "|---|---:|---|---:|",
        (
            f"| draft | `{draft['exists']}` | "
            f"`{', '.join(draft['missing_required_phrases']) or 'none'}` | "
            f"`{len(draft['unqualified_causal_mentions'])}` |"
        ),
        (
            f"| latex | `{latex['exists']}` | "
            f"`{', '.join(latex['missing_required_phrases']) or 'none'}` | "
            f"`{len(latex['unqualified_causal_mentions'])}` |"
        ),
        "",
        "## AC Interpretation",
        "",
        "- 若本 gate 通过，`Problem anatomy depth` 可以基于强 predictive law、",
        "detector-family recurrence、closed-set/OVD 覆盖和诚实边界降级达到 9.5。",
        "- 本 gate 不提升 `Method effectiveness` 或 `Current method novelty` 到 9.5；",
        "这些仍依赖 BASS matched-control 正结果；已完成的 G3-v2 是 boundary/negative。",
        "- 本 gate 明确保留 E-P2 为负结果，防止把 post-NMS scale-swap flip",
        "过度解释为因果机制。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep2-md", default=str(DEFAULT_EP2_MD))
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--latex-main", default=str(DEFAULT_LATEX_MAIN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--result-md", default=str(DEFAULT_RESULT_MD))
    args = parser.parse_args()

    ep2 = parse_ep2_md(Path(args.ep2_md))
    draft = inspect_text("draft", Path(args.draft), REQUIRED_DRAFT_PHRASES)
    latex = inspect_text("latex", Path(args.latex_main), REQUIRED_LATEX_PHRASES)

    ep2_negative_and_specific = (
        ep2.get("exists")
        and ep2.get("gate_pass") is False
        and ep2.get("clean_pass_rate") is not None
        and ep2.get("shuffled_prior_pass_rate") is not None
        and float(ep2.get("clean_pass_rate")) <= float(
            ep2.get("shuffled_prior_pass_rate"))
    )
    text_checks_pass = (
        draft["exists"]
        and latex["exists"]
        and not draft["missing_required_phrases"]
        and not latex["missing_required_phrases"]
        and not draft["unqualified_causal_mentions"]
        and not latex["unqualified_causal_mentions"]
        and draft["has_predictive_framing"]
        and draft["has_boundary_framing"]
        and draft["has_negative_ep2"]
        and latex["has_predictive_framing"]
        and latex["has_boundary_framing"]
        and latex["has_negative_ep2"]
    )
    gate_pass = bool(ep2_negative_and_specific and text_checks_pass)
    payload = {
        "ep2_claim_discipline_gate_pass": gate_pass,
        "formal_noncausal_downgrade_pass": gate_pass,
        "ep2_negative_and_specific": ep2_negative_and_specific,
        "text_checks_pass": text_checks_pass,
        "unqualified_causal_mentions_total": (
            len(draft["unqualified_causal_mentions"])
            + len(latex["unqualified_causal_mentions"])),
        "ep2": ep2,
        "draft": draft,
        "latex": latex,
        "result_md": str(Path(args.result_md)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "ep2_claim_discipline_gate.json", payload)
    build_markdown(Path(args.result_md), payload)
    print(json.dumps({
        "ep2_claim_discipline_gate_pass": gate_pass,
        "formal_noncausal_downgrade_pass": gate_pass,
        "json": str(out_dir / "ep2_claim_discipline_gate.json"),
        "md": str(Path(args.result_md)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
