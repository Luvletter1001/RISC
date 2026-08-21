#!/usr/bin/env python
"""Build paper figures from existing semantic-scale evidence artifacts.

This script is intentionally non-experimental.  It only reads existing JSON/CSV
evidence and renders publication-facing figures/tables.  Missing values remain
missing; no metric is inferred from checkpoints or logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_GATE_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json")
DEFAULT_SUPPORT_CSV = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "support_law_dataset_summary.csv")
DEFAULT_DEPLOYMENT_CSV = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "deployment_utility/deployment_utility_summary.csv")
DEFAULT_RANKDELTA_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_summary.json")
DEFAULT_OUT_DIR = Path("paper/semantic_scale_support_iclr/generated/figures")
DEFAULT_WORK_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_paper_figures_manifest.json")
DEFAULT_TEX = Path(
    "paper/semantic_scale_support_iclr/generated/evidence_figures.tex")
DEFAULT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fres_20260620_semantic_scale_paper_figures.md")

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value in {None, "", "None", "nan", "NaN", "TBD"}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def short_dataset_name(name: str) -> str:
    replacements = {
        "P4 OVD full preselect0.99": "P4 OVD",
        "HRRSD full": "HRRSD",
        "DIOR-R full": "DIOR-R",
        "ShipRS full": "ShipRS",
        "xView full": "xView",
        "DOTA2 full-val LSKNet": "DOTA2",
        "HRRSD closed-set": "HRRSD",
        "DIOR-R closed-set": "DIOR-R",
        "SHIPRS closed-set": "ShipRS",
    }
    return replacements.get(name, name.replace(" closed-set", ""))


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_gate_scores(gate: dict[str, Any], out_path: Path) -> dict[str, Any]:
    gates = gate.get("gates", [])
    rows = [
        row for row in gates
        if safe_float(row.get("current_score")) is not None
    ]
    if not rows:
        return {"created": False, "reason": "no gate rows"}

    labels = [row["dimension"] for row in rows]
    scores = [safe_float(row["current_score"]) or 0.0 for row in rows]
    passed = [str(row.get("gate_pass")) == "True" for row in rows]
    colors = [OKABE_ITO["green"] if ok else OKABE_ITO["orange"]
              for ok in passed]

    fig_h = max(4.2, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    y_pos = list(range(len(rows)))
    ax.barh(y_pos, scores, color=colors, alpha=0.88)
    ax.axvline(9.5, color=OKABE_ITO["red"], linestyle="--",
               linewidth=1.2, label="9.5 target")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(7.8, 10.0)
    ax.set_xlabel("strict current score")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ICLR 9.5 evidence gate: current score vs target")
    for y, score, ok in zip(y_pos, scores, passed):
        ax.text(score + 0.03, y, f"{score:.2f}" + (" pass" if ok else ""),
                va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "created": True,
        "path": str(out_path),
        "num_rows": len(rows),
        "num_passed": gate.get("num_gates_passed"),
        "num_total": gate.get("num_gates_total"),
    }


def plot_support_correlations(rows: list[dict[str, str]],
                              out_path: Path) -> dict[str, Any]:
    data = []
    for row in rows:
        rho = safe_float(row.get("rho_z2_vs_p0199"))
        if rho is not None:
            data.append((short_dataset_name(row.get("dataset", "")), rho))
    if not data:
        return {"created": False, "reason": "no support rows"}

    labels, values = zip(*data)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(labels, values, color=OKABE_ITO["blue"], alpha=0.9)
    ax.axhline(0.5, color=OKABE_ITO["red"], linestyle="--", linewidth=1.0,
               label="support-law gate")
    ax.set_ylim(0.0, max(0.9, max(values) + 0.08))
    ax.set_ylabel(r"Spearman $\rho$ for $z^2$ risk")
    ax.set_title("Semantic-scale support risk predicts high-confidence pair errors")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(fontsize=8)
    for idx, val in enumerate(values):
        ax.text(idx, val + 0.015, f"{val:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"created": True, "path": str(out_path), "num_rows": len(data)}


def _reduction_rate(base: float | None, method: float | None) -> float | None:
    if base is None or method is None or base <= 0:
        return None
    return (base - method) / base


def plot_deployment_utility(rows: list[dict[str, str]],
                            out_path: Path) -> dict[str, Any]:
    data = []
    for row in rows:
        label = short_dataset_name(row.get("dataset", ""))
        map_delta = safe_float(row.get("mAP_delta"))
        total_rate = _reduction_rate(
            safe_float(row.get("total_sise_base")),
            safe_float(row.get("total_sise_method")),
        )
        queue_rate = _reduction_rate(
            safe_float(row.get("review_queue_sise_base")),
            safe_float(row.get("review_queue_sise_method")),
        )
        if map_delta is not None and total_rate is not None:
            data.append((label, map_delta, total_rate, queue_rate))
    if not data:
        return {"created": False, "reason": "no deployment rows"}

    labels = [row[0] for row in data]
    map_delta = [row[1] for row in data]
    total_rate = [row[2] for row in data]
    queue_rate = [row[3] for row in data]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].bar(x, map_delta, color=OKABE_ITO["green"], alpha=0.9)
    axes[0].set_ylabel("mAP delta")
    axes[0].set_title("Deployment utility: AP non-regression and SISE reduction")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1].bar([i - 0.18 for i in x], total_rate, width=0.36,
                color=OKABE_ITO["blue"], alpha=0.9, label="total SISE")
    queue_vals = [val if val is not None else 0.0 for val in queue_rate]
    axes[1].bar([i + 0.18 for i in x], queue_vals, width=0.36,
                color=OKABE_ITO["purple"], alpha=0.85, label="queue SISE")
    axes[1].set_ylabel("reduction rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"created": True, "path": str(out_path), "num_rows": len(data)}


def plot_bass_boundary(rankdelta: dict[str, Any],
                       out_path: Path) -> dict[str, Any]:
    rows = rankdelta.get("rows", [])
    data = []
    for row in rows:
        map_value = safe_float(row.get("mAP"))
        risk_value = safe_float(row.get("risk_weighted_logz"))
        if map_value is not None and risk_value is not None:
            data.append({
                "method": row.get("method", ""),
                "kind": row.get("kind", ""),
                "gate": row.get("gate", ""),
                "mAP": map_value,
                "risk": risk_value,
            })
    if not data:
        return {"created": False, "reason": "no numeric BASS rows"}

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for item in data:
        if item["kind"] == "control":
            color = OKABE_ITO["black"]
            marker = "o"
        elif "delta" in item["method"].lower():
            color = OKABE_ITO["red"]
            marker = "X"
        elif "lite" in item["method"].lower():
            color = OKABE_ITO["green"]
            marker = "s"
        else:
            color = OKABE_ITO["orange"]
            marker = "^"
        ax.scatter(item["mAP"], item["risk"], s=64, color=color,
                   marker=marker, alpha=0.9)
        ax.text(item["mAP"] + 0.00012, item["risk"], item["method"],
                fontsize=7, va="center")
    ax.set_xlabel(r"mAP $\uparrow$")
    ax.set_ylabel(r"risk-weighted log-z $\downarrow$")
    ax.set_title("BASS-GSF boundary: lower SISE risk is not enough")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"created": True, "path": str(out_path), "num_rows": len(data)}


def write_tex_snippet(path: Path, figure_paths: dict[str, str]) -> None:
    rel = {
        key: Path(value).relative_to("paper/semantic_scale_support_iclr")
        for key, value in figure_paths.items()
    }
    lines = [
        "% Generated by M_Tools/analysis/build_semantic_scale_paper_figures.py",
        "% Do not edit figure paths or captions by hand; regenerate instead.",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.95\\linewidth]{{{rel['gate']}}}",
        "\\caption{ICLR 9.5 evidence gate generated from existing audits. "
        "Green bars pass the corresponding gate; orange bars remain blocked. "
        "The figure shows that problem anatomy, literature breadth, and "
        "practicality are at the target, while method and submission gates "
        "remain open.}",
        "\\label{fig:iclr95-gate}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.95\\linewidth]{{{rel['support']}}}",
        "\\caption{Semantic-scale support risk is predictive across OVD and "
        "closed-set remote-sensing settings.  Bars show the dataset-level "
        "correlation between continuous Gaussian support risk and "
        "high-confidence pair error.}",
        "\\label{fig:support-law}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.95\\linewidth]{{{rel['deployment']}}}",
        "\\caption{Deployment utility from existing result summaries.  The "
        "top panel shows AP deltas; the bottom panel shows total and "
        "review-queue SISE reduction rates where the denominator is nonzero.}",
        "\\label{fig:deployment-utility}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=0.95\\linewidth]{{{rel['bass']}}}",
        "\\caption{BASS-GSF boundary evidence.  The scatter plot uses only "
        "rows with real eval JSON plus deployment-risk summaries.  It shows "
        "why the current train-time delta result cannot be promoted to a "
        "9.5 method claim: reducing risk-weighted log-z must also preserve "
        "AP-sensitive ranking.}",
        "\\label{fig:bass-boundary}",
        "\\end{figure}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_md_record(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Semantic-Scale Paper Figures - 2026-06-20",
        "",
        "This file is generated by "
        "`M_Tools/analysis/build_semantic_scale_paper_figures.py`.",
        "It renders paper figures from existing JSON/CSV evidence only.",
        "",
        "| figure | created | path | rows |",
        "|---|---:|---|---:|",
    ]
    for key, item in manifest["figures"].items():
        lines.append(
            f"| {key} | `{item.get('created')}` | "
            f"`{item.get('path', '')}` | {item.get('num_rows', '')} |")
    lines += [
        "",
        "## No-Fabrication Rule",
        "",
        manifest["no_fabrication_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_figures(gate_json: Path = DEFAULT_GATE_JSON,
                  support_csv: Path = DEFAULT_SUPPORT_CSV,
                  deployment_csv: Path = DEFAULT_DEPLOYMENT_CSV,
                  rankdelta_json: Path = DEFAULT_RANKDELTA_JSON,
                  out_dir: Path = DEFAULT_OUT_DIR,
                  out_json: Path = DEFAULT_WORK_JSON,
                  out_tex: Path = DEFAULT_TEX,
                  out_md: Path = DEFAULT_MD) -> dict[str, Any]:
    ensure_out_dir(out_dir)
    gate = read_json(gate_json)
    support_rows = read_csv(support_csv)
    deployment_rows = read_csv(deployment_csv)
    rankdelta = read_json(rankdelta_json)

    figure_paths = {
        "gate": out_dir / "fig_iclr95_gate_scores.pdf",
        "support": out_dir / "fig_support_law_correlations.pdf",
        "deployment": out_dir / "fig_deployment_utility.pdf",
        "bass": out_dir / "fig_bass_boundary_ap_risk.pdf",
    }
    figures = {
        "iclr95_gate_scores": plot_gate_scores(gate, figure_paths["gate"]),
        "support_law_correlations": plot_support_correlations(
            support_rows, figure_paths["support"]),
        "deployment_utility": plot_deployment_utility(
            deployment_rows, figure_paths["deployment"]),
        "bass_boundary_ap_risk": plot_bass_boundary(
            rankdelta, figure_paths["bass"]),
    }

    created_paths = {
        "gate": figures["iclr95_gate_scores"].get("path"),
        "support": figures["support_law_correlations"].get("path"),
        "deployment": figures["deployment_utility"].get("path"),
        "bass": figures["bass_boundary_ap_risk"].get("path"),
    }
    if all(created_paths.values()):
        write_tex_snippet(out_tex, created_paths)

    manifest = {
        "ok": all(item.get("created") for item in figures.values()),
        "figures": figures,
        "tex_snippet": str(out_tex),
        "result_md": str(out_md),
        "inputs": {
            "gate_json": str(gate_json),
            "support_csv": str(support_csv),
            "deployment_csv": str(deployment_csv),
            "rankdelta_json": str(rankdelta_json),
        },
        "no_fabrication_rule": (
            "Figures are rendered only from existing gate, support-law, "
            "deployment, and RankDelta summary artifacts. Missing RankDelta/P2 "
            "rows remain absent or TBD; no checkpoint-only result is inferred."),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_md_record(out_md, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-json", default=str(DEFAULT_GATE_JSON))
    parser.add_argument("--support-csv", default=str(DEFAULT_SUPPORT_CSV))
    parser.add_argument("--deployment-csv", default=str(DEFAULT_DEPLOYMENT_CSV))
    parser.add_argument("--rankdelta-json", default=str(DEFAULT_RANKDELTA_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out-json", default=str(DEFAULT_WORK_JSON))
    parser.add_argument("--out-tex", default=str(DEFAULT_TEX))
    parser.add_argument("--out-md", default=str(DEFAULT_MD))
    args = parser.parse_args()

    manifest = build_figures(
        gate_json=Path(args.gate_json),
        support_csv=Path(args.support_csv),
        deployment_csv=Path(args.deployment_csv),
        rankdelta_json=Path(args.rankdelta_json),
        out_dir=Path(args.out_dir),
        out_json=Path(args.out_json),
        out_tex=Path(args.out_tex),
        out_md=Path(args.out_md),
    )
    print(json.dumps({
        "ok": manifest["ok"],
        "figures": {
            key: value.get("created")
            for key, value in manifest["figures"].items()
        },
        "out_json": str(args.out_json),
        "out_tex": str(args.out_tex),
        "out_md": str(args.out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
