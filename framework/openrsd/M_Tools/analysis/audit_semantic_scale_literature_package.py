#!/usr/bin/env python
"""Audit traceability and coverage of the semantic-scale paper literature set."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_BIB = Path("paper/drafts/semantic_scale_support_refs.bib")
DEFAULT_LATEX = Path("paper/semantic_scale_support_iclr/main.tex")
DEFAULT_MATRIX = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fmatrix_20260620_semantic_scale_closest_competitors.md")
DEFAULT_OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "literature_package_audit/literature_package_audit.json")
DEFAULT_OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "faudit_20260620_semantic_scale_literature_package.md")

CLUSTERS = {
    "scale_aware_detection": {
        "required_keys": [
            "lin2016fpn", "li2019tridentnet", "tian2019fcos",
            "zhang2019atss", "li2020gfl",
        ],
        "matrix_tokens": ["FPN", "TridentNet", "FCOS", "ATSS", "GFL"],
    },
    "remote_sensing_detection": {
        "required_keys": [
            "xia2017dota", "ding2021dotav2", "han2020s2anet",
            "han2021redet", "yu2023h2rboxv2", "li2024lsknet",
        ],
        "matrix_tokens": ["DOTA", "DOTA-v2", "S2A-Net", "ReDet",
                          "H2RBox-v2", "LSKNet"],
    },
    "open_vocabulary_detection": {
        "required_keys": [
            "radford2021clip", "gu2021vild", "li2021glip",
            "minderer2022owlvit", "liu2023groundingdino",
            "zhou2022detic", "cheng2024yoloworld",
            "pan2024laedino", "huang2025openrsd",
        ],
        "matrix_tokens": ["ViLD", "GLIP", "OWL-ViT", "Grounding DINO",
                          "Detic", "YOLO-World", "LAE-DINO", "OpenRSD"],
    },
    "calibration_probabilistic_ood": {
        "required_keys": [
            "kuzucu2024detcalib", "munir2023caldetr",
            "hall2018probod", "du2022vos",
        ],
        "matrix_tokens": ["Calibration", "Cal-DETR", "Probabilistic",
                          "VOS"],
    },
    "gaussian_uncertainty_energy": {
        "required_keys": [
            "choi2019gaussianyolo", "park2021uadet",
            "lee2018mahalanobis", "liu2020energyood",
        ],
        "matrix_tokens": ["Gaussian YOLOv3", "UADET", "Mahalanobis",
                          "Energy-based"],
    },
    "assignment_ranking": {
        "required_keys": ["kim2020paa", "ge2021ota", "feng2021tood"],
        "matrix_tokens": ["PAA", "OTA", "TOOD", "assignment"],
    },
    "conformal_risk_control": {
        "required_keys": ["andeol2023conformalrisk"],
        "matrix_tokens": ["Conformal", "risk-control"],
    },
    "gaussian_box_losses": {
        "required_keys": ["yang2021gwd", "yang2021kld", "wang2021nwd"],
        "matrix_tokens": ["GWD", "KLD", "NWD", "Gaussian box"],
    },
}

FORBIDDEN_PLACEHOLDERS = [
    "citation needed", "TODO", "TBD citation", "待核验", "[fill]",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = match.group(1)
        start = match.start()
        next_match = re.search(r"\n@\w+\s*\{", text[match.end():])
        end = match.end() + next_match.start() if next_match else len(text)
        block = text[start:end]
        title = re.search(r"title\s*=\s*\{([^{}]+)\}", block, re.I)
        year = re.search(r"year\s*=\s*\{([^{}]+)\}", block, re.I)
        url = re.search(r"url\s*=\s*\{([^{}]+)\}", block, re.I)
        entries[key] = {
            "title": title.group(1) if title else "",
            "year": year.group(1) if year else "",
            "url": url.group(1) if url else "",
            "block": block,
        }
    return entries


def parse_citations(text: str) -> set[str]:
    keys = set()
    for cite in re.findall(r"\\cite\{([^{}]+)\}", text):
        keys.update(key.strip() for key in cite.split(",") if key.strip())
    return keys


def cluster_audits(entries: dict, citations: set[str], matrix_text: str) -> dict:
    audits = {}
    for name, spec in CLUSTERS.items():
        required = set(spec["required_keys"])
        present = sorted(required & set(entries))
        missing_bib = sorted(required - set(entries))
        cited = sorted(required & citations)
        missing_cites = sorted(required - citations)
        missing_matrix_tokens = [
            token for token in spec["matrix_tokens"]
            if token.lower() not in matrix_text.lower()
        ]
        passed = (
            not missing_bib and not missing_cites
            and not missing_matrix_tokens)
        audits[name] = {
            "passed": passed,
            "present_bib_keys": present,
            "missing_bib_keys": missing_bib,
            "cited_keys": cited,
            "missing_citation_keys": missing_cites,
            "missing_matrix_tokens": missing_matrix_tokens,
        }
    return audits


def build_audit(bib_path: Path, latex_path: Path, matrix_path: Path,
                out_md: Path) -> dict:
    bib_text = read_text(bib_path)
    latex_text = read_text(latex_path)
    matrix_text = read_text(matrix_path)
    entries = parse_bib_entries(bib_text)
    citations = parse_citations(latex_text)
    cluster_results = cluster_audits(entries, citations, matrix_text)
    missing_urls = sorted(
        key for key, value in entries.items() if not value.get("url"))
    uncited_bib_keys = sorted(set(entries) - citations)
    undefined_citations = sorted(citations - set(entries))
    placeholders = [
        token for token in FORBIDDEN_PLACEHOLDERS
        if token.lower() in (latex_text + "\n" + matrix_text).lower()
    ]
    matrix_has_obligations = "Baseline Obligation Matrix" in matrix_text
    matrix_has_ac_interpretation = "AC Interpretation" in matrix_text
    cluster_pass_count = sum(
        1 for row in cluster_results.values() if row["passed"])
    passed = (
        bib_path.exists()
        and latex_path.exists()
        and matrix_path.exists()
        and len(entries) >= 20
        and not missing_urls
        and not undefined_citations
        and cluster_pass_count == len(CLUSTERS)
        and matrix_has_obligations
        and matrix_has_ac_interpretation
        and not placeholders)
    return {
        "literature_package_pass": passed,
        "bib_path": str(bib_path),
        "latex_path": str(latex_path),
        "matrix_path": str(matrix_path),
        "result_md": str(out_md),
        "bib_entry_count": len(entries),
        "citation_key_count": len(citations),
        "cluster_pass_count": cluster_pass_count,
        "cluster_total": len(CLUSTERS),
        "missing_urls": missing_urls,
        "undefined_citations": undefined_citations,
        "uncited_bib_keys": uncited_bib_keys,
        "placeholders": placeholders,
        "matrix_has_obligations": matrix_has_obligations,
        "matrix_has_ac_interpretation": matrix_has_ac_interpretation,
        "clusters": cluster_results,
    }


def write_markdown(path: Path, audit: dict) -> None:
    lines = [
        "# Semantic-Scale Literature Package Audit - 2026-06-20",
        "",
        "本文件由 `audit_semantic_scale_literature_package.py` 从本地",
        "BibTeX、LaTeX 和 closest-competitor matrix 生成。它只检查可追踪性、",
        "覆盖和未核验占位，不新增或发明文献。",
        "",
        "## Verdict",
        "",
        f"- literature_package_pass: `{audit['literature_package_pass']}`",
        f"- bib_entry_count: `{audit['bib_entry_count']}`",
        f"- citation_key_count: `{audit['citation_key_count']}`",
        f"- cluster_pass_count: `{audit['cluster_pass_count']}/{audit['cluster_total']}`",
        f"- matrix_has_obligations: `{audit['matrix_has_obligations']}`",
        f"- matrix_has_ac_interpretation: `{audit['matrix_has_ac_interpretation']}`",
        "",
        "## Cluster Coverage",
        "",
        "| cluster | pass | missing_bib | missing_citation | missing_matrix_tokens |",
        "|---|---|---|---|---|",
    ]
    for name, row in audit["clusters"].items():
        lines.append(
            f"| {name} | `{row['passed']}` | "
            f"`{', '.join(row['missing_bib_keys'])}` | "
            f"`{', '.join(row['missing_citation_keys'])}` | "
            f"`{', '.join(row['missing_matrix_tokens'])}` |")
    lines += [
        "",
        "## Traceability Checks",
        "",
        f"- missing_urls: `{', '.join(audit['missing_urls'])}`",
        f"- undefined_citations: `{', '.join(audit['undefined_citations'])}`",
        f"- placeholders: `{', '.join(audit['placeholders'])}`",
        "",
        "## Uncited BibTeX Keys",
        "",
        "`" + ", ".join(audit["uncited_bib_keys"]) + "`",
        "",
        "## Interpretation",
        "",
        "若 `literature_package_pass=True`，说明当前 related-work 包已经覆盖",
        "多尺度检测、RS 检测、OVD/RS-OVD、检测校准/概率检测/OOD、",
        "Gaussian box loss 五个必要邻域，并且每个本地 BibTeX 条目都有可追踪 URL。",
        "这不代表文献永远完整；它只证明当前 submission package 中没有明显",
        "未定义引用、缺 URL、缺 cluster 或未核验占位。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", default=str(DEFAULT_BIB))
    parser.add_argument("--latex", default=str(DEFAULT_LATEX))
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    args = parser.parse_args()

    audit = build_audit(
        Path(args.bib), Path(args.latex), Path(args.matrix),
        Path(args.out_md))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), audit)
    print(json.dumps({
        "literature_package_pass": audit["literature_package_pass"],
        "bib_entry_count": audit["bib_entry_count"],
        "cluster_pass_count": audit["cluster_pass_count"],
        "cluster_total": audit["cluster_total"],
        "out_json": str(out_json),
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
