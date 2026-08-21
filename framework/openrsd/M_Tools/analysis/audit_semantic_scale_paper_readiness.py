#!/usr/bin/env python
"""Audit manuscript readiness for the semantic-scale support paper.

The audit is intentionally conservative.  It does not infer experiment results
from logs or checkpoints; it records missing evidence as a blocker and keeps
the manuscript in draft status until the machine gates and RankDelta rows pass.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_LATEX = Path("paper/semantic_scale_support_iclr/main.tex")
DEFAULT_MARKDOWN = Path(
    "paper/drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md")
DEFAULT_BIB = Path("paper/drafts/semantic_scale_support_refs.bib")
DEFAULT_GATE_JSON = Path(
    "work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json")
DEFAULT_REFRESH_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_paper_package_refresh.json")
DEFAULT_SNIPPET_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "bass_rankdelta_paper_snippets.json")
DEFAULT_SNIPPET_TEX = Path(
    "paper/semantic_scale_support_iclr/generated/bass_rankdelta_table.tex")
DEFAULT_SNIPPET_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "fsnippet_20260620_bass_rankdelta_paper_table.md")
DEFAULT_FIGURE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_paper_figures_manifest.json")
DEFAULT_FIGURE_TEX = Path(
    "paper/semantic_scale_support_iclr/generated/evidence_figures.tex")
DEFAULT_APPENDIX_TEX = Path(
    "paper/semantic_scale_support_iclr/generated/reproducibility_appendix.tex")
DEFAULT_SOURCE_PACKAGE_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_source_package_audit.json")
OUT_JSON = Path(
    "work_dirs/semantic_scale_six_experiments_20260620/"
    "semantic_scale_paper_readiness_audit.json")
OUT_MD = Path(
    "resultmd/exp_p4_scale_semantic_validation/"
    "faudit_20260620_semantic_scale_paper_readiness.md")

REQUIRED_LATEX_SECTIONS = [
    "Introduction",
    "Related Work and Closest Competitors",
    "Method",
    "Experiments",
    "Machine-Checkable Gate",
    "Limitations and Next Gates",
    "Conclusion",
    "Reproducibility and No-Fabrication Ledger",
]

REQUIRED_MARKDOWN_HEADINGS = [
    "Abstract",
    "1. Introduction",
    "2. Related Work",
    "3. Problem Formulation",
    "4. Method",
    "5. Experiments",
    "6. Limitations",
    "7. Stricter Devil-Reviewer Scorecard and 9.5 Closure Target",
    "8. Conclusion",
    "9. Reproducibility Appendix",
]

ALLOWED_MARKER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bP2 rows remain\s+`?TBD`?(?:\s+until eval JSON exists\.)?\b",
        re.IGNORECASE),
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def latex_sections(text: str) -> list[str]:
    return re.findall(r"\\section\{([^}]+)\}", text)


def markdown_headings(text: str) -> list[str]:
    headings = []
    for line in text.splitlines():
        match = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if match:
            headings.append(match.group(1))
    return headings


def bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for cite_body in re.findall(r"\\cite\{([^}]+)\}", text):
        for key in cite_body.split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def find_unresolved_markers(text: str) -> list[dict[str, Any]]:
    markers = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if re.search(r"\b(TBD|TODO|FIXME|PLACEHOLDER)\b", line):
            if any(pattern.search(line) for pattern in ALLOWED_MARKER_PATTERNS):
                continue
            markers.append({"line": idx, "text": line.strip()[:240]})
    return markers


def find_allowed_markers(text: str) -> list[dict[str, Any]]:
    markers = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if re.search(r"\b(TBD|TODO|FIXME|PLACEHOLDER)\b", line):
            if any(pattern.search(line) for pattern in ALLOWED_MARKER_PATTERNS):
                markers.append({"line": idx, "text": line.strip()[:240]})
    return markers


def toolchain_status() -> dict[str, Any]:
    candidates = ["latexmk", "pdflatex", "tectonic"]
    found = {name: shutil.which(name) for name in candidates}
    return {
        "available": any(found.values()),
        "tools": found,
    }


def inspect_gate(gate_json: Path) -> dict[str, Any]:
    gate = read_json(gate_json)
    return {
        "exists": gate_json.exists(),
        "all_95_gates_passed": bool(gate.get("all_95_gates_passed", False)),
        "num_gates_passed": gate.get("num_gates_passed"),
        "num_gates_total": gate.get("num_gates_total"),
    }


def inspect_refresh(refresh_json: Path) -> dict[str, Any]:
    refresh = read_json(refresh_json)
    return {
        "exists": refresh_json.exists(),
        "ok": bool(refresh.get("ok", False)),
        "summary_status": refresh.get("summary_status", "missing"),
        "followup_action": refresh.get("followup_action", "missing"),
        "closure_action": refresh.get("closure_action", "missing"),
        "confirmation_action": refresh.get("confirmation_action", "missing"),
        "p2_queue_action": refresh.get("p2_queue_action", "missing"),
        "rankdelta_rows_filled": refresh.get("rankdelta_rows_filled"),
        "rankdelta_rows_total": refresh.get("rankdelta_rows_total"),
        "iclr95_gate": refresh.get("iclr95_gate", "missing"),
    }


def inspect_snippets(snippet_json: Path) -> dict[str, Any]:
    snippets = read_json(snippet_json)
    return {
        "exists": snippet_json.exists(),
        "summary_status": snippets.get("summary_status", "missing"),
        "closure_action": snippets.get("closure_action", "missing"),
        "rankdelta_rows_filled": snippets.get("rankdelta_rows_filled"),
        "rankdelta_rows_total": snippets.get("rankdelta_rows_total"),
        "has_strict_pass": snippets.get("has_strict_pass", False),
        "has_pilot_pass": snippets.get("has_pilot_pass", False),
    }


def inspect_figures(figure_json: Path, figure_tex: Path) -> dict[str, Any]:
    figures = read_json(figure_json)
    figure_items = figures.get("figures", {})
    if not isinstance(figure_items, dict):
        figure_items = {}
    created = {
        key: bool(value.get("created", False))
        for key, value in figure_items.items()
        if isinstance(value, dict)
    }
    return {
        "exists": figure_json.exists(),
        "tex_exists": figure_tex.exists(),
        "ok": bool(figures.get("ok", False)),
        "created": created,
        "created_count": sum(1 for value in created.values() if value),
        "figure_count": len(created),
    }


def inspect_source_package(source_package_json: Path) -> dict[str, Any]:
    source_package = read_json(source_package_json)
    return {
        "exists": source_package_json.exists(),
        "source_package_ready": bool(source_package.get(
            "source_package_ready", False)),
        "pdf_build_ready": bool(source_package.get("pdf_build_ready", False)),
        "missing_file_count": len(source_package.get("missing_files", []))
        if isinstance(source_package.get("missing_files", []), list) else None,
    }


def build_audit(latex_path: Path = DEFAULT_LATEX,
                markdown_path: Path = DEFAULT_MARKDOWN,
                bib_path: Path = DEFAULT_BIB,
                gate_json: Path = DEFAULT_GATE_JSON,
                refresh_json: Path = DEFAULT_REFRESH_JSON,
                snippet_json: Path = DEFAULT_SNIPPET_JSON,
                snippet_tex: Path = DEFAULT_SNIPPET_TEX,
                snippet_md: Path = DEFAULT_SNIPPET_MD,
                figure_json: Path = DEFAULT_FIGURE_JSON,
                figure_tex: Path = DEFAULT_FIGURE_TEX,
                appendix_tex: Path = DEFAULT_APPENDIX_TEX,
                source_package_json: Path = DEFAULT_SOURCE_PACKAGE_JSON
                ) -> dict[str, Any]:
    latex = read_text(latex_path)
    appendix = read_text(appendix_tex)
    latex_for_structure = latex + "\n" + appendix
    markdown = read_text(markdown_path)
    bib = read_text(bib_path)
    sections = latex_sections(latex_for_structure)
    headings = markdown_headings(markdown)
    defined = bib_keys(bib)
    cited = citation_keys(latex)
    unresolved_latex = find_unresolved_markers(latex_for_structure)
    unresolved_markdown = find_unresolved_markers(markdown)
    allowed_latex_markers = find_allowed_markers(latex)
    allowed_markdown_markers = find_allowed_markers(markdown)
    gate = inspect_gate(gate_json)
    refresh = inspect_refresh(refresh_json)
    snippets = inspect_snippets(snippet_json)
    figures = inspect_figures(figure_json, figure_tex)
    source_package = inspect_source_package(source_package_json)
    toolchain = toolchain_status()

    missing_sections = [
        section for section in REQUIRED_LATEX_SECTIONS
        if section not in sections
    ]
    missing_headings = [
        heading for heading in REQUIRED_MARKDOWN_HEADINGS
        if heading not in headings
    ]
    undefined_citations = sorted(cited - defined)
    uncited_bib_keys = sorted(defined - cited)

    structure_pass = not missing_sections and not missing_headings
    citation_pass = not undefined_citations and bool(cited) and bool(defined)
    artifact_pass = all([
        latex_path.exists(),
        markdown_path.exists(),
        bib_path.exists(),
        snippet_tex.exists(),
        snippet_md.exists(),
        snippets["exists"],
        gate["exists"],
        figures["exists"],
        figures["tex_exists"],
        figures["ok"],
        appendix_tex.exists(),
        "\\input{generated/reproducibility_appendix}" in latex,
        source_package["exists"],
        source_package["source_package_ready"],
    ])
    no_unresolved_markers = (
        not unresolved_latex and not unresolved_markdown)
    rankdelta_complete = (
        snippets.get("rankdelta_rows_filled") == snippets.get(
            "rankdelta_rows_total")
        and snippets.get("rankdelta_rows_total") not in {None, 0})
    gate_pass = bool(gate["all_95_gates_passed"])
    compile_tool_available = bool(toolchain["available"])

    diagnostic_draft_ready = (
        structure_pass and citation_pass and artifact_pass
        and refresh.get("ok", False)
    )
    submission_ready = all([
        diagnostic_draft_ready,
        no_unresolved_markers,
        rankdelta_complete,
        gate_pass,
        compile_tool_available,
    ])

    blockers = []
    if not structure_pass:
        blockers.append("missing required manuscript sections/headings")
    if not citation_pass:
        blockers.append("citation/BibTeX mismatch")
    if not artifact_pass:
        blockers.append("missing generated evidence artifacts")
    if not no_unresolved_markers:
        blockers.append("unresolved TBD/TODO markers remain")
    if not rankdelta_complete:
        blockers.append("RankDelta/P2 rows are incomplete")
    if not gate_pass:
        blockers.append("ICLR 9.5 machine gate is not fully passed")
    if not compile_tool_available:
        blockers.append("no LaTeX compiler found in PATH")

    return {
        "diagnostic_draft_ready": diagnostic_draft_ready,
        "submission_ready": submission_ready,
        "blockers": blockers,
        "latex": {
            "path": str(latex_path),
            "line_count": len(latex.splitlines()) if latex else 0,
            "sections": sections,
            "missing_sections": missing_sections,
            "unresolved_markers": unresolved_latex,
            "allowed_markers": allowed_latex_markers,
        },
        "markdown": {
            "path": str(markdown_path),
            "line_count": len(markdown.splitlines()) if markdown else 0,
            "missing_headings": missing_headings,
            "unresolved_markers": unresolved_markdown,
            "allowed_markers": allowed_markdown_markers,
        },
        "bib": {
            "path": str(bib_path),
            "bib_entry_count": len(defined),
            "citation_key_count": len(cited),
            "undefined_citations": undefined_citations,
            "uncited_bib_keys": uncited_bib_keys,
        },
        "artifacts": {
            "snippet_json_exists": snippets["exists"],
            "snippet_tex_exists": snippet_tex.exists(),
            "snippet_md_exists": snippet_md.exists(),
            "gate_json_exists": gate["exists"],
            "refresh_json_exists": refresh["exists"],
            "figure_json_exists": figures["exists"],
            "figure_tex_exists": figures["tex_exists"],
            "figures_ok": figures["ok"],
            "appendix_tex_exists": appendix_tex.exists(),
            "appendix_included": (
                "\\input{generated/reproducibility_appendix}" in latex),
            "source_package_json_exists": source_package["exists"],
            "source_package_ready": source_package["source_package_ready"],
            "pdf_build_ready": source_package["pdf_build_ready"],
        },
        "gate": gate,
        "refresh": refresh,
        "snippets": snippets,
        "figures": figures,
        "source_package": source_package,
        "toolchain": toolchain,
        "no_fabrication_rule": (
            "RankDelta/P2 rows may be filled only from eval JSON plus "
            "deployment-risk summaries. Submission readiness also requires "
            "the machine gate and PDF build checks to pass."),
    }


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Semantic-Scale Paper Readiness Audit - 2026-06-20",
        "",
        "This file is machine-generated and does not infer missing results.",
        "",
        "| field | value |",
        "|---|---|",
        f"| diagnostic_draft_ready | `{audit['diagnostic_draft_ready']}` |",
        f"| submission_ready | `{audit['submission_ready']}` |",
        f"| iclr95_gate | `{audit['gate'].get('num_gates_passed')}/{audit['gate'].get('num_gates_total')}` |",
        f"| rankdelta_rows | `{audit['snippets'].get('rankdelta_rows_filled')}/{audit['snippets'].get('rankdelta_rows_total')}` |",
        f"| paper_figures | `{audit['figures'].get('created_count')}/{audit['figures'].get('figure_count')}` |",
        f"| appendix_included | `{audit['artifacts']['appendix_included']}` |",
        f"| source_package_ready | `{audit['source_package']['source_package_ready']}` |",
        f"| pdf_build_ready | `{audit['source_package']['pdf_build_ready']}` |",
        f"| latex_compile_tool_available | `{audit['toolchain']['available']}` |",
        f"| bib_entries | `{audit['bib']['bib_entry_count']}` |",
        f"| cited_keys | `{audit['bib']['citation_key_count']}` |",
        "",
        "## Blockers",
        "",
    ]
    if audit["blockers"]:
        lines.extend(f"- {blocker}" for blocker in audit["blockers"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## Missing Structure",
        "",
        f"- LaTeX sections: `{audit['latex']['missing_sections']}`",
        f"- Markdown headings: `{audit['markdown']['missing_headings']}`",
        "",
        "## Citation Audit",
        "",
        f"- Undefined citations: `{audit['bib']['undefined_citations']}`",
        f"- Uncited BibTeX keys: `{audit['bib']['uncited_bib_keys']}`",
        "",
        "## Unresolved Markers",
        "",
        f"- LaTeX marker count: `{len(audit['latex']['unresolved_markers'])}`",
        f"- Markdown marker count: `{len(audit['markdown']['unresolved_markers'])}`",
        f"- Allowed evidence-placeholder markers: `{len(audit['latex']['allowed_markers']) + len(audit['markdown']['allowed_markers'])}`",
        "",
        "## No-Fabrication Rule",
        "",
        audit["no_fabrication_rule"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", default=str(DEFAULT_LATEX))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--bib", default=str(DEFAULT_BIB))
    parser.add_argument("--gate-json", default=str(DEFAULT_GATE_JSON))
    parser.add_argument("--refresh-json", default=str(DEFAULT_REFRESH_JSON))
    parser.add_argument("--snippet-json", default=str(DEFAULT_SNIPPET_JSON))
    parser.add_argument("--snippet-tex", default=str(DEFAULT_SNIPPET_TEX))
    parser.add_argument("--snippet-md", default=str(DEFAULT_SNIPPET_MD))
    parser.add_argument("--figure-json", default=str(DEFAULT_FIGURE_JSON))
    parser.add_argument("--figure-tex", default=str(DEFAULT_FIGURE_TEX))
    parser.add_argument("--appendix-tex", default=str(DEFAULT_APPENDIX_TEX))
    parser.add_argument("--source-package-json",
                        default=str(DEFAULT_SOURCE_PACKAGE_JSON))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    audit = build_audit(
        latex_path=Path(args.latex),
        markdown_path=Path(args.markdown),
        bib_path=Path(args.bib),
        gate_json=Path(args.gate_json),
        refresh_json=Path(args.refresh_json),
        snippet_json=Path(args.snippet_json),
        snippet_tex=Path(args.snippet_tex),
        snippet_md=Path(args.snippet_md),
        figure_json=Path(args.figure_json),
        figure_tex=Path(args.figure_tex),
        appendix_tex=Path(args.appendix_tex),
        source_package_json=Path(args.source_package_json))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    write_markdown(Path(args.out_md), audit)
    print(json.dumps({
        "diagnostic_draft_ready": audit["diagnostic_draft_ready"],
        "submission_ready": audit["submission_ready"],
        "blocker_count": len(audit["blockers"]),
        "rankdelta_rows": (
            f"{audit['snippets'].get('rankdelta_rows_filled')}/"
            f"{audit['snippets'].get('rankdelta_rows_total')}"),
        "iclr95_gate": (
            f"{audit['gate'].get('num_gates_passed')}/"
            f"{audit['gate'].get('num_gates_total')}"),
        "paper_figures": (
            f"{audit['figures'].get('created_count')}/"
            f"{audit['figures'].get('figure_count')}"),
        "out_json": args.out_json,
        "out_md": args.out_md,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
