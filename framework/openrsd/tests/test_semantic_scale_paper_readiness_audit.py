import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_semantic_scale_paper_readiness.py")
    spec = importlib.util.spec_from_file_location(
        "audit_semantic_scale_paper_readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_packet(tmp_path, *, gate_pass=False, rows_filled=0,
                          rows_total=6):
    latex = tmp_path / "main.tex"
    markdown = tmp_path / "draft.md"
    bib = tmp_path / "refs.bib"
    gate = tmp_path / "gate.json"
    refresh = tmp_path / "refresh.json"
    snippets = tmp_path / "snippets.json"
    snippet_tex = tmp_path / "snippet.tex"
    snippet_md = tmp_path / "snippet.md"
    figure_json = tmp_path / "figures.json"
    figure_tex = tmp_path / "figures.tex"
    appendix_tex = tmp_path / "appendix.tex"
    source_package_json = tmp_path / "source_package.json"

    latex.write_text(r"""
\section{Introduction}
Text \cite{demo2026}.
\section{Related Work and Closest Competitors}
\section{Method}
\section{Experiments}
\section{Machine-Checkable Gate}
\section{Limitations and Next Gates}
\section{Conclusion}
\input{generated/reproducibility_appendix}
""", encoding="utf-8")
    markdown.write_text("""
## Abstract
## 1. Introduction
## 2. Related Work
## 3. Problem Formulation
## 4. Method
## 5. Experiments
## 6. Limitations
## 7. Stricter Devil-Reviewer Scorecard and 9.5 Closure Target
## 8. Conclusion
## 9. Reproducibility Appendix
""", encoding="utf-8")
    bib.write_text("""@article{demo2026,
  title={Demo},
  author={Author, A.},
  year={2026},
  url={https://example.com}
}
""", encoding="utf-8")
    gate.write_text(json.dumps({
        "all_95_gates_passed": gate_pass,
        "num_gates_passed": 10 if gate_pass else 3,
        "num_gates_total": 10,
    }), encoding="utf-8")
    refresh.write_text(json.dumps({"ok": True}), encoding="utf-8")
    snippets.write_text(json.dumps({
        "summary_status": "done" if rows_filled == rows_total else "waiting",
        "closure_action": "RUN_CONFIRMATION_PACKAGE" if gate_pass else "WAIT_P0",
        "rankdelta_rows_filled": rows_filled,
        "rankdelta_rows_total": rows_total,
        "has_strict_pass": gate_pass,
    }), encoding="utf-8")
    snippet_tex.write_text("table", encoding="utf-8")
    snippet_md.write_text("table", encoding="utf-8")
    figure_json.write_text(json.dumps({
        "ok": True,
        "figures": {
            "iclr95_gate_scores": {"created": True},
            "support_law_correlations": {"created": True},
            "deployment_utility": {"created": True},
            "bass_boundary_ap_risk": {"created": True},
        },
    }), encoding="utf-8")
    figure_tex.write_text("figures", encoding="utf-8")
    appendix_tex.write_text(
        "\\section{Reproducibility and No-Fabrication Ledger}\n",
        encoding="utf-8")
    source_package_json.write_text(json.dumps({
        "source_package_ready": True,
        "pdf_build_ready": False,
        "missing_files": [],
    }), encoding="utf-8")
    return (
        latex, markdown, bib, gate, refresh, snippets, snippet_tex, snippet_md,
        figure_json, figure_tex, appendix_tex, source_package_json)


def test_readiness_allows_diagnostic_draft_but_blocks_submission(tmp_path):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=False, rows_filled=0)

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is True
    assert audit["submission_ready"] is False
    assert "RankDelta/P2 rows are incomplete" in audit["blockers"]
    assert "ICLR 9.5 machine gate is not fully passed" in audit["blockers"]


def test_readiness_passes_when_all_submission_conditions_hold(tmp_path,
                                                             monkeypatch):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=True, rows_filled=6)
    monkeypatch.setattr(mod, "toolchain_status", lambda: {
        "available": True,
        "tools": {"pdflatex": "/usr/bin/pdflatex"},
    })

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is True
    assert audit["submission_ready"] is True
    assert audit["blockers"] == []


def test_readiness_reports_undefined_citations(tmp_path):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path)
    packet[0].write_text(packet[0].read_text(encoding="utf-8").replace(
        "demo2026", "missing2026"), encoding="utf-8")

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is False
    assert audit["bib"]["undefined_citations"] == ["missing2026"]


def test_readiness_allows_no_fabrication_tbd_sentence(tmp_path,
                                                      monkeypatch):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=True, rows_filled=6)
    packet[0].write_text(packet[0].read_text(encoding="utf-8") + (
        "\nP2 rows remain TBD until eval JSON exists.\n"), encoding="utf-8")
    packet[1].write_text(packet[1].read_text(encoding="utf-8") + (
        "\nP2 rows remain `TBD`\nuntil eval JSON exists.\n"),
        encoding="utf-8")
    monkeypatch.setattr(mod, "toolchain_status", lambda: {
        "available": True,
        "tools": {"pdflatex": "/usr/bin/pdflatex"},
    })

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["latex"]["unresolved_markers"] == []
    assert audit["markdown"]["unresolved_markers"] == []
    assert len(audit["latex"]["allowed_markers"]) == 1
    assert len(audit["markdown"]["allowed_markers"]) == 1
    assert audit["submission_ready"] is True


def test_readiness_blocks_when_paper_figures_are_missing(tmp_path):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=False, rows_filled=0)
    packet[8].unlink()

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is False
    assert "missing generated evidence artifacts" in audit["blockers"]
    assert audit["artifacts"]["figure_json_exists"] is False


def test_readiness_blocks_when_appendix_is_missing(tmp_path):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=False, rows_filled=0)
    packet[10].unlink()

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is False
    assert "missing generated evidence artifacts" in audit["blockers"]
    assert audit["artifacts"]["appendix_tex_exists"] is False


def test_readiness_blocks_when_source_package_is_missing(tmp_path):
    mod = _load_module()
    packet = _write_minimal_packet(tmp_path, gate_pass=False, rows_filled=0)
    packet[11].unlink()

    audit = mod.build_audit(
        latex_path=packet[0],
        markdown_path=packet[1],
        bib_path=packet[2],
        gate_json=packet[3],
        refresh_json=packet[4],
        snippet_json=packet[5],
        snippet_tex=packet[6],
        snippet_md=packet[7],
        figure_json=packet[8],
        figure_tex=packet[9],
        appendix_tex=packet[10],
        source_package_json=packet[11])

    assert audit["diagnostic_draft_ready"] is False
    assert "missing generated evidence artifacts" in audit["blockers"]
    assert audit["artifacts"]["source_package_json_exists"] is False
