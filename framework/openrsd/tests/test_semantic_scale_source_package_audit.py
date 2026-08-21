import importlib.util
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/audit_semantic_scale_source_package.py")
    spec = importlib.util.spec_from_file_location(
        "audit_semantic_scale_source_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_package_audit_finds_inputs_graphics_and_bib(tmp_path):
    mod = _load_module()
    paper_dir = tmp_path / "paper"
    generated = paper_dir / "generated"
    figures = generated / "figures"
    figures.mkdir(parents=True)
    (paper_dir / "README.md").write_text("readme", encoding="utf-8")
    (paper_dir / "build.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "refs.bib").write_text("@article{x,}\n",
                                                    encoding="utf-8")
    (figures / "demo.pdf").write_text("%PDF demo\n", encoding="utf-8")
    (generated / "evidence_figures.tex").write_text(
        "\\includegraphics{generated/figures/demo.pdf}\n",
        encoding="utf-8")
    (generated / "reproducibility_appendix.tex").write_text(
        "\\section{Reproducibility and No-Fabrication Ledger}\n",
        encoding="utf-8")
    (paper_dir / "main.tex").write_text(r"""
\begin{abstract}
Demo.
\end{abstract}
\input{generated/evidence_figures}
\input{generated/reproducibility_appendix}
\bibliography{../drafts/refs}
""", encoding="utf-8")

    audit = mod.build_audit(paper_dir)

    assert audit["source_package_ready"] is True
    assert audit["pdf_build_ready"] is False
    assert audit["missing_files"] == []
    assert len(audit["inputs"]) == 2
    assert len(audit["graphics"]) == 1
    assert len(audit["bibs"]) == 1


def test_source_package_audit_reports_missing_graphic(tmp_path):
    mod = _load_module()
    paper_dir = tmp_path / "paper"
    generated = paper_dir / "generated"
    generated.mkdir(parents=True)
    (paper_dir / "README.md").write_text("readme", encoding="utf-8")
    (paper_dir / "build.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "refs.bib").write_text("@article{x,}\n",
                                                    encoding="utf-8")
    (generated / "evidence_figures.tex").write_text(
        "\\includegraphics{generated/figures/missing.pdf}\n",
        encoding="utf-8")
    (generated / "reproducibility_appendix.tex").write_text(
        "\\section{Reproducibility and No-Fabrication Ledger}\n",
        encoding="utf-8")
    (paper_dir / "main.tex").write_text(r"""
\begin{abstract}
Demo.
\end{abstract}
\input{generated/evidence_figures}
\input{generated/reproducibility_appendix}
\bibliography{../drafts/refs}
""", encoding="utf-8")

    audit = mod.build_audit(paper_dir)

    assert audit["source_package_ready"] is False
    assert audit["missing_files"][0]["raw"] == "generated/figures/missing.pdf"
