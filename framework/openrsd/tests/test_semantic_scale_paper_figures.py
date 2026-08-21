import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/build_semantic_scale_paper_figures.py")
    spec = importlib.util.spec_from_file_location(
        "build_semantic_scale_paper_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_figures_writes_pdf_manifest_and_tex(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.chdir(tmp_path)

    gate_json = Path("gate.json")
    support_csv = Path("support.csv")
    deployment_csv = Path("deployment.csv")
    rankdelta_json = Path("rankdelta.json")
    out_json = Path("manifest.json")
    out_tex = Path("paper/semantic_scale_support_iclr/generated/evidence_figures.tex")
    out_md = Path("resultmd/exp_p4_scale_semantic_validation/figures.md")
    out_dir = Path("paper/semantic_scale_support_iclr/generated/figures")

    gate_json.write_text(json.dumps({
        "num_gates_passed": 1,
        "num_gates_total": 2,
        "gates": [
            {
                "dimension": "Problem anatomy depth",
                "current_score": 9.5,
                "gate_pass": True,
            },
            {
                "dimension": "Method effectiveness",
                "current_score": 8.45,
                "gate_pass": False,
            },
        ],
    }), encoding="utf-8")
    support_csv.write_text(
        "dataset,rho_z2_vs_p0199\n"
        "P4 OVD full preselect0.99,0.82\n"
        "HRRSD closed-set,0.64\n",
        encoding="utf-8")
    deployment_csv.write_text(
        "dataset,mAP_delta,total_sise_base,total_sise_method,"
        "review_queue_sise_base,review_queue_sise_method\n"
        "P4 OVD full preselect0.99,0.005,4032,623,100,24\n"
        "HRRSD closed-set,0.0021,219,8,42,0\n",
        encoding="utf-8")
    rankdelta_json.write_text(json.dumps({
        "rows": [
            {
                "method": "density w005 e2",
                "kind": "control",
                "gate": "control",
                "mAP": 0.860457,
                "risk_weighted_logz": 13.7092,
            },
            {
                "method": "RankDelta min-score 0.30",
                "kind": "rankdelta_p0",
                "gate": "waiting/missing",
                "mAP": None,
                "risk_weighted_logz": "TBD",
            },
        ],
    }), encoding="utf-8")

    manifest = mod.build_figures(
        gate_json=gate_json,
        support_csv=support_csv,
        deployment_csv=deployment_csv,
        rankdelta_json=rankdelta_json,
        out_dir=out_dir,
        out_json=out_json,
        out_tex=out_tex,
        out_md=out_md,
    )

    assert manifest["ok"] is True
    assert manifest["figures"]["bass_boundary_ap_risk"]["num_rows"] == 1
    assert out_json.exists()
    assert out_tex.exists()
    assert out_md.exists()
    for pdf in out_dir.glob("*.pdf"):
        assert pdf.stat().st_size > 0
    assert {path.name for path in out_dir.glob("*.pdf")} == {
        "fig_iclr95_gate_scores.pdf",
        "fig_support_law_correlations.pdf",
        "fig_deployment_utility.pdf",
        "fig_bass_boundary_ap_risk.pdf",
    }
    tex = out_tex.read_text(encoding="utf-8")
    assert "\\label{fig:iclr95-gate}" in tex
    assert "\\label{fig:support-law}" in tex
    assert "\\label{fig:deployment-utility}" in tex
    assert "\\label{fig:bass-boundary}" in tex
