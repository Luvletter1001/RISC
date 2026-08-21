import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("M_Tools/analysis/build_iclr95_evidence_gate.py")
    spec = importlib.util.spec_from_file_location(
        "build_iclr95_evidence_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspect_rankdelta_closure_plan_reports_missing(tmp_path):
    mod = _load_module()

    inspected = mod.inspect_rankdelta_closure_plan(tmp_path / "missing.json")

    assert inspected["exists"] is False
    assert inspected["action"] == "missing"
    assert inspected["summary_status"] == "missing"
    assert inspected["execution_priority"] == []


def test_inspect_rankdelta_closure_plan_reads_action_and_queue(tmp_path):
    mod = _load_module()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "action": "RUN_P2_RANK_ASSIGNMENT",
        "reason": "P1 complete without strict pass.",
        "paper_position": "P2 required",
        "summary_status": "done",
        "followup_action": "STOP_P1_DONE_NO_STRICT",
        "p0_done": True,
        "p1_done": True,
        "p2_done": True,
        "strict_pass_methods": [],
        "pilot_pass_methods": ["P1 protected RankDelta"],
        "execution_priority": [{"priority": "P2A"}],
    }), encoding="utf-8")

    inspected = mod.inspect_rankdelta_closure_plan(path)

    assert inspected["exists"] is True
    assert inspected["action"] == "RUN_P2_RANK_ASSIGNMENT"
    assert inspected["p0_done"] is True
    assert inspected["p1_done"] is True
    assert inspected["p2_done"] is True
    assert inspected["pilot_pass_methods"] == ["P1 protected RankDelta"]
    assert inspected["execution_priority"] == [{"priority": "P2A"}]


def test_inspect_rankdelta_summary_reports_p2_completion(tmp_path):
    mod = _load_module()
    summary = tmp_path / "summary.json"
    decision = tmp_path / "decision.json"
    rows = [
        {"method": "RankDelta min-score 0.30", "kind": "rankdelta_p0",
         "gate": "fail", "mAP": 0.85},
        {"method": "RankDelta min-score 0.50", "kind": "rankdelta_p0",
         "gate": "fail", "mAP": 0.84},
        {"method": "P1 gentle RankDelta", "kind": "rankdelta_p1",
         "gate": "pilot pass", "mAP": 0.862},
        {"method": "P1 protected RankDelta", "kind": "rankdelta_p1",
         "gate": "fail", "mAP": 0.857},
        {"method": "P2 assign-rank BASS-GSF", "kind": "rankdelta_p2",
         "gate": "fail", "mAP": 0.858},
        {"method": "P2 posterior-rank BASS-GSF", "kind": "rankdelta_p2",
         "gate": "fail", "mAP": 0.861},
    ]
    summary.write_text(json.dumps({"status": "done", "rows": rows}),
                       encoding="utf-8")
    decision.write_text(json.dumps({"action": "STOP_P1_DONE_NO_STRICT"}),
                        encoding="utf-8")

    inspected = mod.inspect_rankdelta_summary(summary, decision)

    assert inspected["p2_done"] is True
    assert inspected["rankdelta_rows_filled"] == 6
    assert inspected["rankdelta_rows_total"] == 6
    assert inspected["rankdelta_all_complete"] is True


def test_inspect_p3a_summary_reads_main_strict_pass(tmp_path):
    mod = _load_module()
    path = tmp_path / "p3a.json"
    path.write_text(json.dumps({
        "status": "done",
        "strict_pass": True,
        "strict_pass_methods": ["P3A z2 s0.1"],
        "main_method": "P3A z2 s0.1",
        "main_gate": "strict pass",
        "result_md": "resultmd/p3a.md",
        "rows": [{
            "method": "P3A z2 s0.1",
            "kind": "p3a_main",
            "gate": "strict pass",
            "mAP": 0.862961,
            "top5000_precision": 0.7170,
            "top5000_sise_logz": 0,
            "top5000_risk_weighted_logz": 0.0,
            "changed_correct": 0,
            "tp_rank_harm": 0,
        }],
    }), encoding="utf-8")

    inspected = mod.inspect_p3a_summary(path)

    assert inspected["exists"] is True
    assert inspected["strict_pass"] is True
    assert inspected["main_method"] == "P3A z2 s0.1"
    assert inspected["top5000_sise_logz"] == 0
    assert inspected["changed_correct"] == 0


def test_inspect_p3d_transfer_summary_reads_strict_transfer_pass(tmp_path):
    mod = _load_module()
    path = tmp_path / "p3d.json"
    path.write_text(json.dumps({
        "status": "done",
        "dataset": "DIOR-R",
        "strict_transfer_pass": True,
        "strict_transfer_pass_methods": ["P3D DIOR-R z2 s0.1"],
        "main_method": "P3D DIOR-R z2 s0.1",
        "main_gate": "strict transfer pass",
        "main_mAP": 0.645981,
        "main_top5000_precision": 1.0,
        "main_risk_weighted_logz": 1827.56,
        "changed_correct": 0,
        "tp_rank_harm": 0,
        "ap_contributing_fp_removed": 54,
        "result_md": "resultmd/p3d.md",
        "rows": [{
            "method": "P3D DIOR-R z2 s0.1",
            "kind": "p3d_transfer_main",
            "dataset": "DIOR-R",
            "gate": "strict transfer pass",
            "mAP": 0.645981,
            "top5000_precision": 1.0,
            "risk_weighted_logz": 1827.56,
            "changed_correct": 0,
            "tp_rank_harm": 0,
        }],
    }), encoding="utf-8")

    inspected = mod.inspect_p3d_transfer_summary(path)

    assert inspected["exists"] is True
    assert inspected["strict_transfer_pass"] is True
    assert inspected["dataset"] == "DIOR-R"
    assert inspected["main_method"] == "P3D DIOR-R z2 s0.1"
    assert inspected["mAP"] == 0.645981
    assert inspected["tp_rank_harm"] == 0
    assert inspected["ap_contributing_fp_removed"] == 54


def test_inspect_latex_reports_generated_figures_and_appendix(tmp_path):
    mod = _load_module()
    path = tmp_path / "main.tex"
    path.write_text(r"""
\begin{abstract}
Demo.
\end{abstract}
\section{Machine-Checkable Gate}
\begin{table}
\end{table}
\begin{table}
\end{table}
\begin{table}
\end{table}
\input{generated/evidence_figures}
\input{generated/reproducibility_appendix}
\bibliography{refs}
""", encoding="utf-8")

    inspected = mod.inspect_latex(path)

    assert inspected["exists"] is True
    assert inspected["has_generated_figures"] is True
    assert inspected["has_reproducibility_appendix"] is True
    assert inspected["num_tables"] == 3
