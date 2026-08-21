import csv
import json

from M_Tools.analysis.audit_g3_v1_dense_head_results import (
    audit_g3_v1_results,
)


def write_eval_json(path, mAP, nested=False):
    path.mkdir(parents=True, exist_ok=True)
    payload = {"dota/mAP": mAP, "dota/AP50": round(mAP, 3)}
    if nested:
        payload = {"metrics": payload}
    (path / "metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8")


def write_risk_csv(path, variant, pred_path, base_sise, method_sise):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "predictions",
                "localized_correct",
                "localized_precision",
                "sise_logz_wrong_excl_sibling",
                "sise_p0199_wrong_excl_sibling",
                "wrong_excl_sibling_ge_thr_score_ge_0p5",
                "sise_logz_ge_thr_score_ge_0p5",
                "correct_topk_top5000",
                "sise_logz_topk_top5000",
                "sise_p0199_topk_top5000",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "variant": "baseline",
            "predictions": "eval_epoch3_baseline_full/predictions.pkl",
            "localized_correct": "100",
            "localized_precision": "0.50",
            "sise_logz_wrong_excl_sibling": str(base_sise),
            "sise_p0199_wrong_excl_sibling": "200",
            "wrong_excl_sibling_ge_thr_score_ge_0p5": "10",
            "sise_logz_ge_thr_score_ge_0p5": "0",
            "correct_topk_top5000": "90",
            "sise_logz_topk_top5000": "20",
            "sise_p0199_topk_top5000": "40",
        })
        writer.writerow({
            "variant": variant,
            "predictions": pred_path,
            "localized_correct": "110",
            "localized_precision": "0.55",
            "sise_logz_wrong_excl_sibling": str(method_sise),
            "sise_p0199_wrong_excl_sibling": "150",
            "wrong_excl_sibling_ge_thr_score_ge_0p5": "11",
            "sise_logz_ge_thr_score_ge_0p5": "0",
            "correct_topk_top5000": "95",
            "sise_logz_topk_top5000": "21",
            "sise_p0199_topk_top5000": "42",
        })


def test_audit_marks_g3_v1_negative_partial_when_nog3_beats_g3(tmp_path):
    eval_root = tmp_path / "eval_root"
    risk_root = tmp_path / "risk_root"
    out_dir = tmp_path / "out"
    write_eval_json(eval_root / "eval_epoch3_baseline_full", 0.83,
                    nested=True)
    write_eval_json(eval_root / "eval_nog3_ctrl_e2", 0.86)
    write_eval_json(eval_root / "eval_g3_w005_e2", 0.855)
    write_risk_csv(
        risk_root / "hrrsd_nog3_ctrl_e2/deployment_risk_variant_summary.csv",
        "nog3",
        str(eval_root / "eval_nog3_ctrl_e2/predictions.pkl"),
        100,
        70,
    )
    write_risk_csv(
        risk_root / "hrrsd_g3_w005_e2/deployment_risk_variant_summary.csv",
        "g3",
        str(eval_root / "eval_g3_w005_e2/predictions.pkl"),
        100,
        40,
    )

    review = audit_g3_v1_results(
        eval_root=eval_root,
        risk_root=risk_root,
        out_dir=out_dir,
        result_md=out_dir / "g3_audit.md",
    )

    assert review["status"] == "negative_partial"
    assert review["baseline"]["mAP"] == 0.83
    assert review["best_nog3"]["exp_id"] == "nog3_ctrl_e2"
    assert review["best_g3_by_map"]["exp_id"] == "g3_w005_e2"
    assert review["best_g3_beats_best_nog3_map"] is False
    assert review["best_g3_by_sise_reduction"]["sise_logz_total_delta"] == -60
    assert (out_dir / "g3_v1_dense_head_audit.csv").exists()
    assert (out_dir / "g3_audit.md").exists()
