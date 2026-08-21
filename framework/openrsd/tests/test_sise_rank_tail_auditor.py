from M_Tools.analysis.evaluate_sise_deployment_risk import (
    add_rank_fields,
    pair_records_by_aligned_det_idx,
    summarize_rank_tail_audit,
)


def make_record(
        *,
        variant="base",
        img_id="img_a",
        det_idx=0,
        score=0.9,
        pred_class="ship",
        gt_class="ship",
        correct=True,
        wrong_excl_sibling=False,
        log_area_z=0.2,
        logz_implausible=False):
    return {
        "variant": variant,
        "img_id": img_id,
        "det_idx": det_idx,
        "score": score,
        "pred_class": pred_class,
        "gt_class": gt_class,
        "correct": correct,
        "wrong": not correct,
        "wrong_excl_sibling": wrong_excl_sibling,
        "log_area_z": log_area_z,
        "logz_implausible": logz_implausible,
        "p0199_outlier": logz_implausible,
        "risk_logz": score * max(0.0, log_area_z - 4.0)
        if wrong_excl_sibling else 0.0,
        "risk_p0199": score if wrong_excl_sibling and logz_implausible else 0.0,
        "pair": f"{pred_class}->{gt_class}",
    }


def test_add_rank_fields_assigns_class_global_and_image_rank():
    records = [
        make_record(img_id="img_a", det_idx=0, score=0.90, pred_class="ship"),
        make_record(
            img_id="img_b",
            det_idx=1,
            score=0.95,
            pred_class="vehicle",
            gt_class="ship",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=5.1,
            logz_implausible=True),
        make_record(
            img_id="img_a",
            det_idx=2,
            score=0.80,
            pred_class="ship",
            gt_class="vehicle",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=4.5,
            logz_implausible=True),
    ]

    ranked = add_rank_fields(records)
    by_key = {(r["img_id"], r["det_idx"]): r for r in ranked}

    assert by_key[("img_b", 1)]["global_review_rank"] == 1
    assert by_key[("img_a", 0)]["global_review_rank"] == 2
    assert by_key[("img_a", 2)]["global_review_rank"] == 3
    assert by_key[("img_a", 0)]["class_rank"] == 1
    assert by_key[("img_a", 2)]["class_rank"] == 2
    assert by_key[("img_a", 0)]["image_rank"] == 1
    assert by_key[("img_a", 2)]["image_rank"] == 2


def test_pair_records_by_aligned_det_idx_matches_same_original_detection():
    base = add_rank_fields([
        make_record(
            img_id="img_a",
            det_idx=7,
            score=0.91,
            pred_class="ship",
            gt_class="vehicle",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=5.2,
            logz_implausible=True),
    ])
    method = add_rank_fields([
        make_record(
            variant="g1",
            img_id="img_a",
            det_idx=7,
            score=0.31,
            pred_class="ship",
            gt_class="vehicle",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=5.2,
            logz_implausible=True),
    ])

    pairs, stats = pair_records_by_aligned_det_idx(base, method)

    assert stats["paired_count"] == 1
    assert stats["base_unmatched_count"] == 0
    assert len(pairs) == 1
    assert pairs[0]["img_id"] == "img_a"
    assert pairs[0]["det_idx"] == 7
    assert pairs[0]["base_score"] == 0.91
    assert pairs[0]["method_score"] == 0.31


def test_summarize_rank_tail_audit_counts_sise_benefit_and_tp_harm():
    base = add_rank_fields([
        make_record(img_id="img_a", det_idx=1, score=0.90, pred_class="ship"),
        make_record(
            img_id="img_b",
            det_idx=2,
            score=0.80,
            pred_class="ship",
            gt_class="vehicle",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=5.4,
            logz_implausible=True),
        make_record(img_id="img_c", det_idx=3, score=0.70, pred_class="plane"),
    ])
    method = add_rank_fields([
        make_record(
            variant="g1",
            img_id="img_a",
            det_idx=1,
            score=0.20,
            pred_class="ship"),
        make_record(
            variant="g1",
            img_id="img_b",
            det_idx=2,
            score=0.10,
            pred_class="ship",
            gt_class="vehicle",
            correct=False,
            wrong_excl_sibling=True,
            log_area_z=5.4,
            logz_implausible=True),
        make_record(
            variant="g1",
            img_id="img_c",
            det_idx=3,
            score=0.70,
            pred_class="plane"),
    ])
    pairs, _ = pair_records_by_aligned_det_idx(base, method)

    summary = summarize_rank_tail_audit(
        "base", "g1", pairs, score_floor=0.3, rank_drop_tol=0, tail_z_thr=4.0)

    assert summary["base_variant"] == "base"
    assert summary["method_variant"] == "g1"
    assert summary["paired_count"] == 3
    assert summary["correct_rank_drop_count"] == 1
    assert summary["ap_contributing_fp_removed"] == 1
    assert summary["ap_sensitive_sise_rate"] == 1.0
    assert summary["rank_disruptive_delta"] > 0
