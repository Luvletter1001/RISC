from M_Tools.analysis.run_p2_geometry_calibration_baseline import (
    adjusted_score,
    evaluate_calibration,
)


def test_adjusted_score_penalizes_only_flagged_scale_violations():
    assert adjusted_score(0.9, 2.0, True, 1.0) < 0.5
    assert adjusted_score(0.9, 2.0, False, 1.0) == 0.9


def test_evaluate_calibration_reports_wrong_reduction_and_correct_retention():
    wrong_rows = [
        {'score': '0.90', 'scale_prior_violation': 2.0,
         'is_prior_outlier_sise': True},
        {'score': '0.80', 'scale_prior_violation': 0.1,
         'is_prior_outlier_sise': False},
    ]
    correct_rows = [
        {'score': '0.95', 'scale_prior_violation': 0.1,
         'is_prior_outlier_flagged': False},
        {'score': '0.90', 'scale_prior_violation': 2.0,
         'is_prior_outlier_flagged': True},
    ]

    result = evaluate_calibration(
        wrong_rows, correct_rows, lam=1.0, score_thr=0.5,
        wrong_flag_key='is_prior_outlier_sise',
        correct_flag_key='is_prior_outlier_flagged')

    assert result['high_conf_wrong_before'] == 2
    assert result['high_conf_wrong_after'] == 1
    assert result['high_conf_wrong_reduction'] == 1
    assert result['high_conf_correct_before'] == 2
    assert result['high_conf_correct_after'] == 1
    assert result['high_conf_correct_retention'] == 0.5
