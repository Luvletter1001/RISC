from M_Tools.analysis.summarize_e9_counterfactual_results import (
    diagnose_case,
    summarize_cases,
)


def make_row(variant, matched_class, target_matched='1'):
    return {
        'model': 'demo_model',
        'case_index': '0',
        'orig_id': 'P0001',
        'tile_img_id': 'P0001__1024__0___0',
        'variant': variant,
        'target_matched': target_matched,
        'matched_class': matched_class,
        'matched_score': '0.9',
        'target_gt_class': 'large-vehicle',
        'impossible_pred_class': 'ship',
        'error': '',
    }


def test_diagnose_case_accepts_pred_class_scale_variant():
    case_rows = {
        'original_tile': make_row('original_tile', 'ship'),
        'neutral_same_scale': make_row(
            'neutral_same_scale', 'large-vehicle'),
        'neutral_pred_class_scale': make_row(
            'neutral_pred_class_scale', 'ship'),
    }

    diagnosis = diagnose_case(case_rows)

    assert diagnosis['scale_variant'] == 'neutral_pred_class_scale'
    assert diagnosis['neutral_scale_cls'] == 'ship'
    assert diagnosis['diagnosis'] == 'scale_prior_sensitive'


def test_summarize_cases_marks_pred_class_scale_triplets_complete():
    rows = [
        make_row('original_tile', 'ship'),
        make_row('neutral_same_scale', 'large-vehicle'),
        make_row('neutral_pred_class_scale', 'ship'),
    ]

    cases = summarize_cases(rows)

    assert len(cases) == 1
    assert cases[0]['complete_variants'] == 1
    assert cases[0]['scale_variant'] == 'neutral_pred_class_scale'
