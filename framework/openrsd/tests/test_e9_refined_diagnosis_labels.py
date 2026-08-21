from M_Tools.analysis.refine_e9_diagnosis_labels import classify_case


def make_case(original, same, small, diagnosis='other_or_mixed'):
    return {
        'target_class': 'tennis-court',
        'impossible_class': 'small-vehicle',
        'original_cls': original,
        'neutral_same_cls': same,
        'neutral_small_cls': small,
        'diagnosis': diagnosis,
    }


def test_refines_stable_other_attractor_after_context_removal():
    refined = classify_case(make_case('small-vehicle', 'ship', 'ship'))

    assert refined['original_relation'] == 'impossible'
    assert refined['neutral_same_relation'] == 'other'
    assert refined['neutral_small_relation'] == 'other'
    assert refined['neutral_same_family'] == 'maritime_context'
    assert refined['refined_diagnosis'] == 'neutral_stable_other_attractor'


def test_refines_other_class_shift_between_same_and_small_scale():
    refined = classify_case(make_case('small-vehicle', 'harbor', 'ship'))

    assert refined['neutral_same_family'] == 'maritime_context'
    assert refined['neutral_small_family'] == 'maritime_context'
    assert refined['refined_diagnosis'] == 'neutral_other_class_shift'


def test_refines_context_shortcut_with_scale_induced_other_class():
    refined = classify_case(
        make_case('small-vehicle', 'tennis-court', 'ship', 'context_shortcut'))

    assert refined['neutral_same_relation'] == 'target'
    assert refined['neutral_small_relation'] == 'other'
    assert refined['refined_diagnosis'] == 'context_shortcut_small_scale_other'


def test_refines_scale_prior_sensitive_when_small_scale_impossible():
    refined = classify_case(
        make_case('small-vehicle', 'tennis-court', 'small-vehicle',
                  'context_shortcut'))

    assert refined['neutral_small_relation'] == 'impossible'
    assert refined['refined_diagnosis'] == 'context_shortcut_scale_prior_sensitive'


def test_refines_missing_variant_by_location():
    refined = classify_case(
        make_case('small-vehicle', 'harbor', 'no_match',
                  'insufficient_match'))

    assert refined['refined_diagnosis'] == 'insufficient_neutral_small_match'


def test_preserves_scale_decoupling_when_small_scale_is_unmatched():
    refined = classify_case(
        make_case('small-vehicle', 'small-vehicle', 'no_match',
                  'cls_reg_scale_decoupling'))

    assert refined['neutral_same_relation'] == 'impossible'
    assert refined['neutral_small_relation'] == 'no_match'
    assert refined['refined_diagnosis'] == 'cls_reg_scale_decoupling_small_unmatched'
