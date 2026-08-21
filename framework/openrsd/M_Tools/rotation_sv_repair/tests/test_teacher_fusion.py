from M_Tools.rotation_sv_repair.openrsd_head_teacher import head_consensus_label
from M_Tools.rotation_sv_repair.lae_dino_style_prompt import prompt_ensemble_scores


def test_court_override_when_heads_agree():
    r = head_consensus_label(
        'tennis-court', 'tennis-court', 'tennis-court', 'baseball-diamond', 'small-vehicle')
    assert r['consensus_label'] == 'tennis-court'
    assert r['repair_action'] == 'court_pseudo_override'


def test_unstable_sv_single_head():
    r = head_consensus_label('harbor', 'bridge', 'tennis-court', 'ship', 'small-vehicle')
    assert r['is_unstable_sv'] or r['repair_action'] in ('downweight_sv', 'mark_unstable_sv', 'none')


def test_prompt_ensemble_single_prompt_equivalent():
    scores = {'tennis-court': [0.7], 'small-vehicle': [0.2]}
    out = prompt_ensemble_scores(scores, mode='robust_mean')
    assert out['tennis-court'] == 0.7
    assert out['small-vehicle'] == 0.2
