import numpy as np
from M_Tools.rotation_sv_repair.orbit_distillation import map_box_to_reference, view_consensus_teacher


def test_rotation_360_identity():
    b = np.array([10., 20., 100., 200.])
    out = map_box_to_reference(b, 90, 90)
    assert np.allclose(out, b, atol=1e-3)


def test_rotation_90_270_inverse():
    b = np.array([10., 20., 100., 200.])
    b90 = map_box_to_reference(b, 0, 90)
    back = map_box_to_reference(b90, 90, 0)
    assert np.allclose(back, b, atol=2.0)


def test_orbit_teacher_box_reasonable_after_mapping():
    box0 = [100.0, 120.0, 200.0, 240.0]
    box90 = map_box_to_reference(np.array(box0), 0, 90).tolist()
    out = view_consensus_teacher({
        0: [{'bbox': box0, 'class_id': 10, 'score': 0.9}],
        90: [{'bbox': box90, 'class_id': 10, 'score': 0.8}],
    }, iou_thr=0.9, min_views=2)
    assert len(out) == 1
    assert out[0]['class_id'] == 10
    assert np.allclose(out[0]['bbox'], box0, atol=1e-4)
