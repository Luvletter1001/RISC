from M_Tools.rotation_sv_repair.dynamic_pseudo_label_queue import DynamicPseudoLabelQueue

def test_queue_not_sv_dominated():
    q = DynamicPseudoLabelQueue(k_per_class=10)
    for i in range(20):
        q.add(dict(class_name='tennis-court', confidence=0.9))
    for i in range(3):
        q.add(dict(class_name='small-vehicle', confidence=0.9))
    assert q.sv_fraction() < 0.35
