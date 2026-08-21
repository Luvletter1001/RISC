from M_Tools.rotation_sv_repair.common import CLASSES, SMALL

def test_small_vehicle_index():
    assert CLASSES[SMALL] == 'small-vehicle'
    assert SMALL == CLASSES.index('small-vehicle')

def test_dota_class_count():
    assert len(CLASSES) == 15

def test_court_classes_present():
    for c in ('tennis-court', 'baseball-diamond', 'soccer-ball-field'):
        assert c in CLASSES
