from mmengine import Config
from mmengine.utils import import_modules_from_strings

from mmrotate.registry import MODELS


def test_strict_config_registers_ov_capflow_components():
    cfg = Config.fromfile(
        'configs/ov_capflow/ov_capflow_swin-t_strict_visdrone.py')

    import_modules_from_strings(**cfg.custom_imports)

    assert cfg.model.type == 'OVCapFlow'
    assert cfg.model.bbox_head.type == 'OVCapFlowHead'
    assert cfg.model.num_queries == 900
    assert MODELS.get('OVCapFlow') is not None
    assert MODELS.get('OVCapFlowHead') is not None
