_base_ = [
    '../../projects/GroundingDINO/configs/'
    'grounding_dino_swin-t_visdrone_base-set_adamw.py'
]

custom_imports = dict(
    imports=['projects.OVCapFlow.ov_capflow'], allow_failed_imports=False)

model = dict(
    type='OVCapFlow',
    num_queries=900,
    decoder=dict(
        layer_cfg=dict(semantic_fusion_cfg=dict(adapter_init='identity'))),
    bbox_head=dict(type='OVCapFlowHead'),
    # OVCapFlowHead deliberately ignores max_per_img. Removing the inherited
    # test config makes that invariant visible in the resolved configuration.
    test_cfg=dict(_delete_=True))
