_base_ = ['./G02_Baselines_Data1_DOTA1_M5_ORCNN_R50.py']

custom_imports = dict(
    imports=[
        'mmrotate.models.necks.faafusion',
        'mmrotate.models.roi_heads.bbox_heads.faa_head',
    ],
    allow_failed_imports=False)

find_unused_parameters = True

model = dict(
    neck=dict(
        type='FAAFusionFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        fusion_modes=['add', 'add', 'faa'],
        fam_cfg=dict(m=7, c_mid=64)),
    roi_head=dict(bbox_head=dict(type='FAAHead')))
