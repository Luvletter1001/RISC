_base_ = './hrrsd_p16a_p15o_b_dec2_noencoder_topk1_generalization500_gpu89_b32_v5.py'

work_dir = (
    'work_dirs/p16_generalization_hrrsd_20260625/'
    'p16b_p15o_b_dec2_resnet50init_generalization500_gpu89_b32_v5')

# P16B tests the strongest root-cause hypothesis from P16A diagnostics:
# P16A trained a ResNet50 backbone from random init while keeping BN frozen.
# Overfit/train performance was non-zero, but val500 collapsed. This keeps the
# strict E2E/no-NMS decoder path unchanged and only restores ImageNet backbone
# initialization, matching common detection practice.
model = dict(
    backbone=dict(
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            checkpoint='torchvision://resnet50')))

randomness = dict(seed=3407, deterministic=False)
