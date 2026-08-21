import torch

src_ckpt = '/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth'
dst_ckpt = '/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth'


# PyTorch 2.x 支持 weights_only 参数；旧版 PyTorch 不支持，自动退回兼容写法。
try:
    ckpt = torch.load(src_ckpt, map_location='cpu', weights_only=False)
except TypeError:
    ckpt = torch.load(src_ckpt, map_location='cpu')

# 只保留模型权重
new_ckpt = {
    'state_dict': ckpt['state_dict']
}

# 如果你用到了 EMA（很多 mmengine 项目会用）
if 'ema_state_dict' in ckpt:
    new_ckpt['ema_state_dict'] = ckpt['ema_state_dict']

torch.save(new_ckpt, dst_ckpt)

print('Saved weights-only checkpoint to:', dst_ckpt)
