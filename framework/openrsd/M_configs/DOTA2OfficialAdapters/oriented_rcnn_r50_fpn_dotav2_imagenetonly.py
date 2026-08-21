_base_ = './oriented_rcnn_r50_fpn_dotav2_fullinit.py'

# Keep the official ORCNN architecture and DOTA2 adapter, but remove the
# detector-level DOTA checkpoint. The ResNet backbone still uses its
# torchvision ImageNet init from the official base config.
load_from = None

work_dir = 'work_dirs/dotav2_official_adapters/oriented_rcnn_r50_fpn_imagenetonly_12e'
