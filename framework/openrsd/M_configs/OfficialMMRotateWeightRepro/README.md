# Official MMRotate Weight Reproduction Configs

These files are direct copies from `open-mmlab/mmrotate` branch `1.x`.
They are kept as untouched DOTA1 official configs. Do not edit them for
DOTA2 fine-tuning; create adapter configs elsewhere if DOTA2 paths, class
counts, or `load_from` need to change.

## Present weights

| Local weight | Official config copy | Official source |
| --- | --- | --- |
| `weights/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth` | `oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py` | `https://raw.githubusercontent.com/open-mmlab/mmrotate/1.x/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py` |
| `weights/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth` | `kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py` plus `r3det/r3det-oc_r50_fpn_1x_dota.py` | `https://raw.githubusercontent.com/open-mmlab/mmrotate/1.x/configs/kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py` |
| `weights/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth` | `oriented_reppoints/oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py` | `https://raw.githubusercontent.com/open-mmlab/mmrotate/1.x/configs/oriented_reppoints/oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py` |
| `weights/redet_re50_fpn_1x_dota_le90-724ab2da.pth` | `redet/redet-le90_re50_refpn_1x_dota.py` | `https://raw.githubusercontent.com/open-mmlab/mmrotate/1.x/configs/redet/redet-le90_re50_refpn_1x_dota.py` |

## Five baseline counterparts

| Project baseline | Official config copy | Official weight status in `/weights` |
| --- | --- | --- |
| M1 RtnNetOBB | `rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py` | Missing: official weight is `rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth` |
| M2 RoITrans | `roi_trans/roi-trans-le90_r50_fpn_1x_dota.py` | Missing: official weight is `roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth` |
| M3 S2ANet | `s2anet/s2anet-le135_r50_fpn_1x_dota.py` | Missing: official weight is `s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth` |
| M4 R3Det_KFIoU | `kfiou/r3det-oc_r50_fpn_kfiou-ln_1x_dota.py` | Present |
| M5 ORCNN_R50 | `oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py` | Present |

## Not mapped to official MMRotate configs

| Local weight | Reason |
| --- | --- |
| `weights/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth` | LSKNet is not an official `open-mmlab/mmrotate` model zoo config in branch `1.x`. |
| `weights/lsk_s_fpn_1x_dota_le90_20230116-99749191_no_fc_cls.pth` | Derived local filtered checkpoint, not an official model zoo file. |
| `weights/lsk_t_backbone-2ef8a593.pth` | Backbone-only checkpoint, not a detector config. |
| `weights/epoch_8.pth` | Local checkpoint name has no official MMRotate model zoo identifier. |
| `weights/*_no_cls.pth`, `weights/*_no_fc_cls.pth`, `weights/*_mmrotate1xkeys.pth` | Local converted or filtered checkpoints derived from the official files. |

## Copied base files

- `_base_/datasets/dota.py`
- `_base_/schedules/schedule_1x.py`
- `_base_/schedules/schedule_40e.py`
- `_base_/default_runtime.py`

All copied configs parsed successfully with `mmengine.config.Config.fromfile`.

## Local copy audit

The existing root-level `mmrotate_configs/` directory was not trusted blindly.
These official copies were re-downloaded from GitHub raw.

- `oriented_rcnn`, `kfiou`, `redet`, `rotated_retinanet`, and `s2anet`
  matched the local `mmrotate_configs/` files.
- `roi_trans/roi-trans-le90_r50_fpn_1x_dota.py` did not match:
  the local copy had `num_classes = 20`, missed `_base_`, and missed the
  official `optim_wrapper = dict(optimizer=dict(lr=0.005))`.
- `oriented_reppoints/oriented-reppoints-qbox_r50_fpn_mstrain-40e_dota.py`
  differed in the image loading argument name
  (`file_client_args` locally versus official `backend_args`).
