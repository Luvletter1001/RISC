# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from pathlib import Path
import sys

# ==================== 全局配置参数 ====================
proj_root = '/data1/zcy/OpenRSD'

def _path_matches(path_item, target):
    if path_item == '':
        path_item = os.getcwd()
    return os.path.abspath(path_item) == target


# Prefer packages installed in the conda env. The repository also contains a
# partial mmengine checkout, which shadows the installed package and misses
# modules such as mmengine.dist.
sys.path = [
    p for p in sys.path
    if not p.startswith('/home/zcy/.local/')
    and not _path_matches(p, proj_root)
]
os.environ.setdefault('MPLCONFIGDIR', '/data1/zcy/OpenRSD/SimpleRun/.mplconfig')
import json
import pickle as pkl
import traceback
import cv2
import pandas as pd
import numpy as np
def pklsave(obj, file_path, msg=True):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        if msg:
            print('SAVE OBJ: %s' % file_path)
def pklload(file_path, msg=True):
    with open(file_path, 'rb') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
        try:
            return pkl.load(f)
        except Exception:
            traceback.print_exc()
from mmengine.utils import path

# 目录以及文件
sys.path.insert(0, proj_root)
os.chdir(proj_root)
out_ann_pth = os.environ.get(
    'OPENRSD_OUT_RESULTS', './SimpleRun/results/results.pkl')
out_ann_dir = os.path.dirname(out_ann_pth)
if out_ann_dir:
    os.makedirs(out_ann_dir, exist_ok=True)
save_vis_images = os.environ.get('OPENRSD_SAVE_VIS', '1') != '0'
save_empty_vis_images = os.environ.get('OPENRSD_SAVE_EMPTY_VIS', '1') != '0'
vis_out_dir = os.environ.get('OPENRSD_VIS_OUT_DIR', './SimpleRun/results/vis')
vis_img_ext = os.environ.get('OPENRSD_VIS_IMG_EXT', '.jpg')
vis_box_only = os.environ.get('OPENRSD_VIS_BOX_ONLY', '0') != '0'
os.makedirs(vis_out_dir, exist_ok=True)
# 模型配置
work_dir = './SimpleRun/work_dirs'
config_pth = os.environ.get(
    'OPENRSD_CONFIG',
    './M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py')
checkpoint_pth = os.environ.get(
    'OPENRSD_CHECKPOINT',
    './results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24_weights_only.pth')


# GPU设置
gpu = int(os.environ.get('OPENRSD_GPU', '4'))

# 数据集配置
val_batch_size = int(os.environ.get('OPENRSD_BATCH_SIZE', '1'))
val_num_workers = int(os.environ.get('OPENRSD_NUM_WORKERS', '2'))
filter_empty_gt = os.environ.get('OPENRSD_FILTER_EMPTY_GT', '1') != '0'
# DOTA split dataset on this server. Expected image path:
# /data/zcy/dataset/test_ss/images/*.png
data_root = os.environ.get('OPENRSD_DATA_ROOT', '/data/zcy/dataset/test_ss')
img_dir = os.environ.get('OPENRSD_IMG_DIR', 'images')
ann_dir = os.environ.get(
    'OPENRSD_ANN_DIR',
    '/data1/zcy/OpenRSD/SimpleRun/generated_annfiles/dota_test_ss')
base_pkl_content = pklload('./SimpleRun/test_dataset/annfiles/P1860__1024__932___714.pkl')
# dataloader（MMRotate_AD_Pub\M_AD\datasets\dota_online_v1.py）是根据annfiles获取图像信息的
# 因此需要先设置虚拟的annfiles
ann_root = ann_dir if os.path.isabs(ann_dir) else os.path.join(data_root, ann_dir)
os.makedirs(ann_root, exist_ok=True)
for image_file in os.listdir(os.path.join(data_root, img_dir)):
    img_name = Path(image_file).stem
    ann_file = os.path.join(ann_root, f'{img_name}.pkl')
    if not os.path.exists(ann_file):
        pklsave(base_pkl_content, file_path=ann_file, msg=True)

# 待检测的类别，
cls_list = [
    'Baseball_Field', 'Basketball_Court', 'Bridge', 'Ground_Track_Field',
    'Harbor', 'Helicopter', 'Large_Vehicle', 'Plane', 'Roundabout', 'Ship',
    'Small_Vehicle', 'Soccer_Ball_Field', 'Storage_Tank',
    'Swimming_Pool', 'Tennis_Court',
]
cls_list = list(cls_list)
print(cls_list)

# 测试数据集配置
test_img_shape = (1024, 1024)  # test_dataset的img_shape
test_img_scale = (1024, 1024)  # test_pipeline的resize scale

# 采样器配置
sampler_seed = 2024

# Support特征字典配置
support_feat_dict = {
    'Data1_DOTA1': './data/DOTA_800_600/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl',
}

# 类别归一化字典路径
normalized_class_dict = './data/normalized_class_dict.pkl'

# 模型推理配置
val_using_aux = False  # True为使用Fusion Head，False为使用Align Head
support_shot = 8  # support shot数量

# 检测阈值配置
iou_thr = 0.5
score_thr = 0.3

# ====================================================

print(f'Process: gpu {gpu}')
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
os.chdir(proj_root)
print(f'Current working directory: {os.getcwd()}')
print(f'iou_thr: {iou_thr}, score_thr: {score_thr}')
print('#' * 100)


import os.path as osp
import time
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules

from mmcv.ops.roi_align_rotated import RoIAlignRotated
from mmdet.structures.bbox import bbox2roi
from mmrotate.registry import MODELS
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from commonlibs.common_tools import *
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.structures.instance_data import InstanceData

from mmrotate.registry import DATASETS
from torch.utils.data import DataLoader
from ctlib.rbox import obb2poly, obb2xyxy
from ctlib.vis import draw_polys
from copy import deepcopy
import cv2
import math
from ctlib.os import *
import os
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from ctlib.os import *
from mmcv.ops import box_iou_rotated
from tqdm import tqdm
from mmcv.ops.nms import nms_rotated

def obb2hbb(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs : [x_ctr,y_ctr,w,h,angle]

    Returns:
        outer hbb in obb format
    """
    N = obboxes.shape[0]
    if N == 0:
        return obboxes
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    xyxy = torch.cat([center - bias, center + bias], dim=-1)
    x1, y1, x2, y2 = torch.split(xyxy, [1, 1, 1, 1], dim=-1)
    a = torch.zeros_like(x1).to(obboxes.device)
    hbb = torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1, a], dim=-1)

    return hbb


def _vis_color(class_name):
    palette = [
        (56, 56, 255), (80, 220, 60), (255, 140, 40),
        (230, 80, 180), (40, 190, 240), (180, 160, 40),
        (255, 90, 90), (90, 220, 220), (170, 90, 255),
        (70, 170, 120), (220, 120, 40), (120, 120, 255),
    ]
    idx = sum(ord(ch) for ch in str(class_name)) % len(palette)
    return palette[idx]


def save_detection_vis(img_pth, out_pth, polys, texts, scores):
    img = cv2.imread(img_pth)
    if img is None:
        print(f'WARNING: failed to read image for visualization: {img_pth}')
        return False

    h, w = img.shape[:2]
    for poly, text, score in zip(polys, texts, scores):
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
        pts = np.round(pts).astype(np.int32)

        color = _vis_color(text)
        cv2.polylines(img, [pts], True, color, thickness=2,
                      lineType=cv2.LINE_AA)

        if not vis_box_only:
            label = f'{text} {float(score):.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale,
                                                 thickness)
            x = int(np.min(pts[:, 0]))
            y = int(np.min(pts[:, 1]))
            x = min(max(x, 0), max(w - tw - 6, 0))
            y = max(y - 4, th + baseline + 4)
            cv2.rectangle(img, (x, y - th - baseline - 4),
                          (min(x + tw + 6, w - 1), min(y + baseline, h - 1)),
                          color, thickness=-1)
            cv2.putText(img, label, (x + 3, y - 3), font, font_scale,
                        (255, 255, 255), thickness=thickness,
                        lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    ext = os.path.splitext(out_pth)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return cv2.imwrite(out_pth, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.imwrite(out_pth, img)


# TODO: support fuse_conv_bn and format_only
def parse_args(args=''):
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():

    """
    多数据集联合训练
    epoch = 24
    cfg_dir = 'A10_Large_Pretrain_Stage3'
    cfg_name = 'A10_flex_rtm_v3_1_formal'

    使用了Self-Training之后
    epoch = 24
    cfg_dir = 'A12_SelfTrain'
    cfg_name = 'A12_flex_rtm_v3_1_self_training_Labelver5'

    """
    epoch = 24
    cfg_dir = 'A12_SelfTrain'
    cfg_name = 'A12_flex_rtm_v3_1_self_training_Labelver5'
    args = [config_pth,
            checkpoint_pth,
            '--work-dir', work_dir
            ]


    args = parse_args(args)

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    dota1_support_classes = [
        'baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field',
        'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship',
        'small-vehicle', 'soccer-ball-field', 'storage-tank',
        'swimming-pool', 'tennis-court',
    ]
    minimal_neg_support = './SimpleRun/generated_annfiles/Neg_supports_dota1_minimal.pkl'
    if not os.path.exists(minimal_neg_support):
        pklsave(
            {'neg_dict': {'Data1_DOTA1': {c: [] for c in dota1_support_classes}}},
            minimal_neg_support)
    cfg.model.support_feat_dict = support_feat_dict
    cfg.model.val_support_classes = dota1_support_classes
    cfg.model.val_dataset_flag = 'Data1_DOTA1'
    cfg.model.neg_support_data = minimal_neg_support
    cfg.model.normalized_class_dict = normalized_class_dict
    cfg.model.pca_meta_pth = './data/7_25_pca_meta_DINOv2_256.pkl'
    cfg.model.val_using_aux = val_using_aux
    #############################################################################

    from M_AD.datasets.transforms.loading import LoadAnnotationsOnline
    from M_AD.datasets.transforms.formatting import PackDetInputsMM
    from M_AD.datasets.transforms.transforms import ConvertBoxTypeSafe

    test_pipeline = [
        dict(type='mmdet.LoadImageFromFile', file_client_args= dict(backend='disk')),
        dict(type='mmrotate.LoadAnnotationsOnline', with_bbox=True, box_type='qbox'),
        dict(type='mmrotate.ConvertBoxTypeSafe', box_type_mapping=dict(gt_bboxes='rbox')),
        dict(type='mmdet.Resize', scale=test_img_scale, keep_ratio=True),
        dict(
            type='mmdet.Pad', size=test_img_scale,
            pad_val=dict(img=(114, 114, 114))),
        dict(
            type='mmrotate.PackDetInputsMM',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]
    test_dataset =dict(
        type='mmrotate.DOTADatasetOnline',
        data_root=data_root,
        ann_file=ann_dir,
        data_prefix=dict(img_path=img_dir),
        img_shape=test_img_shape,
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=filter_empty_gt),
        pipeline=test_pipeline)

    dataloader_cfg = dict(
        batch_size=val_batch_size,
        num_workers=val_num_workers,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False))
    dataset = DATASETS.build(test_dataset)
    sampler_cfg = dataloader_cfg.pop('sampler')
    sampler = DATA_SAMPLERS.build(
        sampler_cfg,
        default_args=dict(dataset=dataset, seed=sampler_seed))
    batch_sampler = None
    collate_fn_cfg = dict(type='pseudo_collate')
    collate_fn_type = collate_fn_cfg.pop('type')
    if isinstance(collate_fn_type, str):
        collate_fn = FUNCTIONS.get(collate_fn_type)
    else:
        collate_fn = collate_fn_type
    collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **dataloader_cfg)


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.call_hook('before_run')
    runner.load_or_resume()
    model = runner.model
    model.eval()
    ##################################################################################

    # ---- 构造uni_support_data，对类别名称进行归一化
    norm_cls_map = pklload(normalized_class_dict)
    support_data_dict = dict()
    uni_support_data = dict()
    for dataset_flag, support_feat_pth in support_feat_dict.items():
        support_data = pklload(support_feat_pth)
        normed_support_data = {}
        for k, v in support_data.items():
            normed_k = norm_cls_map[k]
            if len(v['visual_embeds']) <= 10:
                v['visual_embeds'] = np.concatenate([v['visual_embeds'] for i in range(10)])
            if len(v['text_embeds']) <= 10:
                v['text_embeds'] = np.concatenate([v['text_embeds'] for i in range(10)])
            normed_support_data[normed_k] = v
            # ----- 构造统一的support
            if normed_k in uni_support_data.keys():
                # ------ 合并同类名
                uni_support_data[normed_k]['visual_embeds'] = np.concatenate([v['visual_embeds'],
                                                                              uni_support_data[normed_k]
                                                                              ['visual_embeds']])
                uni_support_data[normed_k]['text_embeds'] = np.concatenate([v['text_embeds'],
                                                                            uni_support_data[normed_k]
                                                                            ['text_embeds']])
            else:
                uni_support_data[normed_k] = v
        ### ---- 检查前后一致性
        if len(normed_support_data) != len(support_data):
            raise Exception(f'UniSupport Error | pre: {list(support_data.keys())}, '
                            f'Normed: {list(normed_support_data.keys())}')
        support_data_dict[dataset_flag] = normed_support_data

    count = 0

    n_total = len(data_loader)
    start_time = time.time()

    def save_empty(out_ann_pth):
        out_ann = dict(
            visual_embeds=None,
            texts=[],
            scores=[],
            text_embeds=None,
            polys=[],
            cls_list=[],
            # hard_negatives=hard_negatives
        )
        return out_ann

    def get_pred_results(results, id2name,
                         iou_thr=0.1,
                         score_thr=0.3):
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_pred_texts = []
        all_classes = []
        latent_classes = []
        for img_id in range(len(results)):
            sample = results[img_id]
            device = sample.pred_instances.bboxes.device

            pred_boxes = sample.pred_instances.bboxes.detach()
            pred_labels = sample.pred_instances.labels.detach()
            pred_scores = sample.pred_instances.scores.detach()
            ####### NMS
            dets, keep_inds = nms_rotated(pred_boxes, pred_scores, iou_threshold=iou_thr)
            pred_boxes = pred_boxes[keep_inds]
            pred_scores = pred_scores[keep_inds]
            pred_labels = pred_labels[keep_inds]
            if keep_inds == None or torch.sum(keep_inds) == 0:
                all_pred_boxes.append(torch.zeros([0, 5]).to(device))
                all_pred_scores.append(torch.zeros(0).to(device))
                all_pred_labels.append(torch.zeros(0).long().to(device))
                all_pred_texts.append([])
                latent_classes.append([])
                continue


            pred_texts = [id2name[int(l)] for l in pred_labels]
            latent_classes.append(sorted(list(set(pred_texts))))

            ####### Scores
            keep_inds = pred_scores >= score_thr
            pred_boxes = pred_boxes[keep_inds]
            pred_scores = pred_scores[keep_inds]
            pred_labels = pred_labels[keep_inds]

            pred_texts = [id2name[int(l)] for l in pred_labels]
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_pred_labels.append(pred_labels)
            all_pred_texts.append(pred_texts)
            all_classes.extend(list(set(pred_texts)))
        all_classes = sorted(list(set(all_classes)))

        return all_pred_boxes, all_pred_scores, all_pred_labels, \
            all_pred_texts, all_classes, latent_classes

    all_results = dict()
    with torch.no_grad():
        for data_info in data_loader:
            count += 1
            ####### ------- 获得特征
            data = model.data_preprocessor(data_info, False)
            data_samples = data['data_samples']
            feat_x = model.prompt_extract_feats(data['inputs'])

            ####### ------- 设置不同的prompt，获得检测结果
            uni_cls_list = sorted(list(set(uni_support_data.keys())))
            ###### ----------- Fusion Head
            det_support_data = {k: v
                                for k, v in uni_support_data.items() if k in
                                cls_list}
            print(len(det_support_data))
            print(det_support_data.keys())
            name2id = {name: cat_id for cat_id, name in enumerate(det_support_data.keys())}
            id2name = {cat_id: name for cat_id, name in enumerate(det_support_data.keys())}

            fusion_results = model.prompt_predict(
                deepcopy(feat_x),
                data['inputs'],
                deepcopy(data_samples),
                val_support_data=det_support_data,
                val_name2id=name2id,
                val_using_aux=val_using_aux,
                rescale=True,
                support_shot=support_shot,
            )

            fusion_pred_boxes, fusion_pred_scores, fusion_pred_labels, \
                fusion_pred_texts, fusion_classes, _ = \
                get_pred_results(fusion_results, id2name,
                                 iou_thr=iou_thr, score_thr=score_thr)

            n_img = len(data['data_samples'])
            for img_id in range(n_img):
                sample = data_samples[img_id]
                img_pth = sample.img_path
                img_name = Path(img_pth).stem

                nms_bboxes = fusion_pred_boxes[img_id].detach().cpu()
                nms_scores = fusion_pred_scores[img_id].detach().cpu()
                nms_texts = fusion_pred_texts[img_id]
                vis_pth = os.path.join(vis_out_dir, f'{img_name}{vis_img_ext}')
                if len(nms_bboxes) == 0:
                    empty_ann = save_empty('')
                    all_results[img_name] = empty_ann
                    if save_vis_images and save_empty_vis_images:
                        save_detection_vis(img_pth, vis_pth, [], [], [])
                    continue

                polys = obb2poly(nms_bboxes).numpy()
                scores = nms_scores.numpy()
                ######################################
                out_ann = dict(
                    visual_embeds=None,
                    texts=nms_texts,
                    scores=scores,
                    text_embeds=None,
                    polys=polys,
                    cls_list=cls_list,
                    # hard_negatives=hard_negatives
                )
                all_results[img_name] = out_ann
                if save_vis_images:
                    save_detection_vis(img_pth, vis_pth, polys, nms_texts,
                                       scores)
                print(f'{img_name} Done')
                print(out_ann)
                # pklsave(out_ann, out_ann_pth, msg=False)

            end_time = time.time()
            avg_time = (end_time - start_time) / count
            eta_time = int(avg_time * (n_total - count)) / 3600
            print(f'{count} / {n_total}, avg {avg_time: .3f}, eta {eta_time: .3f} hour')

    pklsave(all_results, out_ann_pth)


if __name__ == '__main__':
    main()
