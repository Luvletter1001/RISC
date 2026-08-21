#!/usr/bin/env python3
"""Build DOTA2 OpenRSD support features with the DeCLIP EVA-B encoder."""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DOTA2_CLASSES = [
    'airport', 'baseball-diamond', 'basketball-court', 'bridge',
    'container-crane', 'ground-track-field', 'harbor', 'helicopter',
    'helipad', 'large-vehicle', 'plane', 'roundabout', 'ship',
    'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool',
    'tennis-court',
]

PROMPT_TEMPLATES = [
    'A photo of a {} in the aerial image.',
    'A remote sensing image containing a {}.',
    'An overhead view of a {}.',
    'A satellite image showing a {}.',
    'A cropped aerial object of a {}.',
    'A {} visible from above.',
    'An aerial scene with a {}.',
    'A remote sensing crop centered on a {}.',
]

IMAGE_MEAN = (123.675, 116.280, 103.530)
IMAGE_STD = (58.395, 57.120, 57.375)


def l2_normalize_rows(rows: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    denom = np.linalg.norm(rows, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return rows / denom


def pad_or_truncate_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f'features must be 2D, got shape {features.shape}')
    if features.shape[1] == target_dim:
        return features
    out = np.zeros((features.shape[0], target_dim), dtype=np.float32)
    n_dim = min(features.shape[1], target_dim)
    out[:, :n_dim] = features[:, :n_dim]
    return out


def crop_axis_aligned_from_poly(
        image: np.ndarray,
        poly: Sequence[float],
        pad: int = 2) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f'image must be HxWxC, got shape {arr.shape}')
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    h, w = arr.shape[:2]
    x1 = max(0, int(np.floor(float(pts[:, 0].min()))) - pad)
    y1 = max(0, int(np.floor(float(pts[:, 1].min()))) - pad)
    x2 = min(w, int(np.ceil(float(pts[:, 0].max()))) + pad + 1)
    y2 = min(h, int(np.ceil(float(pts[:, 1].max()))) + pad + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'empty crop for poly {poly}')
    return arr[y1:y2, x1:x2].copy()


def select_topk_rows(
        rows: np.ndarray,
        scores: np.ndarray,
        topk: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(rows) == 0:
        raise ValueError('cannot select support rows from an empty class')
    if len(rows) != len(scores):
        raise ValueError(
            f'rows/scores length mismatch: {len(rows)} vs {len(scores)}')
    order = np.argsort(-scores)
    sorted_rows = rows[order]
    sorted_scores = scores[order]
    if len(sorted_rows) >= topk:
        return sorted_rows[:topk], sorted_scores[:topk]
    repeat_idx = np.arange(topk) % len(sorted_rows)
    return sorted_rows[repeat_idx], sorted_scores[repeat_idx]


def validate_support_schema(support: Dict[str, dict]) -> None:
    required = ('texts', 'text_embeds', 'visual_embeds', 'confidence_scores')
    for cls_name, entry in support.items():
        for key in required:
            if key not in entry:
                raise KeyError(f'{cls_name} missing required key {key}')
        text = np.asarray(entry['text_embeds'])
        visual = np.asarray(entry['visual_embeds'])
        conf = np.asarray(entry['confidence_scores'])
        if text.ndim != 2:
            raise ValueError(f'{cls_name} text_embeds must be 2D')
        if visual.ndim != 2:
            raise ValueError(f'{cls_name} visual_embeds must be 2D')
        if conf.ndim != 1 or len(conf) != len(visual):
            raise ValueError(
                f'{cls_name} confidence_scores must align with visual_embeds')


def iter_limited_paths(
        root: Path,
        pattern: str,
        max_files=None) -> List[Path]:
    paths = sorted(root.glob(pattern))
    if max_files is None or int(max_files) <= 0:
        return paths
    return paths[:int(max_files)]


def contains_needed_class(
        items: Sequence[Tuple[str, np.ndarray]],
        counts,
        max_candidates_per_class: int) -> bool:
    return any(
        counts[cls_name] < max_candidates_per_class
        for cls_name, _ in items
    )


def build_prompts(classes: Sequence[str]) -> Dict[str, List[str]]:
    return {
        cls_name: [template.format(cls_name.replace('-', ' '))
                   for template in PROMPT_TEMPLATES]
        for cls_name in classes
    }


def parse_dota_txt(txt_path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    with txt_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
                continue
            if len(parts) < 10:
                raise ValueError(
                    f'Invalid DOTA line in {txt_path}:{line_no}: {line.strip()}')
            cls_name = parts[8]
            if cls_name not in DOTA2_CLASSES:
                continue
            yield cls_name, np.asarray([float(x) for x in parts[:8]],
                                       dtype=np.float32)


def install_torchvision_v2_compat() -> None:
    try:
        import torchvision.transforms.v2  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType('torchvision.transforms.v2')

        class ScaleJitter:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, image):
                return image

        module.ScaleJitter = ScaleJitter
        sys.modules['torchvision.transforms.v2'] = module


class _LowerTriangularMask:
    pass


class _TorchXopsFallback:
    LowerTriangularMask = _LowerTriangularMask

    @staticmethod
    def memory_efficient_attention(
            q,
            k,
            v,
            p: float = 0.0,
            scale=None,
            attn_bias=None):
        batch, q_len, heads, dim = q.shape
        k_len = k.shape[1]
        q = q.permute(0, 2, 1, 3).reshape(batch * heads, q_len, dim)
        k = k.permute(0, 2, 1, 3).reshape(batch * heads, k_len, dim)
        v = v.permute(0, 2, 1, 3).reshape(batch * heads, k_len, dim)
        scale = scale if scale is not None else dim ** -0.5
        attn = torch.bmm(q * scale, k.transpose(1, 2))
        if isinstance(attn_bias, _LowerTriangularMask):
            mask = torch.ones(
                q_len, k_len, dtype=torch.bool, device=attn.device).triu(1)
            attn = attn.masked_fill(mask, float('-inf'))
        elif attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        if p:
            attn = F.dropout(attn, p=p)
        out = torch.bmm(attn, v)
        return out.reshape(batch, heads, q_len, dim).permute(0, 2, 1, 3)


def import_declip_open_clip(code_root: Path):
    install_torchvision_v2_compat()
    src_root = code_root / 'cat_seg' / 'src'
    if not src_root.is_dir():
        raise FileNotFoundError(
            f'DeCLIP CAT-Seg source not found: {src_root}')
    src = str(src_root.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    import open_clip  # noqa: WPS433
    import open_clip.eva_clip as eva_clip  # noqa: WPS433
    from open_clip.eva_clip import eva_vit_model, transformer  # noqa: WPS433

    if getattr(transformer, 'xops', None) is None:
        transformer.xops = _TorchXopsFallback
    if getattr(eva_vit_model, 'xops', None) is None:
        eva_vit_model.xops = _TorchXopsFallback
    return open_clip, eva_clip


def load_declip_model(
        code_root: Path,
        checkpoint: Path,
        device: str):
    open_clip, eva_clip = import_declip_open_clip(code_root)
    model = eva_clip.create_model(
        model_name='EVA02-CLIP-B-16',
        pretrained=str(checkpoint),
        force_custom_clip=True,
        precision='fp32',
        device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('EVA02-CLIP-B-16')
    return model, tokenizer


def preprocess_crop(crop: np.ndarray, image_size: int) -> torch.Tensor:
    image = Image.fromarray(crop.astype(np.uint8), mode='RGB')
    image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def encode_text_bank(
        model,
        tokenizer,
        prompts: Dict[str, List[str]],
        device: str,
        target_dim: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    native = {}
    padded = {}
    for cls_name, texts in prompts.items():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens).float().detach().cpu().numpy()
        feats = l2_normalize_rows(feats)
        native[cls_name] = feats
        padded[cls_name] = pad_or_truncate_features(feats, target_dim)
    return native, padded


@torch.no_grad()
def encode_crop_batch(
        model,
        crops: List[np.ndarray],
        device: str,
        image_size: int,
        mode: str,
        image_encode: str = 'dense') -> np.ndarray:
    batch = torch.stack([preprocess_crop(crop, image_size) for crop in crops])
    batch = batch.to(device)
    if image_encode == 'global':
        feats = model.encode_image(batch).float()
    elif image_encode == 'dense' and hasattr(model, 'encode_dense'):
        feats = model.encode_dense(
            batch, normalize=True, keep_shape=False, mode=mode).float()
        if feats.ndim == 3:
            feats = feats.mean(dim=1)
    elif image_encode == 'dense':
        feats = model.encode_image(batch).float()
    else:
        raise ValueError(f'unknown image_encode: {image_encode}')
    return l2_normalize_rows(feats.detach().cpu().numpy())


def flush_crop_batch(
        model,
        crops: List[np.ndarray],
        crop_classes: List[str],
        text_native: Dict[str, np.ndarray],
        class_features: Dict[str, List[np.ndarray]],
        class_scores: Dict[str, List[float]],
        device: str,
        image_size: int,
        mode: str,
        image_encode: str) -> None:
    if not crops:
        return
    crop_feats = encode_crop_batch(
        model, crops, device, image_size, mode, image_encode)
    for cls_name, feat in zip(crop_classes, crop_feats):
        text_proto = text_native[cls_name].mean(axis=0)
        text_proto = l2_normalize_rows(text_proto.reshape(1, -1))[0]
        score = float(np.dot(feat, text_proto))
        class_features[cls_name].append(feat.astype(np.float32))
        class_scores[cls_name].append(score)
    crops.clear()
    crop_classes.clear()


def build_declip_support(args) -> Dict[str, dict]:
    data_root = Path(args.data_root)
    image_dir = data_root / 'images'
    ann_dir = data_root / 'annfiles'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'image directory not found: {image_dir}')
    if not ann_dir.is_dir():
        raise FileNotFoundError(f'annotation directory not found: {ann_dir}')

    model, tokenizer = load_declip_model(
        Path(args.declip_code_root), Path(args.checkpoint), args.device)
    prompts = build_prompts(DOTA2_CLASSES)
    text_native, text_padded = encode_text_bank(
        model, tokenizer, prompts, args.device, args.text_target_dim)

    class_features: Dict[str, List[np.ndarray]] = defaultdict(list)
    class_scores: Dict[str, List[float]] = defaultdict(list)
    crops: List[np.ndarray] = []
    crop_classes: List[str] = []
    counts = defaultdict(int)

    txt_paths = iter_limited_paths(ann_dir, '*.txt', args.max_files)
    for file_idx, txt_path in enumerate(txt_paths, 1):
        if all(counts[cls_name] >= args.max_candidates_per_class
               for cls_name in DOTA2_CLASSES):
            break
        if args.progress_interval > 0 and file_idx % args.progress_interval == 0:
            n_done = sum(
                counts[cls_name] >= args.max_candidates_per_class
                for cls_name in DOTA2_CLASSES)
            print(
                f'scanned {file_idx}/{len(txt_paths)} annfiles; '
                f'{n_done}/{len(DOTA2_CLASSES)} classes filled',
                flush=True)

        ann_items = list(parse_dota_txt(txt_path))
        if not contains_needed_class(
                ann_items, counts, args.max_candidates_per_class):
            continue

        image_path = image_dir / f'{txt_path.stem}.png'
        if not image_path.is_file():
            continue
        image = np.asarray(Image.open(image_path).convert('RGB'))
        for cls_name, poly in ann_items:
            if counts[cls_name] >= args.max_candidates_per_class:
                continue
            try:
                crop = crop_axis_aligned_from_poly(image, poly, pad=args.crop_pad)
            except ValueError:
                continue
            crops.append(crop)
            crop_classes.append(cls_name)
            counts[cls_name] += 1
            if len(crops) >= args.batch_size:
                flush_crop_batch(
                    model, crops, crop_classes, text_native, class_features,
                    class_scores, args.device, args.image_size, args.mode,
                    args.image_encode)

    flush_crop_batch(
        model, crops, crop_classes, text_native, class_features, class_scores,
        args.device, args.image_size, args.mode, args.image_encode)

    support = {}
    for cls_name in DOTA2_CLASSES:
        rows = np.asarray(class_features[cls_name], dtype=np.float32)
        scores = np.asarray(class_scores[cls_name], dtype=np.float32)
        selected, selected_scores = select_topk_rows(
            rows, scores, args.support_shot)
        selected = pad_or_truncate_features(selected, args.visual_target_dim)
        support[cls_name] = dict(
            texts=prompts[cls_name],
            text_embeds=text_padded[cls_name].astype(np.float32),
            visual_embeds=selected.astype(np.float32),
            confidence_scores=selected_scores.astype(np.float32),
        )

    validate_support_schema(support)
    return support


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build DOTA2 OpenRSD support PKL from DeCLIP EVA-B.')
    parser.add_argument(
        '--data-root',
        default='data/DOTA2_1024_500/ss_train')
    parser.add_argument(
        '--out',
        default='data/DOTA2_1024_500/ss_train/'
                'Step5_3_Prepare_Visual_Text_DeCLIP_support.pkl')
    parser.add_argument(
        '--checkpoint',
        default='pretrained/declip/DeCLIP_EVA-B_DINOv2-B_csa_0.05_2.0/'
                'epoch_6.pt')
    parser.add_argument(
        '--declip-code-root',
        default='third_party/DeCLIP_CATSeg')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--support-shot', type=int, default=50)
    parser.add_argument('--max-candidates-per-class', type=int, default=200)
    parser.add_argument('--crop-pad', type=int, default=2)
    parser.add_argument('--mode', default='csa', choices=('csa', 'qq'))
    parser.add_argument(
        '--image-encode',
        default='dense',
        choices=('dense', 'global'),
        help='dense uses DeCLIP encode_dense; global uses encode_image.')
    parser.add_argument(
        '--max-files',
        type=int,
        default=0,
        help='debug limit for sorted annfiles; 0 means all files.')
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=500,
        help='print scan progress every N annfiles; 0 disables progress.')
    parser.add_argument('--text-target-dim', type=int, default=768)
    parser.add_argument('--visual-target-dim', type=int, default=1024)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f'output exists; pass --overwrite to replace: {out_path}')
    support = build_declip_support(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        pickle.dump(support, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out_path}')
    for cls_name in DOTA2_CLASSES:
        entry = support[cls_name]
        print(
            f'{cls_name}: text={entry["text_embeds"].shape} '
            f'visual={entry["visual_embeds"].shape} '
            f'conf_mean={float(np.mean(entry["confidence_scores"])):.4f}')


if __name__ == '__main__':
    main()
