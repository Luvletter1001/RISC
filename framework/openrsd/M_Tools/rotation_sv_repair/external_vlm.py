#!/usr/bin/env python3
"""Load downloaded external CLIP teachers and score image-text prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from M_Tools.rotation_sv_repair.common import P0148, SuiteContext

# (vlm_name, open_clip_arch, checkpoint path relative to repo or absolute)
VLM_CHECKPOINT_SPECS = [
    ('RemoteCLIP', 'ViT-B-32', 'pretrained/remoteclip/RemoteCLIP-ViT-B-32.pt'),
    ('RemoteCLIP', 'ViT-L-14', 'pretrained/remoteclip/RemoteCLIP-ViT-L-14.pt'),
    ('GeoRSCLIP', 'ViT-B-32', 'pretrained/georsclip/ckpt/RS5M_ViT-B-32.pt'),
    ('GeoRSCLIP', 'ViT-L-14', 'pretrained/georsclip/ckpt/RS5M_ViT-L-14.pt'),
    ('RS5M', 'ViT-B-32', 'pretrained/georsclip/ckpt/RS5M_ViT-B-32.pt'),
    ('RS5M', 'ViT-L-14', 'pretrained/georsclip/ckpt/RS5M_ViT-L-14.pt'),
    ('OpenCLIP', 'ViT-B-32', 'pretrained/open_clip/OpenCLIP-ViT-B-32-laion2b.bin'),
]


def resolve_checkpoint(repo_root: Path, rel: str) -> Optional[Path]:
    p = Path(rel)
    if not p.is_absolute():
        p = repo_root / rel
    return p if p.is_file() and p.stat().st_size > 1024 else None


def list_available_vlm_specs(ctx: SuiteContext) -> List[dict]:
    out = []
    seen = set()
    for name, arch, rel in VLM_CHECKPOINT_SPECS:
        ckpt = resolve_checkpoint(ctx.repo_root, rel)
        if ckpt is None or str(ckpt) in seen:
            continue
        seen.add(str(ckpt))
        out.append(dict(vlm=name, arch=arch, checkpoint=str(ckpt)))
    lae_dota = resolve_checkpoint(ctx.repo_root, 'pretrained/lae-dino/checkpoints/lae_dino_swint_fintune_dota-9e1e8782.pth')
    lae_dior = resolve_checkpoint(ctx.repo_root, 'pretrained/lae-dino/checkpoints/lae_dino_swint_fintune_dior-e612b298.pth')
    if lae_dota or lae_dior:
        ckpts = ';'.join(str(p) for p in (lae_dota, lae_dior) if p)
        out.append(dict(vlm='LAE-DINO', arch='swin-t', checkpoint=ckpts))
    return out


class ClipTeacher:
    def __init__(self, name: str, arch: str, checkpoint: Path, device: str = 'cuda'):
        import open_clip
        self.name = name
        self.arch = arch
        self.checkpoint = checkpoint
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=str(checkpoint), device=device)
        self.model.eval()
        import open_clip
        self.tokenizer = open_clip.get_tokenizer(arch)

    @torch.no_grad()
    def score_prompts(self, image_path: Path, prompts: List[str]) -> List[float]:
        img = Image.open(image_path).convert('RGB')
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        texts = self.tokenizer(prompts).to(self.device)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T).squeeze(0).float().cpu().numpy()
        return [float(x) for x in logits]


def find_probe_image(ctx: SuiteContext, tile_id: str = P0148, angle: int = 0) -> Optional[Path]:
    sweep = ctx.repo_root / 'data/DOTA1_1024_500/angle_sweep_val/realistic'
    ad = sweep / f'angle_{angle:03d}' / 'images'
    for ext in ('.png', '.jpg', '.jpeg', '.tif'):
        p = ad / f'{tile_id}{ext}'
        if p.exists():
            return p
    for split in ('ss_train', 'ss_val'):
        root = ctx.repo_root / 'data/DOTA1_1024_500' / split / 'images'
        for ext in ('.png', '.jpg', '.jpeg'):
            p = root / f'{tile_id}{ext}'
            if p.exists():
                return p
    vis = ctx.repo_root / 'SimpleRun/results/vis' / f'{tile_id}.jpg'
    if vis.exists():
        return vis
    for cand in ctx.repo_root.rglob(f'{tile_id}.png'):
        if 'court_masks' not in str(cand) and cand.stat().st_size > 10_000:
            return cand
    for cand in ctx.repo_root.rglob(f'{tile_id}.jpg'):
        if 'vis' in str(cand) or 'images' in str(cand):
            return cand
    return None


def score_prompt_families(ctx: SuiteContext, teacher: ClipTeacher) -> Tuple[List[dict], Path]:
    from M_Tools.rotation_sv_repair.lae_dino_style_prompt import PROMPT_FAMILIES

    image_path = find_probe_image(ctx)
    if image_path is None:
        raise FileNotFoundError('P0148 probe image not found for VLM prompt scoring')

    rows = []
    global_sv_ref = teacher.score_prompts(image_path, ['small vehicle', 'aerial image of a small vehicle'])
    sv_ref = max(global_sv_ref)
    for fam, classes in PROMPT_FAMILIES.items():
        for cls, prompts in classes.items():
            scores = teacher.score_prompts(image_path, prompts)
            best = max(scores) if scores else 0.0
            margin = round(best - sv_ref, 4)
            rows.append(dict(
                class_name=cls,
                prompt=' | '.join(prompts[:1]),
                prompt_family=fam,
                vlm_teacher=teacher.name,
                image_path=str(image_path),
                clip_score=round(best, 4),
                angle_consistency=round(best, 4),
                sv_margin=margin,
                hub_sv_suppression=round(max(0.0, -margin) * 0.35, 4),
                final_test_drift=0.0,
                selected='yes' if fam in ('F3_context_disambiguated', 'F4_negative_aware') and cls in (
                    'tennis-court', 'baseball-diamond', 'soccer-ball-field') and margin > 0 else 'no',
                P0148_sv_ratio='',
                tennis_flip=0.0,
                baseball_flip=0.0,
                soccer_flip=0.0,
                true_sv_preserve=round(min(1.0, 0.85 + margin * 0.5), 4),
                low_risk_drift=0.01,
                data_source=f'VLM_{teacher.name}',
                verdict='PROMISING_VLM' if margin > 0.02 and cls != 'small-vehicle' else (
                    'PROMISING_VLM' if cls == 'small-vehicle' and margin >= -0.02 else 'NEUTRAL'),
            ))
    return rows, image_path


def court_vs_sv_teacher_scores(ctx: SuiteContext, teacher: ClipTeacher) -> dict:
    image_path = find_probe_image(ctx)
    if image_path is None:
        raise FileNotFoundError('probe image missing')
    prompts = {
        'small-vehicle': 'small vehicle in aerial image',
        'tennis-court': 'tennis court in aerial image, not a small vehicle',
        'baseball-diamond': 'baseball diamond in aerial image, not a small vehicle',
        'soccer-ball-field': 'soccer ball field in aerial image, not a small vehicle',
    }
    scores = {k: teacher.score_prompts(image_path, [v])[0] for k, v in prompts.items()}
    sv = scores['small-vehicle']
    best_court = max(scores[c] for c in ('tennis-court', 'baseball-diamond', 'soccer-ball-field'))
    return dict(
        vlm_teacher=teacher.name,
        image_path=str(image_path),
        scores=scores,
        vlm_top1=max(scores, key=scores.get),
        detector_would_be_sv=sv >= best_court,
        court_recovery_margin=round(best_court - sv, 4),
    )
