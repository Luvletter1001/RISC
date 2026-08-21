#!/usr/bin/env python3
"""Build swapped support PKL for text/image prompt encoders (not detector backbone)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from commonlibs.common_tools import pklload, pklsave

from tools.encoder_swap_semantic_drift.common import (
    OPENRSD_CLASSES, ORIG_SUPPORT, TEXT_TARGET_DIM, VIS_TARGET_DIM, git_commit,
)
from tools.encoder_swap_semantic_drift.encoder_adapters import EncoderAdapter, build_adapter
from tools.encoder_swap_semantic_drift.projection import project_embeddings
from tools.encoder_swap_semantic_drift.prompts import build_prompt_bank


def load_original_support(path: Path = ORIG_SUPPORT) -> dict:
    d = pklload(str(path))
    return {k.lower(): v for k, v in d.items()}


def reference_text_matrix(support: dict) -> Tuple[np.ndarray, List[str]]:
    names = []
    rows = []
    for c in OPENRSD_CLASSES:
        v = support[c]['text_embeds']
        rows.append(np.asarray(v[0], dtype=np.float32))
        names.append(c)
    return np.stack(rows), names


def reference_visual_matrix(support: dict, per_class: int = 8) -> Tuple[np.ndarray, List[str]]:
    names, rows = [], []
    for c in OPENRSD_CLASSES:
        v = np.asarray(support[c]['visual_embeds'][:per_class], dtype=np.float32)
        names.append(c)
        rows.append(v.mean(0))
    return np.stack(rows), names


def build_text_swapped_support(
        ctx,
        encoder_name: str,
        prompt_family: str,
        projection_method: str,
        device: str = 'cpu',
        original: Optional[dict] = None,
) -> Tuple[Path, dict]:
    """Returns path to new support pkl; only text_embeds replaced."""
    original = original or load_original_support()
    out = copy.deepcopy(original)
    meta = dict(
        swap_type='text_prompt_encoder',
        encoder=encoder_name,
        prompt_family=prompt_family,
        projection=projection_method,
        note='Replaces text_embeds only; visual_embeds unchanged. NOT detector backbone.',
    )
    if encoder_name == 'original':
        path = ctx.support_swap_dir / f'support_text_original_{prompt_family}.pkl'
        pklsave(out, str(path))
        meta['status'] = 'ORIGINAL'
        return path, meta

    adapter = build_adapter(encoder_name, ctx.pretrained_root, device=device)
    if adapter.load_status != 'TEXT_IMAGE':
        raise RuntimeError(f'text encoder {encoder_name}: {adapter.load_status} {adapter.error}')

    ref_mat, class_names = reference_text_matrix(original)
    bank = build_prompt_bank(prompt_family)
    src_rows = []
    for c in class_names:
        texts = bank[c]
        emb = adapter.encode_text(texts, device=device)
        src_rows.append(emb.mean(0))
    src_mat = np.stack(src_rows)
    proj_method = projection_method
    if src_mat.shape[1] == TEXT_TARGET_DIM:
        proj_method = 'l2_only'
    proj_mat, used = project_embeddings(src_mat, ref_mat, proj_method, TEXT_TARGET_DIM)

    for i, c in enumerate(class_names):
        n = out[c]['text_embeds'].shape[0]
        out[c]['text_embeds'] = np.tile(proj_mat[i], (n, 1)).astype(np.float16)
        out[c]['texts'] = bank[c]

    path = ctx.support_swap_dir / f'support_text_{encoder_name}_{prompt_family}_{used}.pkl'
    pklsave(out, str(path))
    meta.update(
        status='DONE', projection_used=used, native_dim=int(src_mat.shape[1]),
        target_dim=TEXT_TARGET_DIM, git_commit=git_commit(ctx.repo_root),
    )
    return path, meta


def build_visual_swapped_support(
        ctx,
        encoder_name: str,
        projection_method: str,
        crop_paths: Dict[str, List[Path]],
        device: str = 'cpu',
        original: Optional[dict] = None,
) -> Tuple[Path, dict]:
    original = original or load_original_support()
    out = copy.deepcopy(original)
    meta = dict(
        swap_type='image_support_prototype_encoder',
        encoder=encoder_name,
        projection=projection_method,
        note='Replaces visual_embeds only; text_embeds unchanged. NOT detector backbone.',
    )
    if encoder_name == 'original':
        path = ctx.support_swap_dir / 'support_visual_original.pkl'
        pklsave(out, str(path))
        return path, meta

    adapter = build_adapter(encoder_name, ctx.pretrained_root, device=device)
    if adapter.load_status not in ('TEXT_IMAGE', 'IMAGE_ONLY') or adapter.modality not in ('image', 'text_image'):
        raise RuntimeError(f'image encoder {encoder_name}: {adapter.load_status} {adapter.error}')

    ref_mat, class_names = reference_visual_matrix(original)
    src_rows = []
    for c in class_names:
        paths = crop_paths.get(c, [])
        if not paths:
            src_rows.append(ref_mat[class_names.index(c)])
            continue
        emb = adapter.encode_image_paths(paths[:20], device=device)
        src_rows.append(emb.mean(0))
    src_mat = np.stack(src_rows)
    proj_method = projection_method
    if src_mat.shape[1] == VIS_TARGET_DIM:
        proj_method = 'l2_only'
    proj_mat, used = project_embeddings(src_mat, ref_mat, proj_method, VIS_TARGET_DIM)

    for i, c in enumerate(class_names):
        n = out[c]['visual_embeds'].shape[0]
        vec = proj_mat[i]
        out[c]['visual_embeds'] = np.tile(vec, (n, 1)).astype(np.float32)
        conf = out[c].get('confidence_scores')
        if conf is not None:
            out[c]['confidence_scores'] = np.ones(n, dtype=np.float32) * 0.9

    path = ctx.support_swap_dir / f'support_visual_{encoder_name}_{used}.pkl'
    pklsave(out, str(path))
    meta.update(status='DONE', projection_used=used, native_dim=int(src_mat.shape[1]),
                target_dim=VIS_TARGET_DIM)
    return path, meta
