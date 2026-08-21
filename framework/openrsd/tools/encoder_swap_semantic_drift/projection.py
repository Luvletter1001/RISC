#!/usr/bin/env python3
"""Project foreign encoder embeddings to OpenRSD support dimensions."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def project_embeddings(
        src: np.ndarray,
        ref: np.ndarray,
        method: str,
        target_dim: int,
) -> Tuple[np.ndarray, str]:
    """src/ref: (N, D_src) and (N, D_ref) aligned by row."""
    method = method or 'none'
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if src.shape[1] == target_dim and method in ('none', 'l2_only'):
        out = src.copy()
        if method == 'l2_only':
            out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out.astype(np.float32), method

    if src.shape[0] < 2:
        method = 'l2_only'

    if method == 'orthogonal_procrustes':
        src_c = src - src.mean(0)
        ref_c = ref - ref.mean(0)
        m = src_c.T @ ref_c
        u, _, vt = np.linalg.svd(m, full_matrices=False)
        r = u @ vt
        out = src_c @ r
        if out.shape[1] > target_dim:
            out = out[:, :target_dim]
        elif out.shape[1] < target_dim:
            pad = np.zeros((out.shape[0], target_dim - out.shape[1]), dtype=np.float64)
            out = np.concatenate([out, pad], axis=1)
        out = out + ref.mean(0)[:out.shape[1]]
        out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out.astype(np.float32), 'orthogonal_procrustes'

    if method == 'ridge_projection':
        lam = 1e-3
        x = src
        y = ref[:, : min(ref.shape[1], target_dim)]
        if y.shape[1] < target_dim:
            ypad = np.zeros((y.shape[0], target_dim - y.shape[1]))
            y = np.concatenate([y, ypad], axis=1)
        w = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)
        out = x @ w
        out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
        return out.astype(np.float32), 'ridge_projection'

    # fallback: truncate/pad + l2
    out = np.zeros((src.shape[0], target_dim), dtype=np.float64)
    d = min(src.shape[1], target_dim)
    out[:, :d] = src[:, :d]
    out = out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-12)
    return out.astype(np.float32), 'truncate_l2'
