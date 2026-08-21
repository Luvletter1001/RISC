from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.ascontiguousarray(value.numpy())
    return np.ascontiguousarray(np.asarray(value))


def tensor_checksum(value: Any) -> str:
    arr = _as_numpy(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _clone(value: Any):
    if hasattr(value, "clone"):
        return value.clone()
    return np.array(value, copy=True)


def _mean_norm(value: Any):
    if hasattr(value, "norm"):
        return value.norm(dim=1).mean().clamp_min(1.0e-12)
    norms = np.linalg.norm(value, axis=1)
    return max(float(norms.mean()), 1.0e-12)


def _normalize_rows(value: Any, target_norm: Any):
    if hasattr(value, "norm"):
        denom = value.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        return value / denom * target_norm
    denom = np.linalg.norm(value, axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0e-12)
    return value / denom * float(target_norm)


def _random_like(value: Any, seed: int):
    if hasattr(value, "new_empty"):
        import torch

        gen = torch.Generator(device=value.device)
        gen.manual_seed(seed)
        return torch.randn(value.shape, generator=gen, device=value.device, dtype=value.dtype)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(value.shape).astype(value.dtype, copy=False)


def apply_openrsd_visual_support_intervention(
    support_feats: Any,
    support_labels: Any,
    *,
    intervention: str,
    small_vehicle_id: int,
    large_vehicle_id: int,
    seed: int = 20260530,
):
    """Apply a real feature-level intervention to OpenRSD support embeddings.

    The function works on either torch tensors or numpy arrays and returns a
    modified copy plus evidence metadata. It intentionally does not edit prompts.
    """

    changed = _clone(support_feats)
    labels = support_labels
    before = tensor_checksum(changed)
    small_mask = labels == small_vehicle_id
    large_mask = labels == large_vehicle_id

    if intervention in {"original", "none", ""}:
        pass
    elif intervention == "zero_sv":
        changed[small_mask] = 0
    elif intervention == "swap_sv_lv":
        small = changed[small_mask].clone() if hasattr(changed, "clone") else np.array(changed[small_mask], copy=True)
        large = changed[large_mask].clone() if hasattr(changed, "clone") else np.array(changed[large_mask], copy=True)
        n = min(int(small.shape[0]), int(large.shape[0]))
        if n:
            changed[small_mask][:n] = large[:n]
            changed[large_mask][:n] = small[:n]
    elif intervention == "norm_sv_mean":
        target_norm = _mean_norm(changed)
        changed[small_mask] = _normalize_rows(changed[small_mask], target_norm)
    elif intervention == "normalize_all":
        target_norm = _mean_norm(changed)
        changed = _normalize_rows(changed, target_norm)
    elif intervention == "random_sv":
        replacement = _random_like(changed[small_mask], seed)
        target_norm = _mean_norm(changed[small_mask])
        changed[small_mask] = _normalize_rows(replacement, target_norm)
    else:
        raise ValueError(f"unknown OpenRSD support intervention: {intervention}")

    after = tensor_checksum(changed)
    return changed, {
        "intervention": intervention or "original",
        "modified_tensor_name": "visual_support_embeddings",
        "target_class_index": int(small_vehicle_id),
        "reference_class_index": int(large_vehicle_id),
        "is_prompt_only": False,
        "is_embedding_level": True,
        "is_visual_support_level": True,
        "actual_embedding_modified": before != after,
        "original_embedding_checksum": before,
        "modified_embedding_checksum": after,
    }
