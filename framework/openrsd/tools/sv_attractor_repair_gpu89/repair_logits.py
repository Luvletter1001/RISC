#!/usr/bin/env python
"""Apply repair stack to cls logits (supports R2B visual embeddings)."""
from __future__ import annotations

from tools.sv_attractor_repair_gpu89.repair_modules import RepairStack


def apply_repair_logits(outs_all, rep: RepairStack):
    cls_adj = []
    pred_embeds = outs_all[3] if len(outs_all) > 3 else None
    for li, cs in enumerate(outs_all[0]):
        b, c, h, w = cs.shape
        flat = cs.permute(0, 2, 3, 1).reshape(-1, c)
        vis = None
        if pred_embeds is not None and rep.neg_bank is not None:
            pe = pred_embeds[li].permute(0, 2, 3, 1).reshape(-1, pred_embeds[li].shape[1])
            if pe.shape[1] == rep.neg_bank.prototypes.shape[1]:
                vis = pe
        flat = rep.apply_logits(flat, vis)
        cls_adj.append(flat.reshape(b, h, w, c).permute(0, 3, 1, 2))
    return cls_adj
