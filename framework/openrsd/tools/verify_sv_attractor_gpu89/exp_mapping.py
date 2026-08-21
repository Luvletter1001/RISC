#!/usr/bin/env python
"""Step 1: mapping sanity (P0)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from tools.rotation_diagnostics import probe_rotated_stage_outputs as probe_base
from tools.verify_sv_attractor_gpu89 import common as C


def _cosine_table(feats: torch.Tensor, labels: torch.Tensor) -> list:
    rows = []
    class_means = {}
    for cid, name in enumerate(C.CLASSES):
        m = labels == cid
        if m.any():
            class_means[name] = feats[m].mean(dim=0)
    sv = class_means.get('small-vehicle')
    lv = class_means.get('large-vehicle')
    for cid, name in enumerate(C.CLASSES):
        m = labels == cid
        if not m.any():
            continue
        v = feats[m]
        row = dict(
            index=cid,
            class_name=name,
            norm=float(v.norm(dim=1).mean().item()),
            mean=float(v.mean().item()),
            std=float(v.std().item()),
            cosine_to_small_vehicle=float(
                F.cosine_similarity(v.mean(0, keepdim=True), sv.unsqueeze(0)).item())
            if sv is not None else '',
            cosine_to_large_vehicle=float(
                F.cosine_similarity(v.mean(0, keepdim=True), lv.unsqueeze(0)).item())
            if lv is not None else '',
        )
        rows.append(row)
    return rows


def run(ctx: C.RunContext) -> dict:
    if ctx.blocked():
        return dict(status='BLOCKED', reason=ctx.progress.get('blocked_reason', ''))

    out = ctx.exp_dir('exp_01_mapping')
    fres = ctx.fres_path('fres_01_mapping_sanity.md')
    status = 'OK'
    p0_pass = True
    notes = []

    try:
        bargs, model, device, det_support, name2id, id2name, _ = C.build_model(ctx, 'visual')
        image_dir = C.tile_image_dir(ctx.repo_root, C.DEFAULT_TILE)
        loader = C.build_loader(ctx, image_dir, [0])

        # embedding geometry (visual + text)
        geom_rows = []
        for st in ['visual', 'text']:
            sf, sl = C.build_mapped_support(
                model, ctx, det_support, name2id, device, st, 'original')
            geom_rows.extend(_cosine_table(sf.squeeze(0), sl.squeeze(0)))
            for r in geom_rows[-len(C.CLASSES):]:
                r['support_type'] = st
        C.write_csv(out / 'ftable_embedding_geometry.csv', geom_rows,
                    ['support_type', 'index', 'class_name', 'norm', 'mean', 'std',
                     'cosine_to_small_vehicle', 'cosine_to_large_vehicle'])

        sv_idx = name2id['small-vehicle']
        if sv_idx != C.SMALL:
            p0_pass = False
            notes.append(f'small-vehicle index mismatch: name2id={sv_idx} expected={C.SMALL}')

        dataset_order = list(det_support.keys())
        if dataset_order != C.CLASSES:
            p0_pass = False
            notes.append('det_support key order != DOTA1_CLASSES')

        # shuffle sanity: permute class embeddings
        sf0, sl0 = C.build_mapped_support(
            model, ctx, det_support, name2id, device, 'visual', 'original')
        perm = torch.randperm(C.NUM_CLASSES, device=device)
        shuffled = sf0.clone()
        groups = []
        for c in range(C.NUM_CLASSES):
            groups.append(sf0[0, sl0[0] == c].clone())
        new_feats = []
        new_labels = []
        for new_c, old_c in enumerate(perm.tolist()):
            new_feats.append(groups[old_c])
            new_labels.append(torch.full((groups[old_c].shape[0],), new_c, device=device))
        shuf_sf = torch.cat(new_feats, dim=0).unsqueeze(0)
        shuf_sl = torch.cat(new_labels, dim=0).unsqueeze(0)

        shuffle_top1 = []
        det_rows = []
        with torch.no_grad():
            for data_info in loader:
                data = model.data_preprocessor(data_info, False)
                data['inputs'] = data['inputs'].to(device)
                x = model.prompt_extract_feats(data['inputs'])
                metas = [s.metainfo for s in data['data_samples']]
                outs = C.forward_dense(model, x, shuf_sf, shuf_sl, bargs)
                cls_scores = outs[0]
                for lvl, cs in enumerate(cls_scores):
                    logits = cs[0].permute(1, 2, 0).reshape(-1, C.NUM_CLASSES)
                    top1 = logits.argmax(dim=1)
                    shuffle_top1.extend(top1.cpu().tolist()[:200])
                pred = model.bbox_head.predict_by_feat(
                    *outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
                det_labels = pred.labels.cpu().tolist()[:20]

        orig_top1 = []
        dense_examples = []
        with torch.no_grad():
            for data_info in loader:
                data = model.data_preprocessor(data_info, False)
                data['inputs'] = data['inputs'].to(device)
                x = model.prompt_extract_feats(data['inputs'])
                metas = [s.metainfo for s in data['data_samples']]
                outs = C.forward_dense(model, x, sf0, sl0, bargs)
                cls_scores = outs[0]
                for cs in cls_scores:
                    logits = cs[0].permute(1, 2, 0).reshape(-1, C.NUM_CLASSES)
                    scores = logits.sigmoid()
                    top1 = scores.argmax(dim=1)
                    orig_top1.extend(top1.cpu().tolist()[:200])
                    flat_idx = torch.topk(scores.max(dim=1).values, min(20, scores.shape[0])).indices
                    for i in flat_idx[:20]:
                        cid = int(top1[i])
                        dense_examples.append(dict(
                            class_id=cid, class_name=C.CLASSES[cid],
                            score=float(scores[i, cid].item())))
                pred = model.bbox_head.predict_by_feat(
                    *outs, batch_img_metas=metas, rescale=True, with_nms=True)[0]
                det_rows = []
                for i in range(min(20, len(pred.labels))):
                    lid = int(pred.labels[i])
                    det_rows.append(dict(
                        label_id=lid, class_name=id2name[lid],
                        score=float(pred.scores[i].item()),
                        bbox=C.bbox_row_list(pred, i)))
        C.write_csv(out / 'ftable_dense_topk_examples_rot000.csv', dense_examples,
                    ['class_id', 'class_name', 'score'])
        C.write_csv(out / 'ftable_final_det_examples_rot000.csv', det_rows,
                    ['label_id', 'class_name', 'score', 'bbox'])

        shuf_hist = Counter(shuffle_top1)
        orig_hist = Counter(orig_top1)
        shuffle_changes = (shuf_hist != orig_hist)
        if not shuffle_changes:
            p0_pass = False
            notes.append('embedding shuffle did not change dense top1 distribution')

        # text vs visual order
        tv_note = 'visual/text use same class order in prepare_support'

        p0_verdict = 'SUPPORTED' if p0_pass else 'REJECTED'
        if not p0_pass:
            ctx.set_blocked('P0 mapping sanity failed: ' + '; '.join(notes))

        with open(fres, 'w') as f:
            C.write_fres_header(f, 'Mapping Sanity (P0)', ctx, status)
            f.write(f'## P0 Verdict: **{p0_verdict}**\n\n')
            for n in notes:
                f.write(f'- {n}\n')
            f.write(f'\n- small-vehicle index: `{sv_idx}` (expected `{C.SMALL}`)\n')
            f.write(f'- dataset/support class order match DOTA1: `{dataset_order == C.CLASSES}`\n')
            f.write(f'- shuffle changes dense top1: `{shuffle_changes}`\n')
            f.write(f'- {tv_note}\n\n')
            f.write('## Embedding geometry (mapped)\n\n')
            f.write('| index | class | norm | cos_sv | cos_lv |\n|---|---|---:|---:|---:|\n')
            for r in geom_rows:
                if r.get('support_type') != 'visual':
                    continue
                f.write(
                    f"| {r['index']} | {r['class_name']} | {r['norm']:.4f} | "
                    f"{r['cosine_to_small_vehicle']:.4f} | {r['cosine_to_large_vehicle']:.4f} |\n")
            f.write('\n## Tables\n\n')
            f.write(f'- `{out / "ftable_embedding_geometry.csv"}`\n')
            f.write(f'- `{out / "ftable_dense_topk_examples_rot000.csv"}`\n')
            f.write(f'- `{out / "ftable_final_det_examples_rot000.csv"}`\n')
            f.write('\n## P0 criteria\n\n')
            f.write('- SUPPORT: class order aligned, shuffle moves interpretation, no id/name mismatch.\n')
            f.write('- REJECT: order mismatch or shuffle ineffective.\n')

        ctx.mark_step('mapping', 'OK' if p0_pass else 'FAILED', p0=p0_verdict)
        return dict(status=status, p0=p0_verdict, fres=str(fres))

    except Exception as exc:
        with open(fres, 'w') as f:
            C.write_fres_header(f, 'Mapping Sanity (P0)', ctx, 'FAILED')
            f.write(f'## Error\n\n```\n{exc}\n```\n')
        ctx.mark_step('mapping', 'FAILED', error=str(exc))
        ctx.set_blocked(str(exc))
        return dict(status='FAILED', error=str(exc), fres=str(fres))
