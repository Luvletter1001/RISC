# P14 Rotated DINO Query Transport Plan

**Goal:** Replace the failed P13N soft Gaussian field with a strict E2E rotated DINO-style query transport detector for HRRSD. Inference must directly output a fixed set of `rbox + cls` predictions with no traditional dense detection head, no NMS, no score threshold fallback, and no teacher model.

**Literature basis:** DETR, Deformable DETR, DAB-DETR, DN-DETR, DINO, AO2-DETR, D2Q-DETR, ARS-DETR, DETRDistill, D3ETR, Group DETR.

## Implementation Checklist

- [x] Search and summarize literature before further training.
- [x] Inspect local DINO/rotated transformer/rotated Hungarian components.
- [x] Implement `P14RotatedDINOQueryTransportHead` as a set posterior module, even if the MMEngine config key remains `bbox_head`.
- [x] Add strict E2E unit tests: fixed set output, no NMS, no score threshold, teacher train-only, no dense RTMDet/S2ANet/Retina-style final branch.
- [x] Add GT denoising query path using noisy `cx, cy, w, h, angle`.
- [x] Add teacher box query path using P13C teacher boxes as train-only hard query priors.
- [x] Add hard transport matching with focal cls, bbox L1, and periodic angle.
- [x] Add mini overfit gate on 20 HRRSD train images before four-card training.
- [x] Add 300-500 iter mini-eval gate before 3e DDP.
- [ ] Only if mini gates pass, launch 4GPU train+multi-GPU eval on HRRSD.
- [x] If smoke has potential, draw the requested flow diagram.

## 2026-06-23 Mini Gate Outcome

Do not launch 4GPU yet. P14-C is the current best strict-E2E candidate, but
`mAP=0.0048` on the 128-image mini gate is still too low for a full 3e run.

| exp_id | key change | mini_mAP | AP50 | decision |
|---|---|---:|---:|---|
| P14-A | hard query transport over stride-size query boxes | 0.0000 | 0.0000 | stop: box scale collapses |
| P14-B | DAB-style class reference width/height | 0.0008 | 0.0010 | keep as scale fix |
| P14-C | class-consistent transport + same-class rank loss | 0.0048 | 0.0050 | current best |
| P14-D | stratified class query seed repetition | 0.0009 | 0.0010 | discard: coverage/ranking worsens |

## Non-Negotiable Constraints

- P13C teacher is train-only.
- Inference path is `features -> fusion/query decoder -> fixed set rbox+cls`.
- No NMS, no score-threshold filtering, no empty fallback.
- No traditional dense detection head as final output.
- No full 4GPU 3e run before the mini gates produce non-zero AP signal.

## Preferred First Variant

`P14-A`: minimal rotated DINO query transport.

- Reuse existing HRRSD RTMDet-L feature/fusion pipeline only as feature extractor/fusion source.
- Replace P13 fixed lattice with learned matching queries plus explicit box reference queries.
- Train with GT denoising first; teacher boxes only as additional high-confidence query priors.
- Keep inference single group fixed query output.

Fallback if P14-A compile cost is too high: `P14-B` keeps the P13 set-head wrapper but replaces soft Gaussian targets with hard teacher/GT Hungarian assignment and DN box queries. This is less ideal architecturally, but still avoids NMS and avoids dense final output.
