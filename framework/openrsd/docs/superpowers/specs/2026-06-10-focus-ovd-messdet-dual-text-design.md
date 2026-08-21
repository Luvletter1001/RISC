# FOCUS-OVD Baseline With MessDet-Inspired Dual Text Branch

Date: 2026-06-10

## Intent

Use FOCUS-OVD as the baseline and add a conservative dual text branch for rotation-aware open-vocabulary detection. The change should keep the validated FOCUS-OVD image/support path as the main method, add text-side rotation features as a bounded residual signal, and remove previously failed modules from the new main path.

The target method is:

1. Native OpenRSD image detector and support bank.
2. FOCUS-OVD Fourier orientation-conditioned visual support adapter.
3. New text branch A: raw text support plus MessDet-style C8 rotation-equivariant text downsampling.
4. New text branch B: text support plus Fourier rotation-invariant and rotation-equivariant summaries.
5. Bounded fusion of text branches, then bounded fusion with FOCUS-OVD visual support before the existing support classifier.

## Baseline And Evidence

FOCUS-OVD remains the experiment baseline. The known safe baseline config is `M_configs/experiments/focus_ovd/focus_ovd_a10_sv_only_full.py`, which loads the A10 Step2 checkpoint and trains only `bbox_head.focus_support_adapter`.

Local experiment records show why this should stay conservative:

- FOCUS-OVD is the current main method in `resultmd/exp_focus_tsafe_20260610/reports/focus_tsafe_final_report.md`, with recorded epoch-24 DOTA2 validation mAP `0.6775` and small-vehicle AP `0.5541`.
- EQText V33 direct text/logit fusion failed at epoch 2 in `resultmd/exp_eqtext_strict_epoch2_audit_20260610/reports/v33_epoch2_failure_audit.md`: mAP `0.4123`, small-vehicle AP `0.0999`, and excessive small-vehicle detections.
- Earlier negative controls and no-effect modules should not be carried into the new main path.

## MessDet Reference Used Conservatively

MessDet contributes design ideas, not a full detector replacement:

- C8 rotation group structure.
- Strict rotation-equivariant downsampling: do orientation-aware mixing before resolution reduction.
- Rotation-equivariant channel attention with shared orientation gates.
- Multi-branch separation before final prediction.

This design does not import the MessDet image backbone, neck, head, or e2cnn dependency into the OpenRSD image path. The MessDet idea is applied only to text support features, where the blast radius is smaller and the FOCUS-OVD baseline can be preserved.

## Architecture

### Existing Path Kept

The existing FOCUS-OVD path stays in place:

```text
image features -> FourierOrientationLearner
native support/text bank -> support mapping
FOCUS-OVD support adapter -> OrientationConditionedContrastiveEmbed
```

The detector backbone, neck, bbox regression path, NMS, DeCLIP switch, support type, and auxiliary bbox head behavior are not changed.

### New Branch A: Raw Text + MessDet-Style Equivariant Downsampling

Add a text-only module, planned name `MessDetTextDownsampleBranch`.

Input:

- Raw mapped text support features shaped like the current support features.
- Support labels and masks.
- Optional orientation prior from `FourierOrientationLearner`, detached by default.

Behavior:

- Project each text support vector into a C8 orientation orbit.
- Apply lightweight cyclic group mixing over the C8 dimension.
- Apply MessDet-style strict downsampling in text/support space: first orientation-preserving mixing, then support-shot or token reduction.
- Apply shared orientation channel gating inspired by MessDet `REChannelAttention`.
- Produce two outputs:
  - rotation-invariant text support by pooling over C8;
  - rotation-equivariant text support retaining the C8 orbit.

The module is initialized to zero residual effect, so enabling it without learned weights must preserve FOCUS-OVD predictions.

### New Branch B: Text + Fourier Invariant/Equivariant Summary

Reuse or narrow the existing Fourier text adapter logic into a safer support-space branch.

Input:

- Current text support features.
- Detached Fourier orientation summaries from `FourierOrientationLearner`.

Behavior:

- Build a rotation-invariant text summary from orientation magnitude statistics.
- Build a rotation-equivariant text residual from phase-conditioned orientation evidence.
- Keep output bounded by norm ratio and scalar caps.
- Do not write directly to class logits.

This branch may reuse code from `M_AD/models/utils/focus_eqtext_adapter.py`, but only as a support-feature residual path.

### Text Branch Fusion

Add a fusion module, planned name `FocusMessFourierTextFusion`.

Input:

- Branch A invariant/equivariant text outputs.
- Branch B invariant/equivariant text outputs.
- Current support labels and masks.

Behavior:

- Fuse invariant summaries with a small learned gate.
- Fuse equivariant summaries only through bounded residuals.
- Normalize fused text support to the native support feature scale.
- Clamp text contribution with `max_text_weight`, default `0.0` for equivalence smoke and at most `0.02` for first nonzero training.

### Fusion With FOCUS-OVD Visual Support

The fused text support is not allowed to replace visual/native support. It enters as a residual to the support features already consumed by `OrientationConditionedContrastiveEmbed`.

The combined support should follow:

```text
support_final = support_focus_ovd + w_text * bounded_text_delta
```

where `w_text` is trainable or scheduled only inside a strict cap, and `bounded_text_delta` is norm-limited per class/support row.

## Modules Removed From The New Main Path

These modules remain available for old experiment reproduction unless a dependency audit proves they can be deleted. They are excluded from all new FOCUS-OVD + dual-text configs and should not run in the new forward path:

- `focus_text_logit_mixer`: direct text/logit mixing is not part of the new method.
- `focus_fourier_head_gate`: head-logit Fourier gating is not part of the new method.
- `focus_text_anchor_calibration`: class-logit anchor calibration is not part of the new method.
- EQ_V33-style direct text/logit fusion.
- Direction hard prompt and random orientation negative controls.
- Anti-attractor-only and preserve-only variants as standalone methods.
- DeCLIP support replacement.
- DOTA1 hard-negative assumptions for DOTA2 training.

Physical deletion is deferred. The implementation should first remove these from new configs and guard their initialization so the new path is clean without breaking historical result folders, generated configs, or tests.

## Configuration Design

Create a new FOCUS-OVD-derived config, planned name:

```text
M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py
```

Required config properties:

- Inherit from the existing FOCUS-OVD baseline config.
- Set `num_gpus = 6`.
- Train only the new text branch, text fusion, and existing FOCUS-OVD support adapter unless an audit requires narrower staging.
- Keep `support_type='text'`, `with_aux_bbox_head=True`, and `use_declip_support=False`.
- Disable `focus_text_logit_mixer`, `focus_fourier_head_gate`, and `focus_text_anchor_calibration`.
- Start with `max_text_weight=0.0` for equivalence smoke.
- First nonzero run uses `max_text_weight<=0.02`, detached orientation, and bounded residual norms.

## Training Plan Shape

Six-card commands must disable unreliable A40 NCCL transports:

```bash
rtk env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 tools/train.py M_configs/experiments/focus_ovd/focus_ovd_a10_mess_fourier_dual_text_6gpu.py
```

If a local launcher script is used, it must preserve the same NCCL environment and use `num_gpus=6` in both config and sampler.

## Verification Gates

Implementation must pass these gates before any long six-card training:

1. Import smoke for all new modules.
2. Unit shape tests for C8 text branch outputs.
3. Zero-equivalence forward smoke: `max_text_weight=0.0` matches FOCUS-OVD output within numerical tolerance.
4. Trainability audit: only declared adapter/fusion parameters are trainable.
5. Single-GPU tiny train smoke.
6. Six-GPU launch smoke with NCCL P2P/IB disabled.
7. Short DOTA2 validation gate: reject if small-vehicle detections explode or small-vehicle AP drops below the FOCUS-OVD reference tolerance.

## Rollback

Rollback must be one config switch:

- Disable the dual-text branch.
- Restore the original FOCUS-OVD config path and trainable parameter list.
- Leave native support features and detector behavior unchanged.

## Non-Goals

- No image backbone or neck replacement with MessDet.
- No e2cnn dependency in this implementation; any e2cnn experiment requires a separate design spec.
- No direct class-logit text calibration.
- No prompt-engineering-only method.
- No physical deletion of historical modules before dependency audit.
- No DOTA1-specific hard-negative design in the DOTA2 baseline run.

## Main Risks

- Text residuals may still destabilize class calibration, as V33 did. Mitigation: zero-equivalence first, strict text caps, no direct logits.
- The MessDet downsampling analogy is text-space rather than image-space. Mitigation: keep it clearly named as MessDet-inspired and test only its support-feature behavior.
- Existing head code has accumulated optional FOCUS modules. Mitigation: new config disables old modules, and implementation should centralize the new path instead of adding another direct-logit hook.
