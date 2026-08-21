# DeCLIP and CCL Config Switches Design

## Goal

Add DeCLIP and CCL as optional OpenRSD training-time integrations controlled from experiment config files, with both features disabled by default and no change to baseline behavior when disabled.

## Confirmed Interface

Training configs expose two top-level booleans:

```python
use_declip = False
use_ccl = False
```

Configs wire the flags into `model` and `model.bbox_head`:

```python
model = dict(
    use_declip_support=use_declip,
    declip_support_feat_dict=dict(
        Data1_DOTA2='./data/.../DeCLIP_support.pkl',
    ),
    bbox_head=dict(
        use_ccl_loss=use_ccl,
        ccl_loss_weight=0.05,
        ccl_temperature=0.1,
    ),
)
```

## Architecture

DeCLIP is integrated as a support-feature source, not as a detector rewrite. `OpenRTMDet` keeps the current support loading path when `use_declip_support=False`; when true, it loads `declip_support_feat_dict` through the same support schema already used by OpenRSD: each class maps to `visual_embeds` and `text_embeds`. Missing DeCLIP support configuration or files fail early with a clear error.

CCL is integrated as an additive loss in `OpenRotatedRTMDetSepBNHead.loss_by_feat`. It uses existing dense prediction embeddings, assigned labels, and support embeddings already passed through the head. When `use_ccl_loss=False`, the branch is skipped. When true, the branch adds `loss_ccl` without replacing `loss_cls`, `loss_bbox`, `loss_aln`, or any existing optional loss.

## Data Flow

1. Training config defines `use_declip` and `use_ccl`.
2. `OpenRTMDet.__init__` chooses either `support_feat_dict` or `declip_support_feat_dict`.
3. Support files are loaded into the existing `uni_support_data` and per-dataset support dictionaries.
4. `OpenRTMDet.loss_labelled` samples support features as before.
5. `OpenRotatedRTMDetSepBNHead.loss_by_feat` computes existing detector losses.
6. If enabled, CCL compares dense embeddings assigned to classes against the class support prototypes and adds `loss_ccl`.

## Error Handling

- `use_declip_support=True` requires a non-empty `declip_support_feat_dict`.
- Every configured DeCLIP support path must exist before training starts.
- DeCLIP support entries must include the same keys as OpenRSD support entries.
- CCL returns a differentiable zero tensor if a batch has no usable positive assignments.

## Testing

A-stage checks:

- Unit test DeCLIP support selection: disabled uses baseline dict, enabled requires and returns DeCLIP dict.
- Unit test CCL loss: disabled branch absent; enabled branch returns finite scalar for a synthetic batch.
- Config smoke test: build the chosen OpenRSD training config with the two switches present.
- Short training smoke: run 1-2 iterations if the local environment and data paths are available.

B-stage checks:

- Baseline, `+DeCLIP`, `+CCL`, and `+DeCLIP+CCL` use the same mini split, seed, schedule, and checkpoint.
- Report loss curves, mAP/AP50, and any rotation diagnostic used for the current OpenRSD experiments.
