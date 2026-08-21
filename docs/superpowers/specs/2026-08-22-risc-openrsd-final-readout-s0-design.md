# RISC-on-OpenRSD Final-Readout S0 Design

**Status:** approved for implementation on 2026-08-22
**Parent design:** `/data1/zcy/OV-CapFlow/.worktrees/risc-er/docs/superpowers/specs/2026-08-22-risc-openrsd-final-readout-m1-design.md`
**Parent design SHA256:** `a2e988bafc0dbf25688c1e42e6a8b3405bb705517674cde4e32e7bce3a1bfb99`

## 1. Objective

S0 prepares a zero-training, auditable interface for testing RISC on the
frozen OpenRSD A10 E24 parent. It does not claim a scientific result and does
not launch inference or training.

The implemented boundary is:

```text
pred_embed
  -> optional class-shared RISC final-readout adapter
  -> existing rtm_cls_heads[idx] / Align Head
```

The adapter must not alter the regression feature, bbox branch, angle branch,
objectness branch, support construction, NMS, or detector post-processing.

## 2. Frozen parent and evaluation authority

The scientific parent remains OpenRSD A10 E24:

- checkpoint: `/data1/zcy/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth`;
- checkpoint SHA256: `097585080a4c95f23370840da546acfdcb85093480454133ad2a34876cc20bd6`;
- reported filtered-6605, scale-1024, text7 mAP:
  `0.7049593925476074`.

S0 does not reproduce this number. N0-O may start only after a separate run
manifest seals the checkpoint, resolved config, dataset root under
`/data1/zcy/datasets`, seven-prompt support tensor and evaluator mouth.

## 3. Adapter contract

The adapter implements the approved minimal readout:

```text
G(h) = alpha * U(GELU(V(LayerNorm(h))))
s    = h - clip_norm(G(h))
```

Contract:

1. It accepts dense `NCHW` embeddings and returns the same shape and dtype.
2. It is class-shared and receives no class id, prompt id, test rotation angle,
   bbox, angle, objectness or post-processing state.
3. `alpha=0` is bitwise identity.
4. Disabled mode returns the original tensor without cloning it.
5. The residual norm is bounded per dense location.
6. Construction uses a local RNG scope so enabling the interface cannot
   change unrelated model initialization.
7. The adapter is optional in the existing
   `OpenRotatedRTMDetSepBNHead`; old configs instantiate no new parameters.

## 4. Head integration

`OpenRotatedRTMDetSepBNHead.forward` applies the adapter immediately after
`self.rtm_cls[idx](cls_feat)` and before either the native Align Head or the
legacy FOCUS path. The regression path continues to use `reg_feat` directly.
The existing `pred_embeds` return remains the unadapted parent tensor so
`loss_align`, CCL and other historical auxiliary consumers are not silently
rerouted through RISC.

The head exposes no new detector type and does not change return tuple shapes.
The S0 diagnostic config explicitly instantiates a rank-8, zero-alpha adapter;
the unchanged A10 config continues to omit it.

## 5. Capture contract

S0 extends the existing experiment-side `OpenRSDHookRecorder` rather than
adding an in-head file writer. A dedicated RISC recorder uses PyTorch forward
hooks to capture, for every feature level:

- parent `pred_embed` from `bbox_head.rtm_cls.*`;
- adapter input and output from `bbox_head.risc_final_readout`;
- adapted embedding, support embeddings, support labels and semantic logits
  from `bbox_head.rtm_cls_heads.*`;
- bbox deltas and angle predictions, plus objectness logits when the parent
  architecture exposes an objectness module. A10 has
  `with_objectness=False`, so its snapshot must seal objectness as
  structurally absent rather than fail capture.

Captured tensors are detached, cloned and moved to CPU. Multiple calls are
kept in call order. The recorder performs no filtering, top-k or NMS and does
not mutate model outputs. Serialization and scene/rotation metadata remain the
responsibility of the existing orbit runner.

## 6. Error handling

- Non-boolean `enabled`, non-positive dimensions/rank, invalid bounds and
  non-NCHW inputs fail explicitly.
- The recorder rejects duplicate registration and reports missing required
  hook families before its snapshot is accepted. Objectness is required only
  when the registered parent exposes that module.
- Capture remains opt-in; normal training and inference allocate no capture
  copies.

## 7. Verification

S0 is complete only when fresh CPU checks show:

1. adapter disabled and zero-alpha modes are bitwise identity;
2. the residual is bounded and gradients reach `alpha`;
3. adapter construction preserves the global CPU RNG state;
4. the A10 legacy config creates no adapter parameters;
5. the S0 config builds with a rank-8 zero-alpha adapter;
6. the recorder captures all required semantic and geometry families on CPU
   without modifying outputs;
7. focused tests, Python compilation and scoped `git diff --check` pass.

No AP, rotation-risk, open-vocabulary or P0148 claim is authorized by S0.
