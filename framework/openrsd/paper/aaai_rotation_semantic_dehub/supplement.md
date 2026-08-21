# Supplement Draft: Rotation Semantic DeHub

This supplement is a working companion to `main.tex`. It records evidence boundaries, full-table placement, blocked results, and negative findings. It should be converted to `supplement.tex` after the AP and true-SV preservation runs finish.

## Evidence Inventory

| Evidence block | Current status | Main-paper use |
| --- | --- | --- |
| Closed-set 10-model S2 12-angle audit | DONE_FULL | Phenomenon support only |
| Open-vocab A10 S2 12-angle benchmark | DONE_FULL | False-hub diagnostic, not AP |
| Causal support-embedding intervention S3 12-angle | DONE_FULL | Mechanism evidence |
| Context counterfactual S3 12-angle | DONE_FULL + NOT_APPLICABLE | Cautious context contribution |
| DeHub baseline vs repair S3 12-angle | DONE_FULL for burden counts | Method candidate, gated by AP/preserve |
| Open-vocab AP50/mAP | BLOCKED | Do not claim until raw predictions are exported |
| DeHub true-SV preservation | BLOCKED | Do not claim safety until GT matching succeeds |

## Full Tables to Move Here

1. Full closed-set false-SV by model and angle.
2. Full closed-set stage decomposition and hook support matrix.
3. Open-vocab anglewise burden curves.
4. Causal intervention paired deltas by angle and risk group.
5. Context counterfactual DONE_FULL and NOT_APPLICABLE audit.
6. DeHub class distribution delta and hub migration audit.
7. Claim ledger with allowed and forbidden wording.

## Claims Allowed in Main Paper

- Closed-set detectors across multiple architectures exhibit measurable false-small-vehicle burden under the 12-angle protocol.
- The OpenRSD A10 open-vocabulary checkpoint persistently predicts small-vehicle across risk groups and angles.
- Support-embedding interventions are paired rerun inference and strongly modulate small-vehicle burden.
- DeHub repair currently reduces small-vehicle prediction burden on paired S3 12-angle rows.

## Claims Not Yet Allowed

- Open-vocab AP50/mAP improves.
- DeHub improves AP.
- DeHub is fully safe.
- Context alone proves hallucination.
- All closed-set failures share the open-vocabulary embedding mechanism.

## Negative and Boundary Results

- Visual-support replacement can suppress the hub but is not deployable without retraining/calibration.
- Normalization-like embedding interventions stay close to original, so the effect is not simply an embedding-norm artifact.
- Repair can redistribute prediction mass to non-SV classes; current migration mass ratio is 0.7693.
- The old 200-crop qualitative pack is quarantined and must not be used as paper evidence.

## Pending Supplement Additions

- AP evaluator implementation notes.
- Raw-prediction schema.
- True-SV matching protocol.
- Checkpoint sweep tables.
- Ablation tables for embedding debias, migration regularizer, safety gate, and full method.
- Cross-dataset transfer smoke results.
