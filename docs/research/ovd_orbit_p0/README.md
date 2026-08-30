# OVD Object-Orbit P0

## Status

`E0_LIVE_HOOK_READY_NO_FORWARD` — focused CPU fake-head and sink-lifecycle
contracts verify the default-off live native-logit hook. The canonical E0
`scores` field contains raw native foreground logits; optional
`calibrated_scores` is diagnostic only. This status is not a scientific result:
no actual model forward, checkpoint load, dataset iteration, GPU allocation, AP
computation was performed. No receipt based on real project assets was
published, and no live-P0 receipt was published in this implementation task.

## Authority

- Current live-hook design:
  `docs/superpowers/specs/2026-08-30-ovd-orbit-p0-live-hook-design.md`
- Current live-hook plan:
  `docs/superpowers/plans/2026-08-30-ovd-orbit-p0-live-hook.md`
- Completed historical E0 contract predecessor design:
  `docs/superpowers/specs/2026-08-30-ovd-orbit-p0-design.md`
- Completed historical E0 contract predecessor plan:
  `docs/superpowers/plans/2026-08-30-ovd-orbit-p0-e0.md`
- Protocol: `docs/research/ovd_orbit_p0/p0_protocol.md`
- Terminal statuses: `P0_INPUT_FAIL_STOP`, `G0_INPUTS_SEALED_NO_FORWARD`,
  `P0_MEASUREMENT_FAIL_STOP`, `E0_CONTRACT_READY`,
  `E0_LIVE_HOOK_READY_NO_FORWARD`

## G0 input-seal boundary

The G0 builder is implemented and tested infrastructure only. No real G0
authority package exists in this implementation work: only an actual successful
G0 builder run can produce `G0_INPUTS_SEALED_NO_FORWARD`.

Before either receipt state, the no-forward builder seals the ordered text
embedding hashes for all prompt families, the oracle mouth adapter and carrier
source identity schema, and the canonical digest of the exact P0-v1 threshold
bundle. This CPU-only protocol authority does not use a GPU or compute a P0
metric; the synthetic implementation does not create a real-asset receipt.

G0 takes a candidate-scene plan and canonical object inventory, validates the
declared candidate image and annotation file bytes, then generates the C4 view
plan. `rot000_a` and `rot000_b` are the distinct identity-repeat view records
in that plan. `P0_INPUT_FAIL_STOP` is the fail-closed receipt value for an
invalid, missing, noncanonical, or hash-mismatched declared input. It has two
fail-closed forms: a malformed/hash/asset mismatch produces a minimal failure
package, while a valid-but-below-G0-scope input produces a full diagnostic
package. The CLI exits 2 in both forms, and neither form authorizes a forward.

Neither receipt value is G1, G2, an effect, AP, or a paper result. No actual
model run, no checkpoint deserialization, no dataset iterator, no GPU use, no
P0 metric computation occurred. No receipt based on real project assets was
published, and no live-P0 receipt was published in this implementation task.

## Scope

The current worktree implements the default-off in-memory OpenRSD head hook,
E0 export infrastructure, and the G0 input-seal builder. It does not execute a
real model, dataset, GPU, or AP computation. No receipt based on real project
assets was published, and no live-P0 receipt was published in this
implementation task. CastDet integration is a separate repository/worktree
task. P0148's source scene is excluded from all future P0 sample manifests.

## Integration

The exact hook boundary and no-GPU E0 restrictions are defined in
`p0_protocol.md`. The hook is activated only by attaching an in-memory sink;
the per-image metadata keys are `ovd_orbit_p0_scene_id` and
`ovd_orbit_p0_view_id`. Missing keys are errors only when a sink is attached.
Without a sink, prediction remains on its ordinary path.
