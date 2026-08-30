# OVD Object-Orbit P0

## Status

`E0_LIVE_HOOK_READY_NO_FORWARD` — focused CPU fake-head and sink-lifecycle
contracts verify the default-off live native-logit hook. The canonical E0
`scores` field contains raw native foreground logits; optional
`calibrated_scores` is diagnostic only. This status is not a scientific result:
no actual model forward, checkpoint load, dataset iteration, GPU allocation, AP
computation, or receipt publication was performed.

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
- Terminal statuses: `P0_INPUT_FAIL_STOP`, `P0_MEASUREMENT_FAIL_STOP`, `E0_CONTRACT_READY`, `E0_LIVE_HOOK_READY_NO_FORWARD`

## Scope

The current worktree implements only the default-off in-memory OpenRSD head
hook and E0 export infrastructure. It does not execute a real model, dataset,
GPU, AP computation, or receipt publication. CastDet integration is a separate
repository/worktree task. P0148's source scene is excluded from all future P0
sample manifests.

## Integration

The exact hook boundary and no-GPU E0 restrictions are defined in
`p0_protocol.md`. The hook is activated only by attaching an in-memory sink;
the per-image metadata keys are `ovd_orbit_p0_scene_id` and
`ovd_orbit_p0_view_id`. Missing keys are errors only when a sink is attached.
Without a sink, prediction remains on its ordinary path.
