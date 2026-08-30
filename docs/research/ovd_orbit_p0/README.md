# OVD Object-Orbit P0

## Status

`E0_CONTRACT_READY` — the CPU-only record, receipt, and dry-run contract passed focused verification. A live OpenRSD full-logit hook is not implemented yet; therefore the scientific P0 remains `P0_BLOCKED_NO_FULL_LOGIT_EXPORT`.

## Authority

- Design: `docs/superpowers/specs/2026-08-30-ovd-orbit-p0-design.md`
- Plan: `docs/superpowers/plans/2026-08-30-ovd-orbit-p0-e0.md`
- Protocol: `docs/research/ovd_orbit_p0/p0_protocol.md`
- Terminal statuses: `P0_INPUT_FAIL_STOP`, `P0_MEASUREMENT_FAIL_STOP`, `P0_BLOCKED_NO_FULL_LOGIT_EXPORT`, `E0_CONTRACT_READY`

## Scope

The current worktree implements OpenRSD export infrastructure only. CastDet integration is a separate repository/worktree task. P0148's source scene is excluded from all future P0 sample manifests.

## Integration

The exact hook boundary and no-GPU E0 restrictions are defined in `p0_protocol.md`. `P0_BLOCKED_NO_FULL_LOGIT_EXPORT` means that top-1 JSON or legacy object-array NPZ artifacts cannot support the oracle mouth and cannot decide G2.
