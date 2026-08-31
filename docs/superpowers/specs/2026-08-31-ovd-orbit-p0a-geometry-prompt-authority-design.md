# OpenRSD P0-A Geometry and Prompt Authority Design

## Status

**Status:** implementation is covered by synthetic CPU-only tests and the real
read-only authority package is
`docs/provenance/ovd_orbit_p0a_authority_20260831_v3/`, with status
`P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD`. It remains preparation evidence,
not model inference or a scientific result.

This step turns the audited P0-A diagnostic inventory into one frozen,
full-vocabulary diagnostic authority. It is a preparation artifact only: no
detector forward, checkpoint load, image read, GPU use, AP, semantic metric,
strict OVD base/novel split, `OV_gap`, `DID_closed`, `DID_shift`, G1/G2/G3
decision, or paper claim is permitted.

## Fixed evidence inputs

- P0-A inventory receipt:
  `docs/provenance/ovd_orbit_p0a_inventory_20260831/receipt.json`, status
  `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD`.
- Inventory diagnostics SHA-256:
  `89e96a1b9f471c719c6a94cd26e46f43e8375d40468d0ea6e54db62adcabc03a`.
- Object inventory SHA-256:
  `a618010efbd8e4c413f2eea9fcc9018309d610ff776ef8178b22543502b46d44`.
- Historical text7 support asset:
  `/data1/zcy/OpenRSD/work_dirs/dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle/DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl`,
  SHA-256 `4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c`.
- Current P0 live-hook code: raw native logits and exact dense source identity
  `(level,row)` from `M_Tools/analysis/ovd_orbit_p0.py`,
  `M_Tools/analysis/ovd_orbit_p0_export.py`, and
  `M_AD/models/dense_heads/Flex_Rrtmdet_head_v3_1.py`.

The support file is hashed as opaque bytes and never unpickled or otherwise
deserialized by this authority builder.

## Sealed diagnostic policy

The primary P0-A diagnostic stratum is defined from the already completed,
GT-only inventory before any model output is seen:

```text
area_min = 43.5       # inventory 10th percentile
area_max = 3265.0     # inventory 90th percentile
max_overlap_iou = 0.05
min_objects_per_class = 5
```

An object is primary-eligible iff its inventory `size` is in the inclusive
area interval and its `overlap` is at most `0.05`. A class is primary-supported
iff at least five eligible objects remain. The primary stratum contains 2094
objects over ten supported classes; two geometry-eligible classes with one
object each (`ground-track-field`, `soccer-ball-field`) remain in diagnostics
but are excluded from primary summaries. This is a sample-support rule, not a
novel/base classification.

The authority records the full 18-class vocabulary and the observed 15-class
inventory coverage. It must explicitly list the three zero-observed vocabulary
classes (`container-crane`, `helicopter`, `helipad`) and must not infer a class
split from their absence.

## Prompt and oracle authority

P0-A freezes one historical prompt condition:

```text
condition_id = historical_text7_support_v1
support_type = text
support_shot = 7
support_asset_sha256 = 4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c
```

This is a single diagnostic condition. The authority must carry
`prompt_stability_status = NOT_TESTED_SINGLE_CONDITION`; it cannot describe
three prompt families or claim prompt robustness.

The oracle contract is:

```text
adapter_type = dense-carrier-fallback
carrier_source_identity_schema = level-row-v1
score_field = raw_native_foreground_logits
```

It also seals SHA-256 for each current live-hook code file. These fields bind
future P0-A smoke output to the code that exported it but do not authorize
inference.

## Output and validation

The CPU-only builder accepts explicit inventory receipt, diagnostics, object
inventory, support asset, and code-file paths. It verifies every supplied
artifact hash, recomputes the primary selection from inventory rows, checks
the stated policy/counts/class lists exactly, and writes a same-parent,
no-overwrite package:

- `p0a_diagnostic_authority.json`;
- `primary_object_ids.jsonl`;
- `authority_diagnostics.json`;
- `receipt.json`;
- Chinese `result.md`.

Success is `P0A_DIAGNOSTIC_AUTHORITY_READY_NO_FORWARD`. Any source mismatch,
policy/count mismatch, changed support/code bytes, object ID duplication, or
authority inconsistency produces a minimal `P0_INPUT_FAIL_STOP` package and
exit code 2. Neither receipt permits inference.

## Tests and handoff

Tests use synthetic inventory and temporary opaque support/code files. They
must verify boundary inclusivity, class-support accounting, zero-observed class
handling, support/code hash failures, immutable selection output, deterministic
artifacts, no-overwrite publication, and AST absence of ML imports or
`torch.load`.

The implementation additionally records a consumed-input SHA-256 ledger for
the inventory receipt, diagnostics, object inventory, implicit candidate plan
and result, source manifest, opaque support, and all three code files. The CLI
rehashes every one of those paths after building and fails closed if the ledger
does not match, including an ABA change that is restored before the post-build
check. All scalar CLI inputs reject both abbreviations and repeated flags; only
the three `--oracle-code-file` occurrences are repeatable.

After a real authority package is ready, the next separate design must decide
how a one-condition P0-A smoke can be reported. It may evaluate raw-logit
rotation change for the supported diagnostic classes, but cannot claim prompt
stability, strict OVD novelty, or a paper result.
