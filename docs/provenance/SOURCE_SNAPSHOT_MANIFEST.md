# RISC Source Snapshot Manifest

**Snapshot time:** 2026-08-22 02:04:52 CST
**Target repository:** `/data1/zcy/RISC`
**Target state before source snapshot:** `6bc4dc6 docs: establish RISC research authority`

## Scope and interpretation

This manifest seals a source-only snapshot of two local, dirty working trees. A recorded
`HEAD` identifies the upstream baseline, while the copied files intentionally include the
then-current uncommitted source/config/test/document edits. Therefore neither target subtree
may be described as a clean upstream release or as a Git submodule.

The copy used `rsync -a` without `--delete`. It did not write to either source tree. The
only source-tree mutation for this task is the separately required appended OpenRSD Codex
work-log entry. The target does not retain that work log because it is runtime process state,
not framework source.

## Source A — OpenRSD framework base

| Field | Recorded value |
|---|---|
| Source path | `/data1/zcy/OpenRSD` |
| Baseline `HEAD` | `12d3fd8b75e8b64ec53fded9cf035a2306d58874` |
| Remote `origin` | `https://ghfast.top/https://github.com/floatingstarZ/OpenRSD.git` |
| Working-tree status at snapshot | `modified_or_deleted=38`; `untracked=146` |
| Target subtree | `framework/openrsd/` |
| Main copy | 5,562 regular files; 124,518,419 bytes transferred |
| Narrow experiment-code copy | 190 regular files; 1,571,796 bytes transferred |
| Final target file count | 5,750 regular files after policy-compliance cleanup |
| Final target disk usage | 136M |

The main copy excludes the full `experiments/` runtime tree. The following four source-code
directories are deliberately restored by a second, narrow copy:

```text
experiments/rotation_semantic_attractor/configs/
experiments/rotation_semantic_attractor/src/
experiments/rotation_semantic_attractor/scripts/
experiments/rotation_semantic_attractor/tests/
```

`outputs/` and `reports/` from that experiment tree are not present.

## Source B — OV-CapFlow / RISC-ER reference

| Field | Recorded value |
|---|---|
| Source path | `/data1/zcy/OV-CapFlow` |
| Baseline `HEAD` | `e87ae43ad294d9918cd6a36a458bf6f03dabb9fe` |
| Remote `origin` | `https://github.com/VisionXLab/CastDet.git` |
| Working-tree status at snapshot | `modified_or_deleted=2`; `untracked=24` |
| Target subtree | `reference/ov-capflow/` |
| Copy result | 792 regular files; 19,127,929 bytes transferred |
| Final target file count | 792 regular files |
| Final target disk usage | 21M |

## Content-integrity checks

The following SHA256 values are the copied target files. Fresh `cmp -s` comparisons against
the corresponding source files passed at snapshot time.

| Target file | SHA256 |
|---|---|
| `framework/openrsd/M_AD/models/dense_heads/risc_orbit_projection_head.py` | `43ed62412f73624088cd947d8affc13610e18cf8eef3efdf8575b54e37c7fd64` |
| `framework/openrsd/M_AD/models/utils/risc_orbit_projection.py` | `d03dae536e58441fe01a14d247a7c0f3bd4f518ae02baeb1a9cb73384d07412d` |
| `reference/ov-capflow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md` | `41a5da385bbe282b466c02185abf634b6eb6c3775821624bdfba5d7aeadcbbe8` |
| `reference/ov-capflow/docs/superpowers/plans/2026-08-11-risc-er-implementation.md` | `a3a84840350af5cb8b18254b7009b9ffa5efc53525959b7f74e7a19382e4547a` |

## Exclusions and link audit

The copy policy is [MIGRATION_POLICY.md](MIGRATION_POLICY.md). It excludes `.git`, IDE/
agent state, datasets, `data` links, `work_dirs`, weights, pretrained files, results,
visualizations, caches, model/checkpoint/prediction binaries, PDFs, archives, images, video
and logs.

The only tracked `.gz` exception is three byte-identical 1.3M BPE vocabulary files in
`framework/openrsd/third_party/DeCLIP_CATSeg`. They are direct relative-path runtime
dependencies of the vendored OpenCLIP/EVA tokenizer; their exact paths and SHA256 are
allowlisted in `MIGRATION_POLICY.md`. No general archive or model allowance is implied.

After copying, `framework/openrsd/results -> /data1/zcy/OpenRSD_results/results` was detected
as an external absolute symlink. It was removed from the target; no external result data was
copied. The retained `.mim` links in both source snapshots are relative links whose targets
remain inside their respective snapshot subtrees. A fresh absolute-link scan found none.

## Formatting residual from source fidelity

The initial source-snapshot commit produced **2,021 diagnostic lines** under a full staged
`git diff --cached --check`. After removing two policy-incompatible runtime/metadata files,
the final committed-tree diff from the empty tree has **1,751 diagnostic lines**. Investigation
showed these are inherited whitespace/newline diagnostics in the copied source/reference files,
not in the RISC authority files written for this repository. For example,
`framework/openrsd/CODEX_WORKLOG.md` is byte-identical to
`/data1/zcy/OpenRSD/CODEX_WORKLOG.md` and both have the same trailing-whitespace lines.

The scoped check over `.gitignore`, `README.md`, `RISC_GOAL.md`, `docs/provenance/`,
`docs/superpowers/`, `task_plan.md`, `findings.md` and `progress.md` exited zero. The source
snapshot is intentionally not mass-formatted: changing thousands of inherited lines would
break its stated provenance and exceed this migration's scope. The final source-code example
for this residual is `reference/ov-capflow/mmrotate/models/detectors/rotated_soft_teacher.py`,
which was compared byte-for-byte against its source and has inherited trailing whitespace.

## Git integrity after source-snapshot commit

The immutable source snapshot was committed as `f8313c3 chore: snapshot RISC source
foundations`, followed by the policy-compliance correction `921d972 chore: tighten RISC
snapshot boundary`. The repository has 4,997 tracked paths after the correction. `git fsck`
exited zero and found no dangling commit or tree. At the source-snapshot point it reported 25
dangling blobs generated by intermediate staging; they are recoverable unreferenced objects,
not corruption, and are intentionally left untouched because no garbage collection was
authorized.

## Reproduction constraints

- Do not treat `framework/openrsd/data` or any historic relative `data/` reference as a valid dataset root.
- Future experiments use absolute dataset roots under `/data1/zcy/datasets` and store runs outside this Git repository.
- Any later source refresh must create a new provenance manifest and a new Git commit; this initial snapshot is immutable history.
