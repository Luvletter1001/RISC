# DOTA-v2 Scene-Only Strict-OV Manifest Preparation

**Date:** 2026-08-03  
**Status:** frozen preparation-only contract; not N0/N1 evidence or authorization

## 1. Purpose and firewall

This task prepares a deterministic, scene-disjoint image inventory while B0
trains. It may reduce later split-selection bias, but it does not implement,
train, or evaluate N0/N1.

The builder must never open or parse any annotation, prediction, evaluator,
checkpoint, or B0 trajectory file. In particular, every `annfiles` tree and
official-validation novel4 geometry/count/frequency/metric is forbidden.
Public class strings and the already frozen base14 whitelist are allowed.

Every report must contain these exact declarations:

```text
preparation_only=true
official_novel4_annotation_opened=false
annotation_files_opened=false
prediction_or_metric_files_opened=false
used_for_training=false
N0_N1_authorized=false
b0_result_influenced_manifest=false
```

## 2. Allowed sources

Primary input is image filenames only from:

```text
/data1/zcy/datasets/DOTA2_1024_500/ss_train/images
```

The logical `ss_train` resolves to
`/data1/zcy/datasets/DOTA2_1024_500/ss_train_full_1024_500_20260620`.
Official validation is used only for a filename-level isolation audit:

```text
/data1/zcy/datasets/DOTA2_1024_500/ss_val/images
```

Its image directory resolves to
`/data1/zcy/datasets/DOTA2_1024_500_valcomplete_20260618/ss_val/images`.
No official-validation image content or annotation is an input to scene
assignment.

The independently observed filename inventory is:

| Inventory | Tiles | Scenes | Invalid stems |
|---|---:|---:|---:|
| train | 47,294 | 1,664 | 0 |
| official validation | 13,833 | 592 | 0 |

Train versus official-validation tile-stem overlap and scene-ID overlap are
both zero.

## 3. Tile identity

Every train and validation image must be a regular non-symlink `.png` whose
stem full-matches:

```regex
^(?P<scene>[^_]+)__(?P<crop_size>[1-9]\d*)__(?P<x>\d+)___(?P<y>\d+)$
```

The parsed crop is the half-open rectangle
`[x, x + crop_size) × [y, y + crop_size)`. Duplicate stems, case-fold
collisions, unexpected suffixes, extra tokens, negative offsets, or ambiguous
scene identities fail closed. The current inventory has only crop size 1024.

## 4. Frozen scene split

Scene assignment uses no annotation statistic, image pixels, B0 observation,
or repeated salt search.

```text
salt = "ov-capflow-task6-scene-v1-20260803"
bucket(scene) = int(SHA256(UTF8(salt + "\0" + scene)), 16) mod 10000
train = [0, 7000)
dev   = [7000, 8500)
test  = [8500, 10000)
```

All tiles from a scene inherit exactly one partition. The first and only
frozen application gives:

| Partition | Scenes | Tiles | Tile fraction |
|---|---:|---:|---:|
| train | 1,157 | 35,109 | 0.7423563242694634 |
| dev | 262 | 5,172 | 0.10935848099124625 |
| test | 245 | 7,013 | 0.1482851947392904 |

The dev tile fraction is accepted as generated. Salt, thresholds, membership,
and ratios must not be changed to improve balance or after observing B0.

## 5. Frozen base14 meta-novel folds

The base14 whitelist, in DOTA class order, is:

```text
baseball-diamond, basketball-court, bridge, ground-track-field,
harbor, large-vehicle, plane, roundabout, ship, small-vehicle,
soccer-ball-field, storage-tank, swimming-pool, tennis-court
```

Fold assignment uses only these public strings:

```text
fold_seed = 20260803
rank classes by (SHA256(UTF8(str(fold_seed) + "\0" + class)), class)
assign ranked class i to fold (i mod 4)
```

The frozen folds are:

1. `plane`, `baseball-diamond`, `tennis-court`, `large-vehicle`
2. `ground-track-field`, `soccer-ball-field`, `bridge`, `storage-tank`
3. `basketball-court`, `swimming-pool`, `ship`
4. `roundabout`, `harbor`, `small-vehicle`

They are protocol metadata only. No annotation may be inspected to estimate
support or revise them.

## 6. Two-level source seal

The image directory is approximately 65 GB, so full hashing must not compete
with the live ten-GPU B0 dataloader.

### M0 — authorized now

Scan directory entries with no-follow semantics and record canonical sorted
`tile_relpath`, stem, parsed scene/size/x/y, half-open crop box, suffix, and
`lstat` byte size. Seal the canonical source inventory and output manifest
with SHA-256. Do not open image bytes. An M0 report must state:

```text
image_content_sha256_complete=false
png_header_audit_complete=false
training_use_forbidden=true
```

Canonical artifacts use UTF-8, lexicographic record order by `tile_relpath`,
lexicographic object-key order, JSON separators `(',', ':')`, lowercase JSON
booleans/null, base-10 integers without leading zeros, no NaN/Infinity, no
ASCII escaping of valid Unicode, and exactly one trailing `\n`. The builder
records its Python and serializer implementation/version. A golden-byte test
must freeze these bytes rather than relying only on semantic JSON equality.

### M1 — deferred I/O window

Only after B0 releases the shared data path, extend the same immutable split
with PNG header/mode/width/height and streaming file SHA-256. Optional decoded
pixel hashes may detect identical pixels under different encodings. M1 may
complete the source-content seal but still does not authorize N0/N1.

M1 applies only to the Section 2 train inventory. It must not open
official-validation image content or discover additional original-image
roots. `original_scene_mapping` is reported as `not_applicable` until a future
contract explicitly authorizes and enumerates such roots.

Neither M0 nor M1 may overwrite an existing output. A new output root is
required for every attempt; all failed evidence is preserved.

## 7. Required audits

The report must verify:

1. every scene maps to exactly one partition;
2. all cross-partition scene-ID intersections are empty;
3. all cross-partition tile-stem intersections are empty;
4. cross-partition half-open crop intersections within a common scene are
   zero;
5. train and official-validation scene/stem intersections are empty;
6. all expected 47,294 train tiles occur exactly once;
7. the frozen scene and tile counts match Section 4;
8. base14 and four folds exactly match Section 5;
9. canonical JSON/JSONL is newline-terminated, finite, duplicate-free, and
   separately hashed from the source inventory.

M1 additionally requires no cross-partition file-SHA intersection, unique
original-scene mapping where source images are available, and PNG header
compatibility with the declared crop size.

Because DOTA scene IDs do not provide geospatial coordinates, this protocol
proves no overlap in declared scene/coordinate space and, after M1, no
byte-identical tiles. It cannot prove that two different scene IDs never show
partially overlapping real-world geography; this limitation must remain in
the report and paper.

## 8. Implementation boundary

The authorized implementation is a CPU-only, annotation-blind builder and
focused tests. It may write only a new preparation output under
`.lab/workspace/exp-8-n0-scene-prep-v1/`. It may not create dataset symlinks or
annotations, mutate an existing cleanstart subset, launch a model, or expose
the output to a dataloader.

Before any run, a static forbidden-source scan must reject source code that
contains `annfiles`, `labelTxt`, evaluator/prediction loaders, B0 log paths, or
checkpoint readers except within an explicit denylist constant and error
messages. Passing M0 is `preparation_only`; N0 still waits for B0 terminal
closure, a separate approved design, and the closest-work hard gate.
