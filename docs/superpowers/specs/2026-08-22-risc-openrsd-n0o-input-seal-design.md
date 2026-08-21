# RISC OpenRSD N0-O Input Seal Design

**Status:** user approved for CPU-only preparation on 2026-08-22
**Protocol:** `risc-openrsd-n0o-input-v1`
**Execution boundary:** seal inputs and ledgers only; no GPU inference, AP replay,
orbit capture, optimization, backward pass or checkpoint write

## 1. Purpose

This phase makes the OpenRSD A10 N0-O experiment reproducible before any new
prediction exists. It seals four independent authorities:

1. the historical paper-mouth AP replay;
2. the frozen A10 E24 parent and RISC S0 code boundary;
3. the previously selected 160 scene-disjoint C4/C8 scene plan;
4. the exact per-scene text7 support selections and mapped tensors.

Completing this seal does not authorize the GPU runner. A later implementation
and preflight must bind its own source hashes to this immutable input seal.

## 2. Historical paper-mouth authority

The baseline AP statement remains a separate full-validation replay:

```text
DOTA2 ss_val
filter_empty_gt = True
dataset_len = 6605
resize/pad = 1024 x 1024
support_type = text
num_val_prompts = 7
val_using_aux = False
checkpoint_mode = raw
dota/mAP = 0.7049593925476074
dota/AP50 = 0.7050
```

Frozen assets:

| Asset | SHA-256 |
|---|---|
| A10 E24 checkpoint | `097585080a4c95f23370840da546acfdcb85093480454133ad2a34876cc20bd6` |
| historical source config | `51fec95ef0c4fff121f7683115caca72721a942a2dfc2afcc72de7686a97cd96` |
| P77E runner | `39054ab7519f9fe3b0490a203faa3b00218a40f289cd019adf3c4c630918ce72` |
| P77E audit | `163478453291ce0c99d3c3eb0c14aaebe25ee8f0f9e7fa23d889fbd3b11fdef3` |
| P77E result | `7202dbab088da8ec66270268b443c98ede5848507e797c6800c7b331b94ac836` |
| P77E predictions | `dedd99f81c84638a93e2d367b52c98774ff22723f4fd76832e37e258e227a6a7` |

The N0-O subset must never be reported as a 6605-image AP evaluation. The AP
replay and orbit-risk diagnosis share a parent and mouth definition but answer
different questions.

## 3. Dataset and scene authority

The dataset root is `/data1/zcy/datasets/DOTA2_1024_500`. Fresh inventory is:

- `ss_val/images`: 13,833 files;
- `ss_val/annfiles`: 13,833 files;
- non-empty annotations: 6,605;
- empty annotations: 7,228.

N0-O reuses the sealed scene plan at
`/data1/zcy/OV-CapFlow/work_dirs/risc_er/n0_set_orbit_20260817/scene_plan_40.json`,
SHA-256
`0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35`.
The plan contains 160 unique scenes split into four non-overlapping 40-scene
folds (`c4_a`, `c4_b`, `c8_a`, `c8_b`), explicitly excludes P0148, and binds
every selected image and annotation by SHA-256.

The input-seal builder must copy the scene plan bytes into RISC provenance,
verify its source SHA, verify every selected image/annotation hash and assert:

- 160 unique scene IDs and image paths;
- four folds with exactly 40 scenes each;
- C4 angles exactly `0,90,180,270`;
- C8 angles exactly `0,45,90,135,180,225,270,315`;
- P0148 absent;
- all selected rows belong to the non-empty 6605 mouth.

No object count, class, prediction, geometry quality or previous N0 result may
change scene or tile selection.

## 4. Parent and S0 code authority

The parent is the raw A10 E24 checkpoint above. Only its `state_dict` is used;
`ema_state_dict` is present but excluded. The text mapping keys must be exactly:

```text
text_support_mapping.0.weight  [1024,768] float32
text_support_mapping.0.bias    [1024]     float32
text_support_mapping.2.weight  [256,1024] float32
text_support_mapping.2.bias    [256]      float32
```

The RISC S0 base commit is `93cabbcf5d383ff4641d07186c1d3c5681443ad7`.
Its relevant file hashes are:

| File | SHA-256 |
|---|---|
| `risc_final_readout.py` | `21c6f768df344bb496faaf5b73cb7f523d336379606f1102493d001adfb982a1` |
| `Flex_Rrtmdet_head_v3_1.py` | `eab2c717eecb70ac5c15ffdde3d3549dc3bf8106dcb3f7bedad1b8a72dda5ed3` |
| `openrsd_hook_registry.py` | `cf71fbd262700f4a3e787591703daa52c5fbf9697e922fb7e1e5b79c05899afc` |
| S0 config | `6318d20b72738b716335cb64134204007144f3d6ae534728fb697ed8a10787c5` |

The future runner may add code, but it must not silently change these S0
semantics. Any changed hash requires a versioned amendment and a new input
seal.

## 5. Per-scene text7 support contract

The source support pickle is:

```text
/data1/zcy/OpenRSD/work_dirs/
dotav2_p4_lowtext_lser_sise_gpu67_20260617_153546/eval_bundle/
DOTA2_1024_500/ss_train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl
```

Its SHA-256 is
`4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c`.
It contains all 18 canonical DOTA2 classes; every class has at least 10
finite float16 text embeddings of width 768. This dtype was verified from the
real pickle during the first fail-closed build attempt.

For each scene, class and source prompt index, define:

```text
key = SHA256(
  "risc-openrsd-n0o-support-v1" + NUL
  + scene_id + NUL
  + class_name + NUL
  + decimal_prompt_index)
```

Sort each class's indices by unsigned digest bytes, with integer index as the
tie-break, and take the first seven. Class iteration and output tensor order
use the canonical 18-class DOTA2 prompt order from the P77E runner. This is
annotation-blind, independent of NumPy RNG implementations and immutable
across views.

The selected `[18,7,768]` float16 source values are cast exactly as the
historical `torch.Tensor(np.concatenate(...))` path to contiguous float32,
then mapped on CPU through the
raw checkpoint's exact `Linear(768,1024) -> ReLU -> Linear(1024,256)` weights.
The mapped `[18,7,256]` tensor is serialized for hashing as little-endian
contiguous float32 in C order. The support ledger records, per scene:

- scene/fold/group identity;
- selected seven source indices for every class;
- source tensor SHA-256;
- mapped tensor SHA-256;
- mapped tensor dtype, shape and byte count.

The top-level manifest records the canonical JSONL ledger SHA-256 and the
SHA-256 of the concatenated mapped tensor bytes in scene-plan row order. The
binary tensors are reconstructed and verified at runtime; they are not tracked
in Git.

## 6. Supporting asset authority

| Asset | SHA-256 |
|---|---|
| text/visual support pickle | `4ea3572d8184bfbfa556d051efaaea06d575bcb9a4e9f05efe3cc3564e5d737c` |
| negative support metadata | `51014c73b587f51f043e7dbe35a716bbcb0f70a57ac39c6ecf2cd5e50fb9a042` |
| normalized class dictionary | `a82b5da5c20ab5f48d93cd4e6a8c84da49254a75b725e764ba776b96af1e276b` |
| PCA metadata | `8eb6eca687939858faa6b2989f4b997a12d518f34d740fb13b7761c51a7f7acd` |

Although N0-O text7 does not use negative support or PCA values directly,
their identities remain recorded because the parent constructor loads them.

## 7. Generated tracked artifacts

The CPU builder produces, with no overwrite:

```text
docs/provenance/risc_openrsd_n0o/scene_plan_40.json
docs/provenance/risc_openrsd_n0o/support_ledger.jsonl
docs/provenance/risc_openrsd_n0o/input_manifest.json
```

The first publication attempt at the paths above is retained but invalid: its
handwritten `dota_mAP` hex string decoded to `0.7049497863006894`, not the
authority value `0.7049593925476074`. It must receive an `INVALIDATED.json`
marker and must never be used. The corrected, separately published authority
uses the no-replace directory:

```text
docs/provenance/risc_openrsd_n0o_v2/
```

The builder derives both metric identities with Python `float.hex()` and a
round-trip unit test; no metric hex value is copied by hand.

All JSON/JSONL uses canonical UTF-8 bytes: sorted keys, compact separators,
`allow_nan=False`, and exactly one trailing LF per JSON object/row.

`input_manifest.json` has status `SEALED_INPUTS_GPU_NOT_AUTHORIZED`. It must
include all source paths, sizes, hashes, counts, protocol constants, generated
artifact hashes, environment versions and a statement that no new prediction,
metric or model tensor beyond the text mapping was read.

## 8. Completion and stop gate

This phase completes only when:

1. focused unit tests verify canonical serialization, scene-plan checks,
   SHA-ranked support selection, raw-state mapping and fail-closed behavior;
2. two independent builder executions in temporary directories produce
   byte-identical artifacts;
3. all recorded source hashes are freshly rechecked;
4. the generated scene copy equals the sealed source bytes;
5. the ledger has 160 rows, each with 18 classes and seven unique indices;
6. every mapped tensor is finite `[18,7,256]` float32;
7. `git diff --check` passes for all new RISC-authored files.

After completion, stop. GPU runner implementation, preflight and N0-O
inference require a separate explicit authorization.
