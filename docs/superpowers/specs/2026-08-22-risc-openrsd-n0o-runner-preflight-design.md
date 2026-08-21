# RISC OpenRSD N0-O Runner and CPU Preflight Design

**Status:** user approved on 2026-08-22
**Protocol:** `risc-openrsd-n0o-runner-v1`
**Input authority:** `docs/provenance/risc_openrsd_n0o_v3/input_manifest.json`
**Input manifest SHA-256:** `646b8702d60a3dbe36a35871b5599459807b8399deafac98d3056a1f04d8351c`
**Execution boundary:** implement runner and execute CPU preflight only; GPU,
model forward, predictions, metrics, training and checkpoint writes remain
unauthorized

## 1. Objective

This phase converts the sealed v3 inputs into a runnable, fail-closed protocol
without crossing the scientific execution boundary. It must prove that:

1. the authoritative manifest, scene plan and support ledger can be consumed;
2. every per-scene mapped support tensor can be reconstructed bitwise;
3. a complete model ledger can be created without GT/annotation access;
4. the OpenRSD model can be constructed on CPU and the A10 checkpoint can be
   load-audited with exactly the expected zero-initialized RISC parameters
   missing;
5. any GPU fold command refuses before CUDA/runtime imports unless a future
   authorization receipt is present and valid.

No N0 result is produced by this phase.

## 2. Authority separation

Three authorities remain distinct:

- **Paper mouth:** filtered non-empty `ss_val=6605`, scale1024, historical
  text7 sampling, `val_using_aux=False`, AP50 `0.7050`.
- **N0-O model ledger:** 160 sealed scenes with per-scene fixed text7 support;
  this is not an AP evaluation.
- **Future audit ledger:** annotations and GT geometry opened only by the CPU
  analyzer after committed prediction shards exist. It is not visible to the
  GPU runner.

The model ledger must contain no annotation path, annotation SHA, GT class,
qbox, object count, previous metric or previous model output.

## 3. Hybrid runtime code authority

No single existing directory is runnable and S0-complete:

- `/data1/zcy/RISC/framework/openrsd` contains the S0 detector/head/adapter/
  capture code but its source-only snapshot intentionally lacks
  `M_AD/datasets`;
- `/data1/zcy/GSOVD/.lab/tmp/openrsd_head_20260706` is the clean
  `12d3fd8b75e8b64ec53fded9cf035a2306d58874` archive used by P77E and contains
  datasets/transforms, but lacks S0.

The runtime therefore uses an explicit Python namespace overlay:

```text
sys.path[0] = /data1/zcy/RISC/framework/openrsd
sys.path[1] = /data1/zcy/GSOVD/.lab/tmp/openrsd_head_20260706
```

`M_AD` is a namespace package in both roots. Preflight must import and seal
origins as follows:

- RISC root: detector, dense head, final-readout adapter, capture recorder;
- clean fallback: `M_AD.datasets`, transforms and samplers absent from RISC;
- installed OpenRSD conda environment: MMEngine/MMCV dependencies;
- RISC root: local MMRotate/MMDetection packages unless an explicit installed
  package is required by the historical preload contract.

Every imported module used by the runner is recorded with absolute path,
byte count and SHA-256. An origin outside the allowed roots is a preflight
failure.

## 4. Input consumer and support reconstruction

The consumer accepts only the exact v3 manifest hash. It rejects:

- missing/noncanonical manifest bytes;
- any sibling `INVALIDATED.json`;
- status other than `SEALED_INPUTS_GPU_NOT_AUTHORIZED`;
- source or artifact hash drift;
- scene/ledger count or identity mismatch;
- support selection mismatch;
- PyTorch reconstructed row or aggregate bundle hash mismatch.

Support reconstruction must use the sealed source indices, float16-to-float32
cast and PyTorch 1.12 CPU `linear -> relu -> linear` mapping. It returns a CPU
`[1,126,256]` tensor and `[1,126]` labels in canonical class order. The same
scene tensor object/bytes are used for `rot000_a`, `rot000_b` and all C4/C8
views.

## 5. Model ledger

The CPU preflight atomically publishes a canonical `model_ledger.jsonl` with
160 rows. Each row contains only:

- protocol/input-manifest/scene-plan/support-ledger identities;
- fold, group order, scene ID, tile name, image path/hash;
- support source/mapped hashes and selected prompt indices;
- ordered view specifications;
- future shard relative identity.

View order is fixed:

```text
C4: rot000_a, rot000_b, rot090, rot180, rot270
C8: rot000_a, rot000_b, rot045, rot090, rot135,
    rot180, rot225, rot270, rot315
```

`rot000_a` and `rot000_b` are independent forward identities over identical
image and support bytes. No rotated view replaces either identity repeat.

## 6. CPU model build and checkpoint load audit

Preflight sets `CUDA_VISIBLE_DEVICES=''` before importing Torch runtime code
and fails if CUDA is initialized. It resolves the historical P77E config, then
overrides only sealed paths/mouth values and adds:

```python
risc_final_readout=dict(
    enabled=True,
    rank=8,
    init_alpha=0.0,
    max_alpha=0.1,
    max_delta_norm_ratio=0.05,
    init_seed=20260822,
)
```

It builds the model on CPU but calls no `forward`, `predict`, `test_step`,
runner loop or dataloader iteration. Raw A10 checkpoint loading must yield:

- unexpected keys: empty;
- missing keys: exactly 17 parameters/buffers: the three
  `bbox_head.risc_final_readout.{raw_alpha,down.weight,up.weight}` entries plus
  14 parameters/buffers from four post-checkpoint modules that the RISC source
  snapshot always constructs;
- every common checkpoint/model tensor loaded exactly;
- adapter alpha zero and debug identity contract intact.

The four additional modules (`focus_text_anchor_calibration`,
`focus_fourier_head_gate`, `focus_text_logit_mixer`,
`counter_support_ratio`) must each expose `enable=False`. The exact 17-key
allowlist is valid only under those disabled assertions; it cannot hide an
active randomly initialized path.

The preflight records parameter counts, common tensor count, missing allowlist,
resolved config hash and module-origin audit. It destroys the model before
publishing its receipt.

## 7. GPU runner contract and authorization guard

The fold runner is implemented as a lazy runtime. Its top-level module may
import only Python stdlib and CPU protocol helpers. Before importing Torch,
MMCV ops, OpenRSD runtime modules or touching CUDA, it requires a future
canonical receipt with:

```text
status = GPU_SMOKE_AUTHORIZED
input_manifest_sha256 = 646b8702...351c
preflight_receipt_sha256 = <exact committed receipt>
allowed_folds = <explicit subset>
max_scenes_per_fold = <explicit positive bound>
```

No such receipt exists in this phase. Unit and CLI tests must prove that
`run-fold` stops before the lazy runtime loader is called.

After future authorization, the runner will reuse the sealed scene/view/
support ledger, fixed-canvas C4/C8 renderer and S0 recorder. It will write
per-scene shards atomically without replacement and keep annotations/GT out of
the model process. No optimizer, backward or checkpoint write API is allowed.

## 8. CPU preflight artifacts

The preflight publishes without replacement:

```text
docs/provenance/risc_openrsd_n0o_preflight_v1/
  resolved_config.py
  module_origins.json
  model_ledger.jsonl
  preflight_report.json
  PREFLIGHT_READY_GPU_NOT_AUTHORIZED.json
```

The final receipt status is exactly `PREFLIGHT_READY_GPU_NOT_AUTHORIZED`. It
chains the v3 manifest, every generated artifact and builder/runner source
hash. No timestamps or output paths that break deterministic rebuilds enter
canonical bytes.

## 9. Completion and stop gate

Completion requires:

1. TDD for manifest rejection, support reconstruction, model-ledger firewall,
   view order, hybrid module origins, model-load allowlist and GPU guard;
2. two temporary CPU preflight builds are byte-identical;
3. a tracked preflight publication passes an independent audit;
4. focused plus adjacent tests, Python compilation and both diff checks pass;
5. independent review reports no Critical/Important findings.

Stop after `PREFLIGHT_READY_GPU_NOT_AUTHORIZED`. A four-scene GPU smoke needs
new explicit user authorization.
