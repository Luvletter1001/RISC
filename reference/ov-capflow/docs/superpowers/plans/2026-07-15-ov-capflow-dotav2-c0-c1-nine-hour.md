# OV-CapFlow DOTA2 C0/C1 Nine-Hour Implementation Plan

> **Execution mode:** Use `executing-plans` to implement this plan task by
> task in the current session. Do not delegate. The unchecked boxes below are
> retained as the immutable pre-execution sequence; post-run completion,
> deviations and evidence are recorded in the matching fplan/flog/faudit/fres
> project-history files.

**Goal:** Build a current-code one-epoch DOTA2 strict C0 parent, train the repaired parent-preserving C1 adapter for one epoch, and produce a fully audited raw-13,833 causal comparison while keeping physical GPUs 4–7 usefully occupied.

**Architecture:** C0 and C1 share one Swin/BERT OV-CapFlow topology, a fixed checkpoint-compatible query count of 200, one deterministic 47,294/13,833 data mouth, and a distributed-aware DN query-budget sampler. C0 is trained first on four GPUs; C0 and zero-update C1 are then replayed in parallel two-GPU lanes; after exact equivalence, C1 is trained on four GPUs and independently replayed twice in parallel.

**Tech Stack:** Python 3.8, PyTorch 1.12, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate 1.0.0rc1, pytest, four NVIDIA A40 GPUs.

---

## File Map

- Create `projects/OVCapFlow/tools/audit_dotav2_mouth.py`: validate image/annotation pairing, counts, empty tiles, class order, and source paths.
- Create `tests/test_projects/ov_capflow/test_dotav2_mouth_audit.py`: unit tests for valid and invalid mouths.
- Create `projects/OVCapFlow/ov_capflow/dn_budget_batch_sampler.py`: deterministic global batching with disjoint rank shards and DN query-area protection.
- Modify `projects/OVCapFlow/ov_capflow/__init__.py`: register and export the sampler.
- Create `tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py`: coverage, determinism, rank-disjointness, shrink, and audit tests.
- Create `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_c0_native_1e.py`: current-code one-epoch strict parent.
- Create `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_c1_parent_preserving_1e.py`: seed-matched frozen-parent adapter.
- Create `tests/test_projects/ov_capflow/test_dotav2_config.py`: effective protocol and single-variable checks.
- Create `tests/test_projects/ov_capflow/test_dotav2_real_batch.py`: opt-in real batch forward/backward gate.
- Modify `projects/OVCapFlow/tools/hrsc_mediator_metrics.py`: add generic score/empty-tile totals without changing existing outputs.
- Modify `tests/test_projects/ov_capflow/test_hrsc_mediator_metrics.py`: verify empty score mass and backward compatibility.
- Create `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/fplan_dotav2_parent_preserving_causal_zh.md`: pre-registered run record.
- Create `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/flog_dotav2_parent_preserving_causal_zh.md`: timestamped execution log.
- Create `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/faudit_dotav2_parent_preserving_causal_zh.md`: data/checkpoint/sampler/strict evidence.
- Create `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/fres_dotav2_parent_preserving_causal_zh.md`: final result and decision.

### Task 1: Audit the exact DOTA2 mouth

**Files:**
- Create: `projects/OVCapFlow/tools/audit_dotav2_mouth.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_mouth_audit.py`

- [ ] **Step 1: Write failing valid/invalid mouth tests**

The tests create temporary `train/images`, `train/annfiles`, `val/images`, and
`val/annfiles` directories. They assert that `audit_mouth()` reports paired
counts, empty annotation files, sorted class names, and missing image/annotation
stems, and that `validate_contract()` raises on a wrong count or class token.

```python
def test_validate_contract_rejects_wrong_mouth(tmp_path):
    report = build_fixture_mouth(tmp_path, train_count=2, val_count=2)
    with pytest.raises(ValueError, match='train image count'):
        validate_contract(
            report, expected_train=3, expected_val=2,
            expected_classes=('plane', 'ship'))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dotav2_mouth_audit.py -q
```

Expected: collection fails because `audit_dotav2_mouth.py` does not exist.

- [ ] **Step 3: Implement the mouth auditor**

Implement `audit_mouth(train_root, val_root)` using `Path.glob`, stem-set
comparison, and annotation token parsing. Its JSON fields are:

```python
{
    'train_image_count': int,
    'train_annotation_count': int,
    'val_image_count': int,
    'val_annotation_count': int,
    'train_empty_count': int,
    'val_empty_count': int,
    'classes': list[str],
    'missing_train_images': list[str],
    'missing_train_annotations': list[str],
    'missing_val_images': list[str],
    'missing_val_annotations': list[str],
}
```

The CLI takes `--train-root`, `--val-root`, `--expected-train 47294`,
`--expected-val 13833`, 18 `--expected-classes`, and `--output`. It exits 1 on
any mismatch and never modifies the dataset.

- [ ] **Step 4: Run focused tests and the real mouth audit**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dotav2_mouth_audit.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/audit_dotav2_mouth.py --train-root /data1/zcy/datasets/DOTA2_1024_500/ss_train --val-root /data1/zcy/datasets/DOTA2_1024_500/ss_val --expected-train 47294 --expected-val 13833 --expected-classes airport baseball-diamond basketball-court bridge container-crane ground-track-field harbor helicopter helipad large-vehicle plane roundabout ship small-vehicle soccer-ball-field storage-tank swimming-pool tennis-court --output work_dirs/ov_capflow_dotav2/audits/mouth.json
```

Expected: tests pass and the real audit exits zero with 47,294/13,833 paired
files.

### Task 2: Add the distributed DN query-budget sampler with TDD

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/dn_budget_batch_sampler.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py`

- [ ] **Step 1: Write failing sampler tests**

Use a fake dataset whose `get_data_info()` returns `instances` lists of known
sizes. Instantiate four samplers with ranks 0–3, the same seed, and
`batch_size=6`. Assert:

```python
all_indices = [index for rank_batches in batches for batch in rank_batches
               for index in batch]
assert sorted(all_indices) == list(range(len(dataset)))
assert len(all_indices) == len(set(all_indices))
assert len({len(rank_batches) for rank_batches in batches}) == 1
assert not (set(flatten(batches[0])) & set(flatten(batches[1])))
```

Also assert same seed/epoch repeats exactly, a different epoch changes order,
a dense sample receives a singleton local batch, and rank 0 writes the audit
JSON.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py -q
```

Expected: import failure for `DNQueryBudgetBatchSampler`.

- [ ] **Step 3: Implement deterministic global batching**

Register with `mmrotate.registry.DATA_SAMPLERS`. Build one global shuffled order
from `sampler.seed + sampler.epoch`; compute each sample's GT count from
`sampler.dataset.get_data_info(index)['instances']`; greedily form global
batches no larger than `batch_size * world_size`. Estimate one rank's attention
area as:

```python
groups = max(1, num_dn_queries // max(1, max_gt))
dn_queries = 2 * max_gt * groups
query_area = local_batch_size * (num_matching_queries + dn_queries) ** 2
```

Distribute GT-descending samples round-robin across ranks so dense examples do
not cluster. Do not emit a global batch with fewer than `world_size` items.
When the final tail has fewer than `world_size` samples, rebalance samples from
preceding global batches into that tail while preserving the per-rank maximum
of six and rechecking the query-area budget. Yield only the current rank's
indices. A single dense sample may exceed the budget, but it must never be
dropped or duplicated.

The rank-0 audit contains epoch, seed, world size, dataset size, update count,
coverage checksum, duplicate count, missing count, global/local batch-size
ranges, shrink update count, max GT, and max estimated query area.

- [ ] **Step 4: Run sampler and portable tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: sampler tests and the full portable suite pass.

### Task 3: Add matched DOTA2 C0/C1 configs

**Files:**
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_c0_native_1e.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_c1_parent_preserving_1e.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_config.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_real_batch.py`

- [ ] **Step 1: Write failing config differential tests**

The tests load both effective configs and assert:

```python
assert c0.model.type == c1.model.type == 'OVCapFlow'
assert c0.model.num_queries == c1.model.num_queries == 200
assert c0.train_dataloader.dataset.type == 'DOTAv2Dataset'
assert c0.val_dataloader.dataset.filter_cfg.filter_empty_gt is False
assert len(c0.val_dataloader.dataset.metainfo.classes) == 18
assert c0.train_cfg.max_epochs == c1.train_cfg.max_epochs == 1
assert c0.randomness.seed == c1.randomness.seed == 20260712
assert not c0.model.decoder.layer_cfg.enable_semantic_fusion
assert c1.model.decoder.layer_cfg.enable_semantic_fusion
assert not c1.model.bbox_head.balanced_cfg.enabled
assert not c1.model.decoder.enable_null_reservoir
assert not c1.model.decoder.layer_cfg.enable_density_capacity
assert c1.load_from.endswith('dotav2_c0_native_1e/epoch_1.pth')
assert c0.train_dataloader.batch_sampler.type == 'DNQueryBudgetBatchSampler'
```

Normalize C0/C1 dictionaries by removing the fusion enable flag, `load_from`,
`custom_hooks`, `work_dir`, and checkpoint output settings; assert every
remaining model/data/optimizer/evaluator field is identical.

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dotav2_config.py -q
```

Expected: file-not-found failure for the C0 config.

- [ ] **Step 3: Implement the C0 config**

Inherit the existing Swin-T GroundingDINO config, override absolute BERT/Swin
paths, use `DOTAv2Dataset`, `return_classes=True`, 800x800 resize/pad, the
airport-first 18-class tuple, Q=200, batch size 6 per rank, the DN budget batch
sampler, `filter_empty_gt=False`, one epoch, validation at epoch 1, seed
20260712, `DOTAMetric(iou_thrs=0.5)`, and local visualization only.

Set OV-CapFlow C0 switches exactly as follows:

```python
model = dict(
    type='OVCapFlow',
    num_queries=200,
    encoder=dict(num_cp=0),
    decoder=dict(
        enable_null_reservoir=False,
        layer_cfg=dict(
            enable_semantic_fusion=False,
            enable_density_capacity=False)),
    bbox_head=dict(
        type='OVCapFlowHead', num_classes=18,
        balanced_cfg=dict(enabled=False),
        readout_cfg=dict(temperature=1.0, power=1.0,
                         use_capacity=False)),
    density_loss_cfg=dict(weight=0.0),
    null_loss_cfg=dict(),
    test_cfg=dict(_delete_=True))
```

- [ ] **Step 4: Implement the C1 config**

Inherit C0 and override only semantic fusion, the parent checkpoint, freeze
hook, checkpoint path, and work directory:

```python
model = dict(decoder=dict(layer_cfg=dict(
    enable_semantic_fusion=True,
    semantic_fusion_cfg=dict(adapter_init='identity'))))
load_from = 'work_dirs/ov_capflow_dotav2/dotav2_c0_native_1e/epoch_1.pth'
custom_hooks = [dict(
    type='FreezeExceptHook',
    trainable_patterns=[r'^decoder\.layers\.\d+\.semantic_fusion\.'])]
```

- [ ] **Step 5: Add the opt-in real-batch test**

Parameterize C0/C1, set workers to zero, build one real training batch, apply
the freeze hook for C1, run finite loss/backward, and assert C1's active
gradient parameters are all semantic-fusion parameters.

- [ ] **Step 6: Run focused and portable tests**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_dotav2_config.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Expected: all tests pass.

### Task 4: Add empty-tile mediator evidence

**Files:**
- Modify: `projects/OVCapFlow/tools/hrsc_mediator_metrics.py`
- Modify: `tests/test_projects/ov_capflow/test_hrsc_mediator_metrics.py`

- [ ] **Step 1: Write failing score-mass tests**

Pass prediction scores into `summarize_image_mediators`. Assert a zero-GT
image increments `empty_image_count`, `empty_prediction_count`, and
`empty_foreground_score_sum`, while a non-empty image does not.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow/test_hrsc_mediator_metrics.py -q
```

Expected: failure because the current function has no score argument or empty
totals.

- [ ] **Step 3: Implement additive empty metrics**

Add optional `scores` without changing current call compatibility. Final JSON
adds `empty_image_count`, `empty_predictions_per_image`, and
`empty_foreground_score_mass_mean`. The CLI passes
`prediction.pred_instances.scores` for each sample.

- [ ] **Step 4: Run focused and portable tests**

Run the focused test and full portable suite; both must pass.

### Task 5: Pre-register and commit the formal run

**Files:**
- Create: `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/fplan_dotav2_parent_preserving_causal_zh.md`
- Create: `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/flog_dotav2_parent_preserving_causal_zh.md`
- Create: `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/faudit_dotav2_parent_preserving_causal_zh.md`
- Create: `docs/project_history/exp_20260715_dotav2_parent_preserving_causal/fres_dotav2_parent_preserving_causal_zh.md`

- [ ] **Step 1: Record immutable protocol and external-reference boundary**

Write the 47,294/13,833 mouth, 18-class order, Q=200, seed, one-epoch C0 and
C1 budgets, GPU layout, parent dependency, strict contract, and decision rules.
Record historical P134B SHA
`f8564985ae4104961dc475300bbf4b4d83f9a6665869c7d51c39e9d5d3776f19`
as an incompatible external reference, not the current parent.

- [ ] **Step 2: Self-review and commit before GPU training**

Run:

```bash
rtk rg -n "[T]BD|[T]ODO|implement [l]ater|fill [i]n" configs/ov_capflow/dotav2 projects/OVCapFlow docs/project_history/exp_20260715_dotav2_parent_preserving_causal
rtk git diff --check
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
```

Commit all formal-run code and tracked records with an Experiment 6 hypothesis
message before the first C0 optimization step.

### Task 6: Run engineering gates

- [ ] **Step 1: Audit environment and mouth**

Record Python, torch, MMEngine, MMDetection, MMRotate, CUDA, GPU model, data
manifest, data symlink targets, and P134B incompatibility evidence.

- [ ] **Step 2: Run real C0/C1 batch forward/backward**

Run the opt-in integration test on GPU4. Expected: both arms finite; C1 only
has semantic-fusion gradients.

- [ ] **Step 3: Run four-rank sampler preflight**

Build the full real train dataloader for ranks 0–3 without loading images,
materialize all batch indices, and assert union 47,294, duplicates 0, missing
0, identical update count, and non-empty local batches. Save
`work_dirs/ov_capflow_dotav2/audits/sampler_preflight.json`.

- [ ] **Step 4: Apply the gate**

Any wrong count, pairing, class order, non-finite result, rank mismatch, or
strict-contract failure is `invalid-engineering`; fix it before training.

### Task 7: Build and verify the C0 parent on GPUs 4–7

- [ ] **Step 1: Launch four-GPU C0 training**

Use `CUDA_VISIBLE_DEVICES=4,5,6,7`, four local ranks, an unused fixed port, the
committed config, offline transformer flags, and a new C0 work directory. Log
the exact command, PID/session, commit, start time, and GPU map.

- [ ] **Step 2: Value-guard the run**

Poll at intervals no longer than 60 seconds while active. Check all four GPU
processes, utilization/memory, iteration progress, finite loss/gradients,
sampler audit, OOM/NCCL/traceback strings, and checkpoint creation. Preserve
all logs if a failure occurs.

- [ ] **Step 3: Audit the C0 checkpoint**

Require `epoch_1.pth`, successful trained-checkpoint load with zero invalid
missing/unexpected keys, finite final loss, sampler-complete JSON, SHA256, and
complete training-time raw metrics.

- [ ] **Step 4: Run parallel full-mouth initialization replays**

Use GPUs4–5 for C0 and GPUs6–7 for zero-update C1, separate ports and output
directories, and prediction pickle output. Both must evaluate 13,833 images
and report identical mAP/AP50. Then run the single-batch seven-tensor parent
equivalence audit and strict-inference audits.

### Task 8: Train and independently verify C1 on GPUs 4–7

- [ ] **Step 1: Launch four-GPU C1 one-epoch training**

Load the audited C0 checkpoint, apply the freeze hook, and verify the log lists
only the expected semantic-fusion parameters before accepting iteration 1.

- [ ] **Step 2: Value-guard the C1 run**

Poll GPU/process/log/checkpoint health at intervals no longer than 60 seconds.
Stop only for a concrete engineering failure; do not change the registered
scientific config after seeing intermediate AP.

- [ ] **Step 3: Run dual independent revalidation**

Evaluate the same C1 `epoch_1.pth` concurrently on GPU pairs 4–5 and 6–7 into
different directories. Require exact metric equality, 13,833 images, expected
Q rows per image, and complete prediction artifacts.

- [ ] **Step 4: Compute mediators and hashes**

Run the mediator tool on C0 and C1, audit the trained C1 checkpoint, and hash
checkpoints/predictions. Record coverage, duplicate extras/GT, empty score
mass, matched/unmatched gate means, gate gap, per-class evidence, duration,
peak memory, sampler shrinking, and final loss.

### Task 9: Decide, report, verify, and merge

- [ ] **Step 1: Apply the pre-registered decision**

Use only raw-13,833 C1 minus C0 mAP and the registered mediator tie-breaks.
Label the outcome promote, park, positive-under-gate, stop-exact-recipe, or
invalid-engineering. Any remaining valid finite outcome is do-not-promote.
For the park rule, material mediator changes are pre-fixed as empty score mass
or duplicate extras/GT down at least 1% relative, GT coverage up at least
0.001 absolute, or C1 gate gap <= -0.005, with coverage delta >= -0.005.
Rare/dense-class recall is excluded unless its deterministic calculator is
committed before result inspection.

- [ ] **Step 2: Complete all records**

Update `flog`, `faudit`, and `fres`, plus `.lab/log.md`, `.lab/results.tsv`,
`.lab/branches.md`, and `.lab/summary.md`. State explicitly that all-class
DOTA2 is structural evidence, not novel-class open-vocabulary proof.

- [ ] **Step 3: Run final verification**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest -p no:cacheprovider tests/test_projects/ov_capflow -q
rtk git diff --check
rtk git status --short --branch
```

Add machine-readable assertions for mouth counts, sampler coverage, checkpoint
loads, strict calls, dual-revalidation metric equality, and report values.

- [ ] **Step 4: Commit and merge**

Commit the final report as `research complete: validate DOTA2 parent-preserving fusion`, merge the research branch into `research/hrsc-calibrated-capacity`, rerun the portable suite on the merged result, and delete the merged feature branch. Preserve the user-owned untracked two-week review directory.
