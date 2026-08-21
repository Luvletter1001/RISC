# DOTA-v2 Clean-Start Open-Vocabulary E2E AP70 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train from a DOTA/remote-sensing-free generic GroundingDINO checkpoint and produce an open-vocabulary, five-dimensional rotated, fixed-set detector whose raw DOTA-v2 `ss_val=13,833` metrics satisfy both `dota/AP50 >= 0.7000` and `dota/mAP >= 0.7000`, with no teacher, distillation, pseudo labels, dense/RoI head, NMS, or prediction-row top-k.

**Architecture:** Keep `OVCapFlow` as the only inference model: Swin-T + BERT text enhancer + fixed rotated decoder queries + token-similarity classification + 5-D box regression. Build a shape-filtered clean-start checkpoint from the generic OGC checkpoint, raise inference capacity to Q600, then screen training-only grouped one-to-one supervision and RHINO Hausdorff/positive-Hungarian denoising on a deterministic real-GT subset. Promote only an evidenced S1 winner to four-card full training; inference always returns the primary 600-query group unchanged.

**Tech Stack:** Python 3.8, PyTorch/MMEngine/MMDetection/MMRotate, pytest, DOTA-v2.0 clean tiles, four NVIDIA A40 GPUs (physical 4–7), Git and the project `.lab` research ledger.

---

## Fixed protocol and safety rules

- Run every shell command through `rtk`; never issue a bare shell command.
- Use `/data/zcy/anaconda3/envs/mmdet/bin/python` with `PYTHONNOUSERSITE=1`.
- Do not modify or stage the user's untracked `docs/project_history/exp_20260714_two_week_review/` directory.
- Before every real training/evaluation experiment, append a `## THINK — before Experiment 8-<stage>` entry to `.lab/log.md`, commit every repository file used by the run, and record the commit in the run manifest.
- A failed scientific intervention is preserved in `.lab`; remove its code from the active branch with `git revert`, not a destructive reset.
- Only these initialization assets are allowed:
  - `/data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth`, SHA256 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`;
  - `/data1/zcy/LAEDINO/weights/bert-base-uncased`;
  - fresh seeded parameters for unmatched Q600 query/angle/new geometry tensors.
- The raw validation mouth is the only completion mouth: `ss_val=13,833`, `filter_empty_gt=False`, IoU 0.5. `filtered-6605` is secondary and named explicitly.
- S1 promotion rule is fixed: AP50 gain at least `+0.020`, or AP50 gain at least `+0.010` and GT coverage gain at least `+0.020`, with duplicate extras/GT rising no more than 10%.
- Do not claim completion until the eight conditions in the design spec are all evidenced, especially raw AP50 and mAP both at least 0.7000.

## Task 1: Freeze the clean-start contract in executable tests

**Files:**

- Create: `tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py`
- Create: `projects/OVCapFlow/tools/prepare_cleanstart_checkpoint.py`
- Test: `tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint-policy tests**

The tests construct tiny synthetic source/target state dictionaries and import the tool as a module. Cover:

```python
def test_filter_keeps_only_exact_name_and_shape_matches():
    source = {
        'backbone.ok': torch.ones(2, 2),
        'bbox_head.reg_branches.0.2.weight': torch.ones(4, 8),
        'query_embedding.weight': torch.ones(900, 8),
    }
    target = {
        'backbone.ok': torch.zeros(2, 2),
        'bbox_head.reg_branches.0.2.weight': torch.zeros(5, 8),
        'query_initializer.query_embedding.weight': torch.zeros(600, 8),
    }
    kept, report = filter_compatible_state_dict(source, target)
    assert tuple(kept) == ('backbone.ok',)
    assert report['shape_mismatches'] == [
        {'key': 'bbox_head.reg_branches.0.2.weight',
         'source_shape': [4, 8], 'target_shape': [5, 8]}]


@pytest.mark.parametrize('key', [
    'teacher.backbone.weight', 'distill_head.weight', 'pseudo_bank.items',
    'dense_head.cls.weight', 'rpn_head.conv.weight', 'roi_head.fc.weight'])
def test_forbidden_namespace_is_rejected(key):
    with pytest.raises(ValueError, match='forbidden checkpoint namespace'):
        reject_forbidden_namespaces({key: torch.ones(1)})
```

Also test source SHA mismatch, a source path containing `dota`, `openrsd`, `hrsc`, `fair1m`, `dior`, `p126`, `p121`, `teacher`, `distill`, or `pseudo`, deterministic JSON ordering, output SHA recording, matched-numel coverage, and absence of optimizer/scheduler/EMA payloads.

- [ ] **Step 2: Verify the test fails before implementation**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py -q
```

Expected: FAIL because the module or required functions do not exist.

- [ ] **Step 3: Implement the self-contained checkpoint preparer**

Implement these public functions with type hints and no external repository import:

- `sha256_file(path: Path) -> str`;
- `reject_forbidden_source(path: Path) -> None`;
- `reject_forbidden_namespaces(state_dict: Mapping[str, Tensor]) -> None`;
- `convert_groundingdino_state_dict(state_dict: Mapping[str, Tensor]) -> dict`;
- `filter_compatible_state_dict(source: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> Tuple[dict, dict]`;
- `prepare_checkpoint(source: Path, config: Path, output: Path, provenance: Path, expected_sha256: str) -> dict`.

Define `FORBIDDEN_PARTS` as `('dota', 'openrsd', 'hrsc', 'fair1m', 'dior', 'p126', 'p121', 'teacher', 'distill', 'pseudo', 'dense_head', 'rpn_head', 'roi_head')`.

Use the official GroundingDINO-to-MMDetection renaming rules already mirrored in `/data1/zcy/GSOVD/tools/model_converters/groundingdino_to_mmdet.py`, but copy the rules into this repository so a later external-tree change cannot alter provenance. Build the target config/model only to obtain target names and shapes; set `model.backbone.init_cfg=None` before construction. Save `{'state_dict': compatible}` only. The provenance JSON records source/config/output absolute paths and hashes, commit, exact matched/missing/unexpected/shape-mismatch lists, matched and target numel, coverage ratio, forbidden hits, and confirms that optimizer/scheduler/EMA are absent.

Require compatible tensors from each generic subsystem prefix: `backbone.`, `encoder.`, `decoder.`, `language_model.`, and `text_feat_map.`. Explicitly leave Q900 horizontal query embeddings and incompatible 4-D final regression tensors out so Q600/angle parameters initialize freshly.

CLI:

```text
prepare_cleanstart_checkpoint.py SOURCE CONFIG OUTPUT PROVENANCE \
  --expected-sha256 SHA256
```

- [ ] **Step 4: Run focused and portable regression tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py tests/test_projects/ov_capflow/test_config_build.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
rtk git add projects/OVCapFlow/tools/prepare_cleanstart_checkpoint.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py
rtk git commit -m "feat: enforce clean-start checkpoint provenance"
```

## Task 2: Build immutable real-GT S0/S1 subsets

**Files:**

- Create: `projects/OVCapFlow/tools/build_dotav2_cleanstart_subset.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_subset.py`
- Create at run time: `work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json`
- Test: `tests/test_projects/ov_capflow/test_dotav2_subset.py`

- [ ] **Step 1: Write failing subset tests using a temporary miniature DOTA tree**

Test parsing real DOTA annotation rows, rejecting non-DOTA/generated annotations, deterministic selection, symlink destinations, manifest hashes, no source-file mutation, class/density counts, and base/novel filtering. Assert the fixed S1 quotas:

```python
assert manifest['s1']['quota'] == {
    'empty': 600, 'sparse_1_10': 900, 'medium_11_50': 300,
    'dense_51_200': 150, 'ultra_gt_200': 50}
assert manifest['s1']['train_count'] == 1600
assert manifest['s1']['val_count'] == 400
assert manifest['novel_classes'] == [
    'airport', 'container-crane', 'helipad', 'helicopter']
assert manifest['uses_pseudo_labels'] is False
```

For S0 assert 64 nonempty + 32 empty, all 18 classes covered, and at least 8 images with more than 200 GT. The builder must fail loudly if the source corpus cannot satisfy any quota or coverage rule.

- [ ] **Step 2: Run the test and confirm RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_subset.py -q
```

Expected: FAIL because the subset builder does not exist.

- [ ] **Step 3: Implement deterministic subset construction**

CLI:

```text
build_dotav2_cleanstart_subset.py \
  --source-root /data1/zcy/datasets/DOTA2_1024_500 \
  --output-root work_dirs/dotav2_cleanstart/subsets/seed20260715 \
  --seed 20260715
```

Use source images by symlink and annotation files by byte-for-byte copy for the 18-class views. For the 14-base training view, write only rows whose label is not one of the four novel labels; do not synthesize any row. Keep all 18 labels in holdout validation. Store per-file SHA256, source path, GT count, labels, density bin, train/val membership, and base/novel removed-row counts in `manifest.json`; store the manifest's own canonical SHA in `manifest.sha256`.

- [ ] **Step 4: Run unit tests and build the real subset**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_subset.py -q
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/build_dotav2_cleanstart_subset.py --source-root /data1/zcy/datasets/DOTA2_1024_500 --output-root work_dirs/dotav2_cleanstart/subsets/seed20260715 --seed 20260715
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool work_dirs/dotav2_cleanstart/subsets/seed20260715/manifest.json
```

Expected: tests pass; manifest reports S0=96 and S1=2,000 with the locked quotas and no missing symlink target.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
rtk git add projects/OVCapFlow/tools/build_dotav2_cleanstart_subset.py tests/test_projects/ov_capflow/test_dotav2_subset.py
rtk git commit -m "feat: build deterministic DOTA2 cleanstart subsets"
```

Do not commit the image/annotation subset under `work_dirs`; its manifest and hash are copied into the experiment report in Task 8.

## Task 3: Add Q600 clean-start configurations

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_base.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py`
- Create: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_real_batch.py`
- Test: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`

- [ ] **Step 1: Write failing config-contract tests**

Load all three configs with `Config.fromfile`. Assert:

- `model.type == 'OVCapFlow'`, `model.num_queries == 600`, `model.train_query_groups == 1`;
- `model.backbone.init_cfg is None`, `resume is False`, and no DOTA/HRSC/OpenRSD/P126/P121 path exists in `load_from`;
- `load_from` is exactly the generated filtered generic checkpoint;
- `bbox_head` remains `OVCapFlowHead` and uses token-similarity classification;
- sampler has `type='DNQueryBudgetBatchSampler'` and `num_matching_queries=600`;
- raw full train/val keep empty GT; S0 and S1 use only their manifest-backed real-GT roots;
- S0 is iteration-based for exactly 50 optimizer updates;
- S1 uses the same seed, optimizer, schedule, transform, prompt, and validation split for all future candidate configs;
- no test config contains `nms`, `topk`, `max_per_img`, or prediction-row limit;
- full prompt has 18 classes in the canonical order and holdout prompt has base14/novel4 records.

Extend the opt-in real-batch test so `OVCAPFLOW_RUN_DOTA2_REAL_BATCH=1` builds a Q600 clean-start batch, checks DN budget accounting, finite losses, successful backward, and exactly 600 inference rows.

- [ ] **Step 2: Confirm RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py -q
```

Expected: FAIL because the configs do not exist.

- [ ] **Step 3: Implement the configs**

The base config inherits the current DOTA-v2 protocol but overrides all training-critical fields. Set:

```python
num_queries = 600
batch_size = 1
model = dict(
    type='OVCapFlow', num_queries=num_queries, train_query_groups=1,
    backbone=dict(init_cfg=None),
    bbox_head=dict(type='OVCapFlowHead', matching_query_groups=1),
    test_cfg=dict(_delete_=True))
train_dataloader = dict(
    batch_size=batch_size,
    batch_sampler=dict(
        type='DNQueryBudgetBatchSampler',
        num_matching_queries=num_queries, num_dn_queries=100,
        max_query_area=50000000))
resume = False
load_from = 'work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_compatible.pth'
```

Keep `find_unused_parameters=True`, `static_graph=True`, resize 800, real rotations/flips, and all model parameters trainable. Do not retain the old one-epoch scheduler: S1 uses 12 epochs with validation at every epoch and a cosine schedule with linear warmup; optimizer is AdamW with effective-batch-adjusted learning rate. S0 overrides the dataset and uses `IterBasedTrainLoop(max_iters=50, val_interval=50)`; S1 uses 1,600 train and 400 validation files.

- [ ] **Step 4: Generate and audit the compatible checkpoint**

Run:

```bash
rtk mkdir -p work_dirs/dotav2_cleanstart/checkpoints
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/prepare_cleanstart_checkpoint.py /data/zcy/GroundingDINO/weights/groundingdino_swint_ogc.pth configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_base.py work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_compatible.pth work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_provenance.json --expected-sha256 3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m json.tool work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_provenance.json
```

Expected: source SHA matches; forbidden hits are empty; required generic subsystems have compatible tensors; Q600 and 5-D incompatible tensors are explicitly fresh.

- [ ] **Step 5: Run config and real-batch tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_config_build.py -q
rtk env PYTHONNOUSERSITE=1 OVCAPFLOW_RUN_DOTA2_REAL_BATCH=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
```

Expected: PASS and no OOM for batch 1 on physical GPU4.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_base.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_dotav2_real_batch.py
rtk git commit -m "feat: add Q600 DOTA2 cleanstart configs"
```

## Task 4: Implement training-only grouped one-to-one queries

**Files:**

- Create: `projects/OVCapFlow/ov_capflow/grouped_queries.py`
- Create: `tests/test_projects/ov_capflow/test_grouped_queries.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py`
- Test: `tests/test_projects/ov_capflow/test_grouped_queries.py`

- [ ] **Step 1: Write failing pure-tensor tests**

Define and test these exact interfaces:

- `repeat_matching_queries(query: Tensor, references: Tensor, groups: int) -> Tuple[Tensor, Tensor]`;
- `expand_dn_attention_mask(mask: Tensor, num_dn: int, queries_per_group: int, groups: int) -> Tensor`;
- `split_matching_groups(tensor: Tensor, queries_per_group: int, groups: int) -> Sequence[Tensor]`;
- `average_matching_loss_dicts(losses: Sequence[Mapping[str, Tensor]]) -> dict`.

Assert group=1 is value- and gradient-identical to the current path; group=3 yields Q×3 only during training; groups share the same underlying query initializer and decoder/head parameters; off-diagonal matching-group attention blocks are masked; DN-to-matching restrictions are repeated correctly; inference remains Q exactly; losses equal the arithmetic mean of three independently matched group losses rather than one Hungarian assignment over 1,800 queries.

- [ ] **Step 2: Confirm RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_grouped_queries.py -q
```

Expected: FAIL because the helpers and model switches do not exist.

- [ ] **Step 3: Implement model-side grouping**

Add `train_query_groups: int = 1` to `OVCapFlow.__init__`, validate it is positive, and in `pre_decoder`:

1. obtain the primary fixed Q queries and references;
2. only when `self.training and train_query_groups > 1`, repeat them group-major;
3. generate DN queries once;
4. expand the DN mask to `num_dn + Q * groups`, isolating matching groups;
5. set `matching_query_count=Q * groups` and pass `matching_query_groups` plus `queries_per_group=Q` to the head metadata;
6. keep evaluation unchanged at exactly Q.

Do not register a second classifier, regressor, decoder, or query embedding.

- [ ] **Step 4: Implement head-side independent matching**

Add `matching_query_groups: int = 1` to `OVCapFlowHead`. In `loss`, split decoder outputs into the DN prefix and group-major Q slices. Call the existing GroundingDINO/RHINO loss path once per matching group with `dn_meta=None`; average each matching and auxiliary-decoder loss key across groups. Compute DN loss once using the primary group's matching predictions plus the original DN prefix. Cache the primary group's final matching mask for calibration. With group=1, execute the original method without tensor reconstruction so the equivalence test is exact.

The grouped config inherits the S1 control and changes only:

```python
model = dict(
    train_query_groups=3,
    bbox_head=dict(matching_query_groups=3))
train_dataloader = dict(
    batch_sampler=dict(num_matching_queries=1800))
```

Use a distinct work directory and leave inference Q600.

- [ ] **Step 5: Run unit, strict, and real-batch tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_grouped_queries.py tests/test_projects/ov_capflow/test_strict_detector.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_parent_equivalence.py -q
rtk env PYTHONNOUSERSITE=1 OVCAPFLOW_RUN_DOTA2_REAL_BATCH=1 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
```

Expected: PASS; training sees three groups and inference emits exactly 600 rows.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
rtk git add projects/OVCapFlow/ov_capflow/grouped_queries.py projects/OVCapFlow/ov_capflow/ov_capflow.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/__init__.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py tests/test_projects/ov_capflow/test_grouped_queries.py
rtk git commit -m "feat: add training-only grouped O2O queries"
```

## Task 5: Integrate RHINO Hausdorff and adaptive positive-Hungarian DN

**Files:**

- Modify: `mmrotate/models/task_modules/assigners/__init__.py`
- Create: `projects/OVCapFlow/ov_capflow/adaptive_dn.py`
- Modify: `projects/OVCapFlow/ov_capflow/ov_capflow_head.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py`
- Create: `tests/test_projects/ov_capflow/test_adaptive_dn.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_hausdorff_dn.py`
- Test: `tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py`
- Test: `tests/test_projects/ov_capflow/test_adaptive_dn.py`

- [ ] **Step 1: Write failing Hausdorff tests**

Test registry construction through `TASK_UTILS.build(dict(type='HausdorffCost'))`, import from the assigner package, identity cost 0, symmetry, `pi`-periodic equivalent boxes, finite gradients, empty-GT shape, and dense GT CPU fallback. Fix the existing typo/documentation only when required by a failing test; preserve its geometric definition.

- [ ] **Step 2: Write failing adaptive-DN tests**

Port the RHINO `DNGroupHungarianAssigner` behavior into open-vocabulary token targets without copying a closed-set classifier. Test:

- positive DN queries compete jointly with primary matching queries per group;
- reassigned positives receive `gt_instances.positive_maps`, not an 18-wide learned class vector;
- unmatched DN positives become background token targets;
- periodic angle noise follows the shortest path modulo `pi` and widths/heights remain positive;
- `enabled=False` is exactly the inherited CDN loss;
- no teacher/dense/RPN/RoI parameter namespace is added.

- [ ] **Step 3: Confirm RED**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py tests/test_projects/ov_capflow/test_adaptive_dn.py -q
```

Expected: FAIL on missing package export and adaptive open-vocabulary DN integration.

- [ ] **Step 4: Export Hausdorff and implement adaptive DN**

Export `HausdorffCost` from `mmrotate.models.task_modules.assigners`. Add `OpenVocabularyAdaptiveDNMixin` in `adaptive_dn.py`; reuse `DNGroupHungarianAssigner` and the RHINO positive-Hungarian control flow, but construct token-level binary classification targets from `positive_maps`. Keep one shared decoder/head. The mixin receives `adaptive_dn_cfg=dict(enabled=True)` and overrides only DN target assignment/loss; standard matching remains in `OVCapFlowHead`.

The S1 Hausdorff-DN config inherits the Q600 control and changes only the geometry package:

```python
model = dict(
    bbox_head=dict(
        adaptive_dn_cfg=dict(enabled=True),
        train_cfg=dict(
            assigner=dict(match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='RBoxL1Cost', weight=5.0, box_format='xywha'),
                dict(type='GDCost', weight=2.0, loss_type='kld',
                     fun='log1p', tau=1, sqrt=False),
                dict(type='HausdorffCost', weight=2.0,
                     box_format='xywha')]),
            dn_assigner=dict(
                type='DNGroupHungarianAssigner',
                match_costs=[
                    dict(type='BinaryFocalLossCost', weight=2.0),
                    dict(type='RBoxL1Cost', weight=5.0,
                         box_format='xywha'),
                    dict(type='HausdorffCost', weight=2.0,
                         box_format='xywha')]))))
```

Do not enable grouped O2O in this candidate: S1 candidates differ from the control by one package and are not stacked before evidence.

- [ ] **Step 5: Run focused and portable tests**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py tests/test_projects/ov_capflow/test_adaptive_dn.py tests/test_projects/ov_capflow/test_strict_head.py tests/test_projects/ov_capflow/test_parent_equivalence.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
rtk git add mmrotate/models/task_modules/assigners/__init__.py projects/OVCapFlow/ov_capflow/adaptive_dn.py projects/OVCapFlow/ov_capflow/ov_capflow_head.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py tests/test_projects/ov_capflow/test_adaptive_dn.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_hausdorff_dn.py
rtk git commit -m "feat: add open-vocabulary RHINO geometry training"
```

## Task 6: Add open-vocabulary and no-hidden-postprocess audits

**Files:**

- Create: `projects/OVCapFlow/tools/audit_open_vocabulary.py`
- Create: `projects/OVCapFlow/tools/audit_decoder_alignment.py`
- Create: `tests/test_projects/ov_capflow/test_open_vocabulary_audit.py`
- Modify: `tests/test_projects/ov_capflow/test_strict_audit.py`
- Test: `tests/test_projects/ov_capflow/test_open_vocabulary_audit.py`

- [ ] **Step 1: Write failing audit tests**

Use a tiny constructed model and synthetic prompts to assert:

- canonical, reordered, synonym-expanded, and one-new-class prompts run without parameter-shape changes;
- actual prompt, entity order, token spans, positive map, model state shapes, and result counts are saved;
- no trainable linear layer with `out_features=18` is used for final class prediction;
- checkpoint keys reject teacher/distill/pseudo/dense/RPN/RoI prefixes;
- primary decoder boxes and exported prediction boxes align after the documented coordinate rescale;
- each image has exactly 600 rows in the same query order;
- strict runtime hooks record zero calls to `topk`, NMS variants, and `minAreaRect`.

- [ ] **Step 2: Confirm RED, implement, and run tests**

Run before implementation:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_open_vocabulary_audit.py tests/test_projects/ov_capflow/test_strict_audit.py -q
```

Implement both CLIs with JSON output and nonzero exit status on any failed invariant. Then run the same command again; expected PASS.

- [ ] **Step 3: Commit Task 6**

Run:

```bash
rtk git add projects/OVCapFlow/tools/audit_open_vocabulary.py projects/OVCapFlow/tools/audit_decoder_alignment.py tests/test_projects/ov_capflow/test_open_vocabulary_audit.py tests/test_projects/ov_capflow/test_strict_audit.py
rtk git commit -m "feat: audit OV prompts and decoder-aligned output"
```

## Task 7: Execute S0 and the GPU memory staircase

**Files:**

- Modify: `.lab/log.md`
- Create: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s0-result.md`
- Create at run time: `work_dirs/dotav2_cleanstart/s0_control/`

- [ ] **Step 1: Record the pre-experiment hypothesis**

Append to `.lab/log.md`: experiment ID `8-S0`, exact commit/config/checkpoint SHA/manifest SHA, hypothesis that compatible generic features plus fresh Q600/angle parameters produce finite learning, expected failure signals, and the fixed 50-update stop.

- [ ] **Step 2: Commit the S0 report skeleton before the run**

Create `s0-result.md` with protocol, hashes, command, and blank metric table represented as `not-run` values rather than placeholders. Run:

```bash
rtk git add docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s0-result.md
rtk git commit -m "docs: register DOTA2 cleanstart S0 run"
```

- [ ] **Step 3: Check GPU ownership and run batch 1→2→3 preflights on physical GPU4**

Run:

```bash
rtk nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py OVCAPFLOW_PREFLIGHT_BATCH=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py OVCAPFLOW_PREFLIGHT_BATCH=2 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 OVCAPFLOW_DOTA2_CONFIG=configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py OVCAPFLOW_PREFLIGHT_BATCH=3 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_dotav2_real_batch.py -q
```

Choose the largest batch with peak allocated memory no more than 42 GiB. At the first OOM, stop the staircase and keep the last passing batch; increase accumulation so the intended effective batch is unchanged.

- [ ] **Step 4: Run the 50-update S0 smoke**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s0.py --work-dir work_dirs/dotav2_cleanstart/s0_control --cfg-options resume=False
```

Then run the strict, open-vocabulary, and decoder-alignment audits on the S0 checkpoint. S0 passes only if losses are finite, backward and save/load work, inference is exactly 600 rows, prompt variants do not change parameter shapes, and forbidden-call hits are zero.

- [ ] **Step 5: Record result and commit evidence**

Fill `s0-result.md` with loss range, memory staircase, batch/accumulation choice, hashes, audit JSON paths and pass/fail. Append the outcome to `.lab/log.md` and `.lab/results.tsv`. Run:

```bash
rtk git add docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s0-result.md
rtk git commit -m "docs: record DOTA2 cleanstart S0 result"
```

If S0 fails, diagnose and fix the engineering failure before any S1 run; it is not a scientific candidate result.

## Task 8: Run the locked 2,000-image S1 screen

**Files:**

- Modify: `.lab/log.md`
- Modify: `.lab/results.tsv`
- Create: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s1-screen.md`
- Create: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/subset-manifest.json`
- Create at run time: `work_dirs/dotav2_cleanstart/s1_control/`
- Create at run time: `work_dirs/dotav2_cleanstart/s1_grouped/`
- Create at run time: `work_dirs/dotav2_cleanstart/s1_hausdorff_dn/`

- [ ] **Step 1: Freeze S1 evidence before training**

Copy the canonical subset manifest into the experiment directory with `apply_patch`, preserving its canonical JSON bytes, and create `s1-screen.md` with the exact control/grouped/Hausdorff-DN configs, commit hashes, checkpoint provenance hash, 12-epoch budget, validation mouth, class split, promotion formula, and `not-run` result rows. Record `8-S1-C`, `8-S1-G`, and `8-S1-H` THINK entries immediately before their respective launches.

Run:

```bash
rtk git add docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s1-screen.md docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/subset-manifest.json
rtk git commit -m "docs: freeze DOTA2 cleanstart S1 protocol"
```

- [ ] **Step 2: Run Q600 control on one available A40**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_control.py --work-dir work_dirs/dotav2_cleanstart/s1_control --cfg-options resume=False
```

Evaluate both all-18 and base14/novel4 prompt holdout. Save AP50, mAP, per-class AP/recall, GT coverage, extras/GT, fixed-row count, and prompt audit.

- [ ] **Step 3: Run grouped O2O and apply the promotion rule**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_grouped.py --work-dir work_dirs/dotav2_cleanstart/s1_grouped --cfg-options resume=False
```

Use the same evaluator and audits. Record whether the locked promotion rule passes. Do not stack it with Hausdorff-DN yet.

- [ ] **Step 4: Run Hausdorff adaptive-DN and apply the same rule**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/mmdet/bin/python tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_s1_hausdorff_dn.py --work-dir work_dirs/dotav2_cleanstart/s1_hausdorff_dn --cfg-options resume=False
```

Use the same evaluator and audits. Select the highest AP50 candidate among those that pass every safety/OV/E2E audit and the promotion rule. If neither intervention passes, promote the control only for a short full-data diagnostic, not for a 24-epoch claim, and activate the O2-DEIM structural fork after three valid no-gain candidates as specified in the design.

- [ ] **Step 5: Finalize and commit S1 evidence**

Update `.lab`, include actual commands, seeds, wall time, GPU memory/utilization, best epoch, metric JSON hashes, and the promotion decision in `s1-screen.md`. Run:

```bash
rtk git add docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/s1-screen.md docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/subset-manifest.json
rtk git commit -m "docs: record DOTA2 cleanstart S1 screen"
```

## Task 9: Create the promoted full-data 24-epoch config and monitor

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py`
- Create: `projects/OVCapFlow/tools/monitor_gpu_run.py`
- Create: `tests/test_projects/ov_capflow/test_gpu_monitor.py`
- Modify: `tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py`
- Create: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/full24e-protocol.md`

- [ ] **Step 1: Write failing tests for full protocol and monitoring**

Assert full train has 47,294 tiles, raw val has 13,833 tiles, both retain empty GT, epochs are 24, evaluation checkpoints include 1/6/12/18/24, `resume=False`, no forbidden checkpoint, physical GPU list is `[4,5,6,7]`, sampler query area reflects the chosen per-GPU batch, and effective batch is 32 through accumulation. Monitor tests parse mocked `nvidia-smi` CSV, write JSONL every 30 seconds, detect dead rank/NCCL/OOM/NaN signatures, and summarize per-card median utilization and memory.

- [ ] **Step 2: Implement the promoted config without silently stacking losers**

Inherit the exact S1 winner, switch only dataset to full train/raw val, schedule to 24 epochs, batch/accumulation to the GPU4 staircase result, and work directory to a new full-run path. Implement checkpoint intervals that guarantee 1/6/12/18/24 artifacts and validation at those epochs. The monitor accepts `--pid`, `--gpu-indices 4 5 6 7`, `--interval 30`, `--train-log`, `--output`, and exits nonzero when all ranks die or a fatal pattern appears.

- [ ] **Step 3: Run tests and commit before launch**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow/test_gpu_monitor.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py tests/test_projects/ov_capflow/test_strict_audit.py -q
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py projects/OVCapFlow/tools/monitor_gpu_run.py tests/test_projects/ov_capflow/test_gpu_monitor.py tests/test_projects/ov_capflow/test_dotav2_cleanstart_config.py docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/full24e-protocol.md
rtk git commit -m "feat: prepare four-card DOTA2 cleanstart full training"
```

## Task 10: Launch and supervise four-card full training

**Files:**

- Modify: `.lab/log.md`
- Modify: `.lab/results.tsv`
- Create at run time: `work_dirs/dotav2_cleanstart/full24e_<winner>/`
- Modify: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/full24e-protocol.md`

- [ ] **Step 1: Preflight GPUs 4–7 and freeze Experiment 8-S2 THINK**

Verify no foreign process owns GPUs4–7, record UUIDs, available memory, selected per-GPU batch, accumulation, effective batch, commit, config hash, source/compatible checkpoint hashes, and S1 promotion evidence.

Run:

```bash
rtk nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,pstate --format=csv
rtk git status --short
rtk git rev-parse HEAD
rtk sha256sum configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_compatible.pth work_dirs/dotav2_cleanstart/checkpoints/groundingdino_swint_ogc_q600_provenance.json
```

- [ ] **Step 2: Launch with the required NCCL settings**

Run from the repository root:

```bash
rtk env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29670 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=4 tools/train.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/full24e_promoted --cfg-options resume=False
```

Start `monitor_gpu_run.py` against the launcher PID in a second terminal. Sample at 30 seconds, never slower than 60 seconds. Target median utilization at least 90% and stable 35–42 GiB per A40. If utilization is low but data loading is healthy, raise workers/prefetch; if memory has safe headroom, raise per-GPU batch and reduce accumulation proportionally only after a new preflight and config commit.

- [ ] **Step 3: Apply bounded recovery rules**

- OOM: stop once, reduce per-GPU batch by one, increase accumulation to preserve effective batch, update sampler budget, test, commit, and restart from epoch 0 unless the same unmodified run has a valid own checkpoint and provenance.
- NCCL failure: keep model/data/seed unchanged, verify ranks and required env flags, use a new port/work log, and resume only the same run's checkpoint.
- NaN/Inf: stop, preserve first bad batch IDs and log, reproduce on one GPU, diagnose numerics, add a test, then launch a new work directory.
- Data/sampler failure: preserve audit JSON and sample IDs, fix and test the sampler; never silently drop dense or empty samples.
- Scientific plateau: do not mutate a running experiment. Evaluate the scheduled checkpoint, record it, then decide a new committed experiment.

- [ ] **Step 4: Evaluate at epochs 1, 6, 12, 18, and 24**

Use the raw full validation mouth for every scheduled decision. If epoch24 is still rising and either metric is below 0.7000, extend the same recipe to 36, then 48 epochs in committed config revisions. If the curve is flat, return to the S1 screen/O2-DEIM fork instead of spending four cards on an unchanged recipe.

## Task 11: Run both evaluation mouths and final integrity audit

**Files:**

- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_eval_filtered6605.py`
- Create: `docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/final-result.md`
- Modify: `.lab/log.md`
- Modify: `.lab/results.tsv`

- [ ] **Step 1: Run raw-13,833 evaluation first**

Run:

```bash
rtk env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29671 /data/zcy/anaconda3/envs/mmdet/bin/python -m torch.distributed.run --nproc_per_node=4 tools/test.py configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_full24e.py work_dirs/dotav2_cleanstart/full24e_promoted/best_dota_mAP_epoch_24.pth --launcher pytorch --work-dir work_dirs/dotav2_cleanstart/eval_raw13833 --out work_dirs/dotav2_cleanstart/eval_raw13833/predictions.pkl
```

Before accepting metrics, assert evaluator `dataset_len==13833`, prediction record count 13,833, every record has exactly 600 rows, IoU threshold is 0.5, and 18 per-class records exist. Save metric JSON/log/config dump/prediction SHA/checkpoint SHA.

- [ ] **Step 2: Run filtered-6605 separately**

The filtered config differs only by `filter_empty_gt=True` and work/output naming. Run the same checkpoint and prompt, assert `dataset_len==6605`, and label every artifact and table row `filtered-6605`.

- [ ] **Step 3: Run final OV/E2E/provenance audits**

Run the prompt reorder, synonyms, new-class, base14/novel4, strict forbidden-call, 600-row, decoder-alignment, checkpoint namespace, and source provenance audits. Scan the expanded config and state dict. Record `forbidden_call_hits=0` and `forbidden_state_keys=[]`; absence of evidence is not a pass.

- [ ] **Step 4: Write and verify the final result**

`final-result.md` must report raw AP50/mAP first, filtered result second, all 18 AP/recall, novel AP/recall, dense `GT>600` coverage, extras/GT, GPU utilization/memory summary, run duration, all hashes, and a checklist of the eight completion conditions. If either raw metric is below 0.7000, title the report `intermediate result` and keep the project active.

Run the full portable suite and diff checks:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest tests/test_projects/ov_capflow tests/test_models/test_task_modules/test_assigners/test_hausdorff_cost.py -q
rtk git diff --check
rtk git status --short
```

- [ ] **Step 5: Commit final evidence only after verification**

Run:

```bash
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-t_dotav2_cleanstart_q600_eval_filtered6605.py docs/project_history/exp_20260715_dotav2_cleanstart_ov_e2e_ap70/final-result.md
rtk git commit -m "docs: record cleanstart DOTA2 OV E2E result"
```

## Plan self-review gate

Before implementation starts, verify this plan against the design spec:

```bash
rtk rg -n "T[B]D|T[O]DO|implement[ ]later|fill[ ]in|待[定]" docs/superpowers/plans/2026-07-15-dotav2-cleanstart-ov-e2e-ap70.md
rtk rg -n "raw-13,833|0.7000|Q600|grouped|Hausdorff|adaptive|GPU4|GPU5|GPU6|GPU7|NMS|top-k|teacher|pseudo|novel" docs/superpowers/plans/2026-07-15-dotav2-cleanstart-ov-e2e-ap70.md
rtk git diff --check
```

The first command must return no matches. The second must cover every design pillar. Only then commit this plan and begin Task 1.
