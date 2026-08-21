# Strict-OV N0 Negative-Gradient Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `using-git-worktrees` before edits, then `executing-plans` task-by-task; apply `test-driven-development` to every production change and `verification-before-completion` before each claimed gate.

**Goal:** Implement and execute a leakage-resistant strict-OV N0 diagnostic that decides whether harmful false-background gradients exist strongly enough at both generic zero-step and a frozen seen-only Epoch 1 endpoint to make a later N1 intervention scientifically eligible.

**Architecture:** Start from the last pre-RISC N0 evidence commit, reuse the committed M0/M1 integrity chain, and materialize physically separate per-fold model/audit annotation views. Run the unmodified detector loss path while temporarily observing its exact Hungarian assignments and final Q600 state, recompute the configured focal gradients only with respect to detached final query states, and compare withheld-adjacent rows with fixed-budget random, spatial-shift, scene-shuffle, and seen-object controls. Publish immutable sufficient statistics and fail-closed decisions; do not implement N1 unless the combined zero-step/E1 N0 gate passes.

**Tech Stack:** Python 3.8, PyTorch 1.12.1+cu113, MMCV 2.1.0, MMEngine 0.10.4, MMDetection 3.3.0, MMRotate/OV-CapFlow, NumPy, pytest, canonical JSON/JSONL, SHA-256, Git worktrees.

---

## 1. Scope, authority, and non-goals

### 1.1 Execution base and environment

- Implementation base commit: `9dbb3f03c49692f7b90c18db49b4cb5295bbff37` (`fix: preserve N0-RI invalid authority`). It is the last N0-RI evidence commit before later RISC implementation work.
- New branch/worktree: `research/strict-ov-n0-ng` at `/data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng`.
- Runtime environment: `/data/zcy/anaconda3/envs/mmdet`; every Python invocation uses `rtk conda run -n mmdet python`.
- Repository commands in this plan always begin with `rtk`, per the workspace `AGENTS.md`.
- The dirty `.worktrees/n0-literature-m1` worktree is evidence-only input. Do not edit it, clean it, or branch from its working tree state.

### 1.2 Frozen scientific authority

| Authority | Frozen value |
|---|---|
| M0 root | `.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438` |
| M0 manifest SHA256 | `0c2382ed886d384c363e4b3ae68ec2cdc04e14a12cc942421a8358f24c7d13aa` |
| M0 inventory SHA256 | `bafb17b75911a89c25b893b8c45a1420283714ee0888e545b149909c03309ba8` |
| M1 root | `.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1` |
| M1 manifest SHA256 | `0deb6e3ca635abc0a3491c6abfb8bf1dc00753f56f949af7f42862fcd39fdd1d` |
| M1 content inventory SHA256 | `2b1bb3b3da30688d50cef6f0fbcc0a22d4a554e6ac6c6e5e007f836ccd3caef2` |
| M1 records | 47,294 RGB images; train/dev/test = 35,109/5,172/7,013 |
| M1 cross-partition content overlap | 0 |
| N0 preregistration | `docs/superpowers/specs/2026-08-03-n0-negative-gradient-diagnostic.md`, SHA256 `2d5281ca6870bd3f86ac6db1c75c10490dc8dc59b0dfbb291a8dc70634d3b228` |
| Closest-work record | `docs/literature-search-20260803-negative-safe-fixed-set/papers.md`, SHA256 `98e8753f49e4381cf2a7d0336316af97904b4b8da256d35fe9aea070841a6422`, decision `NARROW` |
| Generic Swin-B source | `/data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth` |
| Generic source SHA256 / bytes | `46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7` / `938057991` |
| Generic source URL | `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth` |
| Query/inference mouth | fixed Q600, rotated 5D, all rows, no proposal/top-k/NMS |

The historical M1 report still says `preparation_only=true`, `training_use_forbidden=true`, and `N0_N1_authorized=false`. This plan converts that completed integrity prerequisite into a new, separately committed strict-view authority; it does not reinterpret the old M1 marker as training authorization.

### 1.3 Four immutable meta-novel folds

1. Fold 1 withheld: `plane`, `baseball-diamond`, `tennis-court`, `large-vehicle`.
2. Fold 2 withheld: `ground-track-field`, `soccer-ball-field`, `bridge`, `storage-tank`.
3. Fold 3 withheld: `basketball-court`, `swimming-pool`, `ship`.
4. Fold 4 withheld: `roundabout`, `harbor`, `small-vehicle`.

The official novel4 (`airport`, `container-crane`, `helipad`, `helicopter`) is never a training, selection, threshold, checkpoint, stopping, rescue, or result source. A sanitizer may recognize those four denylisted labels solely to discard their complete source rows; it must not persist their counts, coordinates, difficulties, per-image presence, or any derivative statistic.

### 1.4 Decision order

```text
CPU fixtures and authority seals
  -> Fold 1 generic zero-step pilot
  -> four-fold generic zero-step confirmation
  -> four frozen seen-only E1 trainings
  -> four-fold E1 confirmation
  -> combined N0 decision
  -> N1 design eligibility only
```

- Pilot failure: `FAIL_CLOSE_N1`; do not run the four-fold confirmation.
- Four-fold zero-step failure: `FAIL_CLOSE_N1`; do not train E1.
- Any E1 endpoint failure: `FAIL_CLOSE_N1`; do not design the intervention as if the mechanism were established.
- Both endpoints pass independently: publish `N1_ELIGIBLE.json`. This is mechanism evidence, not AP improvement evidence.
- Infrastructure/provenance failure: `INVALID_NO_DECISION`; repair only the demonstrated contract problem and use a fresh attempt directory.

### 1.5 Explicit non-goals

- Do not implement, tune, or train N1 in this plan.
- Do not reuse B0, RISC, POQ, rotation-control, OMQ, ODQ, or other project-trained weights, caches, metrics, samplers, or thresholds.
- Do not transport the first 600 generic content queries; `--transport-first-queries` remains off.
- Do not use rare-positive repeat sampling or `RandomRotate`.
- Do not evaluate official validation/test or official novel4.
- Do not modify the production Hungarian assigner, detection head, or focal-loss source merely to expose diagnostics; use temporary context-managed observation and restore it in `finally`.

## 2. Claim-evidence matrix

| Claim or decision | Required evidence | Artifact | Status before execution |
|---|---|---|---|
| M1 content/header seal is valid | exact committed marker, manifest/content hashes, zero cross-partition collision | existing M1 `COMMITTED.json` | verified prerequisite |
| Closest work leaves a narrow gap | 21 screened, 15 primary-source verified, frozen narrow claim | existing `papers.md` | `NARROW`; refresh before N1 |
| Model branch is strict seen-only | per-fold prompt and annotations contain seen classes only; exact scene manifest | strict-view `COMMITTED.json` | not run; must be read from artifact |
| Official novel4 did not influence N0 | denylist-only discard, no persisted counts/coordinates, forbidden-read/import audits | strict-view and runner decisions | not run; must be read from artifact |
| Assignment identity is authoritative | observed return of the exact assigner call uniquely aligned to final matching Q600 rows | capture sufficient statistics | not run; must be read from artifact |
| Negative gradients are concentrated | eligible/random ratio, controls, scene bootstrap and permutation evidence | endpoint `decision.json` | not run; must be read from artifact |
| Negative gradients conflict with foreground-compatible updates | median `-cos(g_neg,g_pos)`, controls and intervals | endpoint `decision.json` | not run; must be read from artifact |
| Mechanism persists after seen-only learning | full four-fold gate independently passes at E1 | combined `decision.json` | not run; must be read from artifact |
| N1 may be designed | zero-step and E1 full gates both pass | `N1_ELIGIBLE.json` | unauthorized before execution |
| N1 improves detection | matched N0/N1 AP experiment | outside this plan | no evidence |

Allowed research wording remains:

> We investigate whether cross-view-stable evidence can identify and suppress harmful false-background classification gradients in a fixed-Q600 rotated open-vocabulary set predictor, using a negative-only training intervention with no positive pseudo-box targets and no proposal/top-k/NMS path.

Do not convert a pass into “first”, “effective method”, “zero-shot AP improvement”, or “strict-OV generalization improved”.

## 3. Planned file map

| File | Role |
|---|---|
| `projects/OVCapFlow/tools/n0_ng_protocol.py` | constants, authority validation, canonical publication, decision schemas |
| `docs/superpowers/specs/2026-08-08-n0-negative-gradient-firewall-amendment.md` | pre-execution clarification for denylist-only official-novel sanitation |
| `projects/OVCapFlow/tools/build_n0_strict_views.py` | four physically separated scene/fold model and audit views |
| `projects/OVCapFlow/ov_capflow/strict_ov_dataset.py` | explicit manifest-backed DOTA dataset without annotation globbing |
| `projects/OVCapFlow/ov_capflow/__init__.py` | register/export `SealedSceneDOTAv2Dataset` |
| `projects/OVCapFlow/tools/seal_n0_ng_initialization.py` | fresh generic Swin-B-compatible initialization seal |
| `projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py` | weights-only E1 endpoint and exact-coverage seal |
| `projects/OVCapFlow/tools/n0_ng_capture.py` | exact final-state and assignment observer |
| `projects/OVCapFlow/tools/n0_ng_gradients.py` | exact configured negative/audit-positive focal gradients |
| `projects/OVCapFlow/tools/n0_ng_controls.py` | rotated eligibility, deduplication, and four controls |
| `projects/OVCapFlow/tools/n0_ng_metrics.py` | image macro, clustered inference, Holm gates, combined decision |
| `projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py` | fail-closed pilot/full CLI and immutable artifacts |
| `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_base.py` | frozen Swin-B/Q600/E1 and diagnostic recipe |
| `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold{1,2,3,4}_e1.py` | fold prompts/views/work dirs |
| `tests/test_projects/ov_capflow/test_n0_ng_*.py` | focused protocol, data, capture, gradient, control, metric, runner tests |
| `docs/project_history/exp_20260808_strict_ov_n0_ng/fres_strict_ov_n0_ng_zh.md` | final factual result record; created only from committed decisions |

## Task 0: Create an isolated pre-RISC implementation worktree

**Files:**
- Read: `/data1/zcy/OV-CapFlow/docs/superpowers/plans/2026-08-08-strict-ov-n0-negative-gradient-gate.md`
- Create later through Git: `/data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng/`

- [ ] **Step 1: Verify the base and target are unambiguous**

```bash
rtk git rev-parse 9dbb3f03c49692f7b90c18db49b4cb5295bbff37
rtk git show --no-patch --oneline 9dbb3f03c49692f7b90c18db49b4cb5295bbff37
rtk git show-ref --verify refs/heads/research/strict-ov-n0-ng
rtk git worktree list --porcelain
```

Expected: the first command prints the exact 40-hex base; the second prints `fix: preserve N0-RI invalid authority`; the target branch does not yet exist; the target worktree path is absent. If the target exists, inspect it and stop instead of overwriting or deleting it.

- [ ] **Step 2: Create the worktree from the exact base commit**

```bash
rtk git worktree add -b research/strict-ov-n0-ng /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng 9dbb3f03c49692f7b90c18db49b4cb5295bbff37
rtk git -C /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng rev-parse HEAD
rtk git -C /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng status --short
```

Expected: exact base hash and an empty status. Do not copy dirty files from `.worktrees/n0-literature-m1`.

- [ ] **Step 3: Verify the selected runtime**

```bash
rtk conda run -n mmdet python -c "import sys, torch, mmcv, mmengine, mmdet; print(sys.version.split()[0]); print(torch.__version__, mmcv.__version__, mmengine.__version__, mmdet.__version__)"
```

Expected: Python `3.8.19`; packages `1.12.1+cu113 2.1.0 0.10.4 3.3.0`.

## Task 1: Implement the frozen protocol and authority validator

**Files:**
- Create: `docs/superpowers/specs/2026-08-08-n0-negative-gradient-firewall-amendment.md`
- Create: `projects/OVCapFlow/tools/n0_ng_protocol.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_protocol.py`

- [ ] **Step 1: Freeze the narrow sanitation amendment before code or annotation access**

The original preregistration forbids official-novel access, while a strict model view cannot be derived from all-18 source text without recognizing rows that must be discarded. Add an amendment that permits exactly one pre-model, streaming sanitation boundary to compare a source label against the four-name denylist and discard the entire row. It forbids persisting or reporting class-specific names, counts, coordinates, difficulty, image presence, hashes derived from the row, or any branch based on those observations. It changes no fold, sample, threshold, estimand, control, checkpoint, or gate. Bind the tracked amendment's bytes/SHA in every later view and attempt authority before any annotation is opened.

- [ ] **Step 2: Write failing tests for exact constants and decisions**

Cover the four folds, exact hashes above, Q600, `64/128` pilot minimums, `128` target tiles per full fold, `1,024` random draws, `10,000` bootstrap/permutation draws, `1.25` ratio, `0.10` conflict, `1e-12` cosine epsilon, official-novel denylist, and the three-state decision vocabulary.

```python
def test_frozen_fold_partition_is_exact_and_disjoint():
    assert tuple(item.meta_novel for item in FOLDS) == (
        ('plane', 'baseball-diamond', 'tennis-court', 'large-vehicle'),
        ('ground-track-field', 'soccer-ball-field', 'bridge', 'storage-tank'),
        ('basketball-court', 'swimming-pool', 'ship'),
        ('roundabout', 'harbor', 'small-vehicle'),
    )
    for item in FOLDS:
        assert set(item.seen).isdisjoint(item.meta_novel)
        assert set(item.seen) | set(item.meta_novel) == set(BASE14_CLASSES)
```

- [ ] **Step 3: Run the tests and observe the intended import failure**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_protocol.py -q
```

Expected: fail because `n0_ng_protocol.py` does not exist.

- [ ] **Step 4: Implement immutable contracts and canonical publication**

Reuse `load_committed_m1`, `canonical_json_bytes`, and the repository's no-replace primitives. Do not reimplement M1 parsing. Expose typed records and validators such as:

```python
@dataclass(frozen=True)
class FoldContract:
    fold: int
    seen: Sequence[str]
    meta_novel: Sequence[str]

class N0NGDecision(str, Enum):
    PASS = 'PASS'
    FAIL_CLOSE_N1 = 'FAIL_CLOSE_N1'
    INVALID_NO_DECISION = 'INVALID_NO_DECISION'

def validate_prerequisites(repo_root: Path) -> Mapping[str, Any]:
    """Bind the exact M0/M1/spec/literature/source identities or raise."""

def publish_committed_attempt(root: Path, payloads: Mapping[str, bytes]) -> None:
    """Publish files without replacement and COMMITTED.json strictly last."""
```

The validator must reject changed hashes, an untracked/changed sanitation amendment, a missing M1 completion marker, M1 source-identity drift, a literature decision other than `NARROW`, a source checkpoint with the wrong bytes/hash, non-canonical JSON, any pre-existing output path, and any code authority rooted in a different Git commit than the runner records.

- [ ] **Step 5: Add CLI-only prerequisite verification**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_protocol.py verify-prerequisites --repo-root /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng --m0-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438 --m1-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --generic-source /data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

Expected after implementation: canonical JSON on stdout with `status=PASS_PREREQUISITES`, the exact hashes, and no writes.

- [ ] **Step 6: Run focused tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_protocol.py -q
rtk git diff --check
rtk git add docs/superpowers/specs/2026-08-08-n0-negative-gradient-firewall-amendment.md projects/OVCapFlow/tools/n0_ng_protocol.py tests/test_projects/ov_capflow/test_n0_ng_protocol.py
rtk git commit -m "feat: bind strict-OV N0 protocol authority"
```

Expected: focused tests pass and the commit contains only the pre-data amendment, protocol, and tests.

## Task 2: Materialize four sealed, physically separate strict views

**Files:**
- Create: `projects/OVCapFlow/tools/build_n0_strict_views.py`
- Create: `tests/test_projects/ov_capflow/test_n0_strict_views.py`

- [ ] **Step 1: Write failing synthetic-view tests**

Build fixtures containing seen, withheld meta-novel, official novel4, difficult, ignored, empty-after-filter, duplicate, malformed, and symlinked annotations. Assert:

- only M1-listed tiles are admitted and their scene/partition/content identities remain exact;
- all 35,109 train identities and all 5,172 dev identities are represented in a real build, including empty-after-filter tiles;
- each fold's `model/train` and `model/dev` annotation files contain only its seen classes;
- each fold's `audit/dev` ledger contains only its withheld meta-novel rows;
- the audit ledger has stable hashed GT IDs and hexadecimal source qboxes, while the model ledger cannot import or locate it;
- official novel4 rows leave no count, coordinate, difficulty, tile-presence, or class-key trace in any output;
- model and audit roots are real, disjoint, non-symlink directories;
- test-partition annotations are never opened;
- a failure leaves no `COMMITTED.json`; rerunning into an existing path fails without overwrite.

- [ ] **Step 2: Run the failing fixture suite**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_strict_views.py -q
```

Expected: fail because the builder is absent.

- [ ] **Step 3: Implement two-phase annotation sanitization**

The builder first validates committed M1 and ranks M1 rows. It then opens only train/dev annotation files named by those rows. For each source line:

```python
if class_name in OFFICIAL_NOVEL4:
    continue  # no counter and no derivative record
if class_name in fold.seen:
    emit_model_row_verbatim()
elif partition == 'dev' and class_name in fold.meta_novel:
    emit_audit_object_with_hashed_identity()
```

No output filename, JSON key, counter, warning, exception text, or log line may reveal which tile contained an official novel4 row. Unknown labels, malformed lines, unsafe paths, or duplicate source objects fail the entire fresh attempt.

Use this layout:

```text
strict_views_v1/
  fold1/
    model/train/annfiles/*.txt
    model/dev/annfiles/*.txt
    model/model_manifest.json
    audit/dev/audit_manifest.json
    COMMITTED.json
  fold2/
  fold3/
  fold4/
  report.json
  COMMITTED.json
```

Do not copy or symlink image files. Model manifests contain explicit M1-bound image paths and annotation paths. Audit coordinates remain only under the isolated audit root and are never copied into diagnostic results.

- [ ] **Step 4: Implement authority and publication checks**

Each fold marker binds code/spec/M0/M1 hashes, seen-prompt SHA256, model manifest SHA256, audit manifest SHA256, train/dev identity counts, zero scene/content overlap, and `test_partition_opened=false`. Publish fold markers first and the root marker last. A committed root must be immutable.

- [ ] **Step 5: Pass tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_strict_views.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/build_n0_strict_views.py tests/test_projects/ov_capflow/test_n0_strict_views.py
rtk git commit -m "feat: seal strict-OV fold annotation views"
```

Expected: fixture suite passes; no real dataset has been opened by the tests.

## Task 3: Add the explicit manifest-backed strict dataset

**Files:**
- Create: `projects/OVCapFlow/ov_capflow/strict_ov_dataset.py`
- Modify: `projects/OVCapFlow/ov_capflow/__init__.py`
- Create: `tests/test_projects/ov_capflow/test_strict_ov_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Test registry construction and require that `load_data_list()` follows the committed model manifest order exactly instead of `glob`. The dataset must reject:

- an uncommitted or hash-changed view;
- `partition` outside `train/dev`;
- a manifest tile whose M1 identity, scene, content hash, image root, or annotation filename differs;
- audit-root paths, symlinked annotation files, absolute paths escaping the committed model root, duplicate image IDs, an unexpected class, or a model prompt differing from `metainfo.classes`;
- `test_mode=False` on dev and any request for the M0 test partition.

Assert that an annotation containing zero remaining seen objects returns `instances=[]` and remains in the dataset when `filter_empty_gt=False`.

- [ ] **Step 2: Run tests and observe the missing class**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_strict_ov_dataset.py -q
```

Expected: fail because `SealedSceneDOTAv2Dataset` is absent.

- [ ] **Step 3: Implement the dataset with no implicit discovery**

Subclass the repository's DOTA dataset only for shared instance conversion, but override discovery completely:

```python
@DATASETS.register_module()
class SealedSceneDOTAv2Dataset(DOTADataset):
    def __init__(self, view_manifest: str, partition: str, **kwargs):
        self.view_manifest = Path(view_manifest)
        self.partition = partition
        self._sealed_records = validate_model_view(
            self.view_manifest, partition=partition)
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        return [self._load_one_sealed_record(item)
                for item in self._sealed_records]
```

Open annotation files with no-follow regular-file checks. Parse only classes in the manifest-bound seen prompt. Include `scene_id`, `partition`, `content_sha256`, and the immutable tile identity in `data_info`, then carry `scene_id` through `PackDetInputs.meta_keys`.

- [ ] **Step 4: Register and pass tests**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_strict_ov_dataset.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/ov_capflow/strict_ov_dataset.py projects/OVCapFlow/ov_capflow/__init__.py tests/test_projects/ov_capflow/test_strict_ov_dataset.py
rtk git commit -m "feat: load strict-OV data from sealed manifests"
```

Expected: explicit-order, empty-tile, path-safety, and registry tests pass.

## Task 4: Freeze fold configs and seal an independent generic initialization

**Files:**
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_base.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py`
- Create: `configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py`
- Create: `projects/OVCapFlow/tools/seal_n0_ng_initialization.py`
- Create: `projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_configs.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_checkpoint_seals.py`

- [ ] **Step 1: Write failing config-contract tests**

Load all four configs and assert:

```python
assert cfg.model.num_queries == 600
assert cfg.selected_world_size == 10
assert cfg.train_dataloader.batch_size == 2
assert cfg.optim_wrapper.accumulative_counts == 1
assert cfg.train_cfg.max_epochs == 1
assert cfg.randomness.seed == 20260808
assert cfg.train_dataloader.dataset.filter_cfg.filter_empty_gt is False
assert all(step.type != 'RandomRotate' for step in cfg.train_pipeline)
assert not hasattr(cfg, 'rare_repeat_factor')
```

Also require Swin-B `embed_dims=128`, depths `[2,2,18,2]`, heads `[4,8,16,32]`, window `12`, Q600 all-query prediction, exact fold prompts, distinct fold work dirs, no validation/test dataloader, and no B0/RISC/POQ checkpoint ancestry.

The training pipeline is frozen to load qboxes, convert to rboxes, resize to `1024x1024`, filter only degenerate boxes, apply the existing flip recipe (`prob=0.75`, horizontal/vertical/diagonal), and pack the sealed scene identity. The diagnostic dev pipeline contains no stochastic transform.

- [ ] **Step 2: Write failing initialization/E1 seal tests**

Require the generic seal to:

- invoke the existing clean-start conversion against the strict Swin-B target architecture;
- reject a nonfresh root, wrong source hash/bytes, training state, forbidden namespaces, and `--transport-first-queries`;
- publish weights/provenance/SHA sidecars and `COMMITTED.json` last;
- prove the compatible state omits `query_initializer.query_embedding.weight`, so Q600 starts from the config's seeded generic fresh initialization rather than D11's closed first-600 transport;
- prove one generic sealed checkpoint is shape-compatible with all four fold configs.

Require each E1 seal to bind exactly one fold config, the generic initialization seal, `epoch_1.pth`, and the epoch sampler audit with `dataset_size=35109`, `duplicate_count=0`, `missing_count=0`, `epoch=0`. Its published checkpoint contains a model `state_dict` only, never optimizer/scheduler/scaler state.

- [ ] **Step 3: Run the intentionally failing tests**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_configs.py tests/test_projects/ov_capflow/test_n0_ng_checkpoint_seals.py -q
```

Expected: fail because configs and sealers are absent.

- [ ] **Step 4: Implement the base and four fold configs**

The base config derives architecture/optimizer primitives from the existing clean-start Swin-B recipe but deletes rare4x data paths and all official validation components. Freeze:

```python
n0_ng_role = 'strict_ov_seen_only_epoch1_control'
selected_world_size = 10
randomness = dict(seed=20260808, deterministic=False, diff_rank_seed=False)
train_cfg = dict(
    _delete_=True, type='VariableBatchEpochBasedTrainLoop',
    max_epochs=1, val_interval=2)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
```

Use `DNQueryBudgetBatchSampler` with `update_count_multiple=1` and no-replace per-fold audit. It already gives exact non-padded, no-duplicate distributed coverage for a nondivisible dataset. Freeze the optimizer to AdamW `lr=1e-4`, weight decay `1e-4`, backbone LR multiplier `0.1`, clip norm `0.1`; freeze the schedulers before the pilot to LinearLR over the first 500 iterations and CosineAnnealingLR over the single epoch. Save only the Epoch 1 endpoint.

- [ ] **Step 5: Implement the two sealers**

`seal_n0_ng_initialization.py` wraps `prepare_cleanstart_checkpoint.prepare_checkpoint` inside a fresh root, enforces `transport_first_queries=False`, canonicalizes provenance, checks all four config shapes, and commits last. `seal_n0_ng_e1_checkpoint.py` opens the training checkpoint once on CPU, validates config/sampler/generic-parent hashes and epoch metadata, writes a weights-only state, and commits last.

- [ ] **Step 6: Pass focused tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_configs.py tests/test_projects/ov_capflow/test_n0_ng_checkpoint_seals.py -q
rtk git diff --check
rtk git add configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_base.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py projects/OVCapFlow/tools/seal_n0_ng_initialization.py projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py tests/test_projects/ov_capflow/test_n0_ng_configs.py tests/test_projects/ov_capflow/test_n0_ng_checkpoint_seals.py
rtk git commit -m "feat: freeze strict-OV N0 endpoints"
```

Expected: config expansion and synthetic checkpoint seals pass without opening real training annotations or launching training.

## Task 5: Capture the exact final Q600 assignment and state

**Files:**
- Create: `projects/OVCapFlow/tools/n0_ng_capture.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_capture.py`

- [ ] **Step 1: Write failing capture tests**

Use a small fake head plus a real-head fixture to cover:

- direct model and one supported wrapper resolve uniquely;
- nested/concurrent capture is rejected;
- exactly one detector loss forward and a one-image batch are required;
- DN rows are removed by taking the final matching suffix of exactly 600 rows;
- the selected assignment is the unique real call whose exact normalized class/box inputs equal the final matching layer;
- `gt_inds==0` is the unmatched authority and positive values preserve exact GT identity;
- encoder and earlier-decoder assignments cannot be mistaken for the final layer;
- hidden/logit/box/text masks share batch/query/token dimensions and all active values are finite;
- hooks/wrappers restore original methods after both success and exceptions;
- capture does not change parent losses or any prediction tensor.

- [ ] **Step 2: Run tests and observe the missing module**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_capture.py -q
```

Expected: fail because `n0_ng_capture.py` is absent.

- [ ] **Step 3: Implement context-managed observation without production edits**

The observer temporarily wraps the head's exact `_get_targets_single` and assigner's exact `assign` call, and registers a bbox-head forward hook. It records the assigner result returned inside each real target call; it never recomputes Hungarian matching. Record `final_layer_index = hidden_states.shape[0] - 1`; the final decoder classifier is `cls_branches[final_layer_index]`, not `cls_branches[-1]` (the latter may be the extra encoder branch).

```python
@dataclass
class CapturedN0State:
    final_layer_index: int
    final_hidden: Tensor       # detached after exact call alignment
    final_logits: Tensor
    final_boxes_normalized: Tensor
    memory_text: Tensor
    text_token_mask: Tensor
    final_gt_inds: Tensor
    cls_avg_factor: float
    img_meta: Mapping[str, Any]

@contextmanager
def capture_n0_loss(model: nn.Module) -> Iterator[CaptureSession]:
    """Observe one unmodified model.loss call and restore every patch."""
```

Run the parent loss forward under `torch.no_grad()` with frozen parameters. After it returns, uniquely align the exact `_get_targets_single` inputs to the captured final matching layer. If there are zero or multiple matches, fail. Compute `cls_avg_factor=max(num_pos + num_neg*bg_cls_weight, 1)` from that exact assignment and the actual head setting. Detach/clone only after alignment; the only later autograd leaf is the explicit detached final hidden state used in Task 6.

- [ ] **Step 4: Add parameter-gradient and mutation assertions**

Before and after capture, assert all model parameter `.grad` fields are `None`, no optimizer/scheduler object exists in the session, state-dict tensor hashes are unchanged, query count is 600, and no proposal/NMS/top-k path was called.

- [ ] **Step 5: Pass tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_capture.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/n0_ng_capture.py tests/test_projects/ov_capflow/test_n0_ng_capture.py
rtk git commit -m "feat: observe exact N0 assignment gradients"
```

Expected: tests prove exact call identity and zero behavioral mutation.

## Task 6: Recompute exact negative and audit-positive focal gradients

**Files:**
- Create: `projects/OVCapFlow/tools/n0_ng_gradients.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_gradients.py`

- [ ] **Step 1: Write failing focal-gradient equivalence tests**

For CPU and available CUDA, compare value and final-hidden gradient against the actual head's `FocalLoss` using its runtime `gamma`, `alpha`, `loss_weight`, valid-token mask, `bg_cls_weight`, and `cls_avg_factor`. Cover float32, padding, hyphenated class names, zero/multiple positive-map tokens, low norms, nonfinite values, and unexpected focal configurations.

Prove that differentiating the sum of independent per-row losses returns each row's own gradient exactly:

```python
all_at_once = torch.autograd.grad(
    row_losses.sum(), hidden, retain_graph=True)[0]
for row in checked_rows:
    isolated = torch.autograd.grad(
        row_losses[row], hidden, retain_graph=True)[0][row]
    torch.testing.assert_close(all_at_once[row], isolated)
```

- [ ] **Step 2: Run tests and observe the missing implementation**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_gradients.py -q
```

Expected: fail because `n0_ng_gradients.py` is absent.

- [ ] **Step 3: Implement the negative branch**

Create a detached leaf `hidden = final_hidden.detach().requires_grad_(True)`. Reuse the frozen final `ContrastiveEmbed` branch with detached seen text and the actual token mask. Select valid tokens before calling `py_sigmoid_focal_loss` with `reduction='none'`; multiply by actual `loss_weight` and divide by the ordinary `cls_avg_factor`. For every authoritative unmatched row, use an all-zero target and compute all row gradients with one sum-gradient call. Matched rows remain recorded but cannot enter eligible/control pools.

- [ ] **Step 4: Implement the audit-positive branch after the model freeze point**

Only after final model assignments, hidden states, boxes, and negative losses are detached may code open the audit object. Encode one withheld class at a time through the same frozen tokenizer/language model/text projection conventions under `torch.no_grad()`. Build the exact positive map returned by `get_positive_map`; positive class-token pieces receive the ordinary positive map weights and other valid prompt tokens receive zero. Differentiate only with respect to the selected detached hidden rows.

```python
final_branch = head.cls_branches[frozen_state.final_layer_index]
g_neg = negative_query_gradients(frozen_state, final_branch)
g_pos = audit_positive_gradients(
    model=model, hidden_rows=paired_hidden,
    class_names=paired_class_names, avg_factor=frozen_state.cls_avg_factor)
conflict = -torch.nn.functional.cosine_similarity(
    g_neg_rows, g_pos, dim=-1, eps=1e-12)
```

Destroy audit prompt strings, token tensors, text embeddings, positive maps, and raw class names before returning sufficient statistics. A gradient norm below `1e-12` or any nonfinite active value invalidates the tile. Assert again that every parameter `.grad is None`.

- [ ] **Step 5: Pass tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_gradients.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/n0_ng_gradients.py tests/test_projects/ov_capflow/test_n0_ng_gradients.py
rtk git commit -m "feat: measure exact N0 focal conflicts"
```

Expected: focal values/gradients match the configured parent path; padding and parameter gradients never enter the statistic.

## Task 7: Implement rotated eligibility, deduplication, and fixed-budget controls

**Files:**
- Create: `projects/OVCapFlow/tools/n0_ng_controls.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_controls.py`

- [ ] **Step 1: Write failing geometry and identity tests**

Cover clockwise MMRotate angles, boundary points, toroidal wrap, qbox-to-rbox conversion, ties, multiple GTs claiming one query, unsupported GTs, insufficient controls, identical centers, stable GT IDs, and deterministic results under input permutation.

The exact eligibility rule is:

```text
authoritative unmatched row AND
(rotated IoU(row, withheld GT) >= 0.10
 OR predicted center lies inside the same-angle GT expanded 1.5x in width/height)
```

Tie order is maximum rotated IoU, minimum center distance divided by image diagonal, smallest zero-based query index; cross-GT dedup then keeps maximum IoU, minimum normalized distance, smallest stable GT ID. Never backfill an unsupported GT.

- [ ] **Step 2: Write failing control tests**

For `K` eligible rows require exactly `K` distinct unmatched rows in every control:

- random-unmatched: 1,024 without-replacement draws from other unmatched rows, seed derived from M1 manifest hash, fold, image ID, control name, and draw index;
- spatial-shift: add half image width/height with toroidal wrap and rerun the exact selector;
- scene-shuffle: use the next hash-ranked valid dev donor in the same fold, never the same parent scene, map normalized geometry, and rerun the selector;
- seen-object: use the same selector around model-view seen GTs; descriptive in pilot and a full-N0 specificity control outside the six-test family.

Tests must reject budget shortfalls instead of sampling with replacement or relaxing thresholds.

Sort every random draw by its deterministic sample rank, then pair its rows in eligible-rank order with the corresponding withheld GT and audit token. This preserves exactly `K` paired comparisons without allowing random geometry or class identity to change the target side.

- [ ] **Step 3: Run failing tests**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_controls.py -q
```

Expected: fail because the control module is absent.

- [ ] **Step 4: Implement pure deterministic control functions**

Use MMRotate/MMCV rotated primitives with explicit angle-direction tests. Keep model boxes as pre-NMS/pre-top-k absolute `(cx,cy,w,h,angle)` rows. Functions take immutable arrays/records and return query indices plus hashed GT identities; they never receive a model or mutate tensors.

```python
@dataclass(frozen=True)
class EligiblePair:
    query_index: int
    stable_gt_id: str
    reason: str
    iou: float
    normalized_center_distance: float

def select_eligible(unmatched_boxes, audit_objects, image_shape) \
        -> Sequence[EligiblePair]:
    candidates = enumerate_frozen_candidates(
        unmatched_boxes, audit_objects, image_shape,
        iou_threshold=0.10, expansion=1.5)
    return deduplicate_candidates(candidates)
```

- [ ] **Step 5: Pass tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_controls.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/n0_ng_controls.py tests/test_projects/ov_capflow/test_n0_ng_controls.py
rtk git commit -m "feat: add N0 fixed-budget gradient controls"
```

Expected: deterministic geometry, tie, dedup, and all control budgets pass.

## Task 8: Implement image-macro statistics and conjunctive decisions

**Files:**
- Create: `projects/OVCapFlow/tools/n0_ng_metrics.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_metrics.py`

- [ ] **Step 1: Write failing estimand tests**

Freeze these image-level sufficient statistics:

```text
log concentration = log(mean ||g_neg|| eligible / mean ||g_neg|| control)
conflict = median eligible -cos(g_neg, g_pos)
negative-mass share = sum ||g_neg|| eligible / sum ||g_neg|| all unmatched
```

Fold and pooled authorities are unweighted means of tile statistics; exponentiate only to report concentration ratios. Row-micro and strata by opaque withheld-class ID, square-root pixel area (`<=8`, `(8,16]`, `>16`), fold, and scene-density quartile are descriptive only.

- [ ] **Step 2: Write failing inference/gate tests**

Use hand-computable fixtures to prove:

- all tiles from one parent scene move together in each of 10,000 deterministic bootstrap resamples;
- paired one-sided sign/permutation tests preserve tile pairing;
- seeds use SHA-256-derived unsigned integers and repeated execution is byte-identical;
- pilot has no multiplicity correction because all conditions are conjunctive;
- full N0 applies Holm jointly to exactly six comparisons: two estimands against random, spatial-shift, and scene-shuffle;
- ordered Holm adjusted p-values and corresponding step-down one-sided lower bounds use `alpha/(6-rank)` for zero-based ascending raw-p rank;
- seen-object remains outside those six tests but the withheld pooled effect must exceed it;
- insufficient folds/tiles/GTs, NaN, empty bootstrap replicates, or missing controls yield `INVALID_NO_DECISION`, never scientific fail.

Pilot passes only when all eight preregistered effect/interval/placebo conditions pass, with exactly 64 tiles and at least 128 eligible withheld GTs. Full endpoint passes only when all preregistered pooled, fold-direction, fold-CI, placebo, and specificity conditions pass; each fold targets 128 valid tiles and fails closed below 64 valid tiles or 128 eligible GTs.

Spell the full endpoint conjunction directly in tests and production decision code:

```text
pooled eligible/random concentration ratio >= 1.25
Holm-corrected one-sided lower bound for pooled ratio > 1
pooled eligible conflict >= 0.10
Holm-corrected one-sided lower bound for pooled conflict > 0
all four fold point estimates positive for both primary estimands
at least three of four fold lower bounds above null for both estimands
eligible beats random, spatial-shift, and scene-shuffle for both estimands after Holm
withheld-eligible pooled effect exceeds the seen-object control for both primary estimands
```

- [ ] **Step 3: Run the failing suite**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_metrics.py -q
```

Expected: fail because metrics are absent.

- [ ] **Step 4: Implement pure metric and decision layers**

Keep statistical code independent from torch/model code. Encode floats as binary64 hexadecimal strings in canonical decisions, while human reports may additionally display decimal values. Store the full deterministic random-null and bootstrap vectors in separately hashed NPZ artifacts, not inline JSON.

```python
def decide_combined(zero_step: EndpointDecision,
                    e1: EndpointDecision) -> CombinedDecision:
    if 'INVALID_NO_DECISION' in (zero_step.status, e1.status):
        return CombinedDecision('INVALID_NO_DECISION', n1_eligible=False)
    if 'FAIL_CLOSE_N1' in (zero_step.status, e1.status):
        return CombinedDecision('FAIL_CLOSE_N1', n1_eligible=False)
    return CombinedDecision('PASS_BOTH_ENDPOINTS', n1_eligible=True)
```

Never let descriptive strata or negative-mass share enter the primary decision.

- [ ] **Step 5: Pass tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_metrics.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/n0_ng_metrics.py tests/test_projects/ov_capflow/test_n0_ng_metrics.py
rtk git commit -m "feat: gate N0 gradient evidence"
```

Expected: all synthetic pass/fail/invalid and deterministic-byte tests pass.

## Task 9: Build the fail-closed diagnostic runner and immutable artifact chain

**Files:**
- Create: `projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py`
- Create: `tests/test_projects/ov_capflow/test_n0_ng_runner.py`

- [ ] **Step 1: Write failing runner-state tests**

Mock the model/data boundary and cover:

- CLI accepts only `pilot/full`, `zero_step/e1`, dev partition, Q600, and `--no-optimizer`;
- Fold 1 only for pilot; folds 1–4 exactly once for a full endpoint;
- pilot scans dev tiles in SHA-256 rank order and stops at the first 64 eligible/control-valid tiles;
- full scans to the first 128 eligible/control-valid tiles per fold when available;
- model forward/assignment/state freeze precedes the first audit-manifest read for every recipient or scene-shuffle donor tile;
- official novel4, M0 test, B0/RISC/POQ ancestry, prefiltered rows, top-k/NMS/proposals, optimizer construction/step, scheduler step, and diagnostic checkpoint writes are runtime stops;
- resume only accepts committed shard hashes from the same exact attempt authority; fresh output and no-overwrite publication are mandatory;
- scientific pass/fail writes a valid decision and then `COMMITTED.json`; infrastructure failure writes a bounded `error.json` with no committed marker;
- exit `0` = scientific pass, `3` = valid scientific fail, `2` = invalid/infrastructure.

The sample rank digest is the UTF-8/NUL concatenation of the M1 manifest SHA256, literal `pilot` or `full`, decimal fold ID, and image ID. Tests pin the byte payload and resulting order, not only the selected count.

- [ ] **Step 2: Run failing runner tests**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_runner.py -q
```

Expected: fail because the runner is absent.

- [ ] **Step 3: Implement the per-tile state machine**

```text
validate all authorities
  -> open model-view image/seen annotation only
  -> run exact model.loss under capture, with no backward/optimizer
  -> freeze assignment/Q600/negative gradients/model hashes
  -> open isolated audit record
  -> select eligible and controls
  -> encode detached audit prompt and compute g_pos
  -> erase raw audit tensors/strings
  -> write hashed sufficient statistics
```

Batch size is exactly one for diagnostics. Before building a generic zero-step model, apply seed `20260808`, build the model, load the partial generic seal, and freeze every parameter with `requires_grad_(False)`; a test proves independently built fold models have identical shared initial tensors. Keep the model in training mode only to exercise the real training loss/assignment path, and wrap that forward in `torch.no_grad()`. Before opening a scene-shuffle donor audit record, call the same model-freeze routine for that donor and cache its detached state; if it later becomes a recipient, reuse the same state rather than rerunning stochastic layers. Derive and record a per-tile torch/CUDA RNG seed from M1 hash, endpoint, fold, and tile identity so model-view training-mode dropout is order-independent and reproducible. The runner may checkpoint its own completed sufficient-statistic shards, but never a model checkpoint. Each output row may contain query index, stable hashed GT ID, eligibility/control role, gradient norm/conflict, scene hash, fold, and opaque stratum IDs; it may not contain image pixels, qbox coordinates, raw annotations, or raw audit class/prompt strings.

- [ ] **Step 4: Bind every artifact and publish last**

`attempt_manifest.json` binds repository/config/code/prereg hashes, M0/M1/strict-view authority, model checkpoint authority, fold/endpoint/mode, image and scene identities, prompt hashes, deterministic settings, library/CUDA versions, seeds, shard names/bytes/hashes, and counts. `decision.json` contains every point estimate, bound, raw/adjusted p-value, Boolean gate, failed gate, and the final status. `COMMITTED.json` is always last for valid scientific completion.

- [ ] **Step 5: Pass runner tests and commit**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_runner.py -q
rtk git diff --check
rtk git add projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py tests/test_projects/ov_capflow/test_n0_ng_runner.py
rtk git commit -m "feat: run fail-closed N0 gradient diagnostics"
```

Expected: ordering, no-optimizer, publication, resume, and exit-code tests pass.

## Task 10: Run integration, forbidden-dependency, and regression verification

**Files:**
- Modify only if a demonstrated bug requires it: files created in Tasks 1–9
- Create: `tests/test_projects/ov_capflow/test_n0_ng_integration.py`

- [ ] **Step 1: Add a complete tiny end-to-end fixture**

Use a synthetic two-scene/two-fold fixture with a tiny fake detector that still exercises the real capture, gradient, geometry, statistics, and publication layers. Assert byte-identical repeated results in separate roots, valid scientific fail, invalid contract behavior, no parameter gradients, and no audit-before-freeze event.

- [ ] **Step 2: Add AST-level scientific-boundary tests**

Parse imports in every `n0_ng` production file and reject modules or symbols containing `risc`, `orbit`, `poq`, `omq`, `odq`, project-trained checkpoint loaders, proposal generators, NMS, or top-k row selection. Allow the literal `B0` only inside explicit rejection/error-contract data, never as an imported module or accepted path.

- [ ] **Step 3: Run all focused tests**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ng_protocol.py tests/test_projects/ov_capflow/test_n0_strict_views.py tests/test_projects/ov_capflow/test_strict_ov_dataset.py tests/test_projects/ov_capflow/test_n0_ng_configs.py tests/test_projects/ov_capflow/test_n0_ng_checkpoint_seals.py tests/test_projects/ov_capflow/test_n0_ng_capture.py tests/test_projects/ov_capflow/test_n0_ng_gradients.py tests/test_projects/ov_capflow/test_n0_ng_controls.py tests/test_projects/ov_capflow/test_n0_ng_metrics.py tests/test_projects/ov_capflow/test_n0_ng_runner.py tests/test_projects/ov_capflow/test_n0_ng_integration.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Run relevant existing regressions**

```bash
rtk conda run -n mmdet python -m pytest tests/test_projects/ov_capflow/test_n0_ri_protocol.py tests/test_projects/ov_capflow/test_n0_ri_capture.py tests/test_projects/ov_capflow/test_n0_ri_metrics.py tests/test_projects/ov_capflow/test_n0_ri_runner.py tests/test_projects/ov_capflow/test_dn_budget_batch_sampler.py tests/test_projects/ov_capflow/test_cleanstart_checkpoint.py -q
```

Expected: existing integrity, capture, sampler, and checkpoint behavior remains green.

- [ ] **Step 5: Run static and diff verification**

```bash
rtk rg -n 'TO[D]O|FIX[M]E|T[B]D|choose[ ]later|best[ ]checkpoint|transport-first-queries[=]True' projects/OVCapFlow/tools/n0_ng_protocol.py projects/OVCapFlow/tools/build_n0_strict_views.py projects/OVCapFlow/ov_capflow/strict_ov_dataset.py projects/OVCapFlow/tools/seal_n0_ng_initialization.py projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py projects/OVCapFlow/tools/n0_ng_capture.py projects/OVCapFlow/tools/n0_ng_gradients.py projects/OVCapFlow/tools/n0_ng_controls.py projects/OVCapFlow/tools/n0_ng_metrics.py projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_base.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py
rtk git diff --check
rtk git status --short
```

Expected: placeholder scan has no matches, diff check is clean, and status contains only intended N0 files.

- [ ] **Step 6: Commit the integration boundary**

```bash
rtk git add tests/test_projects/ov_capflow/test_n0_ng_integration.py
rtk git commit -m "test: enforce strict-OV N0 boundaries"
```

Expected: no experiment has run yet; code readiness is independently verifiable.

## Task 11: Build and audit the real strict views and generic initialization

**Files/artifacts:**
- Create by tool: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1/`
- Create by tool: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1/`

- [ ] **Step 1: Revalidate prerequisites immediately before the real build**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_protocol.py verify-prerequisites --repo-root /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng --m0-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438 --m1-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --generic-source /data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

Expected: `PASS_PREREQUISITES`. Any mismatch stops the build.

- [ ] **Step 2: Build all four model/audit views once**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/build_n0_strict_views.py --m0-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438 --m1-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --image-root /data1/zcy/datasets/DOTA2_1024_500/ss_train_full_1024_500_20260620/images --annotation-root /data1/zcy/datasets/DOTA2_1024_500/ss_train_full_1024_500_20260620/annfiles --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1
```

Expected: root and four fold `COMMITTED.json` markers, 35,109 train and 5,172 dev identities per fold, zero test access, zero cross-view path overlap, no persisted official-novel statistic. If the output root already exists, inspect it; never overwrite it.

- [ ] **Step 3: Seal a separate clean generic Swin-B initialization**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/seal_n0_ng_initialization.py --source /data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth --source-url https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth --expected-sha256 46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7 --expected-bytes 938057991 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1
```

Expected: weights-only `model.pth`, canonical provenance, SHA sidecars, and `COMMITTED.json`; query transport disabled; no project-trained ancestry.

- [ ] **Step 4: Verify real authorities without GPU work**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_protocol.py verify-views --strict-view-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_protocol.py verify-initialization --initialization-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1 --config-root configs/ov_capflow/dotav2
```

Expected: `PASS_STRICT_VIEWS` and `PASS_GENERIC_INITIALIZATION` with exact hashes. Record these hashes before the pilot; they cannot change afterward.

## Task 12: Run the Fold 1 generic zero-step pilot

**Artifacts:**
- Create by runner: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/`

- [ ] **Step 1: Confirm one GPU is genuinely idle and the authority still matches**

```bash
rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_protocol.py verify-prerequisites --repo-root /data1/zcy/OV-CapFlow/.worktrees/strict-ov-n0-ng --m0-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438 --m1-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --generic-source /data/zcy/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

GPU 0 is the frozen launch slot in the command below, but it is an operational resource, not part of the estimand. If GPU 0 is occupied, wait; do not silently change the scientific config, sample, seed, or endpoint.

- [ ] **Step 2: Launch only the no-optimizer pilot**

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 conda run -n mmdet python projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py --mode pilot --fold 1 --endpoint zero_step --partition dev --query-count 600 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py --scene-manifest /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438/manifest.json --content-seal /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --strict-view-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1 --initialization-manifest /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1/COMMITTED.json --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1/model.pth --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1 --no-optimizer
```

Expected valid completion: exactly 64 eligible/control-valid tiles, at least 128 eligible withheld GTs, `optimizer_steps=0`, and a committed `decision.json`. Exit `0` authorizes Task 13 only; exit `3` closes N1; exit `2` is invalid and authorizes only an infrastructure diagnosis.

- [ ] **Step 3: Read the complete decision instead of inferring from exit code**

```bash
rtk conda run -n mmdet python -m json.tool /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/decision.json
rtk sha256sum /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/decision.json /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/COMMITTED.json
```

Expected pass components: ratio `>=1.25`, conflict `>=0.10`, both bootstrap lower bounds above null, and both estimands beat both spatial-shift and scene-shuffle at one-sided `p<=0.05`.

## Task 13: Run the full four-fold generic zero-step confirmation

**Artifacts:**
- Create by runner: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/`

- [ ] **Step 1: Enforce the pilot promotion gate**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_metrics.py require-pilot-pass --decision /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/decision.json --commit-marker /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/pilot_fold1_zero_step_v1/COMMITTED.json
```

Expected: `PASS_TO_FULL_N0`; otherwise stop.

- [ ] **Step 2: Run all four folds from the same generic authority**

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 conda run -n mmdet python projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py --mode full --fold 1 2 3 4 --endpoint zero_step --partition dev --query-count 600 --config-root configs/ov_capflow/dotav2 --scene-manifest /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438/manifest.json --content-seal /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --strict-view-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1 --initialization-manifest /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1/COMMITTED.json --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1/model.pth --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1 --no-optimizer
```

Expected valid completion: every fold has at least 64 valid tiles and 128 eligible GTs, with a target of 128 tiles; full pooled/fold/Holm/specificity gates are explicit in `decision.json`. Exit `0` authorizes the already-frozen E1 schedule; exit `3` closes N1 and skips Task 14.

- [ ] **Step 3: Verify the zero-step full decision**

```bash
rtk conda run -n mmdet python -m json.tool /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/decision.json
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_metrics.py require-endpoint-pass --endpoint zero_step --decision /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/decision.json --commit-marker /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/COMMITTED.json
```

Expected for promotion: `PASS_ZERO_STEP`; no single fold or descriptive stratum can rescue a failed conjunction.

## Task 14: Train and seal the four frozen seen-only E1 controls

**Artifacts:**
- Training roots: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold{1,2,3,4}_train_v1/`
- Sealed endpoints: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold{1,2,3,4}_sealed_v1/`

- [ ] **Step 1: Verify zero-step pass, all ten GPUs idle, and frozen configs unchanged**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_metrics.py require-endpoint-pass --endpoint zero_step --decision /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/decision.json --commit-marker /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/COMMITTED.json
rtk nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
rtk nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
rtk sha256sum configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_base.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py
```

Expected: promotion pass, ten idle GPUs, and hashes equal those sealed before the pilot. If all ten GPUs are not simultaneously available, wait; do not change world size/effective batch.

- [ ] **Step 2: Train Fold 1 for exactly one no-repeat epoch**

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 PYTHONNOUSERSITE=1 bash tools/dist_train.sh configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py 10
```

Expected: one endpoint checkpoint and sampler audit covering every one of 35,109 train tiles exactly once.

- [ ] **Step 3: Train Folds 2–4 sequentially with the same frozen recipe**

```bash
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 PYTHONNOUSERSITE=1 bash tools/dist_train.sh configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py 10
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 PYTHONNOUSERSITE=1 bash tools/dist_train.sh configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py 10
rtk env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 PYTHONNOUSERSITE=1 bash tools/dist_train.sh configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py 10
```

Expected: four independent endpoints, each initialized from the same generic seal and trained only on its fold's seen model view. Do not select among intermediate checkpoints; only Epoch 1 exists.

- [ ] **Step 4: Seal each E1 endpoint and exact sampler coverage**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py --fold 1 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold1_e1.py --generic-initialization-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1 --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold1_train_v1/epoch_1.pth --sampler-audit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold1_train_v1/sampler_epoch_00.json --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold1_sealed_v1
rtk conda run -n mmdet python projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py --fold 2 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold2_e1.py --generic-initialization-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1 --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold2_train_v1/epoch_1.pth --sampler-audit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold2_train_v1/sampler_epoch_00.json --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold2_sealed_v1
rtk conda run -n mmdet python projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py --fold 3 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold3_e1.py --generic-initialization-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1 --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold3_train_v1/epoch_1.pth --sampler-audit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold3_train_v1/sampler_epoch_00.json --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold3_sealed_v1
rtk conda run -n mmdet python projects/OVCapFlow/tools/seal_n0_ng_e1_checkpoint.py --fold 4 --config configs/ov_capflow/dotav2/ov_capflow_swin-b_dotav2_n0_ng_fold4_e1.py --generic-initialization-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/generic_init_swinb_v1 --checkpoint /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold4_train_v1/epoch_1.pth --sampler-audit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold4_train_v1/sampler_epoch_00.json --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold4_sealed_v1
```

Expected: four weights-only committed seals; every coverage audit reports `duplicate_count=0` and `missing_count=0`.

## Task 15: Run full four-fold E1 confirmation and combine endpoints

**Artifacts:**
- Create: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_e1_v1/`
- Create: `/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/combined_n0_v1/`

- [ ] **Step 1: Run the identical full diagnostic on the fold-matched E1 seals**

```bash
rtk env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 conda run -n mmdet python projects/OVCapFlow/tools/run_n0_negative_gradient_diagnostic.py --mode full --fold 1 2 3 4 --endpoint e1 --partition dev --query-count 600 --config-root configs/ov_capflow/dotav2 --scene-manifest /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m0_20260803_0438/manifest.json --content-seal /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-scene-prep-v1/m1_n0_ri_v1 --strict-view-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/strict_views_v1 --e1-checkpoint-root 1=/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold1_sealed_v1 --e1-checkpoint-root 2=/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold2_sealed_v1 --e1-checkpoint-root 3=/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold3_sealed_v1 --e1-checkpoint-root 4=/data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/e1_fold4_sealed_v1 --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_e1_v1 --no-optimizer
```

Expected: a standalone E1 decision using the same thresholds, sample rules, controls, inference, and multiplicity family as zero-step.

- [ ] **Step 2: Combine only committed endpoint decisions**

```bash
rtk conda run -n mmdet python projects/OVCapFlow/tools/n0_ng_metrics.py combine-endpoints --zero-step-decision /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/decision.json --zero-step-commit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_zero_step_v1/COMMITTED.json --e1-decision /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_e1_v1/decision.json --e1-commit /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/full_e1_v1/COMMITTED.json --output-root /data1/zcy/OV-CapFlow/.lab/workspace/exp-8-n0-ng-v1/combined_n0_v1
```

Expected:

- both pass: `decision=PASS_BOTH_ENDPOINTS`, `n1_eligible=true`, `N1_ELIGIBLE.json` is included, and `COMMITTED.json` is published last;
- either valid endpoint fails: `decision=FAIL_CLOSE_N1`, `n1_eligible=false`, no eligibility marker;
- either endpoint invalid/uncommitted: combined result is invalid and no scientific close/pass is emitted.

- [ ] **Step 3: Inspect the complete evidence table**

| Endpoint | Eligible/random ratio | Conflict | Fold stability | Holm placebos | Seen specificity | Decision |
|---|---:|---:|---|---|---|---|
| Fold 1 zero-step pilot | 未运行；从 pilot `decision.json` 导入 | 未运行；从 pilot `decision.json` 导入 | 仅 Fold 1 | 未运行；pilot 无 Holm | 未运行；仅描述 | 未运行；从 artifact 导入 |
| Four-fold zero-step | 未运行；从 zero-step `decision.json` 导入 | 未运行；从 zero-step `decision.json` 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 |
| Four-fold E1 | 未运行；从 E1 `decision.json` 导入 | 未运行；从 E1 `decision.json` 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 | 未运行；从 artifact 导入 |
| Combined N0 | 不合并数值 | 不合并数值 | 两端各自满足 | 两端各自满足 | 两端各自满足 | 未运行；从 combined artifact 导入 |

Do not manually type measured values from console logs; import only committed canonical decision fields and preserve their hashes.

## Task 16: Write the factual result record and enforce the N1 boundary

**Files:**
- Create: `docs/project_history/exp_20260808_strict_ov_n0_ng/fres_strict_ov_n0_ng_zh.md`
- Modify: `docs/project_history/README.md`
- Modify: `docs/project_history/exp_20260729_20260807_ten_day_review/fres_20260729_20260807_full_review_reflection_zh.md`

Run this documentation task in the authoritative main worktree `/data1/zcy/OV-CapFlow`, where the P0 review and index live. Do not assume those later memory files exist in the pre-RISC implementation branch, and do not merge unrelated implementation history merely to update the record.

- [ ] **Step 1: Use the experiment-result organization skill and import committed decisions**

The record must label facts, inferences, decisions, open items, claim boundaries, authority paths/hashes, counts, all primary gates, and failures. Preserve invalid attempts separately. If execution stopped before a stage, say “未运行；上游门未授权”, never fill a numeric cell from training loss or intuition.

- [ ] **Step 2: Apply the branch-specific outcome**

If combined N0 passes:

- record only that the harmful-gradient mechanism is supported at both endpoints and N1 design is eligible;
- refresh the primary-source literature search before a formal N1 claim;
- run a new brainstorming turn comparing at least three minimal negative-only interventions;
- require user approval of a separate N1 spec/plan;
- keep teacher, pseudo boxes, positive pseudo targets, proposals, top-k, NMS, rotation, POQ, and calibration out of the first matched N1.

If combined N0 fails:

- record `FAIL_CLOSE_N1` and the exact failed gates;
- close this mechanism under the frozen protocol;
- do not threshold-tune, resample, select a favorable fold, extend E1, or train N1 as rescue.

- [ ] **Step 3: Verify and commit documentation**

```bash
rtk rg -n '\[FACT\]|\[INFERENCE\]|\[DECISION\]|\[OPEN\]|\[CLAIM-BOUNDARY\]' docs/project_history/exp_20260808_strict_ov_n0_ng/fres_strict_ov_n0_ng_zh.md
rtk rg -n '未运行|PASS_BOTH_ENDPOINTS|FAIL_CLOSE_N1|INVALID_NO_DECISION|N1_ELIGIBLE' docs/project_history/exp_20260808_strict_ov_n0_ng/fres_strict_ov_n0_ng_zh.md
rtk git diff --check
rtk git add docs/project_history/exp_20260808_strict_ov_n0_ng/fres_strict_ov_n0_ng_zh.md docs/project_history/README.md docs/project_history/exp_20260729_20260807_ten_day_review/fres_20260729_20260807_full_review_reflection_zh.md
rtk git commit -m "docs: record strict-OV N0 gradient decision"
```

Expected: one canonical experiment subdirectory, an updated P0 index, and a dated append-only revision pointer in the ten-day authority.

## 4. Stop conditions and recovery policy

Stop immediately and publish no committed scientific decision on:

- any official novel4 derivative leaving the sanitizer;
- audit access before the model branch freeze point;
- M0/M1/spec/literature/config/code/checkpoint hash mismatch;
- source ancestry from B0, RISC, POQ, OMQ, ODQ, or any project-trained model outside the four sealed E1 endpoints;
- query count other than exactly 600 or loss assignment that cannot be uniquely tied to final matching rows;
- proposal, NMS, top-k, confidence filtering, or any premeasurement row removal;
- optimizer/scheduler/checkpoint construction inside diagnostic execution;
- model parameter gradients, model-state mutation, nonfinite active value, or gradient norm below `1e-12`;
- duplicate scene/tile/query/GT identity, cross-partition collision, insufficient control budget, or sample-size failure;
- output collision or attempted overwrite.

Allowed recovery is narrow: diagnose the infrastructure cause, add a failing regression test, fix only that cause, rerun the affected CPU verification, commit, and use a new versioned attempt root. Scientific failure is not an infrastructure defect and receives no rescue run.

## 5. Final implementation verification checklist

- [ ] Branch begins at the exact pre-RISC base and contains no unrelated dirty state.
- [ ] Every production behavior was introduced by a failing test.
- [ ] M0/M1 and source hashes match the frozen table.
- [ ] Four strict views are physically separate, scene-disjoint, and test-closed.
- [ ] Official novel4 leaves no persisted derivative.
- [ ] Q600 assignment is observed from the exact parent call, not recomputed.
- [ ] Focal gradients numerically match the configured parent loss.
- [ ] Diagnostic parameter gradients and optimizer steps are zero.
- [ ] Fixed budgets, 1,024 random draws, 10,000 clustered resamples, and Holm family are exact.
- [ ] Pilot/full/E1 ordering is enforced by committed artifacts.
- [ ] E1 sees every 35,109 train tile once, keeps empty-after-filter tiles, and uses no rare4x/rotation.
- [ ] Results come only from canonical `decision.json` files.
- [ ] N1 remains code-free unless both full endpoints pass and the user approves a separate design.

## 6. Plan self-review commands

Run from `/data1/zcy/OV-CapFlow` before committing this plan:

```bash
rtk rg -n 'TO[D]O|FIX[M]E|T[B]D|待[定]|choose[ ]later|implement[ ]later|similar[ ]to' docs/superpowers/plans/2026-08-08-strict-ov-n0-negative-gradient-gate.md
rtk rg -n '^## Task [0-9]+:' docs/superpowers/plans/2026-08-08-strict-ov-n0-negative-gradient-gate.md
rtk rg -n '64|128|1\.25|0\.10|1,024|10,000|Holm|Q600|zero-step|E1|N1_ELIGIBLE' docs/superpowers/plans/2026-08-08-strict-ov-n0-negative-gradient-gate.md
rtk git diff --check
```

Expected: the first scan has no matches; Tasks 0–16 are present; all frozen numbers and gates are discoverable; diff check is clean.

## 7. Execution handoff

This plan is intentionally one N0 implementation/execution plan. N1 is a separate research decision because its admissible mechanism depends on the observed N0 effect and because implementing N1 before the gate would turn a falsifiable diagnostic into post-hoc method justification.

At handoff, choose one execution mode:

1. **Inline execution:** continue in the main agent, applying the worktree/TDD/verification skills task by task.
2. **Subagent-driven execution:** only after the user explicitly authorizes delegation, dispatch independent bounded tasks with review checkpoints; never parallelize the sequential scientific gates or four E1 jobs.
