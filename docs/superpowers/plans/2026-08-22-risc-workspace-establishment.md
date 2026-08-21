# RISC Workspace Establishment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an independent, source-only Git repository for RISC that preserves OpenRSD and OV-CapFlow code artifacts with auditable provenance.

**Architecture:** The repository has two nested source snapshots, `framework/openrsd` and `reference/ov-capflow`, plus root-level research authority and provenance documents. A non-destructive `rsync -a` copy filters out version-control metadata, datasets, models, run outputs, cache, media, and archives; a manifest seals the two source commits and copied-tree facts.

**Tech Stack:** Git, Bash, rsync, sha256sum, Markdown; existing OpenRSD and OV-CapFlow Python/MMRotate source code.

---

### Task 1: Establish the repository authority and migration boundary

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `RISC_GOAL.md`
- Create: `docs/provenance/MIGRATION_POLICY.md`
- Create: `docs/superpowers/specs/2026-08-22-risc-repository-migration-design.md`
- Create: `task_plan.md`, `findings.md`, `progress.md`

- [x] **Step 1: Initialize the independent root**

Run:

```bash
rtk git init -b main
```

Expected: Git reports an empty repository rooted at `/data1/zcy/RISC/.git`.

- [x] **Step 2: Write the authoritative goal and source-only policy**

Create the listed Markdown files using the confirmed snapshot design. The goal must separate B* from RISC-ER, preserve the E12 diagnostic gate, define the `+0.3` AP promotion threshold, and forbid unproved OV or causal claims.

- [x] **Step 3: Verify the policy is internally complete**

Run:

```bash
rtk run 'pattern="TO""DO|T""BD|fill ""in|implement ""later"; if rg -n "$pattern" RISC_GOAL.md docs/superpowers docs/provenance task_plan.md findings.md progress.md; then printf "PLACEHOLDER_FOUND\\n"; exit 1; else rc=$?; test "$rc" -eq 1 && printf "NO_PLACEHOLDERS\\n"; fi'
```

Expected: exit code 1 because no placeholder text exists.

- [x] **Step 4: Commit the authority documents**

Run:

```bash
rtk git add .gitignore README.md RISC_GOAL.md docs task_plan.md findings.md progress.md
rtk git commit -m "docs: establish RISC research authority"
```

Observed: root commit `6bc4dc6` contains only text policies and no source snapshot.

### Task 2: Snapshot OpenRSD as the framework base

**Files:**
- Create: `framework/openrsd/**`
- Create: `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md`

- [x] **Step 1: Seal source identity before copying**

Run:

```bash
rtk run 'git -C /data1/zcy/OpenRSD rev-parse HEAD; git -C /data1/zcy/OpenRSD config --get remote.origin.url; git -C /data1/zcy/OpenRSD status --porcelain=v1 | awk "BEGIN {m=0; u=0} /^\?\?/ {u++; next} {m++} END {printf \"modified_or_deleted=%d\\nuntracked=%d\\n\", m, u}"'
```

Expected: base SHA `12d3fd8b75e8b64ec53fded9cf035a2306d58874`, the recorded remote, and a nonzero dirty summary that is captured as provenance rather than discarded.

- [x] **Step 2: Copy source files without nested Git metadata or artifacts**

Run the exact `rsync -a` command specified in `MIGRATION_POLICY.md`, with source `/data1/zcy/OpenRSD/` and destination `framework/openrsd/`. Include `--exclude` arguments for each policy pattern, including `experiments/`, and omit `--delete`. Then run a second `rsync -a` for only `experiments/rotation_semantic_attractor/{configs,src,scripts,tests}` into the corresponding target directory.

Expected: framework files and the four experiment-code directories appear under `framework/openrsd/`; neither source timestamps nor source Git state are changed.

- [x] **Step 3: Verify required framework markers**

Run:

```bash
rtk run 'test -f framework/openrsd/README.md && test -f framework/openrsd/setup.py && test -f framework/openrsd/M_AD/models/dense_heads/risc_orbit_projection_head.py && test -f framework/openrsd/M_AD/models/utils/risc_orbit_projection.py'
```

Expected: exit code 0.

### Task 3: Snapshot OV-CapFlow as the RISC-ER reference implementation

**Files:**
- Create: `reference/ov-capflow/**`
- Modify: `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md`

- [x] **Step 1: Seal source identity before copying**

Run:

```bash
rtk run 'git -C /data1/zcy/OV-CapFlow rev-parse HEAD; git -C /data1/zcy/OV-CapFlow config --get remote.origin.url; git -C /data1/zcy/OV-CapFlow status --porcelain=v1 | awk "BEGIN {m=0; u=0} /^\?\?/ {u++; next} {m++} END {printf \"modified_or_deleted=%d\\nuntracked=%d\\n\", m, u}"'
```

Expected: base SHA `e87ae43ad294d9918cd6a36a458bf6f03dabb9fe`, the recorded remote, and its dirty summary.

- [x] **Step 2: Copy with the same source-only filter**

Run the same `rsync -a` filter set with source `/data1/zcy/OV-CapFlow/` and destination `reference/ov-capflow/`, again without `--delete`.

Expected: RISC-ER formal design, implementation plan, project source, configurations and tests are available under the reference subtree.

- [x] **Step 3: Verify the RISC-ER authority paths**

Run:

```bash
rtk run 'test -f reference/ov-capflow/docs/superpowers/specs/2026-08-11-risc-er-formal-design.md && test -f reference/ov-capflow/docs/superpowers/plans/2026-08-11-risc-er-implementation.md && test -f reference/ov-capflow/projects/OVCapFlow/ov_capflow/ov_capflow.py'
```

Expected: exit code 0.

### Task 4: Produce manifest and verify the migration boundary

**Files:**
- Create: `docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md`
- Modify: `task_plan.md`, `findings.md`, `progress.md`

- [x] **Step 1: Write the manifest from fresh command output**

Record both absolute source paths, base SHA, remote URL, dirty counts, migration timestamp, selected code files, target file counts, policy link and SHA256 of the four key RISC source/design files.

- [x] **Step 2: Scan for forbidden tracked material**

Run:

```bash
rtk run 'base="(^|/)(data|datasets|data_local|work_dirs|weights|pretrained|results|visual|vis_infer_epoch12|resultmd|pdf|\\.git|\\.agents|\\.codex|\\.cursor|\\.lab|\\.vscode|\\.dist_test|\\.pytest_cache|\\.mypy_cache|\\.ruff_cache)(/|$)|(^|/)CODEX_WORKLOG\\.md$|\\.(pth|pt|ckpt|pkl|pickle|npy|npz|onnx|h5|zip|tar|pdf|png|jpe?g|gif|mp4|log)$"; allow_gz="^framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/src/open_clip/bpe_simple_vocab_16e6\\.txt\\.gz$|^framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/src/open_clip/eva_clip/bpe_simple_vocab_16e6\\.txt\\.gz$|^framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/third_party/bpe_simple_vocab_16e6\\.txt\\.gz$"; git ls-files | rg "$base" >/dev/null; base_rc=$?; git ls-files | rg "\\.gz$" | rg -v "$allow_gz" >/dev/null; gz_rc=$?; test "$base_rc" -eq 1 && test "$gz_rc" -eq 1'
```

Expected: exit code 0 because no tracked path matches a forbidden policy pattern and only the three allowlisted tokenizer vocabularies end in `.gz`.

- [x] **Step 3: Run structural and whitespace verification**

Run:

```bash
rtk run 'test -f framework/openrsd/README.md && test -f reference/ov-capflow/README.md && test -f docs/provenance/SOURCE_SNAPSHOT_MANIFEST.md && git diff --cached --check -- .gitignore README.md RISC_GOAL.md docs/provenance docs/superpowers task_plan.md findings.md progress.md'
```

Expected: exit code 0 for new RISC authority files. If a complete snapshot check reports inherited whitespace, record its count and source comparison in `SOURCE_SNAPSHOT_MANIFEST.md`; do not mass-format source snapshots.

- [x] **Step 4: Commit the immutable source snapshot**

Run:

```bash
rtk git add framework reference docs/provenance task_plan.md findings.md progress.md
rtk git commit -m "chore: snapshot RISC source foundations"
```

Observed: `f8313c3` contains only source/text files permitted by the policy, with the three explicit tokenizer vocabulary exceptions.

### Task 5: Verify the handoff state

**Files:**
- Modify: `task_plan.md`, `findings.md`, `progress.md`

- [x] **Step 1: Inspect the committed tree**

Run:

```bash
rtk git status --short --branch
rtk git log --oneline --decorate -3
rtk run 'git ls-files | wc -l'
```

Observed: clean `main`, three documented commits (`6bc4dc6`, `f8313c3`, `921d972`), and 4,997 tracked paths.

- [x] **Step 2: Re-read the completion definition**

All items in `RISC_GOAL.md` §10 were checked against fresh output: independent `main`, both source subtrees, provenance manifest, strict path policy with three narrow BPE exceptions, scoped authority whitespace, committed records, and source-only mutation discipline.
