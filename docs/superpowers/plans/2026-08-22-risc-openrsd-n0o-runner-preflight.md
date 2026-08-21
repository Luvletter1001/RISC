# RISC OpenRSD N0-O Runner and CPU Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a v3-only N0-O protocol consumer, model ledger, CPU model-build/load preflight and GPU fold guard without executing a model forward or touching GPU.

**Architecture:** A protocol module owns manifest/ledger/support/view contracts; a preflight module builds and load-audits the hybrid OpenRSD runtime entirely on CPU; a fold-runner entry point guards all lazy runtime imports behind a future authorization receipt.

**Tech Stack:** Python 3.10, PyTorch 1.12 CPU, MMEngine config/registry, canonical JSON/JSONL, pytest.

---

### Task 1: Freeze runner/preflight authority

**Files:**
- Create: `docs/superpowers/specs/2026-08-22-risc-openrsd-n0o-runner-preflight-design.md`
- Create: `docs/superpowers/plans/2026-08-22-risc-openrsd-n0o-runner-preflight.md`
- Modify: `task_plan.md`, `findings.md`, `progress.md`

- [x] Record v3 identity, hybrid roots, model-ledger firewall, CPU build/load
  audit, GPU authorization schema and stop boundary.
- [x] Scan markers/placeholders and run scoped `git diff --check`.
- [x] Commit with `docs: freeze OpenRSD N0-O runner preflight`.

### Task 2: Implement protocol consumer and model ledger with TDD

**Files:**
- Create: `framework/openrsd/tools/risc_n0o/openrsd_n0o_protocol.py`
- Create: `framework/openrsd/tests/test_openrsd_n0o_protocol.py`

- [x] Write RED tests for exact v3 hash/status, invalid marker rejection,
  scene/ledger join, PyTorch support reconstruction and aggregate hash.
- [x] Add RED tests requiring exact C4/C8 view order and a model ledger with no
  key containing `annotation`, `gt`, `qbox`, `class`, `metric` or `prediction`.
- [x] Implement the minimal consumer, support cache, view specs and canonical
  model-ledger builder.
- [x] Verify GREEN and commit `feat: add OpenRSD N0-O protocol consumer`.

### Task 3: Implement CPU hybrid-runtime preflight with TDD

**Files:**
- Create: `framework/openrsd/tools/risc_n0o/preflight_openrsd_n0o.py`
- Create: `framework/openrsd/tests/test_preflight_openrsd_n0o.py`

- [ ] Write RED tests for allowed hybrid origins, RISC-first namespace order,
  deterministic resolved config and exact missing-key allowlist.
- [ ] Add a fake-model/load audit test proving no forward/predict/test-step or
  CUDA API is called.
- [ ] Implement origin sealing, resolved config materialization, CPU model
  construction/load audit and atomic preflight publication.
- [ ] Run a synthetic preflight twice and prove byte identity.
- [ ] Verify GREEN and commit `feat: add OpenRSD N0-O CPU preflight`.

### Task 4: Implement GPU fold authorization guard with TDD

**Files:**
- Create: `framework/openrsd/tools/risc_n0o/run_openrsd_n0o_fold.py`
- Create: `framework/openrsd/tests/test_run_openrsd_n0o_fold_guard.py`

- [ ] Write RED tests showing absent, malformed, wrong-manifest or excessive
  authorization refuses before a monkeypatched lazy runtime loader is called.
- [ ] Implement canonical receipt validation, fold/scene bounds and lazy import
  boundary. Keep the actual runtime callable unreachable without receipt.
- [ ] Verify the current CLI exits fail-closed with no CUDA initialization.
- [ ] Verify GREEN and commit `feat: guard OpenRSD N0-O fold runner`.

### Task 5: Execute and seal the real CPU preflight

**Files:**
- Create: `docs/provenance/risc_openrsd_n0o_preflight_v1/**`

- [ ] Run two temporary real CPU preflights with `CUDA_VISIBLE_DEVICES=''` and
  compare every artifact byte-for-byte.
- [ ] Publish the tracked preflight directory once; verify repeat invocation is
  rejected.
- [ ] Independently audit config, module origins, 160-row model ledger,
  checkpoint load allowlist and absence of annotation/GT fields.
- [ ] Commit `chore: seal OpenRSD N0-O CPU preflight`.

### Task 6: Final verification and stop-boundary handoff

**Files:**
- Modify: `task_plan.md`, `findings.md`, `progress.md`
- External append-only: `/data1/zcy/OpenRSD/CODEX_WORKLOG.md`

- [ ] Run protocol/preflight/guard plus S0 adjacent tests under the OpenRSD
  conda interpreter with `PYTHONNOUSERSITE=1`.
- [ ] Run `py_compile`, branch-wide and working-tree `git diff --check`.
- [ ] Obtain independent read-only review; fix all Critical/Important issues.
- [ ] Record exact hashes/test counts and append finish worklog.
- [ ] Commit `docs: record verified OpenRSD N0-O preflight` and stop. Do not
  launch GPU smoke.
