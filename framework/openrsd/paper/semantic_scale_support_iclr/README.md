# Semantic-Scale Support ICLR Draft Package

Date: 2026-06-21

## Files

- `main.tex`: evidence-safe LaTeX manuscript skeleton.
- `../drafts/semantic_scale_support_refs.bib`: traceable BibTeX anchors.
- `../drafts/semantic_scale_support_mismatch_iclr_draft_v0_4_20260620.md`: source Markdown draft with v0.8 Gaussian support-field contract and launch-readiness update.
- `generated/evidence_figures.tex`: generated figure snippet.
- `generated/reproducibility_appendix.tex`: generated reproducibility and no-fabrication appendix.
- `build.sh`: compiler-dispatch wrapper for `latexmk`, `pdflatex`, or `tectonic`.

## Compile

This machine currently does not expose `pdflatex` or `xelatex` in `PATH`.
When a LaTeX environment is available, compile from this directory:

```bash
bash build.sh
```

The draft uses only standard packages: `geometry`, `booktabs`, `amsmath`,
`amssymb`, `graphicx`, `hyperref`, and `xcolor`.

## Static Source-Package Audit

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python \
  M_Tools/analysis/audit_semantic_scale_source_package.py
```

This checks that `main.tex`, generated inputs, figures, BibTeX, README, and
`build.sh` are present. It does not compile the PDF and does not infer missing
RankDelta/P2 metrics.

## Evidence Status

The manuscript intentionally reports the current machine gate:

```text
num_gates_passed = 3 / 10
```

Passed 9.5 gates:

- Literature breadth
- Problem anatomy depth
- Practicality

Open gates:

- optional E-P2-clean-v2 only if the paper restores a causal/path claim
- BASS matched-control method evidence beyond completed boundary G3-v2
- keep Platt/Isotonic calibration as a non-replacement baseline; add temperature scaling only if needed
- broader method transfer across datasets/detector families

Do not present this package as submission-ready until the open gates are
resolved and the machine audit agrees.

Historical GPU6/7 readiness/status is tracked in:

```text
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_gpu67_rankdelta_wait_blocker.md
resultmd/exp_p4_scale_semantic_validation/faudit_20260621_bass_gsf_gpu67_launch_readiness.md
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_bass_gsf_gpu67_waiter_liveness.md
```

The current rerouted GPU0/1 queue is tracked in:

```text
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_bass_gsf_gpu01_live_queue.md
```

As of the latest refresh, P0 RankDelta is running on GPU0/GPU1 and the
follow-up/P2 waiters are active. The package refresh manifest now reports
`active_route=gpu01`; GPU6/GPU7 fields are retained only as historical audit
fields and must not trigger a restart of the old waiters. RankDelta/P2 rows
must remain missing until the queued eval JSON and deployment-risk summaries
exist. `P0_RUNNING_GPU01_WAITERS_ACTIVE` is a liveness/prerequisite state, not
a method result.
