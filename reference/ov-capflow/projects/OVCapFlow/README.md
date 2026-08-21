# OV-CapFlow

OV-CapFlow is an isolated research extension of CastDet's oriented
GroundingDINO implementation. The current strict substrate changes two
inference-critical behaviors:

1. matching query content and 5-D rotated references are learned parameters,
   not top-ranked encoder proposals;
2. inference emits one class decision for every matching query instead of a
   global query-class top-k list.

## Research status (2026-08-01)

- The holistic Query Allocation Quality Flow source preflight is immutable and closed with status `FAIL`; training remains restricted to physical GPUs 8 and 9, but this branch launches no training.
- D13-N closed at the registered E12 endpoint: 0.838305831 candidate versus 0.847486973 control, delta -0.009181142.
- Immutable source report: `.lab/workspace/exp-8-qaf-source-v1/source_report.json`; external report SHA256: `d51f88a677af48ddbdc0455f4300452bcaf51e44bbddec7c22176555b034ca33`.
- Exact reachabilities: overall `0.07178882557149407`; base14 `0.0717736256898161`; novel4 `1.0` (`total=1`, reported faithfully); spatial shuffle `0.07262396017554201`; semantic shuffle `0.07183795113643807`.
- Failed gates: `reachability_at_least_25pct`, `beats_prior_e_by_3pp`, `placebo_sensitivity`. Passed gates: `finite_reproducible_provenance`, `novel_base_gap_at_most_10pp`.
- Primary and placebo reachabilities are effectively indistinguishable, while primary reachability is far below `0.25` and prior E plus 3 percentage points. Approach A / Query Allocation Quality Flow is closed for the current source family, and Plan 2 is not executed.
- No production module or config, proxy/raw/full training, scale rescue, or alternate evidence level, threshold, seed, or subset is authorized.

The decoder now keeps the initial matching-query tensor as immutable native
semantics, treats the upstream image/text-attended query as transported
evidence, and fuses only the matching-query suffix through a zero-initialized
gate. Every matching query also receives a continuous capacity, while pooled
encoder memory predicts global scene count. Capacity and count are supervised
without sorting or pruning queries.

This code has passed isolated and constructor tests, but it has not yet passed
a real image/annotation training batch. Runtime and performance claims remain
blocked until that smoke test and the DOTA2 protocol are configured.

## Verified environment

Use `/data/zcy/anaconda3/envs/mmdet/bin/python`:

- Python 3.8.19
- PyTorch 1.12.1+cu113
- MMCV 2.1.0
- MMEngine 0.10.4
- MMDetection 3.3.0
- MMRotate 1.0.0rc1 from this checkout

Run the focused tests with:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow -q
```

The initial integration config is
`configs/ov_capflow/ov_capflow_swin-t_strict_visdrone.py`. It exists to prove
compatibility with the public Oriented GroundingDINO baseline. DOTA2 full-val
configuration and the 18-class open-vocabulary protocol are the next delivery
phase.
