# HRRSD SGP-Head Smoke Implementation Plan

## Goal

Implement SGP-Head / Gaussian Product Posterior Head for HRRSD, using the same dataset and evaluation family as P10/P11, then run local verification and prepare a smoke runner.

## Constraints

- Prefix shell commands with `rtk`.
- Do not use repository-local `data/`; use `/data1/zcy/datasets`.
- Use `/data/zcy/anaconda3/envs/openrsd/bin/python`.
- Do not modify OpenRSD's existing fusion head as the primary route.
- Keep the new head importable through `custom_imports`.

## Plan

- [x] Add RED tests for Gaussian product scoring, geometry-prior scoring, and the head forward path.
- [x] Implement `SGPRotatedRTMDetSepBNHead` as a new RTMDet-compatible head.
- [x] Add HRRSD train/eval configs that reuse the P10/P11 dataset split and priors.
- [x] Add a smoke runner mirroring the P11 train/eval/risk pattern.
- [x] Run unit tests.
- [x] Run config import/build checks.
- [x] Run a short forward/loss smoke if full training is not launched in this turn.
- [x] Append OpenRSD worklog completion entry.

## Verification Commands

```bash
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mpl /data/zcy/anaconda3/envs/openrsd/bin/python -m pytest tests/test_sgp_rtmdet_head.py -q
rtk env PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/openrsd_mpl /data/zcy/anaconda3/envs/openrsd/bin/python - <<'PY'
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS
cfg = Config.fromfile('M_configs/Diagnostics/hrrsd_rtmdet_l_dota_init_internal_sgp_head_train_gpu67.py')
init_default_scope('mmrotate')
model = MODELS.build(cfg.model)
print(type(model.bbox_head).__name__)
PY
```

## Smoke Runner

```bash
rtk bash M_Tools/experiments/run_hrrsd_sgp_head_train_20260622.sh
```
