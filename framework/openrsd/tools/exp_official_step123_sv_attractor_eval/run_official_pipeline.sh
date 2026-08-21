#!/usr/bin/env bash
set -euo pipefail
REPO=/data1/zcy/OpenRSD
PY=/data/zcy/anaconda3/envs/openrsd/bin/python
TOOL=$REPO/tools/exp_official_step123_sv_attractor_eval
RD=$REPO/resultmd/exp_official_step123_sv_attractor_eval
export PYTHONPATH=$REPO

log() { echo "[$(date -Iseconds)] $*" | tee -a "$RD/log_official_pipeline_master.txt"; }

log "=== Official Step123 pipeline start ==="
nvidia-smi >> "$RD/log_official_pipeline_master.txt" 2>&1

$PY $TOOL/discover_weights.py 2>&1 | tee -a "$RD/log_discover_weights.txt"
$PY $TOOL/run_preflight.py 2>&1 | tee -a "$RD/log_preflight.txt"

# Smoke: Step2 P0148 + one high/low
log "Smoke Step2"
CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_mechanism_eval.py --stage Step2 --task p0148 --gpu 9 2>&1 | tee -a "$RD/log_smoke.txt" || true
CUDA_VISIBLE_DEVICES=9 $PY -c "
import os,sys
from pathlib import Path
REPO=Path('$REPO')
sys.path.insert(0,str(REPO))
os.environ['CUDA_VISIBLE_DEVICES']='9'
from tools.exp_mechanism_sv_attractor_gpu89 import common_attractor_utils as mech
from tools.exp_official_step123_sv_attractor_eval.common_official_utils import STAGES
spec=STAGES['Step2']
ctx=mech.MechContext(gpu=9,config=spec.config,checkpoint=spec.primary_ckpt)
b,m,d,s,n,_,_=mech.build_model(ctx)
for tile,ang in [('P0682__1024__553___0',0),('P0132__1024__0___0',0)]:
    r=mech.eval_one(ctx,m,b,d,s,n,tile,ang,'baseline')
    print('smoke',tile,r.get('final_sv_ratio'),r.get('notes',''))
" 2>&1 | tee -a "$RD/log_smoke.txt"

# Parallel eval
log "GPU8: Step2 highlow + P0148"
(
  CUDA_VISIBLE_DEVICES=8 $PY $TOOL/run_mechanism_eval.py --stage Step2 --task highlow --gpu 8 --full-matrix
  CUDA_VISIBLE_DEVICES=8 $PY $TOOL/run_mechanism_eval.py --stage Step2 --task p0148 --gpu 8 --full-matrix
) 2>&1 | tee -a "$RD/log_gpu8_step2.txt" &
PID8=$!

log "GPU9: Step3 highlow + heldout AP"
(
  CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_mechanism_eval.py --stage Step3 --task highlow --gpu 9 --full-matrix
  CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_heldout_ap.py --stage Step2 --heldout-n 200 --gpu 9
  CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_heldout_ap.py --stage Step3 --heldout-n 200 --gpu 9
  CUDA_VISIBLE_DEVICES=9 $PY $TOOL/run_mechanism_eval.py --stage Step3 --task p0148 --gpu 9 --full-matrix
) 2>&1 | tee -a "$RD/log_gpu9_step3_ap.txt" &
PID9=$!

wait $PID8 || log "GPU8 exit non-zero"
wait $PID9 || log "GPU9 exit non-zero"

# Step1: heldout AP attempt only
log "Step1 heldout AP attempt"
CUDA_VISIBLE_DEVICES=8 $PY $TOOL/run_heldout_ap.py --stage Step1 --heldout-n 100 --gpu 8 2>&1 | tee -a "$RD/log_step1_ap.txt" || true

$PY $TOOL/build_summaries.py 2>&1 | tee -a "$RD/log_build_summaries.txt"
log "=== pipeline end ==="
