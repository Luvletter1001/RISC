# DOTA-v2.0 AP50 Strong Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GPU 4–7 上把 P126C epoch40 检查点按 raw-13,833 同口径重新评估，并以完整证据证明 DOTA2 AP50 至少为 0.50。

**Architecture:** 当前仓库只保存薄评估配置、实验台账和结果；模型注册代码由历史兼容的 GSOVD 源码提供。评估配置继承经验证的 P126C 配置，仅覆盖输出目录、测试 batch/worker 和训练专用 hooks，避免改动模型与指标口径。

**Tech Stack:** MMEngine 0.10.x、MMDetection/MMRotate、PyTorch distributed、NCCL、A40 GPU、DOTA-v2.0 raw-13,833。

---

## 文件结构

- `configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py`：定义可重复的 P126C raw-full-val 评估覆盖项。
- `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/fplan_dotav2_ap50_strong_anchor_zh.md`：记录实验假设、输入、命令和门槛。
- `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/flog_dotav2_ap50_strong_anchor_zh.md`：记录启动、资源、异常和恢复。
- `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/fres_dotav2_ap50_strong_anchor_zh.md`：记录最终指标与达标判定。
- `.lab/config.md`、`.lab/log.md`、`.lab/results.tsv`、`.lab/branches.md`：维护研究状态。

### Task 1: 建立可解析的强锚点评估配置

**Files:**
- Create: `configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py`

- [ ] **Step 1: 写最小继承配置**

```python
_base_ = (
    '/data1/zcy/GSOVD/.lab/workspace/'
    'p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708.py'
)

work_dir = (
    'work_dirs/dotav2_ap50_strong_anchor/'
    'p126c_epoch40_raw13833_gpu4567'
)
custom_hooks = []
test_dataloader = dict(batch_size=32, num_workers=8, persistent_workers=True)
val_dataloader = dict(batch_size=32, num_workers=8, persistent_workers=True)
```

- [ ] **Step 2: 解析展开配置**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/GSOVD/.lab/workspace:/data1/zcy/OpenRSD:/data1/zcy/GSOVD:/data1/zcy/GSOVD/codex_hooks /data/zcy/anaconda3/envs/openrsd/bin/python -c "from mmengine import Config; c=Config.fromfile('/data1/zcy/OV-CapFlow/configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py'); print(c.test_dataloader.batch_size, c.test_dataloader.dataset.ann_file, c.test_dataloader.dataset.filter_cfg, c.test_evaluator)"
```

Expected: batch 为 32，ann_file 为 `ss_val/annfiles`，`filter_empty_gt=False`，evaluator 为 `DETAILDOTAMetric`。

- [ ] **Step 3: 提交配置**

```bash
rtk git add configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py docs/superpowers/specs/2026-07-15-dotav2-ap50-strong-anchor-design.md docs/superpowers/plans/2026-07-15-dotav2-ap50-strong-anchor.md
rtk git commit -m "research: add DOTA2 AP50 strong-anchor protocol"
```

### Task 2: 预注册并完成运行前审计

**Files:**
- Create: `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/fplan_dotav2_ap50_strong_anchor_zh.md`
- Create: `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/flog_dotav2_ap50_strong_anchor_zh.md`
- Modify: `.lab/config.md`
- Modify: `.lab/log.md`
- Modify: `.lab/branches.md`

- [ ] **Step 1: 登记不可变输入**

记录 checkpoint 绝对路径、SHA256 `5babf6e...af979`、验证集 13,833 张、18 类顺序、
GPU 4–7、batch 32、环境和 AP50/mAP 双 0.50 门槛。

- [ ] **Step 2: 检查运行环境与模型构建**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/GSOVD/.lab/workspace:/data1/zcy/OpenRSD:/data1/zcy/GSOVD:/data1/zcy/GSOVD/codex_hooks CUDA_VISIBLE_DEVICES=4 /data/zcy/anaconda3/envs/openrsd/bin/python -c "from mmengine import Config; from mmengine.registry import init_default_scope; from mmdet.registry import MODELS; c=Config.fromfile('/data1/zcy/OV-CapFlow/configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py'); init_default_scope(c.default_scope); model=MODELS.build(c.model); print(type(model).__name__)"
```

Expected: 打印 `OpenRTMDet`，无 import/registry 异常。

- [ ] **Step 3: 检查 GPU 绑定和空闲状态**

Run:

```bash
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: GPU 4–7 无外部占用；若出现外部进程，不终止用户进程，等待或改排程。

- [ ] **Step 4: 提交预注册记录**

```bash
rtk git add .lab docs/project_history/exp_20260715_dotav2_ap50_strong_anchor
rtk git commit -m "research: preregister DOTA2 raw AP50 anchor replay"
```

### Task 3: 在 GPU 4–7 运行完整 raw-13,833 评估

**Files:**
- Create: `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/test_tmux.log`
- Create: `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/predictions.pkl`

- [ ] **Step 1: 启动四卡分布式评估**

Run from `/data1/zcy/OpenRSD`:

```bash
rtk env PYTHONNOUSERSITE=1 PYTHONPATH=/data1/zcy/GSOVD/.lab/workspace:/data1/zcy/OpenRSD:/data1/zcy/GSOVD:/data1/zcy/GSOVD/codex_hooks CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TMPDIR=/tmp MPLCONFIGDIR=/tmp/ov_capflow_mpl_p126c /data/zcy/anaconda3/envs/openrsd/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=29647 tools/test.py /data1/zcy/OV-CapFlow/configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py /data1/zcy/GSOVD/work_dirs/p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708/best_dota_mAP_epoch_40.pth --launcher pytorch --work-dir /data1/zcy/OV-CapFlow/work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567 --out /data1/zcy/OV-CapFlow/work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/predictions.pkl
```

Expected: 四个 rank 启动并实际占用物理 GPU 4–7。

- [ ] **Step 2: 监控首批迭代**

Run:

```bash
rtk nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
rtk tail -n 80 work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/test_tmux.log
```

Expected: GPU 4–7 有持续负载，日志无 OOM/traceback。若 OOM，将 batch 依次降至 16、8、2 并重新提交配置后重启。

- [ ] **Step 3: 等待完整评估结束**

Expected: 日志出现 `Epoch(test)` 的最终 `dota/mAP` 与 `dota/AP50`，并生成 predictions 文件。

### Task 4: 验证结果并完成目标审计

**Files:**
- Create: `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/fres_dotav2_ap50_strong_anchor_zh.md`
- Modify: `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/flog_dotav2_ap50_strong_anchor_zh.md`
- Modify: `.lab/log.md`
- Modify: `.lab/results.tsv`
- Modify: `.lab/branches.md`

- [ ] **Step 1: 抽取最终指标和覆盖证据**

Run:

```bash
rtk rg -n "Dataset Instances|Epoch\(test\).*dota/mAP|dota/AP50|Traceback|CUDA out of memory|NCCL" work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567 -g "*.log" -g "*.json"
rtk sha256sum /data1/zcy/GSOVD/work_dirs/p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708/best_dota_mAP_epoch_40.pth
```

Expected: 13,833 张、无异常、SHA256 一致、mAP/AP50 均至少 0.50。

- [ ] **Step 2: 检查预测条目数**

Run:

```bash
rtk env PYTHONNOUSERSITE=1 /data/zcy/anaconda3/envs/openrsd/bin/python -c "import pickle; p=pickle.load(open('work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/predictions.pkl','rb')); print(len(p))"
```

Expected: `13833`。

- [ ] **Step 3: 写入最终报告和研究台账**

报告必须明确区分：当前 C1 仍是 0.125；A 路线强检测锚点的新复验结果是本轮过 0.50 的模型。不得把外部历史日志冒充本轮结果。

- [ ] **Step 4: 运行完成前验证并提交**

```bash
rtk git diff --check
rtk git status --short
rtk git add .lab configs/strong_anchors docs/project_history/exp_20260715_dotav2_ap50_strong_anchor docs/superpowers
rtk git commit -m "research: verify DOTA2 raw AP50 strong anchor"
```

Expected: 只提交本实验文件，不提交用户拥有的 `docs/project_history/exp_20260714_two_week_review/`。

### Task 5: 失败恢复（仅在 Task 3 未达标时执行）

**Files:**
- Modify: `configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py`
- Modify: `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/flog_dotav2_ap50_strong_anchor_zh.md`

- [ ] **Step 1: 分类失败**

只允许四类：环境/注册失败、checkpoint 键不匹配、OOM、完整指标低于 0.50。每类都记录原始错误，不通过改验证口径掩盖。

- [ ] **Step 2: 按最小修复恢复**

- 环境/注册失败：改用生成历史 0.661 日志时的 GSOVD 源码和 openrsd 环境；
- 键不匹配：核对 P126C 展开配置与 checkpoint SHA，不启用 `strict=False` 掩盖关键键；
- OOM：batch 32→16→8→2；
- 指标低于 0.50：比较本轮展开 config 与历史 config，若一致则启动 H2RBox-v2 raw-13,833 复验，目标仍保持 0.50。

- [ ] **Step 3: 重新执行 Task 3 和 Task 4**

Expected: 只有同一 raw-13,833 口径实际达到 0.50 才结束目标。
