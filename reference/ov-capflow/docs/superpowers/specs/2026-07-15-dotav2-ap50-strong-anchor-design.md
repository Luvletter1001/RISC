# DOTA-v2.0 AP50 强锚点设计

## 目标

在物理 GPU 4、5、6、7 上，用 DOTA-v2.0 原始完整验证口 `ss_val=13,833`
完成可复现评估，使项目拥有一个经当前机器重新验证的 `AP50 >= 0.50` 模型锚点。
达标证据必须同时包含检查点哈希、展开配置、完整日志、13,833 张覆盖证明和
`dota/mAP`/`dota/AP50`。

## 已知起点

- 当前 OV-CapFlow C1：`mAP=0.1253`、`AP50=0.1250`，两次完整回放一致。
- 本机历史 P126C epoch40：相同 raw-13,833 口径记录
  `mAP=0.660601`、`AP50=0.6610`。
- P126C 检查点：
  `/data1/zcy/GSOVD/work_dirs/p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708/best_dota_mAP_epoch_40.pth`。
- 检查点 SHA256：
  `5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979`。
- 官方 H2RBox-v2 DOTA2 配置给出 `AP50=0.5033`，只作为 P126C 无法加载时的
  第二候选；它不是 raw-13,833 达标证据。

## 路线比较与决定

### A. 强检测父模型复验（已批准）

直接复验本机已有的强 DOTA2 模型，然后把检查点、配置和结果登记为当前项目的
强检测锚点。该路线最可能在一次完整验证中跨过 0.50，也为后续语义容量实验提供
不会被低质量检测底座限制的 parent。

### B. 当前 OV-CapFlow 从 0.125 长训

保持现有 Swin/DINO substrate，继续训练多个 epoch。结构连续性最好，但从 0.125
跳到 0.50 的短期成功概率低，不能作为本轮首选。

### C. H2RBox-v2 官方模型

模型结构简单且仓库原生支持，公开 AP50 略高于 0.50；但公开口径不是当前 raw
13,833 口径，因此仅在 A 的本地依赖损坏时作为恢复路线。

决定执行 A。A 达标后再决定是否把 OV-CapFlow 的语义机制迁移到强底座；迁移不得
破坏已经验证的 `AP50 >= 0.50`。

## 实验结构

当前仓库新增一个很薄的评估配置，它继承 P126C 的冻结展开配置，但覆盖：

- `work_dir` 到当前仓库；
- 验证 batch size 初值为每卡 32，使用四卡提高吞吐和 GPU 利用率；
- dataloader workers 为每卡 8；
- 关闭仅训练时需要、且会写回旧 GSOVD 目录的 custom hooks；
- 数据固定为 `/data1/zcy/datasets/DOTA2_1024_500/ss_val`；
- `filter_empty_gt=False`、`test_mode=True`、18 类顺序不变；
- evaluator 固定为 `DETAILDOTAMetric(metric='mAP')`。

执行从 `/data1/zcy/OpenRSD` 启动，并复用历史 P126C 的完整模块搜索顺序：
`GSOVD/.lab/workspace`、`OpenRSD`、`GSOVD`、`GSOVD/codex_hooks`。这是历史
启动脚本实际使用的源码边界；配置、输出、台账和最终结论保存在 OV-CapFlow。
运行环境固定为历史兼容的 `/data/zcy/anaconda3/envs/openrsd/bin/python`，并设置
`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`。

## GPU 策略

- 只绑定物理卡 `4,5,6,7`，四个进程一一对应。
- 首跑每卡 batch 32；它比历史每卡 batch 2 大 16 倍，用于提高占用率。
- 若发生 OOM，只允许依次降为 16、8、2；数据、权重和指标口径不变。
- 若 GPU 利用率因 CPU 供数不足而长期低于 70%，先提高 workers，再调整 batch。
- 不占用正在执行外部任务的 GPU 0–3、8–9。

## 达标门槛

只有以下条件全部满足，才算用户的 0.50 目标完成：

1. 运行进程实际绑定 GPU 4、5、6、7；
2. 日志无 traceback、OOM、NCCL 或漏加载关键权重；
3. 验证集实例数为 13,833，empty-GT tile 未被过滤；
4. `dota/AP50 >= 0.5000`；
5. `dota/mAP >= 0.5000`；
6. 日志、预测文件、配置快照、checkpoint SHA256 和结果台账齐全。

历史日志中的 0.6610 只是先验，不替代本轮复验。若本轮低于 0.50，则目标仍未完成，
转入 checkpoint/config 差异审计或 H2RBox-v2 恢复路线。

## 产物

- `configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py`
- `work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/`
- `docs/project_history/exp_20260715_dotav2_ap50_strong_anchor/`
- `.lab/log.md`、`.lab/results.tsv`、`.lab/branches.md`
