# DOTA-v2.0 AP50 强锚点复验计划

## 目标

在物理 GPU `4,5,6,7` 上复验本机 P126C epoch40 checkpoint，使用原始完整
DOTA-v2.0 `ss_val=13,833`，要求本轮实际输出同时满足：

- `dota/AP50 >= 0.5000`；
- `dota/mAP >= 0.5000`；
- predictions 条目数为 13,833；
- 无关键权重漏载、OOM、NCCL 或 traceback。

当前 OV-CapFlow C1 的正式结果仍为 `mAP=0.1253`、`AP50=0.1250`。本实验
执行用户批准的 A 路线，先建立强检测父模型锚点，不改写 C1 的因果结论。

## 固定输入

| 项目 | 固定值 |
|---|---|
| checkpoint | `/data1/zcy/GSOVD/work_dirs/p126c_p121_semantic_state_b12x4_80e_val20_fullval_gpu0189_20260708/best_dota_mAP_epoch_40.pth` |
| SHA256 | `5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979` |
| config | `configs/strong_anchors/dotav2_p126c_raw13833_ap50_eval.py` |
| data | `/data1/zcy/datasets/DOTA2_1024_500/ss_val` |
| ann | `ss_val/annfiles` |
| empty tiles | `filter_empty_gt=False` |
| classes | DOTA2 canonical 18 classes |
| evaluator | `DETAILDOTAMetric(metric='mAP')` |
| GPUs | physical `4,5,6,7` |
| initial batch | 32/GPU |
| workers | 8/GPU |
| environment | `/data/zcy/anaconda3/envs/openrsd/bin/python` |

## 运行边界

历史 P126C 启动脚本在 `/data1/zcy/OpenRSD` 执行，模块搜索顺序固定为：

```text
/data1/zcy/GSOVD/.lab/workspace
/data1/zcy/OpenRSD
/data1/zcy/GSOVD
/data1/zcy/GSOVD/codex_hooks
```

四卡必须设置 `NCCL_P2P_DISABLE=1` 和 `NCCL_IB_DISABLE=1`。结果写到：

```text
/data1/zcy/OV-CapFlow/work_dirs/dotav2_ap50_strong_anchor/
  p126c_epoch40_raw13833_gpu4567/
```

## 失败处理

- OOM：只按 `32→16→8→2` 降低每卡 batch，不改数据、权重或后处理；
- import/registry：与历史启动脚本逐项比对路径，不复制或临时改写模型；
- checkpoint key：审计配置和 SHA，不用宽松加载掩盖关键缺失；
- AP50 低于 0.50：保留原始结果，转入展开配置差异审计；仍未恢复时启用官方
  H2RBox-v2 的 raw-13,833 恢复路线，目标不降低。

## 验收

历史 `AP50=0.6610` 只是假设依据。只有本轮日志、配置快照、13,833 条预测、
checkpoint 哈希和双 0.50 指标全部齐全后才通过。

