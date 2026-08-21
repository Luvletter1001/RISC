# DOTA-v2.0 AP50 强锚点执行日志

## 2026-07-15 运行前审计

- 建立分支 `research/dotav2-ap50-strong-anchor`。
- 定位 P126C epoch40 checkpoint，大小约 553 MiB，SHA256 为
  `5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979`。
- 历史 raw-13,833 记录为 `mAP=0.660601`、`AP50=0.6610`；仅作先验。
- wrapper 配置解析确认：batch 32、workers 8、`ss_val/annfiles`、
  `filter_empty_gt=False`、`test_mode=True`、`DETAILDOTAMetric`。
- 首次解析因 `PYTHONPATH` 缺少 `/data1/zcy/GSOVD/.lab/workspace` 失败，未加载
  checkpoint、未启动 GPU。历史启动脚本证明完整路径顺序；按原顺序后解析通过。
- 模型 registry/build 命令退出码为 0。
- 当前项目 OV-CapFlow portable 测试在修改后通过。
- 用户拥有的未跟踪目录
  `docs/project_history/exp_20260714_two_week_review/` 不纳入本实验暂存或提交。

下一动作：提交预注册配置与记录，然后在 GPU 4–7 启动完整评估。

## 2026-07-15 正式运行

- 预注册提交：`1ad768d`。
- 首次在受限沙箱内启动时，PyTorch TCPStore 无权绑定 localhost `29647`，四个
  worker 尚未创建，checkpoint 和 GPU 推理均未开始。这一条记为无效工程启动，
  不算实验结果。
- 相同命令在获准执行环境重启，四个 rank 实际绑定物理 GPU `4,5,6,7`。
- batch 32/GPU、workers 8/GPU；计算段共 `109` steps。
- GPU 4–7 计算期利用率为 `98–100%`，`nvidia-smi` 观察到每卡约
  `20,293 MiB`，MMEngine 日志峰值为 `9,511 MiB`。
- 推理完成后进入 IoU 0.5 的 CPU 类别汇总，最终进程退出码为 0。

## 结果与审计

| 项目 | 本轮证据 |
|---|---:|
| raw val images | 13,833 |
| prediction rows | 13,833 |
| `dota/mAP` | 0.6568131446838379 |
| `dota/AP50` | 0.6570 |
| AP50 超过目标 | +0.1570 |
| mAP 超过目标 | +0.156813 |
| checkpoint SHA256 | `5babf6e5a8c335299705158ec49bb8b88f70508ff0e72c75323abb4d894af979` |
| runtime | 约 460 秒 |

最终 MMEngine JSON：
`work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/20260715_082204/20260715_082204.json`。

最终日志：
`work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/20260715_082204/20260715_082204.log`。

预测文件：
`work_dirs/dotav2_ap50_strong_anchor/p126c_epoch40_raw13833_gpu4567/predictions.pkl`，
大小约 54.2 MiB，pickle 顶层长度为 13,833。

日志扫描未发现 traceback、CUDA OOM、NCCL error、RuntimeError、missing key 或
unexpected key。当前用户数值目标已通过；尚需提交结果并合并研究分支。

