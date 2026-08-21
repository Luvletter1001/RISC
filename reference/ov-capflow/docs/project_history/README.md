---
title: "OV-CapFlow Project History Index"
document_role: "project_history_navigation"
ai_scan_priority: "P0"
updated_at: "2026-08-07 Asia/Shanghai"
---

# OV-CapFlow 项目历史扫描入口

> [AI-SCAN-P0] 后续 AI 在恢复实验、扫描 work_dirs、提出新实验或撰写论文前，
> 必须先读取下面的当前权威工作记忆。

## 当前 P0 权威文档

- 时间窗：2026-07-29 至 2026-08-07
- 状态：ACTIVE_AUTHORITY
- 当前研究优先级：strict open vocabulary + negative-safe
- 文档：
  [十日实验全面复盘、思考与后续工作记忆](exp_20260729_20260807_ten_day_review/fres_20260729_20260807_full_review_reflection_zh.md)

该文档包含：

- 16 个研究节点的统计与逐项裁决；
- raw/proxy/closed-set/strict-OV 指标边界；
- 已关闭路线与禁止重复清单；
- POQ 动态状态及强制刷新要求；
- P0/P1/P2 work_dirs 扫描地图；
- 下一阶段 strict-OV/negative-safe 证据防火墙。

## 历史背景

- [2026-07-01 至 2026-07-14 两周复盘](exp_20260714_two_week_review/fres_20260701_20260714_full_review_zh.md)
- [OV-CapFlow 后续研究与实验执行指南](exp_20260714_two_week_review/fplan_ov_capflow_next_stage_zh.md)
- [DOTA-v2 E24 路线审计](exp_20260723_e24_route_audit/fres_dotav2_e24_route_audit_zh.md)
- [D11 matched proxy 结果](exp_20260715_dotav2_cleanstart_ov_e2e_ap70/fres_20260730_d11_content_query_world3_zh.md)

## 扫描纪律

1. 详细实验内容必须保留在各自 exp_* 子目录；本索引只负责导航。
2. 不从目录名、训练 loss 或 GPU 利用率推断科学结论。
3. 动态任务必须重新读取最新日志与进程。
4. 冻结指标以 scalars.json、audit JSON 和 .lab/results.tsv 为准。
5. 不删除 checkpoint、清理 work_dirs、覆盖台账或终止未知进程。
6. 所有 shell 命令必须以 rtk 开头。
