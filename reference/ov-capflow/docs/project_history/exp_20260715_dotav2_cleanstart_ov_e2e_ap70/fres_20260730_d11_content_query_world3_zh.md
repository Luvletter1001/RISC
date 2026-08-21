# D11 Generic Content Query Matched Proxy 结果

## 结论

D11 未通过预注册晋级门。它只恢复 generic GroundingDINO OGC 的前 600 行
content query，并保持 Q600、旋转 5D reference、数据、loss、optimizer 和严格
all-query 推理协议不变。候选在部分早期 epoch 有短暂优势，但在冻结的 epoch12
终点显著落后，因此判定为 `discard`，禁止 retune、rescue、额外 epoch、挑选最佳
checkpoint 或与其他候选堆叠。

| exp_id | candidate_mAP | control_mAP | delta_mAP | candidate_AP50 | control_AP50 | delta_AP50 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 8-D149-D11-W3-V2-FINAL | 0.4521 | 0.4660 | -0.0139 | 0.4520 | 0.4660 | -0.0140 | discard |

| group | candidate | control | delta |
|---|---:|---:|---:|
| novel4 | 0.26425 | 0.32475 | -0.06050 |
| base14 | 0.473571 | 0.473071 | +0.000500 |

## 曲线判读

候选在 epoch3 相对对照领先 `+0.0351 mAP`，在候选自身最佳的 epoch10 仍领先
`+0.0110`，但 epoch11、epoch12 分别反转为 `-0.0180`、`-0.0139`。这支持
“generic content initialization 改变早期优化路径”，但不支持“它改善稳定收敛
终点”。预注册裁决口径是 epoch12，不能事后改成最佳 epoch。

## 完整性

- candidate/control 均完成 12 epoch，训练退出码均为 0；
- postrun 的退出码 1 是预期 `discard` 返回，不是工程崩溃；
- strict audit 均为 `pass=true`，每图保持 600 个输出；
- `uses_nms=false`、`uses_topk=false`、`uses_min_area_rect=false`；
- candidate/control 的 open-vocabulary audit 均为 `pass=true`；
- gate SHA256:
  `2c39231b6b974d01a5918a2e5722d7b3b824bea09724602f2a046ebb4e6152d6`。

## 证据路径

- Gate:
  `.lab/workspace/exp-8-d149/final/d11_v2_paired_proxy_gate_epoch12.json`
- Candidate work dir:
  `work_dirs/dotav2_cleanstart/d11_v2_world3_candidate_seed20260716_gpu012_batch2`
- Control work dir:
  `work_dirs/dotav2_cleanstart/d11_v2_world3_control_seed20260716_gpu345_batch2`
- Queue state:
  `.lab/workspace/exp-8-d149/d11_v2_postrun_queue_state.jsonl`

所有 checkpoint、日志和审计产物保留，不执行删除或覆盖。
