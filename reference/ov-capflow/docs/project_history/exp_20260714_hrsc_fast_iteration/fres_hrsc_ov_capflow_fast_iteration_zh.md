# OV-CapFlow 在 HRSC2016 上的快速结构验证

## 1. 结论先行

本轮 HRSC 快速迭代完成了从工程门、严格父模型到三臂因果干预的完整闭环。

- 上游 P0 在 epoch 10 达到 `AP50=0.6620`，证明本地 HRSC 数据口、训练栈和评估器有效。
- 严格固定查询父模型 C0 达到 `AP50=0.5870`，超过预注册 H1 门槛 `0.50`；同一检查点三次独立复评和一次 H2 后置 replay 均为 `0.5870`。
- C1/C2/C3 的 AP50 分别为 `0.5580/0.3930/0.4460`，相对 C0R 分别下降 `0.0290/0.1940/0.1410`，没有干预臂满足晋级条件。
- 三个干预臂都显著减少重复 ownership，并使 `unmatched_gate - matched_gate` 为负；但 AP 与 coverage 同时下降，说明这些 mediator 改善尚未转化为有效检测质量。
- 本轮不执行 seed `20260713` 的 H3 复现，也不把 C1/C2/C3 中任何模型机制推进到 DOTA2。保留严格 C0 底座、工程审计和 mediator 评估工具。

HRSC 是单类闭集舰船检测口。本结果只验证结构可训练性和查询校准行为，**不能证明开放词汇迁移能力**。

## 2. 实验问题与预注册门槛

本轮依次回答三个问题：

1. H0：真实 HRSC batch 能否贯通 C0/C1/C2/C3 的前向、反向和严格推理路径？
2. H1：不使用 semantic/density/null 干预的固定学习查询 C0，能否形成 AP50 至少为 `0.50` 的严格父模型？
3. H2：在冻结父模型后，C1 语义融合、C2 平衡分类、C3 显式 null reservoir 是否能相对无更新 C0R 提升 AP 或在 AP 持平时改善至少两个 ownership/null mediator？

预注册晋级规则为：

- 首选：相对 C0R 的 AP50 增益至少 `+0.002`；
- 备选：AP50 与 C0R 相差不超过 `0.001`，recall 下降不超过 `0.005`，且至少两个 mediator 达到材料性改善。

详细设计和执行计划见：

- `docs/superpowers/specs/2026-07-14-ov-capflow-hrsc-fast-iteration-design.md`
- `docs/superpowers/plans/2026-07-14-ov-capflow-hrsc-fast-iteration.md`

## 3. 数据口、环境与统一设置

### 3.1 数据与评估口

- 数据根目录：`/data/zcy/dataset/HRSC_unzip/`
- 训练划分：`ImageSets/trainval.txt`，617 张图通过数据浏览检查。
- 测试划分：`ImageSets/test.txt`，453 张图、1228 个舰船 GT。
- 输入缩放：`800 × 512`，保持宽高比。
- 训练增强：水平、垂直、对角翻转，概率 `0.75`。
- 评估：旋转框 IoU `0.5`，主指标 `dota/AP50`。
- 严格预测：C0/C1/C2/C3 每图固定输出 600 个查询行，不做 top-k、NMS、rotated NMS、multiclass NMS 或 min-area-rectangle 回退。

### 3.2 软件与硬件

- Python `3.8.19`
- PyTorch `1.12.1+cu113`
- MMCV `2.1.0`
- MMEngine `0.10.4`
- MMDetection `3.3.0`
- MMRotate `1.0.0rc1`
- Transformers `4.46.3`
- 本地文本模型：`/data1/zcy/LAEDINO/weights/bert-base-uncased`
- GPU：NVIDIA A40 48 GB；H1 使用 GPU 4/5，H2 使用 GPU 5/6/7。

### 3.3 优化预算

- 随机种子：`20260712`。
- 查询数：600。
- H1：10 epochs，epoch 5/10 验证；batch size 3，梯度累积 2。
- H2：从同一 C0 epoch-10 检查点初始化，冻结父模型，仅训练预注册 adapter/null 参数 5 epochs。
- C1/C2/C3 的可训练参数张量数：18/18/20。
- 模型实验代码提交：`d03f29b`；只读 mediator 工具提交：`ffdb3f3`。

## 4. H0 工程门

H0 在投入训练预算前完成：

- 初始便携测试：`38 passed, 4 skipped`；本轮收口后为 `41 passed, 4 skipped`。
- C0/C1/C2/C3 真实 HRSC batch CUDA 前向/反向：4 个配置全部通过。
- HRSC 数据浏览：617/617 样本通过。
- 严格推理审计：`pass=true`，单图 600 行，禁用调用命中数 0。
- 修复查询别名后，AdamW 中 897 个参数对象全部唯一。

工程迭代中还修复了两个不计入科学结论的问题：移除非必要 TensorBoard 后端；把 `query_embedding` 改为只读 property，避免同一参数被优化器重复注册。

## 5. H1：P0 与严格 C0

P0 和 C0 是并行健康锚点，不是一个可直接归因的单变量消融。P0 保留上游 Oriented GroundingDINO 行为；C0 改为固定学习查询与固定参考点，并关闭 semantic/density/null 干预。

| 模型 | Epoch 5 mAP | Epoch 5 AP50 | Epoch 5 recall | Epoch 10 mAP | Epoch 10 AP50 | Epoch 10 recall | 时长/s | 峰值显存/MB | 最终 loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.4006 | 0.4010 | 0.889 | 0.6619 | 0.6620 | 0.967 | 1958 | 9509 | 8.8745 |
| C0 | 0.3088 | 0.3090 | 0.866 | 0.5868 | 0.5870 | 0.941 | 1935 | 9351 | 9.0953 |

C0 相对 P0 的 epoch-10 AP50 成本为 `-0.0750`。这是严格固定查询底座的综合成本，不能归因于后续 adapter。

检查点：

- P0：`a1e46af29b012d1cef2256f774b43bc0a350536a5eddfe8f16af7a4de6fe7692`
- C0：`a7c93158c22e7af4f858c43655f1a5e69c1f91ff3aab3137d480198190e78311`

### 5.1 C0 稳定性与严格性

| 评估 | mAP | AP50 | recall |
|---|---:|---:|---:|
| 训练内 epoch-10 验证 | 0.5868 | 0.5870 | 0.941 |
| 独立复评 1 | 0.5868 | 0.5870 | 0.941 |
| 独立复评 2 | 0.5868 | 0.5870 | 0.941 |
| H2 后置 replay | 0.5868 | 0.5870 | 0.941 |

最大 AP50 差为 0，评估器漂移为 0。后置严格审计仍为：每图 600 行、无禁用操作。

C1/C2/C3 对 C0 的加载审计均通过：预期 missing keys 为 18/18/20，invalid missing 与 unexpected keys 均为 0。因此 H1 判定为通过。

## 6. H2：冻结父模型的因果矩阵

Mediator 使用验证集上每个预测框对 GT 的最佳旋转 IoU；最佳 IoU 不小于 0.5 的查询记为 matched。`duplicate extras/GT` 是每个 GT 被分配的额外 matched 查询数之和除以 GT 数。这是验证集 ownership 诊断，不等同于训练时 Hungarian assignment。

| 模型 | 有效干预 | AP50 | ΔAP50 vs C0R | Recall/coverage | Duplicate extras/GT | Null Brier | Matched gate | Unmatched gate | Gate gap | 时长/s | 峰值显存/MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0R | 无更新 replay | 0.5870 | 0.0000 | 0.9414 | 23.1669 | N/A | N/A | N/A | N/A | 97 | N/A |
| C1 | query-preserving semantic fusion | 0.5580 | -0.0290 | 0.9210 | 10.6482 | N/A | 0.5448 | 0.5397 | -0.0050 | 553 | 1558 |
| C2 | C1 + balanced classification | 0.3930 | -0.1940 | 0.9267 | 13.1596 | N/A | 0.5502 | 0.5315 | -0.0187 | 670 | 1558 |
| C3 | C2 + explicit null reservoir | 0.4460 | -0.1410 | 0.9064 | 10.1107 | 0.0347 | 0.5252 | 0.5138 | -0.0114 | 678 | 1558 |

所有模型均在 453 张图上输出 271800 个结果，即每图严格 600 行。Density capacity 在本矩阵中关闭，因此 capacity mass 记为 N/A。

检查点 SHA256：

- C1：`689bf20e4a01d58805328f5d025c4526475e1326cf213bd5f304fd70e2482d88`
- C2：`c548f53ff85e823578050d52dd34793fe4d4adfe805ead14dff170439c29c245`
- C3：`e32fa7fdba14c3b109aed45860efbdf7c9e1c0585364098746a18af303629d8d`

### 6.1 因果解释

1. **C1：重复 ownership 改善，但父轨迹未保留。** 重复额外框减少约 54%，gate gap 变为负；然而 coverage 下降约 0.0204，AP50 下降 0.029。当前零 gate 返回 native query，而非精确复现 C0 transported query，因此它是显著结构干预，不是函数保持型 adapter。
2. **C2：平衡分类主要破坏得分排序。** 相对 C1，coverage 反而略升，但 AP50 从 0.558 降到 0.393，重复额外框也回升。这说明相等 matched/unmatched 组归约改变了全查询得分标定和排序，框覆盖不足不是主要原因。
3. **C3：null 能部分修复 C2，但不能恢复 C0。** C3 相对 C2 恢复 0.053 AP，并获得 0.0347 的 null Brier；但 coverage 和 recall 进一步下降，最终仍比 C0R 低 0.141 AP。
4. **Mediator 成功不等于主任务成功。** 三臂均得到非正 gate gap，并显著降低重复 ownership；但预注册规则要求 AP 先改善或基本持平。本轮所有臂均远离该条件。

## 7. 晋级、复现与转移决定

| 对象 | 决定 | 原因 |
|---|---|---|
| P0 | 保留为数据口/上游健康锚点 | AP50 0.662，证明 HRSC 栈有效；不是严格父模型 |
| C0 | 保留为严格父模型与后续诊断起点 | AP50 0.587，重复复评稳定，严格审计通过 |
| C1 | 淘汰模型候选 | AP50 -0.029，recall/coverage 下降超过门槛 |
| C2 | 淘汰模型候选 | AP50 -0.194，得分排序显著受损 |
| C3 | 淘汰模型候选 | 虽部分修复 C2，但 AP50 -0.141 且 coverage 更低 |
| H3 seed 20260713 | 跳过 | 没有 H2 晋级者，不允许事后放宽规则 |

**允许转移到 DOTA2 的模型机制：无。** 本轮只允许保留并迁移工程基础设施：严格 600 行审计、checkpoint load audit、真实 batch gate 和 mediator 评估工具。C1/C2/C3 的学习机制均不得以本轮结果作为 DOTA2 推进依据。

## 8. 下一轮建议

下一轮仍应先在 HRSC 做最小、单变量验证：

1. 把语义融合改为父函数保持形式，使初始化时输出逐元素等于 C0 transported query；在训练前增加 checkpoint replay 等价测试和 AP replay 门。
2. 暂停 C2 的等权 group reduction。先分析 C0/C2 的正负查询 score 分布、排序相关性和校准曲线，再设计不改变正样本排序的 loss reweighting。
3. Null 不再叠加在已失败的 C2 上。若父函数保持融合过门，再用 `C0 + null-only` 或严格单变量臂隔离其真实贡献。
4. 只有 HRSC 主指标和复现门同时通过后，才重新讨论 DOTA2；HRSC 仍不能替代开放词汇数据与跨类评估。

## 9. 可复现工件

- 训练与评估目录：
  - `work_dirs/ov_capflow_hrsc_fast/p0_parent_10e/`
  - `work_dirs/ov_capflow_hrsc_fast/c0_native_10e/`
  - `work_dirs/ov_capflow_hrsc/hrsc_c1_fusion_5e/`
  - `work_dirs/ov_capflow_hrsc/hrsc_c2_balanced_5e/`
  - `work_dirs/ov_capflow_hrsc/hrsc_c3_null_5e/`
- 审计：`work_dirs/ov_capflow_hrsc_fast/c0_native_10e/audits/`
- Mediator JSON：各模型目录下的 `mediators.json`。
- 完整实验台账：`.lab/log.md`、`.lab/results.tsv`、`.lab/summary.md`。

