# RISC M1：旋转条件低秩语义投影（GPU 8/9）

## 目标与边界

本阶段验证一个最小、单机制命题：旋转方向信息能否通过一个共享低秩子空间，抑制 OpenRSD 语义相似度中的旋转条件干扰，并在不改变定位分支、NMS 和类别原型定义的前提下提升 DOTA2 raw-13,833 mAP。

父模型固定为 C2B：`best_dota_mAP_epoch_1.pth`，其 raw-13,833 mAP 为 0.6596778631。C2C 仅保留为 BN 统计病理控制，不作为训练父模型。本阶段是 RISC 的 M1 actionability 实验，不把它表述为完整的成对旋转轨道监督 N1。

## 单一机制

对每个位置的方向傅里叶码 `z_theta` 和置信度 `c`，学习秩为 `r` 的共享语义基 `B` 以及方向门 `g(z_theta)`：

`t' = t - alpha * c * B diag(g(z_theta)) B^T t`

其中 `B` 经 QR 正交化，`alpha=max_strength*tanh(raw_strength)`，`raw_strength=0`。因此初始化时 `t'=t`，严格复现父模型。所有类别经过同一个投影，不设置 small-vehicle 专属参数。等价地，点积满足 `q^T t' = ((I-alpha*B diag(g) B^T)q)^T t`，所以它是分类空间中的查询投影，但无需触碰回归特征或框分支。

仅训练 `bbox_head.focus_support_adapter`。损失仍是探测器原有分类/回归目标；低秩投影只通过真实语义 logits 接收梯度，不使用缓存 logits，也不新增校准、路由、后处理或类别偏置。残差范数限制为原支持向量范数的 5%，强度上限 0.10。

## 数据、训练和评价

- 训练：DOTA2 47,294 个 1024/500 切片，当前 832 输入、RandomRotate=1.0，2 epoch、400 iter/epoch。
- 父权重：C2B epoch-1 best；BN 全程 eval/frozen。
- 优化：AdamW，只含低秩投影参数；初始学习率 3e-4；batch=2/GPU，sampler batch=2。
- 设备：仅物理 GPU 8、9；`CUDA_VISIBLE_DEVICES=8,9`；NCCL P2P/IB 均禁用。
- 主口径：DOTA2 `ss_val` 全部 13,833 切片，`filter_empty_gt=False`，scale=1024，18 类，IoU=0.5。
- 参照：C2B mAP 0.6596778631；候选至少不得回退，`+0.003` 以上才视为值得进入 N1 的积极信号。

## 失败门与完整性门

1. 初始化输出必须与输入逐元素严格相等。
2. 分类点积损失对 `raw_strength` 必须有非零梯度，防止旧低秩脚本“固定 logits 与适配特征脱钩”的无效训练。
3. 投影必须对所有类别共享，且不改变回归路径、框数、NMS 或评测口径。
4. 若训练后 mAP 下降、强度保持零、出现非有限损失，或 small-vehicle 改善伴随广泛类别坍缩，则拒绝该 M1，不叠加补丁救援。
5. 只有 M1 通过，才进入真正的 scene-disjoint 旋转轨道成对监督 N1。

## 可复现性例外

OpenRSD 当前主工作区包含用户未提交的历史改动，且 C2B 依赖其中的分类路径。为避免覆盖或错误提交这些改动，本阶段在当前状态新建 `research/risc-orbit-lowrank-gpu89` 分支，只提交新增模块、测试、配置和文档；同时记录依赖文件哈希。该提交不是独立 clean-clone 构建，报告中必须保留这一限制。
