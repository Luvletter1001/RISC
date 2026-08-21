# DOTA-v2 AP70 方向复盘与 GPU2389 续跑记录

## 1. 目标与不可变合同

最终目标是 raw `ss_val=13,833`、`filter_empty_gt=False`、IoU 0.5 口径同时达到 `dota/mAP >= 0.7000` 与 `dota/AP50 >= 0.7000`。模型必须是 generic-only clean start、开放词汇、旋转框、端到端 set prediction；禁止 teacher、distillation、pseudo label、dense/RoI head、NMS/rotated-NMS 和 inference top-k。每图固定输出 Q600，代理集、filtered mouth 或旧 remote-sensing checkpoint 都不能作为完成证据。

## 2. 过去几天实际推进的方向

| stage | direction | evidence | judgment |
|---|---|---|---|
| HRSC H3 | parent-preserving semantic fusion | AP50 `0.587 -> 0.629`，第二 seed `0.621` | 证明 zero-update parent equality 与小幅语义融合可兼容，但不是 DOTA2 AP70 解法 |
| DOTA2 causal C1 | frozen-parent one-epoch transfer | raw mAP `0.1107 -> 0.1253` | coverage 增益真实，绝对性能太低 |
| strong anchor | replay 旧 P126C | raw mAP/AP50 `0.6568/0.6570` | 说明数据与 evaluator 可到高分，但 checkpoint 不合规，不能当 clean-start parent |
| clean-start S1 | grouped O2O | 2k proxy AP50 `0.381 -> 0.407`，repeat 仅 `+0.004` | 主 seed 有效但 seed-sensitive；仍保留为 matching 基线 |
| geometry | Hausdorff-DN / Chamfer | AP50 `0.341/0.375` | 降低 geometry cost 或 duplicates 没有转化为 AP，停止该方向 |
| scale800 full | full 47,294 training | raw AP50 Epoch1 `0.344`、Epoch6 `0.484` | 全量训练有效，但小目标分辨率与 rare/novel 仍是瓶颈 |
| scale1024 proxy | matched resolution screen | control Epoch12 AP50 `0.403` | resolution 有帮助但单独未过 total/novel 双门 |
| text LR | BERT LR `1e-4 -> 1e-5` | Epoch12 AP50 `0.410`，novel4 `0.11225` | base 稳定，novel 无净增益；终止该变量 |
| checkpoint averaging | uniform E10–E12 | AP50 `0.415`，novel4 `0.1175` | 总 AP 恢复但 novel 门失败，不再 checkpoint-pick |
| rare-positive 4x | 36 rare-positive proxy / 824 full-data rare-positive | proxy Epoch12 AP50 `0.469`、novel4 `0.3675`、base14 `0.4646` | 首个同时通过 total/novel/base 冻结门的干预，晋级 full-data |
| rare4x full | scale1024 full 47,294 | raw Epoch1 `0.417`，Epoch6 `0.554` | 当前最强且仍在上升的合规路线，应继续同配方长训 |

核心结论不是“多训就能到 70”，而是过去的失败已经排除了三个局部方向：单纯改善几何 matching、降低 text LR、末段 checkpoint averaging。真正稳定的新信号来自 rare-positive exposure，它先在代理集修复 novel 激活，再在 raw full-data 把 Epoch6 AP50 推到 `0.5540`。不过距离 `0.7000` 仍有 `0.1460`，当前证据只支持继续，不能支持完成声明。

## 3. 当前 full-data 曲线与风险

| milestone | mAP | AP50 | delta_AP50 | gap_to_0.70 |
|---:|---:|---:|---:|---:|
| Epoch 1 | 0.4168 | 0.4170 | — | 0.2830 |
| Epoch 6 | 0.5537 | 0.5540 | +0.1370 | 0.1460 |

Epoch1→6 的提升与旧 scale800 路线的 `+0.1400` 历史增量几乎一致，说明 evaluator 口径和 full-data 学习趋势相互吻合。风险在于后半程是否平台化，以及 rare oversampling 是否在更晚 epoch 损伤 base 类；因此仍以 Epoch12/18/24 的 raw 全量评测为准，不根据 training loss 或 proxy AP 提前改配方。

## 4. GPU 2/3/8/9 迁移设计

原训练为 `batch2 × world2 × accum8 = effective batch 32`。从完整 Epoch6 checkpoint 切换为 `batch2 × world4 × accum4 = effective batch 32`，配置级 recipe、数据 mouth、seed、模型、optimizer 和 epoch scheduler 均不变；每 rank 每轮 microsteps 从 11,832 降为 5,916。这里的“不变”不代表 bitwise stochastic trajectory 等价：rank/worker 重排会改变逐样本 augmentation、DN noise 和 accumulation 内分组，后文单独量化。

world-size 迁移不能直接加载原 metadata。24 个 world4 sampler epoch 已逐轮审计：每轮都是 5,916 updates/rank，总计 141,984；duplicate=0、missing=0，最大 accumulation split=1。重基准 checkpoint 将 Epoch6 `iter` 从 70,992 改为 35,496、`max_iters` 从 283,968 改为 141,984。909 个 model tensors 和 2,659 个 optimizer tensors 的内容 digest 与原 checkpoint 完全相同，warmup 已在 step500 结束，epoch scheduler 仍停在 Epoch6。

切换会丢弃当前 world2 Epoch7 的部分未保存进度，但四卡将后续每轮 microsteps 减半，预计在不到一轮内回收这部分时间。新启动必须设置 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`，并继续独立监控 OOM、NaN/Inf、NCCL failure、dead rank、显存和进程组存活状态。

## 5. 决策

继续 rare4x full24 冻结配方，迁移到 physical GPU `2,3,8,9`。下一科学判据是 raw Epoch12；若仍显著上升则继续 Epoch18/24，若明确平台化再根据 per-class、novel4/base14、coverage 与 empty-tile diagnostics 设计下一分支。只有 raw 13,833 上 `mAP/AP50` 同时达到 0.70 才结束目标。

## 6. 四卡接管结果

第一次四卡启动虽然加载了重基准 checkpoint，但启动命令漏传 CLI `--resume`。`tools/train.py` 会用 CLI 默认值覆盖 config 中的 `resume=True`，因此日志错误地从 Epoch 1 开始、LR 也回到 warmup 区间。该进程在 20 steps 后停止，没有 checkpoint 或科学评测结果；这是启动层 engineering crash，不是模型、checkpoint 或 NCCL 失败。

修正后的启动只增加 `--resume`，日志明确记录 `resumed epoch: 6, iter: 35496`，并从 Epoch 7 开始，LR 恢复为 `8.5502e-5`。physical GPU `2,3,8,9` 四个 rank 均持续推进，loss 与 grad norm 有限，独立 monitor 的 `fatal_pattern=null`。Epoch24 guard 已绑定四卡、`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1` 与完整 checkpoint publication gate；异常退出时只从 `last_checkpoint` 指向的完整文件恢复。

## 7. Epoch 6 类别诊断

| diagnostic | Epoch 1 | Epoch 6 | change |
|---|---:|---:|---:|
| novel4 rounded class AP | 0.306000 | 0.431500 | +0.125500 |
| base14 rounded class AP | 0.448500 | 0.588571 | +0.140071 |
| container-crane AP | 0.190 | 0.052 | -0.138 |
| small-vehicle AP | 0.207 | 0.246 | +0.039 |

与旧 scale800 full-data Epoch 6 相比，当前 novel4 提升 `+0.227500`，base14 只提升 `+0.024500`。这表明 rare-positive 4x + 1024 的跨路线增益主要来自 novel exposure，和代理集晋级证据一致；但它并没有同步解决所有类别。

两个主要薄弱点不能混为一类：`container-crane` 只有 71 GT、recall `0.366`、308 detections，更像低支持度 semantic activation/coverage；`small-vehicle` 有 150,145 GT，但 AP `0.246`、recall `0.490`，更像密集小目标的定位、匹配和覆盖瓶颈。`helipad` 只有 6 GT，单次 AP 方差过高，不能独立决定路线。`airport` 虽因 Q600/no-NMS 产生大量 predictions，AP 已达 `0.898`，也说明检测数本身不是失败判据。

对 raw annotations 的固定容量审计进一步表明：243,632 个 GT 中只有 33 个 tiles 超过每图 600 GT，联合 Q600 theoretical recall cap 为 `0.934578`；small-vehicle 单类 cap 为 `0.896580`，远高于当前 recall `0.490`。所以固定 Q600 会损失一部分极密集目标，但没有让 AP70 结构上不可达，也不能解释当前主要缺口。rare4x train 已提供 3,936 次 container-crane instance exposure、724 次 positive-image exposure（181 unique source images），其后期退化也不能只归因于从未见过该类。

按 rounded 18 类 AP 求和，从 Epoch 6 宏平均约 `0.554` 到 `0.700` 还需累计补足 `2.634` 个 class-AP 点。Epoch 12 前不重试已经失去因果解释的 text-LR，不采用损伤 novel 的均匀 checkpoint averaging，也不回到未转化为 AP 的 Hausdorff/Chamfer 支线。若 Epoch 12 明显平台化，再将 semantic recall 与 dense-small-object coverage 分成两个单变量、互不堆叠的候选，同时保持开放词汇、旋转框、E2E、Q600、无 NMS。

## 8. 下一分支的只读预诊断

semantic 侧，当前 `clean_label_name` 不会把 DOTA hyphen 转为空格，BERT positive map 实际包含共享 punctuation token，例如 `container-crane → [container, -, crane]`。readout 会平均一个类别全部 positive tokens 的 log-probability，再为每个 query 选择最大类别，因此自然语言空格 prompt 是可检验的 class-competition 变量。但 `tennis-court` 在相同机制下已达 AP `0.887`，hyphen 不能被直接宣布为普遍根因。若 Epoch 12 需要新路线，先对同一 checkpoint 做一次预注册的全类别 canonical-vs-natural-space prompt A/B，同时约束 total、novel4 与 base14，不针对单类反复挑 prompt。

dense 侧，Q600 审计已经把硬容量与学习缺口分开。候选优先级是额外的高分辨率只读评测和更强 grouped matching 的 coverage 诊断；两者各自保持单变量，不能靠增加 Q、top-k 或 NMS 改写合同。现阶段这些都只是 Epoch 12 后的候选，不改变正在 GPU2/3/8/9 上运行的 T7。

工程侧，Epoch24 guard 已增加自动恢复后的 monitor rebind：新 launcher 会同时获得新的 process-tree、physical-GPU 与 fatal-log monitor。该修改通过 8 个本地回归，只重启 guard，没有触碰训练 launcher 或四个 rank。

## 9. Generic initialization 利用率审计

generic→Epoch1→Epoch6 的 module drift 表明，Epoch1→6 学习主要发生在 detector encoder/decoder（relative L2 均约 `0.061`）和 rotated regression head（`0.0586`）；text projection 为 `0.0335`，BERT encoder 只有 `0.0143`，word embeddings 只有 `0.00114`。因此 container-crane 退化不支持“大幅 BERT 漂移”解释，也进一步降低了重试 text-LR 的优先级。

更重要的是，原始 OGC 含有 `module.transformer.tgt_embed.weight`，shape 为 `900×256`；clean-start provenance 明确因 Q900/Q600 行数不同而把 converted `query_embedding.weight` 列为 unexpected，并让 target `600×256` content queries 随机初始化。OGC Q900 与 first600 的 norm mean 为 `1.6260/1.6059`、pairwise cosine std 为 `0.2524/0.2515`，说明前600行对全体结构具有代表性。当前 Epoch1/Epoch6 queries 对 OGC 的 mean nearest cosine `0.1833/0.1846` 与 seeded random `0.1844` 无区别；六轮训练只在随机 query basis 周围细调，没有恢复 generic content-query manifold。

这产生一个新的合规候选：精确迁移 OGC 前600个 content embeddings，但继续使用现有 deterministic 5-D rotated grid references。它不会恢复 encoder proposal ranking，不改变 Q600，不使用 top-k/NMS，也不引入遥感 checkpoint；唯一变量是更充分利用允许的 generic GroundingDINO prior。该候选不能直接替换 live T7，应在正式设计批准后先做 zero-update 与 matched real-2000 rare4x proxy，并同时约束 total、novel4、base14 和 strict OV/E2E/no-NMS audits。

## 10. Rotated terminal regression 初始化审计

相同的 24 个 fresh tensors 清单里还有 14 个 terminal regression tensors：六个 decoder heads 和一个 encoder-output head 的 weight/bias 都因 generic `4D` 与 rotated target `5D` shape 不同而被 converter 整体丢弃。只读逐层对齐发现，这并不代表回归头整体不兼容。六个 decoder heads 的两层 hidden MLP 共 `24/24` 个 weight/bias tensors 都已从 raw OGC 精确迁移，唯一断点就是最后的 `4×256 → 5×256` 输出层。

六个 OGC decoder terminal weights 彼此完全相同，每组 L2 为 `19.5626`。当前 T7 Epoch1 的 xywh 前四行合并 L2 只有 source 的 `1.1328%`，对 source 同行 mean cosine `0.1093`、relative L2 `0.9994`；到 Epoch6 也只有 `2.0954%`，mean cosine `0.1660`、relative L2 `0.9982`。相反，E1→E6 对自身同行的 mean cosine 为 `0.7903`，说明训练在扩展随机初始化的小幅 terminal basis，而没有重新学回 generic bbox-output manifold。

因此出现第二个、与 content-query transport 独立的 generic-only 候选：只将 raw OGC 六个 decoder terminal heads 的 xywh 四行复制进 rotated `5×256` target，第五行 angle 继续使用 seeded fresh initialization。第七个 encoder-output branch 在 E1/E6 的五行 terminal weights 都保持全零；固定 query 路线必须继续保持它为零，不能借此恢复 encoder proposal ranking 或 top-k。

这个候选的风险高于 content-query transport：source terminal norm 比 Epoch1 target 大约 88 倍，直接迁移可能改变初始 reference delta 和框分布。任何实现前都必须先过设计批准；实现后先做 zero-update finite/box-distribution audit 与真实 GT smoke，再在同一 real-2000 rare4x proxy 上单变量比较。它不能与 content-query transport 同时启用，除非两个独立 proxy 都已经给出因果增益证据。

剩余 fresh tensors 没有形成第三个合法直迁候选。raw OGC 不含 target contrastive cls scalar bias；当前 bias 使用标准 1% foreground prior `-4.595120`，六个 decoder branches 到 Epoch1 只偏移约 `0.0047–0.0071`。OGC `module.label_enc.weight` 虽有 `2001×256`，但它按源训练类别索引取行；DOTA DN table 是另一套 18 类索引，source first18 与 target E1 同行 cosine 均值仅 `-0.0229`，裁前18行会制造类别语义错配。fixed Q600 rotated references 在 raw OGC 中没有对应 tensor；公式重建 grid 与 Epoch1/Epoch6 的 relative L2 仅 `0.0016/0.0034`，应继续保留。

因此 24 个 fresh tensors 的审计已闭环：只保留 content-query first600 与 decoder xywh first4 rows 两个互斥单变量；cls bias、DN label table、fixed rotated reference 和 encoder-output branch 均不得迁移。

## 11. World-size 随机轨迹等价性复核

为判断 world4 较高的早期 geometry loss 是否只是样本密度差异，按375个 optimizer steps 对齐重放了 sampler epoch6：world2 用3000 microsteps/accum8，world4 用1500 microsteps/accum4。两条前缀分别覆盖 `11,988/11,994` 张，sample-set Jaccard `0.999500`；world4 的 GT total、GT/sample 与 mean local query-area 相对比值为 `1.000136/0.999636/1.008893`。所以两者看到的样本和密度几乎相同。

但 scalar 轨迹仍然分叉：world4 mean total loss 比 world2 高 `5.93%`，cls 低 `6.38%`，bbox/IoU 高 `8.21%/25.63%`，grad norm 低 `16.52%`。前125 optimizer steps total loss 还低 `3.53%`，差异从后续 steps 逐渐形成，排除了错误 checkpoint 起点或样本 mouth 作为直接原因。更合理的解释是 physical rank/worker 改变后，RandomFlip、DN label/box noise 和 accumulation 分组不再逐样本相同。

因此四卡续跑是合同有效、数据覆盖有效、optimizer state 有效的 stochastic continuation，而不是 world2 的逐步重演。这个修正不授权按 train loss 选择 checkpoint，也不使 raw result 无效；继续到 Epoch12，并只用 raw 13,833 mAP/AP50 决定保留或回退。

后续到 world4 microstep1660 时，用两条 run 各自最后40个 optimizer steps 再对齐，total/bbox/IoU 相对差异已变为 `-3.92%/+8.31%/+3.49%`，DN bbox/IoU 为 `-0.42%/-2.26%`。早先 `+25.63%` 的 mean IoU gap 没有继续扩大，不支持数值发散；但这个 trailing window 同样不能预测 raw AP，决策门仍保持不变。

## 12. Epoch12 曲线形状判读

过去完成的三条 matched 12e proxy 提供了后段斜率衰减的经验参照。定义 stage ratio `r=(AP_E12-AP_E6)/(AP_E6-AP_E1)`：direct control、same-recipe repeat、rare4x 的 `r` 分别为 `0.152174/0.168224/0.256579`；换算成每个 epoch interval，后段只保留前段斜率的 `12.68%/14.02%/21.38%`。

T7 raw AP50 从 E1 `0.4170` 到 E6 `0.5540`，early gain `0.1370`。若 E12 达到 `0.7000`，late gain 必须是 `0.1460`，即 `r=1.065693`、每 interval slope retention `88.81%`，约为最佳历史 proxy ratio 的 `4.153×`；mAP 的 required ratio 同样为 `1.068663`。按三个历史 `r` 做 curve-shape analogy，E12 AP50 落点为 `0.574848–0.589151`，mAP 为 `0.574533–0.588826`；若前段每 interval 斜率完全不衰减，线性参照 AP50 则为 `0.7184`。

这些数不是预测置信区间：400-image proxy 与 raw13,833/full47,294 不可交换，三条曲线也不足以建模概率。它们只冻结分流判据。Epoch12 若未过双0.70且 observed `r<=0.256579`（AP50 约不高于 `0.589151`），判为没有超越最佳历史后段保留率，优先启动获批的 D11 content-query 单变量 proxy；若 `r>0.256579`，说明 full-data persistence 确实更强，保留 T7 到 Epoch18。无论哪一种，权威证据仍是 scheduled raw evaluation。

## 13. AP70 类别差距集中度

按 E6 的 rounded 18类 AP 求和，当前是 `9.966`，到宏平均0.70对应的 `12.600` 还差 `2.634`。这部分不能归结为单一的新类问题：novel4 缺口 `1.074`，占 `40.77%`；base14 缺口 `1.560`，占 `59.23%`。container-crane 与 small-vehicle 两类合计缺 `1.102`（`41.84%`）；再加 helipad 是 `1.657`（`62.91%`），但 helipad 在 raw val 只有6 GT，单次 AP 的方差太高，不能用于决定整条路线。

E1→E6 gain 不超过0.04的类别是 baseball-diamond、container-crane、small-vehicle、storage-tank；它们的 deficit 合计 `1.507`，占总差距 `57.21%`。问题机制也不相同：container-crane AP/recall 为 `0.052/0.366`，是语义激活或 query 分配同时不足；small-vehicle 为 `0.246/0.490`，主要是 dense coverage；baseball-diamond recall 已到 `0.904` 而 AP 只有 `0.466`，更接近 precision/ranking；storage-tank 也只从 `0.524` 到 `0.529`。因此单一 semantic patch 或单一 dense patch 都不能解释全部差距。

当前超过0.70的 airport、plane、ship、tennis-court 即使全部升到1.0，也只剩 `0.633` class-AP headroom，相当于总 gap 的 `24.03%`；其他 classes 仍至少要贡献 `2.001`。另一方面，rare4x+1024 相对旧 scale800 E6 的 class-AP sum gain 为 `1.253`（宏平均 `+0.06961`），其中 novel4 贡献 `0.910`（`72.63%`），证明 rare exposure 路线的主效应真实存在，但基础类广泛平台化仍未解决。

这使 Epoch12 后的顺序更清楚：若 D15 判为历史形状平台化，先验证待批准的 D11 generic content-query 单变量，因为它可能同时改善 novel/base 的 query basis；其后再把 container semantic、small dense coverage 与高-recall低-AP ranking 分开做互斥诊断。这里的 rounded class AP 只用于归因，最终成功仍必须是 raw 13,833 official mAP 与 AP50 同时达到 `0.7000`。

## 14. 与不合规强锚点的逐类互补性

P126C epoch40 曾在相同 raw13,833、18类、IoU0.5 mouth 上复验到 mAP/AP50 `0.6568/0.6570`，但它使用旧 OpenRSD/GSOVD 权重与不同推理路线，仍严格禁止作为 Experiment8 parent、teacher、pseudo source、控制臂或最终证据。这里仅利用已经保存的逐类表判断 DOTA2 难类是否跨架构持续存在，不重新加载权重或运行推理。

P126C 与 T7 E6 的18类 AP Pearson correlation 为 `0.9010`。T7 只在 airport、bridge、container-crane 三类更高，P126C 在其余15类更高；但逐类取两者最大值的不可实现 oracle 也只有宏平均 `0.664883`，距0.70仍差 `0.035117`，class-AP sum 仍缺 `0.6321`。因此即使完全忽略合同，也不能靠回放旧 anchor、按类挑 checkpoint 或组合两套既有能力达到目标；AP70 必须产生超越两条路线的新能力。

两条路线共同低于0.70的是 baseball-diamond、bridge、container-crane、ground-track-field、helipad、roundabout、small-vehicle、soccer-ball-field，说明这些类具有持续难度。D16 的四个停滞类里，P126C 相对 T7 可将 baseball/small/storage 提高 `0.1298/0.2858/0.2476`，证明这三类不是不可学习，T7 的通用 query/detector basis 仍有 headroom；container-crane 在 P126C 反而只有 AP/recall `0.0179/0.1972`，低于 T7 的 `0.0520/0.3660`，是跨路线的独立语义/分配难点。

这个结果强化而不改变候选顺序：若 Epoch12 平台化，获批后先以 D11 单变量测试 generic content-query basis；无论 D11 成败，都不能把 container 自动归入通用初始化收益，必须另做全类别受约束的 semantic/query-allocation 诊断。旧锚点只作为难度参照，严格 OV、旋转框、fixed Q600、E2E、无 NMS/无 top-k 合同不变。

## 15. Epoch12/Epoch24 运行保障

在 world4 Epoch7 microstep2280 快照下，114条 scalar 的全局 mean time 为 `2.0987 s/microstep`；最近100条的 mean/median/p90/p95 为 `2.1015/2.1137/2.2217/2.2738 s`。距离 Epoch12 与 Epoch24 train boundary 分别还有 `33,216/104,208` microsteps。按 recent mean 外推，两者约在 `2026-07-20 22:01` 和 `2026-07-22 15:27 CST`；按 p95 保守外推则约在 `2026-07-20 23:36` 和 `2026-07-22 20:27 CST`。

guard 的 Epoch24 deadline 是 `2026-07-25 23:59 CST`，因此 p95 外推仍有 `75.53h` margin。过去 E1/E6 raw evaluator 平均为 `0.38855 s/batch`；world4、val batch2 下每 rank 约1730 batches，对应纯计算约 `11.20 min`。这不包括 metric aggregation 与 I/O，只用于值守规划，不能当作 metric 到达时间或 AP 预测。

现有 epoch1、epoch6 和 rebased epoch6 三个 checkpoint 共约 `6.337 GB`；E12/E18/E24 三个剩余 milestone checkpoint 按当前最大文件保守估算需 `6.382 GB`。当前 filesystem free约 `1.326 TB`，是需求的 `207.79×`。160/160 monitor samples alive、fatal0，当前全部 scalar finite。故时间、guard 和磁盘都不是 AP70 阻塞因素，不需要改变 GPUs2/3/8/9、checkpoint cadence 或科学配方。

## 16. 高梯度窗口根因审计

Epoch7 microstep2240–2420 出现一段连续10个 logger windows 的高 `grad_norm`，mean/min/max 为 `170.48/161.03/177.95`，明显高于 world4 median `62.50`。沿 expanded config 和 MMEngine source 回溯可知，live optimizer 使用 L2 `max_norm=0.1`；`OptimWrapper` 先调用 `torch.nn.utils.clip_grad_norm_`，把该函数返回的裁剪前 total norm写入 scalar，随后才执行 optimizer step。因此日志中的170并不是实际更新仍带有170范数。

该段全部 finite，mean loss `5.7708`，比前10窗 `6.7631` 低 `14.67%`；time与memory只高 `2.06%/0.85%`。world2 E1–E6 历史上还出现过 finite grad norm `328.49`。到 microstep2480时高段已经自然结束，174/174 monitor samples alive、fatal0。故它是被正常裁剪的困难 batch/accumulation cluster，不是梯度爆炸；不暂停、不降 LR、不改 clip，继续只按 non-finite、loss持续同步恶化、fatal 与 raw metric判断。

## 17. 过去方向的因果总账

过去几天的核心进展不是把许多技巧叠在一起，而是用逐步冻结的 metric mouth 和复验门排除错误方向。HRSC 上 parent-preserving fusion 从 strict control `0.587` 提到两个种子的 `0.629/0.621`，两种子均值 `0.625`，证明“保留父模型行为再引入新分支”可以产生可重复增益；但它不是 DOTA2，也不是当前 generic-init raw mouth，只能作为架构支持。迁到 DOTA2 后，同类 parent-preserving 由 raw mAP `0.1107` 提到 `0.1253`，增益 `+0.0146` 可精确复验，但绝对水平和父拓扑都不能通向本次 strict AP70。

旧 P126C 在 raw13,833 上曾复验到 mAP/AP50 `0.6568/0.6570`，但旧权重与推理路径违反 generic-init 路线，始终禁止作为 parent、teacher、pseudo source 或最终证据；D17 还证明即使违规地逐类取 P126C/T7 最大值，不可实现 oracle 也只有 `0.664883`。因此旧 anchor 既不合规，也不足以通过组合达到70。

clean-start proxy 的 geometry matching 分支已经关闭：Hausdorff 相对 control `0.381` 降到 `0.341`，Chamfer 为 `0.375`，相对 control 低 `0.006`、相对 grouped 低 `0.032`。grouped matching 首种子曾到 `0.407`、表面增益 `+0.026`，但 matched repeat 只从 `0.380` 到 `0.384`，剩 `+0.004`；首种子 novel4 还从 `0.125` 降到 `0.030`。同一 grouped checkpoint 从400-image proxy换到 raw13,833 后又从 `0.407` 掉到 `0.309`，transfer gap `-0.098`。所以 grouped 只能说明一个弱且高方差的 matching 信号，不能再作为主路线。

full-data scale800 raw E1/E6 AP50 为 `0.344/0.484`，虽有 `+0.140` 学习，但未过预注册的E6总量与novel门，因而按计划切到scale1024。scale1024 matched control 的冻结E12 endpoint是 `0.403`；E10曾有 `0.413`，但不能挑 checkpoint 代替endpoint。名义 text-LR 路线 endpoint是 `0.410`，比control高 `0.007`且novel4完全不变；随后逐参数 optimizer audit 发现两者372个显式LR entry完全相同、197个language参数本来就已被 `backbone` 子串规则设为 `1e-5`。因此数值可保留为同配方repeat，text-LR因果解释必须撤销。末三轮 uniform averaging 得到 AP50/base14 `0.415/0.470214`，但novel4只有 `0.1175`，低于冻结门 `0.14225`，也已关闭。

唯一同时通过冻结 total/novel/base 三门并合法晋级 raw full-data 的干预是 `scale1024 + rare-positive exposure 4x`。它在matched E12上从control AP50/novel4/base14 `0.403/0.11225/0.456857` 提到 `0.469/0.3675/0.464643`，对应 `+0.066/+0.25525/+0.007786`；进入raw后E1/E6为 `0.417/0.554`。这证明 rare exposure 主效应真实，但当前离双0.70仍差 mAP/AP50 `0.1463/0.1460`，不能把proxy成功写成最终成功。

因此当前处置保持单一：T7 在 GPU2/3/8/9 原配方继续到 Epoch12 raw13,833；Hausdorff、Chamfer、text-LR、averaging与旧anchor不重开，grouped不当稳定主效应。D11 generic first-600 content-query transport 与 D12 decoder xywh row transport仍只是待批准、互斥的设计候选，既没有实施，也不计为已有结果。若Epoch12未过双0.70，再按D15的斜率判据决定保留到E18或启动一个获批单变量，仍不允许无NMS合同外的top-k、teacher、pseudo或checkpoint组合。

## 18. scale1024 与 rare4x 的因果边界

当前 raw 路线同时包含两个变化：输入尺度从800提高到1024，并将 rare-positive exposure 调到4x。必须把“组合路线有效”和“单个变量的raw主效应”分开。scale1024 matched proxy中，direct control与rare4x除exposure外完全一致，E12 AP50/novel4/base14从 `0.403/0.11225/0.456857` 提到 `0.469/0.3675/0.464643`，因此rare4x在该proxy mouth上的 `+0.066/+0.25525/+0.007786` 是可识别的单变量因果效应。

raw E6则比较旧scale800 standard exposure与当前scale1024+rare4x，mAP/AP50从 `0.4842/0.4840` 到 `0.5537/0.5540`，组合增益为 `+0.0695/+0.0700`。novel4/base14分别增加 `0.2275/0.0245`；按四个novel与14个base的class-sum加权，novel贡献路线增益的 `72.63%`。但由于没有完整训练一条full47,294/raw13,833的scale1024/no-rare控制臂，这个raw +0.070不能继续拆成resolution主效应、rare主效应和interaction，也不能全部归给任何一个变量。

resolution仍有独立机制证据。800→1024使validation小于4像素的GT比例从 `0.18044` 降至 `0.05440`，相对减少 `69.85%`；小于8像素从 `0.51646` 降至 `0.33337`，减少 `35.45%`；small-vehicle小于8像素从 `0.73769` 降至 `0.49372`，减少 `33.07%`；train小于4像素减少 `80.01%`。这些证明1024显著改善可见像素几何，但不是isolated raw AP估计。

因此允许的结论只有三条：rare4x造成matched proxy改善；1024显著缓解极小目标像素压缩；二者组合改善raw E6。Epoch12前保持组合路线冻结。若平台化，不机械增加rare aliases：当前路线收益已经高度偏novel，而剩余AP70 gap仍有 `59.23%` 来自base14；更合理的下一步仍是获批后以D11检验跨novel/base的generic query basis。缺失的raw control cell只对归因有价值，不直接提高当前模型，不能打断权威E12路径补做。

## 19. proxy 到 raw 的逐类迁移边界

rare4x 的400-image proxy成功预测了“路线值得晋级”，但不是raw逐类能力的缩小复刻。对齐T6 E12 proxy与T7 E6 raw的18类表，class AP Pearson/Spearman只有 `0.6880/0.6381`，平均绝对偏差 `0.1771`。两边top5只共同包含plane与tennis-court，bottom5共同包含container-crane、helipad与soccer-ball-field。harbor/airport/large-vehicle在raw分别比proxy高 `0.439/0.428/0.381`；helicopter则从proxy `1.000`降到raw `0.631`，而它在proxy只有3 GT、raw有284 GT。

还有一个口径差异：proxy helipad为0 GT，因此官方 `mAP=0.4691` 实际等于17个有GT类别的rounded macro `0.46912`；如果为了跨mouth对齐强制把helipad AP0纳入18类，macro只有 `0.44306`。raw13,833的18类都有GT，rounded macro `0.55367`与官方 `0.5537`一致。novel4内部Pearson虽为 `0.7299`，但只有四类且MAE高达 `0.2485`；base14 Pearson/MAE为 `0.6173/0.1566`。

因此proxy的合法角色仍是廉价、matched、预注册的路线筛选：official total与手工novel4/base14三门一起通过时才晋级。它不能支持为helicopter、container-crane或helipad单独挑checkpoint、prompt或方法，也不能替代raw class diagnosis。后续若获批运行D11 proxy，继续使用同一aggregate gates；任何单类机制结论必须在scheduled raw checkpoint验证，最终仍只认raw13,833 mAP/AP50同时达到0.7000。

## 20. proxy 支持集修正不改变历史路线决策

D22发现proxy的helipad为0 GT后，进一步把T4/T5/T6的total gate从17个supported classes的official AP50映射到对齐的18类macro。control按novel4/base14重建的all18 macro是 `0.380278`；原预注册official paired增益 `+0.010` 按 `17/18` 映射后，all18 total target是 `0.389722`。

重放结果没有改变任何处置。T4 same-recipe repeat的all18为 `0.387334`，相对修正total门差 `-0.002388`，novel门还差 `-0.030000`；仍失败total与novel。T5 uniform averaging的all18为 `0.391833`，total高 `+0.002111`、base高 `+0.016357`，但novel仍差 `-0.024750`；仍因novel退化关闭。T6 rare4x的all18为 `0.443056`，相对total/novel/base门分别高 `+0.053334/+0.225250/+0.010786`，仍是唯一三门全过并晋级raw的路线。

因此零GT类别揭示的是proxy绝对总分mouth需要双重报告，而不是过去筛选错误。今后的proxy证据同时保留official supported-class total与aligned all-class macro，并继续单列novel/base门；这可以避免support集合或少数高AP低支持类掩盖分组退化。D20因果总账、D21组合归因和当前T7冻结运行均不变，最终成功仍只由raw13,833上18类有GT的mAP/AP50双 `0.7000` 判定。

## 21. AP70 的实际召回上界与排序效率缺口

同一raw13,833 evaluator的18类表显示，T7 Epoch1的rounded mean AP/recall为 `0.416833/0.697056`，aggregate `AP/recall` efficiency为 `0.597992`。如果保持Epoch1召回不变、只把每类所有已召回目标完美排序，其宏平均上界仍比0.70低 `0.002944`；所以早期阶段确实同时需要recall增长。

到Epoch6，mean AP/recall已变成 `0.553667/0.769278`，aggregate efficiency为 `0.719723`。这时fixed-recall perfect-ranking oracle高于0.70达 `0.069278`，说明宏平均AP70不再被实际召回数学封死；但当前还差 `2.634` 个class-AP sum，而AP到recall之间总headroom只有 `3.881`，需要回收其中 `67.87%`，不能把理论可达误写成容易实现。

用 `AP=mean_recall×aggregate_efficiency` 对Epoch1→6的 `+0.136833` AP增益做对称Shapley分解，recall增长贡献 `0.047584`（`34.78%`），efficiency增长贡献 `0.089249`（`65.22%`）。E6最大的单类AP→recall headroom集中在baseball-diamond `0.438`、ground-track-field `0.391`、bridge `0.335`、container-crane `0.314`、roundabout `0.301`、storage-tank `0.256`与small-vehicle `0.244`。

这进一步修正后续优先级：若Epoch12的fixed-recall oracle仍高于0.70而AP平台，主矛盾更偏general query basis、matching/ranking、定位质量与class competition，不应继续机械增加rare exposure，更不能通过NMS/top-k或增加Q改合同；若Epoch12 mean recall反而不足0.70，coverage才成为数学必需。container、small-vehicle等低recall类仍必须单独诊断，强类盈余只支持宏平均上界，不能证明每类健康。D11/D12仍需显式批准且保持单变量。
