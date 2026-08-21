# RISC M1 低秩投影实施计划

> 已由用户指令“现在做低秩投影，仅在 89 双卡”批准；按本计划连续执行到双卡训练确认启动。

1. 数据与父模型锁定
   - 核验 C2B 时间、权重、raw-13,833 指标。
   - 从完整 `ss_train/annfiles` 生成独立 Step6 pkl，不改写旧目录。

2. TDD 红灯
   - 新增 `tests/test_risc_orbit_projection.py`。
   - 覆盖零初始化恒等、共享低秩投影公式、残差范数上限和真实点积梯度连通。
   - 在模块不存在时运行测试并确认预期失败。

3. 最小实现
   - 新增 `M_AD/models/utils/risc_orbit_projection.py`。
   - 新增 `M_AD/models/dense_heads/risc_orbit_projection_head.py`，继承现有 OpenRSD head，只替换语义打分前的 support adapter。
   - 不修改已有脏工作区文件，不新增独立 loss、路由器或后处理。

4. 配置与预检
   - 新增 `M_configs/Diagnostics/dotav2_risc_orbit_lowrank_r8_c2b_frozenbn_2e_gpu89_20260807.py`。
   - 绝对数据路径；batch/sampler batch 均为 2；仅低秩模块可训练；C2B 加载；BN 冻结。
   - 运行 focused pytest、py_compile、配置构建、单真实 batch 前后向和参数白名单检查。

5. 固化与运行
   - 记录依赖源哈希，精确提交新增文件。
   - 用 `CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 启动两卡训练。
   - 确认两个 rank、GPU 8/9 占用、首批有限 loss 和非零投影梯度；训练自动完成 2 epoch 及 raw-13,833 评测。

6. 结果裁决
   - 与同口径 C2B 0.6596778631 比较，不与 filtered-6,605 的 70.50 混比。
   - 报告全类 AP、small-vehicle AP、投影强度和是否通过 +0.003 晋级门。
