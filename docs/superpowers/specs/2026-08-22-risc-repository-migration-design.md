# RISC 独立仓库与源码迁移设计

**状态：** 已由用户确认采用“独立 Git 仓库 + 筛选代码快照”。

## 目标

从两个有未提交实验改动的历史工作树中建立一个干净、可审计、可继续开发的 RISC
仓库，同时避免数据、checkpoint 和运行产物污染 Git。

## 方案比较与选择

| 方案 | 优点 | 风险 | 结论 |
|---|---|---|---|
| 直接复制整个工作树 | 快速获得一份可见副本 | 混入数百 GB 运行产物、缓存和隐式状态，无法审计 | 不采用 |
| 仅用 OpenRSD 上游 submodule | 上游历史清晰 | 丢失本地未提交代码成果，外部远端也不能表达当前实验状态 | 不采用 |
| 基线工作树加受控源码快照 | 保留框架与本地代码成果，且来源、范围和禁入物可验证 | 需要明确排除规则与 manifest | **采用** |

## 架构

新仓库是一个独立 Git 根：

```text
RISC
├── framework/openrsd      # OpenRSD 运行底座的代码快照
├── reference/ov-capflow   # RISC-ER 规格、代码、测试与历史参考
└── docs                   # 决策、计划和 provenance
```

`framework/openrsd` 和 `reference/ov-capflow` 都不保留嵌套 `.git`。来源的提交 SHA、
远端、工作树 dirty 摘要、迁入文件计数和关键文件哈希在 provenance manifest 中固定。
这使该仓库可在源工作树以后变化时仍能解释自己所含代码的来源。

## 数据流

```text
OpenRSD current worktree ──[source-only rsync filters]──> framework/openrsd
OV-CapFlow current worktree ─[same policy]──────────────> reference/ov-capflow
                                                          │
                                                          └─> provenance manifest + Git commit
```

过滤器排除 `.git`、数据、权重、checkpoint、预测、可视化、缓存、二进制媒体与归档，
并且不使用 `--delete`。因此源工作树不会被修改，目标也不会因刷新脚本而删除已有
受保护文件。

## 失败处理

- 目标目录非空或已有 Git 历史：停止并请求用户决定是否合并，不覆盖。
- 任一来源目录或 Git SHA 无法读取：不产生不完整快照。
- 复制后发现禁止文件：从新仓库的明确目标路径移除该文件，更新过滤规则，重新扫描；不触碰来源。
- `git diff --check`、关键文件检查或禁止产物扫描失败：不创建初始提交，先保留失败记录并修复。

## 验证

1. 检查两个目标子树的 README、Python 源码、配置和测试路径存在。
2. 扫描已跟踪路径中是否出现政策表的禁止模式。
3. 记录 Git 跟踪文件数、子树文件数和关键 RISC 文件 SHA256。
4. 运行 `git diff --check`，确认工作树并复查首次提交的树。

## 非目标

- 不在本次迁移中启动训练、下载数据、移动/删除数据集或 checkpoint。
- 不改变 OpenRSD/OV-CapFlow 的代码、配置、分支、提交或远端。
- 不宣称当前快照通过完整训练；本次只验证存档边界和 Git 可审计性。
