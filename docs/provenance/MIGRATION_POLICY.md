# RISC 源码迁移政策

## 目的

保留可读、可编辑、可审计的代码性成果，同时不把不可复用的训练产物和大文件带入
新的 Git 历史。该政策适用于 OpenRSD 与 OV-CapFlow 两个输入工作树。

## 迁入范围

- Python、Shell、配置、测试、文档、许可证、构建/依赖描述和项目元数据；
- 两个工作树中已提交与未提交的上述文本源码；
- 代码所需的轻量、文本化的示例配置与注册文件；
- 与 RISC-ER 有关的历史设计、实施计划和实验日志 Markdown。

## 明确排除范围

| 类别 | 排除模式 |
|---|---|
| Git 与 IDE 元数据 | `.git/`、`.agents/`、`.codex/`、`.cursor/`、`.lab/`、`.vscode/` |
| 数据与运行目录 | `data`、`data/`、`datasets/`、`data_local/`、`work_dirs/`、`results`、`results/`、`visual/`、`vis_infer_epoch12/` |
| 模型与预训练资源 | `weights/`、`pretrained/`、`*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.h5` |
| 评测/缓存二进制 | `*.pkl`、`*.pickle`、`*.npy`、`*.npz`、`__pycache__/`、`.pytest_cache/` |
| 文档二进制与归档 | `pdf/`、`*.pdf`、`*.zip`、`*.tar`、`*.tar.gz`、`*.gz`、`*.png`、`*.jpg`、`*.jpeg`、`*.gif`、`*.mp4` |
| 运行结果 | `resultmd/`、`resultmd_backup_*/`、`experiments/`、`*.log` |

复制使用 `rsync -a` 的排除列表，且**绝不使用 `--delete`**。因此重复执行只更新迁入
文件，不会从目标快照删除已有内容。每次正式刷新都必须先新建一份带时间戳的
`SOURCE_SNAPSHOT_MANIFEST.md` 或先取得用户书面同意。

对可能被 `rsync -a` 当作文件保留的顶层符号链接，目录名和带尾斜杠的目录模式必须
同时出现，例如 `results` 与 `results/`、`data` 与 `data/`。迁移后还要单独检查目标
中的绝对链接；除可证明指向快照内部的相对包链接外，一律不保留。

## 格式保真规则

初始 source snapshot 的目标是保留来源代码/文档的字节内容，而不是对来源进行批量
格式化。因此如果完整 `git diff --cached --check` 报告只存在于 `framework/` 或
`reference/` 的来源继承文件，必须记录其数量、代表性路径和来源对比，不得为了让
检查变绿而重写快照。相反，本仓库新写的根目标、`docs/provenance/`、
`docs/superpowers/`、`task_plan.md`、`findings.md` 和 `progress.md` 必须通过 scoped
whitespace check。后续改动任何快照文件时，不得在所触及行新增 whitespace 问题。

## CATSeg tokenizer 词表的精确例外

以下三份文件是 OpenRSD vendored `DeCLIP_CATSeg` 的 OpenCLIP/EVA tokenizer 以相对
路径直接加载的 BPE 词表。它们是静态运行时代码依赖，不是模型权重、数据集、预测或
实验产物；每份为 1.3M，SHA256 均为
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`：

```text
framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/src/open_clip/bpe_simple_vocab_16e6.txt.gz
framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/src/open_clip/eva_clip/bpe_simple_vocab_16e6.txt.gz
framework/openrsd/third_party/DeCLIP_CATSeg/cat_seg/third_party/bpe_simple_vocab_16e6.txt.gz
```

这些精确路径是唯一允许跟踪的 `.gz` 文件。`.gitignore` 忽略所有其他 `.gz`；禁入
扫描也必须将 allowlist 限制为这三条路径，不能将该例外扩展为通用压缩包白名单。

### OpenRSD 实验代码的窄例外

`/data1/zcy/OpenRSD/experiments/rotation_semantic_attractor/` 同时包含可复用源码和
大型结果。主复制排除整个 `experiments/`，随后仅以独立 `rsync -a`（同样无
`--delete`）迁入以下四个代码目录：

- `configs/`
- `src/`
- `scripts/`
- `tests/`

不得迁入其 `outputs/` 或 `reports/`。这一例外保留 RISC/orbit 分析代码，却不把
训练输出、预测、可视化和报告数据写入 Git。

## 来源和追溯

迁移后 manifest 必须记录：来源绝对路径、来源 `HEAD` SHA、远端 URL、dirty 状态计数、
目标子树、迁入文件总数、排除扫描结果和关键 RISC 文件的 SHA256。源仓库中未提交
代码的具体行级来源由目标仓库的初始提交永久封存；不得把它误标为上游已发布代码。

## 数据路径约束

OpenRSD 历史 `data/` 链接不可作为数据来源。后续 DOTA 或其他实验必须使用
`/data1/zcy/datasets` 下的绝对数据根路径，并将实际路径记录在仓库外的运行 manifest。
