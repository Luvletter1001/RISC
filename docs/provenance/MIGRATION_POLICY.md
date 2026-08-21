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
| 数据与运行目录 | `data`、`data/`、`datasets/`、`data_local/`、`work_dirs/`、`results/`、`visual/`、`vis_infer_epoch12/` |
| 模型与预训练资源 | `weights/`、`pretrained/`、`*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.h5` |
| 评测/缓存二进制 | `*.pkl`、`*.pickle`、`*.npy`、`*.npz`、`__pycache__/`、`.pytest_cache/` |
| 文档二进制与归档 | `pdf/`、`*.pdf`、`*.zip`、`*.tar`、`*.tar.gz`、`*.png`、`*.jpg`、`*.jpeg`、`*.gif`、`*.mp4` |
| 运行结果 | `resultmd/`、`resultmd_backup_*/`、`*.log` |

复制使用 `rsync -a` 的排除列表，且**绝不使用 `--delete`**。因此重复执行只更新迁入
文件，不会从目标快照删除已有内容。每次正式刷新都必须先新建一份带时间戳的
`SOURCE_SNAPSHOT_MANIFEST.md` 或先取得用户书面同意。

## 来源和追溯

迁移后 manifest 必须记录：来源绝对路径、来源 `HEAD` SHA、远端 URL、dirty 状态计数、
目标子树、迁入文件总数、排除扫描结果和关键 RISC 文件的 SHA256。源仓库中未提交
代码的具体行级来源由目标仓库的初始提交永久封存；不得把它误标为上游已发布代码。

## 数据路径约束

OpenRSD 历史 `data/` 链接不可作为数据来源。后续 DOTA 或其他实验必须使用
`/data1/zcy/datasets` 下的绝对数据根路径，并将实际路径记录在仓库外的运行 manifest。
