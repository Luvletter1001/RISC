# OpenRSD Skills

项目级 AI Skills，用于在 OpenRSD 仓库中增强特定领域能力。

## Cursor（自动生效）

Cursor Agent 会从以下路径**自动发现并按场景应用** skills（与 Codex 的 `~/.codex/skills/` 类似，但作用域仅限本仓库）：

```text
.cursor/skills/<skill-name>/SKILL.md
```

当前已配置：

| Skill | 路径 | 何时自动使用 |
|-------|------|----------------|
| git-expert | `.cursor/skills/git-expert/` | Git 分支、合并、提交、PR、历史排查 |
| python-expert | `.cursor/skills/python-expert/` | 编写/修改/调试本仓库 Python 代码 |

无需在对话里手动 `/skill`；描述匹配任务时 Agent 会读取并遵循对应 `SKILL.md`。

## 旧版 `skills/*.md`（兼容）

本目录下的 `git-expert.md`、`python-expert.md` 为早期手写说明，**权威内容已迁移到** `.cursor/skills/`。新增或修改 skill 时请只改 `.cursor/skills/<name>/SKILL.md`。

## 添加新 Skill

1. 创建目录：`.cursor/skills/<skill-name>/`
2. 添加 `SKILL.md`，包含 YAML frontmatter：

```markdown
---
name: your-skill-name
description: What it does. Use when the user ...
---

# Title
...
```

3. 需要**自动触发**时：不要设置 `disable-model-invocation: true`（默认仅显式调用时才加载）。
4. 更新本 README 表格。

## Planning with Files（已安装）

来自 [OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files)（v2.38.1，Manus 式三文件规划）。

- **Skill**：`.cursor/skills/planning-with-files/`
- **Hooks**（已与 Superpowers 合并）：`userPromptSubmit`、`preToolUse`、`postToolUse`、`stop` + 保留 `sessionStart`
- **规划文件**（复杂任务时）：`task_plan.md`、`findings.md`、`progress.md`（或 `.planning/<日期-slug>/`）
- **重装**：`bash scripts/install-planning-with-files.sh`
- **中文 skill**（可选）：`npx skills add OthmanAdi/planning-with-files --skill planning-with-files-zh -g`

与 `experiment-result-md-organization` 分工：实验**过程/结果归档**走 `resultmd/exp_*/fres_*.md`；**进行中任务拆解**可走 planning 三文件。

## Superpowers（已安装）

来自 [obra/superpowers](https://github.com/obra/superpowers) 的 **14 个** skill 已安装到 `.cursor/skills/` 与 `~/.codex/skills/`。

- **Session 引导**：`.cursor/hooks.json` → 新对话注入 `using-superpowers`
- **重装**：`bash scripts/install-superpowers.sh`
- **官方方式**（可选）：Cursor 里 `/add-plugin superpowers`

核心 skill：`brainstorming`、`writing-plans`、`test-driven-development`、`systematic-debugging`、`using-superpowers`。

目录整理相关（Superpowers，已安装）：

| Skill | 用途 |
|-------|------|
| `using-git-worktrees` | 大规模改目录前开隔离工作区 |
| `requesting-code-review` | 整理后做 review（导入路径、误删、测试） |
| `finishing-a-development-branch` | 收尾：提交、合并、PR、清理 |

## Matt Pocock skills（已安装）

来自 [mattpocock/skills](https://github.com/mattpocock/skills)：

| Skill | 用途 |
|-------|------|
| `zoom-out` | 先理解目录/代码区块整体结构 |
| `improve-codebase-architecture` | 找模块边界、重复逻辑、可测试性等结构整理机会 |
| `scaffold-exercises` | 按章节/题目/答案/讲解规范化练习目录 |

- **路径**：`.cursor/skills/` 与 `~/.codex/skills/`
- **重装**：`bash scripts/install-matt-pocock-skills.sh`（需能访问 GitHub；当前环境若被网关拦截，可用手动拷贝的 skill 目录）

## Hermes RTK + Caveman（已安装）

来自 [hermes-agent-rtk-caveman](https://github.com/adityahimaone/hermes-agent-rtk-caveman) 的 **33 个** skill 已安装到：

- **Cursor（本仓库）**：`.cursor/skills/`（与 `git-expert`、`python-expert` 等同目录）
- **Codex（全局）**：`~/.codex/skills/`
- **脚本 / 模板**：`~/bin/*.sh`、`~/templates/*.txt`
- **Shell 别名**：已写入 `~/.bashrc`（`cgs`、`gs` 等）

重装或更新：

```bash
bash scripts/install-hermes-rtk-caveman.sh
```

Agent 跑 shell 时应遵循根目录 `AGENTS.md`：**非交互命令加 `rtk` 前缀**；需要极简 git/lint/test 输出时用 `caveman-rtk-integration` skill 中的 Caveman 流程。

核心 development skills：`caveman-rtk-integration`、`context-optimization`。

## Researcher（已安装）

来自 [krzysztofdudek/ResearcherSkill](https://github.com/krzysztofdudek/ResearcherSkill)（v1.6.0，自主实验循环：假设 → 改动 → 度量 → keep/discard，历史在 `.lab/`）。

- **Skill**：`.cursor/skills/researcher/` 与 `~/.codex/skills/researcher/`
- **重装**：`bash scripts/install-researcher-skill.sh`
- **触发**：说「进入 researcher 模式」「overnight 优化某指标」「迭代实验直到达到 N」等；Agent 会按 `SKILL.md` 建 `research/<slug>` 分支与 `.lab/` 日志
- **忽略项**：`.lab/`、`run.log`（已写入根目录 `.gitignore`）
- **完整说明**：[GUIDE.md](https://github.com/krzysztofdudek/ResearcherSkill/blob/main/GUIDE.md)

与 `planning-with-files` / `experiment-result-md-organization` 分工：Researcher 管**可度量目标的自动试错**；planning 管任务拆解；`resultmd/exp_*/` 管已定型实验的归档报告。

## 计划中（尚未实现）

以下名称曾在表格中列出，尚无对应 `SKILL.md`：`code-review`、`ml-expert`。需要时可按上节在 `.cursor/skills/` 下新增。
