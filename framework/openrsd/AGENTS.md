# OpenRSD Agent Instructions

## Shell commands (RTK)

**Golden rule:** Prefix non-interactive shell commands with `rtk` to reduce CLI output tokens.

Examples:

```bash
rtk git status
rtk git diff --stat
rtk find . -name '*.py' | head -50
```

When a Hermes Caveman wrapper exists and fits the task, prefer compact output:

```bash
rtk "$HOME/bin/caveman_wrapper.sh" git-status
# or after shell aliases: cgs, cgl, clint, ctest
```

**Exceptions (no `rtk` prefix):**

- Interactive commands (`vim`, `less`, `python -i`, long-running training you must not wrap)
- Commands that must return full unstructured output the user explicitly asked to keep raw
- Already-wrapped `rtk ...` commands

## Skills

- Project Cursor skills: `.cursor/skills/`
- **Superpowers** ([obra/superpowers](https://github.com/obra/superpowers)): brainstorming, TDD, debugging, plans, code review, git worktrees — reinstall: `bash scripts/install-superpowers.sh`
- **Planning with Files** ([OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files)): Manus-style `task_plan.md` / `findings.md` / `progress.md` — reinstall: `bash scripts/install-planning-with-files.sh`
- **Hermes RTK+Caveman** ([hermes-agent-rtk-caveman](https://github.com/adityahimaone/hermes-agent-rtk-caveman)): reinstall: `bash scripts/install-hermes-rtk-caveman.sh`
- Legacy notes: `skills/readme.md`
- Codex global skills: `~/.codex/skills/`

### Superpowers workflow (when building features)

1. `brainstorming` — clarify design before code
2. `using-git-worktrees` — isolated branch workspace (if applicable)
3. `writing-plans` — bite-sized implementation plan
4. `subagent-driven-development` or `executing-plans` — execute with review
5. `test-driven-development` — red/green/refactor
6. `requesting-code-review` / `finishing-a-development-branch` — wrap up

User instructions in this file override skill defaults when they conflict.

## Dataset paths

- User-confirmed fact on 2026-06-18: datasets previously under
  `/data1/zcy/OpenRSD/data` were accidentally deleted. Do not treat
  repository-local `data/` as a valid dataset source.
- For future OpenRSD experiments, search for datasets only under
  `/data1/zcy/datasets`.
- When DOTA v2 is needed, re-extract raw archives and rebuild the 1024/500
  sliced dataset under `/data1/zcy/datasets`, then pass absolute dataset roots
  into configs or command-line overrides.
- If an existing config references `data/...`, override it with an absolute
  path under `/data1/zcy/datasets` instead of following the repo `data` symlink.
