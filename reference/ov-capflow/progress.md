# OV-CapFlow 72h Progress Log

## Session: 2026-07-21 (Asia/Shanghai)

### Phase 0: Protect and Audit Current State

- **Status:** in_progress
- **Started:** 2026-07-21
- Actions taken:
  - 完整读取用户粘贴的 72 小时 ICLR 实验目标与限制。
  - 确认系统已有 active 持续目标，未重复或覆盖。
  - 读取 `using-superpowers`、`planning-with-files`、`vibe-research-workflow`、`distributed-training-nccl-a40` 与 `coding-agent-eight-honors` 约束。
  - 运行 planning session catchup；无待恢复输出。
  - 列出项目根目录；未发现根目录 `AGENTS.md` 或既有 planning files。
  - 建立 `task_plan.md`、`findings.md`、`progress.md`。
  - 核验 Git：记录当前分支、确认无 tracked diff、保护既有 untracked 脚本/测试/文档。
  - 阅读 root README；判定其为上游 OVAD 项目背景，不是当前 T7 严格实验权威记录。
  - 枚举 `.lab` 并定位 d133–d143 与 E24 相关的高价值状态/审计文档。
  - 阅读 `.lab/config.md` 与 `summary.md`：确认 E18 authority、plateau、E24 last-authority/no-new-train gate、1280 路线关闭及 HRSC parent-preserving 先验。
  - 阅读 `.lab/parking-lot.md` 并记录 2026-07-21 路线优先级覆盖：E24 证据决定 quality/ranking 或 geometry；D11/D12 不再默认优先。
  - 开始读取 `.lab/log.md`；已到 Experiment 6 中段，尚需续读到 EOF。
  - 续读 `.lab/log.md`：记录 DOTA2 C1 coverage-driven gain 及 duplicate/empty tradeoff、P126C 不合规强锚点、Experiment 8 S0 engineering-only 证据。
  - 首次直接读取 `.lab/results.tsv` 因体量过大被截断；后续改用 line count 与最新尾部/定向过滤。
  - 核验 `.lab/log.md` 6,640 行、`results.tsv` 403 行；从最新结果提取 E18 oracle/headroom、scale1280 exact FP/matched decomposition、query/ref prior 与 E24 priority reconciliation。
  - 完整读取 clean-start AP70 design 与 `full24e-protocol.md`；区分早期 scale800 路线和当前 scale1024+rare4x T7，并登记 GPU/E25+ supersession。
  - 阅读 GPU2389 continuation：核验 E6→world4 checkpoint tensor/optimizer 等价、sampler 完整、首次漏 `--resume` 无效尝试和当前有效 stochastic continuation。
  - 完整读取 E24 autoschedule plan；核对四门 launch、duplicate protection、CPU validator、8,299,800-row contract 与实时 tmux command 一致。
  - 分两段完整读取 567 行 daily summary；登记 grouped seed-sensitivity、novel/coverage tradeoff、proxy→raw -0.098 AP50 与 empty foreground 恶化反例。
  - 分两段完整读取 199 行 GPU2389 continuation；登记 rare4x proxy 因果边界、raw组合不可分解、proxy zero-GT mouth 与历史 D11 priority 已被覆盖。
  - 读取当前 GPU2389 resume config 与 E24 non-resume eval base；确认 batch/accum/resume/work_dir/scale1024/raw mouth，开始沿 base chain 展开协议。
  - 读取 full24/grouped base：确认24 epochs、1/6/12/18/24 milestones、raw mouth、grouped training-only delta；继续追到 S1 control。
  - 读取 S1 control/S0 config；确认核心模型/推理合同全部继承自 `q600_base.py`。
  - 读取 q600 base 并定向扫描：确认 Q600、动态文本分类、18/14/4 split、empty tiles、empty test_cfg；继续向 c0/model code 展开。
  - 读取 c0 native base：确认 IoU0.5 DOTAMetric、non-roundup raw val、禁用旧 intervention；配置链审计完成，转代码级 predict/export。
  - 逐行读取 `ov_capflow.py` 与 `ov_capflow_head.py`：确认 group/DN 仅训练、eval Q600、逐query类别选择、无query排序裁剪或NMS。
  - 读取 calibration 与 strict auditor：确认 class-only max、per-query calibration、AST/runtime forbidden-call hooks；记录 auditor 的一般 sort/slice 盲点由代码审读补足。
  - 枚举 audit artifacts 与 focused tests；定位 grouped strict/OV/alignment 证据，未发现 T7-specific strict JSON，计划用现有结构产物和 CPU/config tests补强。
  - 读取同构 grouped strict/OV artifacts：strict pass with exact 600 and zero forbidden calls；OV pass/state shapes unchanged，待核对 variant tail。
  - 读取 grouped decoder alignment：prediction 与 decoder 逐行同序、误差0；OV tail 显示19-class仍600 rows，待定向取 variant names。
  - 按 conda selection skill 核验环境；选择现有 `mmdet` Python 3.8.19，验证 Torch/MMEngine/workspace mmrotate，禁用 user site。
  - 运行 CPU-only focused tests：51 passed；确认四 prompt variants 均固定600 rows，无固定分类层/禁用 checkpoint keys。
  - 完整读取 E18 plateau JSON 与 E1/E6/E12/E18 curve audit；核验 exact metrics、checkpoint hash、group/class tradeoffs、same-route dump缺失和E24 routing gate。
  - 完整读取 E24 same-route prediction dump runbook；登记 queue 单一所有者、启动/完成门槛、CPU-remap validator、诊断顺序与按证据分支的唯一候选规则。
  - 完整读取 D141 quality-priority reconciliation；确认早期 D11-first 已撤回，quality/coverage/D11/D12/prompt 各自有互斥的 canonical E24 准入证据。
  - 完整读取 D141 AP70 headroom concentration 与 E18 allocation/init-priority audits；量化 +0.091833 gap、49.703% oracle headroom、弱类 84.645% gap 负担、airport allocation hub，并登记 D11/D12 结构事实但按后出 reconciliation 撤销其旧优先级。
  - 完整读取 D142 prompt/literature audits；登记 hyphen allocation signal、合法 alias-prompt A/B 口径和 +0.005/组退化门槛，并把本地论文清单标为等待独立原始来源复核，禁止直接形成 novelty/first claim。
  - 完整读取 D143 scale1280 pre-registration/result；确认路线按门槛关闭、SV 仅局部 AP/recall tradeoff、dump CUDA-GT 便携性问题，并声明历史 GPU4/5 使用不覆盖当前仅2/3/8/9合同。
  - 依 Real Literature Trace 做 2021–2026 原始来源检索；官方核验 Stable Matching/Rank/Align/DQ/DetCLIP/PET-DINO，并发现 PaQ-DETR 已正式发表于 CVPR 2026。
  - 扩展最新近邻：CVPR 2026 HSA-DINO、ViTPrompt 与 2025 VK-Det；更新 novelty 风险，撤销所有宽泛“query+quality/prompt/航拍 first”可能表述。
  - 完整读取 D139 cross-scale/center/shape audits；确认 mixed topology 跨尺度稳定，否定“tiny中心离网格远”，并把 geometry 假设收缩到 41px monomorphic prior 的 refinement distance 与高密度局部竞争。
  - 完整读取 D139 query trajectory、matched trajectory、exact evaluator FP decomposition 与 Q600/quality代码差距；确认 query-id 对齐、decoder可大幅refine且无dead queries、micro几何失败与macro排序双瓶颈、AP-support FP构成及binary token Focal的真实缺口。
  - 阅读 project README 与三图 pipeline design；识别 README 状态已陈旧，冻结 Figure1 active T7 / Figure2候选架构 / Figure3机制公式的证据边界和现成出版规范。
  - 17:14复查 runtime：E21 2820/5916、四卡100%、ETA9:36:48；只读确认 E24 queue tmux pane alive，未启动任何新任务。
  - 复核 Codex goal 仍为 active；当前无 generic automation-update capability，沿用 active goal + monitor + guard + gated queue 的无重复监控拓扑。
  - 按 OpenRSD DOTA2 mouth skill 只读审计5月rotation/false-SV资产；明确6605 paper-mouth与13833 raw不可混报，P0148仅diagnostic、历史true-SV preservation仍是proxy。
  - 定向聚合orbit raw rows：P0148 mean SV ratio 0.7510→0.5798时P0682也0.9683→0.8606，且无GT recall字段；把P0682设为今后任何hub抑制候选的强安全门。
  - 阅读OpenRSD DOTA2 text-only 12-angle、MESS/Fourier与48h narrow-adaptation总结；记录rotation mean drop约0.045、大融合崩溃及8条微调全负，强化“先零训练/同mouth/no-op control”的边界。
  - 复核Git HEAD=`314c131...`，tracked diff仍为空；未stage/commit/清理任何用户已有untracked资产。
  - Goal continuation于17:27重新审计外部状态：E21 3120/5916、ETA9:29:32、monitor alive/fatal-null；E24 checkpoint/dump未出现，queue仍安全等待。
  - 盘点D139复用资产：exact matched/evaluator分析没有保留脚本，确认需在E24前把canonical CPU-only诊断工具与测试固化，避免临时手算。
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

## Runtime Snapshot

| Time | T7/E24 | GPU 2/3/8/9 | tmux/process | log growth | checkpoint | disk |
|---|---|---|---|---|---|---|
| pending | unknown | unknown | unknown | unknown | unknown | unknown |
| 2026-07-21 snapshot-1 | tmux chains present; exact E24 state unknown | 2/3/8/9 each ~40–41 GiB used | 5 T7-related tmux sessions | pending | pending | pending |
| 2026-07-21 snapshot-2 | rare4x resume active; E24 dump queue configured | mmdet workers 689012/14/15/17 | master tracked as 688939; monitor+guard active | pending | queue waits for epoch_24.pth | pending |
| 2026-07-21 snapshot-3 | Epoch 21 observed; exact latest iter pending short tail | repeated high utilization samples | process group alive; fatal_pattern null through ≥16:56 | growing | epoch_24 not yet checked | pending |
| 2026-07-21 16:56 | E21 2400/5916; ETA 9:46:49 | busy, not idle | training alive | confirmed growing | queue: unpublished/not launched | pending |
| 2026-07-21 17:14 | E21 2820/5916; ETA 9:36:48 | 2/3/8/9 each 100% | queue tmux pane alive; training active | confirmed growing | queue waits; no launch | pending |
| 2026-07-21 17:25 | E21 3080/5916; ETA 9:30:31 | ~40–41 GiB each; active utilization | process group alive; fatal null | confirmed growing | queue unchanged | pending |
| 2026-07-21 storage | checkpoint through E18 | — | — | — | no epoch_24 | 1.3 TB free, 86% used |

## Experiment Runs

| Run | Hypothesis | Variable | Config/commit | GPU/tmux/PID | Status | Result |
|---|---|---|---|---|---|---|
| None started | — | — | — | — | Phase 0 audit | — |

## Test Results

| Check | Expected | Actual | Status |
|---|---|---|---|
| Persistent goal | Existing active goal retained | Active goal read successfully | PASS |
| Session catchup | Recover pending context if present | No pending output | PASS |
| Planning files | Absent before initialization | Created without touching code/config | PASS |
| Strict/config/queue focused tests | CPU-only current code | 51 passed in 10.39s | PASS |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|---|---|---:|---|
| 2026-07-21 | `create_goal` rejected because an unfinished goal already exists | 1 | Read and retained existing active goal |
| 2026-07-21 | Physical `AGENTS.md` not found by root file search | 1 | Enforce user-provided inline AGENTS rule; continue hidden-directory audit |
| 2026-07-21 | Sandbox `pgrep` cannot enumerate host GPU PIDs | 1 | Use tmux pane metadata/output and on-disk logs instead |
| 2026-07-21 | tmux pane inspection denied by sandbox socket permissions | 1 | Re-ran the same read-only inspection with approved escalation |
| 2026-07-21 | Orbit CSV aggregation used nonexistent `sv_ratio` key | 1 | Read header, corrected to `final_sv_ratio`, reran successfully |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | Phase 0, protection and real-state audit |
| Where am I going? | E24 diagnosis → one causal route → minimal verified implementation → promotion → ICLR evidence package |
| What's the goal? | Strict DOTA2 70+/70+ plus a defensible causal and OV evidence loop |
| What have I learned? | See `findings.md`; current model/runtime facts remain mostly unknown pending audit |
| What have I done? | Read full mandate, loaded workflows, retained active goal, initialized durable records |
## 2026-07-21 17:xx — E24 analyzer pre-design

- Read the existing Q600 dump validator and its tests, plus the official rotated mAP evaluator and D140 diagnostic runbook.
- Confirmed that no reusable canonical dump-diagnostics script is present; current E24 queue/monitor remains untouched and live.
- Entered the `brainstorming` design gate. No production code or tests were added. Next action requires design approval, after which a detailed implementation plan and red-first tests will be written.
## 2026-07-21 17:33 — T7/E24 continuation check

- Located the authoritative live artifacts rather than relying on sandbox-local `pgrep`: launcher log `work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2_gpu2389_e6_resume_a1_launcher.log`, scalar JSON, monitor JSONL, queue JSONL, and guard JSONL.
- T7 advanced from Epoch 21 `[3120/5916]` at 17:26:40 to `[3260/5916]` at 17:32:47. Losses and gradient norms remained finite; ETA reported by MMEngine was 9:26:11.
- The 17:32:43 monitor sample reports PID 688939 `process_group_alive=true`, `fatal_pattern=null`, and only the authorized physical GPUs 2/3/8/9, each retaining about 40–41 GiB training memory.
- `last_checkpoint` still points to `epoch_18.pth`; `epoch_24.pth` and the E24 prediction dump are absent. The armed queue remains unlaunched and was last recorded waiting for checkpoint, validation, training exit, and GPU idleness.
- No process was stopped, restarted, or launched; no training/evaluation configuration changed.
## 2026-07-21 17:34 — blocked-audit threshold

- Rechecked the authoritative scalar, monitor, queue, and output paths. The latest scalar remains Epoch 21 / step 121580; the 17:33:14 monitor sample remains healthy (`process_group_alive=true`, `fatal_pattern=null`) on GPUs 2/3/8/9.
- `epoch_24.pth` and `eval_t7_epoch24_raw13833_gpu2389_dump/` remain absent; the gated queue has not launched.
- The proposed canonical analyzer design has now awaited explicit approval for three consecutive goal turns. The `brainstorming` skill forbids implementation before that approval, while the experimental path also awaits the external E24 completion event.
- Preserved all live processes and artifacts and marked the persistent goal blocked per the three-turn blocked-audit rule. Resume conditions: user approves design option 3, or E24 checkpoint/dump becomes available for existing-validator audits.
## 2026-07-22 00:xx — analyzer design approved and committed

- User explicitly approved approach 3 (focused analysis module plus thin CLI), resuming the previously blocked goal.
- Wrote and self-reviewed `docs/superpowers/specs/2026-07-22-dotav2-q600-canonical-dump-analyzer-design.md`.
- Self-review found and corrected an important dataset-boundary issue: historical `P0148__1024__651___0` and `P0682__1024__553___0` are cross-dataset DOTA1 crop diagnostics, not members of the DOTA-v2 raw-val 13,833 mouth. They are now an optional auxiliary case dump and cannot contaminate canonical mAP or integrity counts.
- Placeholder scan passed; staged-diff whitespace check passed; document SHA256 is `ce0e57395552472cf2dee0064f64557848f2cb8997db59f01b8c1e3a9f4f37ec`.
- Committed only the design document as `6c3c896` (`docs: design canonical Q600 dump analyzer`); all unrelated untracked assets remain untouched.
- Live T7 remained healthy and advanced to Epoch 23 / step 131152 while the design was prepared. E24 and its dump remain pending.
- Next gate is explicit user review of the written spec. Do not invoke `writing-plans` or implement code until that review is approved.

## 2026-07-22 — analyzer specification approved and implementation plan written

- User explicitly approved the written specification (`规范通过`).
- Applied the `writing-plans` workflow and created
  `docs/superpowers/plans/2026-07-22-dotav2-q600-canonical-dump-analyzer.md`.
- Split implementation into eight independently committable TDD tasks:
  protocol/input gates, exact evaluator replay, AP/GT decomposition,
  calibration/query/refinement/strata, optional historical cases,
  deterministic reporting, atomic CLI publication, and D143/E24 integration.
- The plan fixes exact paths, interfaces, synthetic red tests, selected Python
  environment, all shell commands, D143 numerical anchors, and the eventual
  canonical E24 command. It makes no production-code or experiment mutation.
- Self-review found and removed implementation placeholders, corrected the
  refinement-equation fixture, made path-based dataclass imports valid,
  added missing/ambiguous metric and checkpoint gates, covered every fixed bin,
  and verified every shell command in the plan starts with `rtk`.
- Next gate is execution-mode selection: subagent-driven task-by-task execution
  or inline execution in this task.

## 2026-07-22 — Q600 analyzer Task 1 complete

- User selected Subagent-Driven execution. Task 1 was implemented with strict
  red-green TDD and independent specification/code-quality reviews.
- Added the frozen diagnostic records plus class-protocol, JSONL metric,
  checkpoint-reference, parity, and CPU record-preparation gates.
- Review loops found and fixed duplicate base/novel definitions, invalid metric
  step types, nonfinite metric/parity bypasses, and nonfinite/non-real reference
  tensors. The row-order test now uses nonmonotonic scores and a genuinely
  noncontiguous source tensor.
- Focused result: 33 tests passed. Final spec and quality reviews reported no
  remaining issues.
- Task commits: `ca621549`, `3fae74d0`, `e119b838`, `d9b1fb44`.
- T7 was observed healthy in Epoch 24 at `[400/5916]`; `epoch_24.pth` and the
  E24 dump were still absent at that snapshot. No live process was changed.

## 2026-07-22 — Q600 analyzer Task 2 complete

- Reconstructed the exact CPU rotated evaluator with immutable query IDs,
  ordinary-before-ignored matching, local/global NumPy ordering, six exclusive
  row outcomes, MMDetection VOC07 AP, and official empty-class mean behavior.
- Review hardening added discriminating default-vs-stable tie coverage,
  mixed ordinary/ignored argmax and coverage cases, official `tpfp_default`
  parity, auxiliary scatter/original-GT-index assertions, and scalar-safe tuple
  image IDs.
- Focused result: 41 tests passed. A quality-review randomized audit matched
  official TP/FP, precision/recall, AP, and mAP in 30/30 cases; a 100-image
  canonical-shape sample took 1.737 seconds, with no measured scale blocker.
- Final spec and quality reviews reported no remaining issues.
- Task commits: `24e87d79`, `7ff5ceae`.

## 2026-07-22 — Q600 analyzer Task 3 complete

- Added exact VOC07 support ranks, class AP-support endpoints, a fixed-recall
  perfect-ranking oracle, four mutually exclusive GT reachability states,
  compact witness summaries, and fixed-Q global/per-class capacity evidence.
- Ownership is scoped to ordinary GT of the same label; ignored GT never enters
  reachability. The oracle preserves evaluator-neutral `[0,0]` rows.
- Review hardening now rejects contradictory global/per-class capacity counts,
  impossible state/witness combinations, complex/unbounded/nonmonotonic PR
  curves, and complex TP/FP inputs. Empty prediction, zero-GT, and witness-tie
  cases have direct regression coverage.
- Focused result: 81 tests passed. Final spec and quality reviews reported no
  remaining issues.
- Task commits: `d6028c4c`, `a6bc36af`.

## 2026-07-22 — Q600 analyzer Task 4 complete

- Added deterministic bounded-memory stable sampling, tie-aware Spearman,
  calibration summaries, per-query diagnostics, normalized refinement deltas,
  and the complete fixed strata matrix.
- The specification review found and fixed floating-point instability at exact
  periodic angle-bin boundaries. The code-quality loop then hardened mixed-type
  image-ID tie breaking without changing the frozen primary SHA payload, and
  enforced prior/reference normalization plus full GT/witness evidence alignment.
- Focused result: 185 tests passed in the independent final re-review. Sampling
  remains `O(N log sample_size)` time and `O(sample_size)` memory, with no extra
  per-witness rotated-IoU calls.
- Final specification and code-quality reviews reported no remaining Critical,
  Important, or Minor issues and explicitly approved progression to Task 5.
- Task commits: `64f01a2`, `c974ae8`, `149f50ef`.

## 2026-07-22 — Q600 analyzer Task 5 complete

- Added the optional two-image P0148/P0682 case path with an exact immutable
  manifest contract, explicit `not_run` state, exact-evaluator outcome and
  AP-support attribution, true-SV reachability witnesses, and false-SV nearest
  ordinary-GT evidence. The path remains isolated from canonical records and
  mAP by construction.
- The specification loop added stable top-1/5/20 truncation tests, high-valued
  SV class inference, and fail-closed image-ID handling. The quality loop then
  rejected duplicate/aliased/adversarial manifests, canonicalized equal-score
  case record order, hardened shared real-valued score validation, and added
  discriminating cross-class/tied-nearest-GT tests.
- Independent final result: 233 focused tests passed. Prepare, evaluate, and
  decompose run exactly once for supplied cases; no remaining Critical,
  Important, or Minor issues were reported.
- Task commits: `391c854b`, `c09988a4`, `9fea2ed4`.

## 2026-07-22 — Q600 analyzer Task 6 complete

- Added detached finite-only diagnostics assembly, deterministic per-type error
  witness capping, five fixed CSV schemas, canonical JSON/CSV/Markdown bytes,
  and a manifest that hashes exactly the seven non-manifest payloads.
- Review hardening rejects duplicate retained witness keys, custom numeric
  coercions, negative oracle headroom, malformed report schemas, recursive or
  over-deep inputs, invalid case-gate states, and adversarial Mapping/iterator
  failures. Complete case rows are normalized to manifest order.
- Error capping is now a single-pass bounded heap with `O(N log K)` time and
  `O(K)` persistent memory. The final error table retains an exact duplicate
  gate while the streaming cap avoids any all-row tuple or set.
- Independent final result: 374 focused tests passed; no remaining Critical,
  Important, or Minor issues, and explicit approval to proceed to Task 7.
- Task commits: `e6d45a7e`, `d0a02576`, `0c004ef3`, `b8093d50`.

## 2026-07-22 — Q600 analyzer Task 7 complete

- Added the CPU-only CLI with strict canonical/noncanonical mode gates,
  complete D143/E24 summary assembly, provenance hashing, isolated auxiliary
  cases, deterministic manifest construction, and eight-file atomic publish.
- Atomic review hardening covers no-replace races, dangling symlinks,
  post-rename durability failures, failure-only expected-error bundles, and
  zero input reads when the destination is already occupied.
- Scientific gate hardening adds explicit optimized-interpreter-safe record and
  row counts, canonical case 2xQ checks, pre-load/post-load path+SHA+size input
  binding, unwrapped resource-exhaustion errors, replayable direct-run commands,
  and complete programmatic Namespace validation.
- Independent final result: 446 focused tests passed, two `python -O` gates
  passed, direct help exited zero, and no Critical/Important/Minor issues
  remained. The CLI/core introduced no model, CUDA, training, or queue work.
- Task commits: `3a5cc971`, `0560cc52`, `78b30a02`, `84ac1ee7`.

## 2026-07-22 — Q600 analyzer Task 8 and canonical E24 gate complete

- Fresh analyzer/validator/queue/GPU-guard regression passed with `477 passed`.
- D143 full 13,833×600 CPU replay reproduced all frozen numerical anchors;
  the corrected compact/replayable bundle manifest is
  `49f5ccd054f1adc1219f5089a00e84a35dcfca6b13ae83bd02967215b59b6dd9`.
- Diagnosed the armed E24 queue stall before mutation: whole-cmdline substring
  matching treated the tmux server's historical shell command as a live
  `tools/train.py` rank. A strict argv-token matcher was implemented under TDD,
  independently specification- and quality-reviewed, and verified with 27
  queue/guard tests before replacing only the original queue session.
- The original queue launched once on physical GPUs 2,3,8,9, exited zero at
  raw `dota/mAP=0.6064`, `dota/AP50=0.6060`, wrote a 433,478,685-byte dump,
  and validated 13,833 unique records / 8,299,800 finite Q600 rows before
  recording `queue_complete`.
- Canonical E24 replay passed: exact mAP `0.6064053488274416`, same-dump error
  `2.856055891786724e-08`, training delta `0.0`, base14 mAP
  `0.6540175995656422`, novel4 mAP `0.43976247124373913`, and
  `case_gate=not_run`.
- Final review found two output-contract defects despite correct science: the
  report duplicated 500,000 sample keys/query rows and the manifest command
  lacked an executable interpreter/cwd. Strict RED→GREEN repair commit
  `abd860d157d562aa9d30422c05e41e246effc8a7` reduced the canonical report to
  18,939 bytes and made main/direct/failure commands replayable.
- Full D143 and E24 reruns preserved every scientific value. The final E24
  manifest is
  `bd1ce0a7b98a9e03f4b647fe194290bab4c229225e53304c270b2fc3d4166dfe`;
  the superseded bundle and one failure-only backup-scan artifact remain
  preserved under `.lab/workspace/exp-8-d144/`.
- The experiment ledger records input provenance, old/new manifest hashes,
  compact report, D143 hash, and explicitly retains no-E25+/no-new-training.
  Final independent review reported no Critical/Important/Minor issues and
  approved closing analyzer Task 8. The broader ICLR/AP70 goal remains active.

## 2026-07-30 22:xx — D11 matched proxy closed as a valid discard

- Recovered the post-reboot state without changing any process or artifact.
  Physical GPUs 0–9 were idle; both D11 world3 v2 training panes had exited
  zero after completing epoch12.
- Diagnosed the postrun pane exit status 1 as the queue's intentional
  scientific-discard return contract. The state log reached `queue_complete`,
  and all four candidate/control strict/open-vocabulary audits were published
  successfully.
- Frozen endpoint: candidate/control `mAP=0.4521/0.4660`,
  `AP50=0.4520/0.4660`, `novel4=0.26425/0.32475`. D11 is discarded with no
  retune, rescue, extra epoch, best-checkpoint cherry-pick, or stacking.
- Added the append-only lab result/log and Chinese result record. All
  checkpoints, console logs, gate outputs, dead tmux panes, and unrelated
  untracked assets remain preserved.

## 2026-07-30 22:30 — D12 approval gate and novelty audit

- No D12 implementation, config, checkpoint mutation, or training was started:
  the proposed matched 5-GPU candidate versus 5-GPU control design is still
  waiting for explicit user approval under the active brainstorming gate.
- Used the idle interval for a read-only 2021–2026 primary-source audit of the
  claimed novelty boundary. The broad claims “first aerial OV detector,”
  “first oriented aerial OV detector,” and “first end-to-end aerial OV
  detector” are all already occupied by prior work.
- The surviving working position is substantially narrower: teacher-free,
  pseudo-label-free, fixed-Q600 rotated-5D direct set prediction with all-query
  scoring and no inference NMS/proposal top-k. This remains provisional until
  a full systematic first-claim search is complete.
- Key protocol collisions were recorded: RT-OVAD is end-to-end RT-DETR with
  top-500 query initialization but 4D boxes; OpenRSD supports OBB but is
  RTMDet-R with self-training and class-agnostic NMS; oriented CastDet is
  RPN/RoI plus multiple teachers and pseudo-labels; SOAR is query-based but
  generates 4D pseudo boxes.
- Physical GPUs 0–9 remain deliberately idle while approval is absent. No
  files were deleted, overwritten, reset, or cleaned.

## 2026-07-30 22:32 — third consecutive D12 approval hold

- Rechecked authoritative runtime state: every physical GPU 0–9 is at 17 MiB,
  0% utilization, P8; every A0/D11 tmux pane is dead; no new training,
  checkpoint, queue, or external process has appeared.
- Rechecked the worktree: it still contains only the protected untracked
  assets already enumerated; no tracked diff and no destructive operation.
- This is the third consecutive goal turn with the same explicit-design
  approval missing. Further mAP work would require creating and executing the
  D12 checkpoint/config pair, which the active brainstorming gate forbids
  before explicit approval.
- Persistent goal is therefore placed in blocked/waiting-for-user state, not
  completed. Resume trigger is the exact user response `批准 D12`; after that,
  run the frozen preflight and, only if all gates pass, launch the matched
  GPU0–4 versus GPU5–9 proxy.

## 2026-08-01 — Holistic QAF source preflight freeze

- The holistic Query Allocation Quality Flow design is frozen; training is restricted to physical GPUs 8 and 9.
- D13-N closed at the registered E12 endpoint: 0.838305831 candidate versus 0.847486973 control, delta -0.009181142.
- Active hypothesis: one query-to-image evidence coupling jointly governs center transport, null validity, matching/training quality, and all-row score calibration.
- Current authorized action: frozen E24 stride-32 source preflight only.
- Hard stop: no production model/config code and no distributed training before all five source gates pass.

## 2026-08-01 — Holistic QAF source preflight FAIL; Approach A closed

- Immutable source report: `.lab/workspace/exp-8-qaf-source-v1/source_report.json`; external report SHA256: `d51f88a677af48ddbdc0455f4300452bcaf51e44bbddec7c22176555b034ca33`; status: `FAIL`.
- Frozen reachability observations: overall `0.07178882557149407`; base14 `0.0717736256898161`; novel4 `1.0` (`total=1`, so this stratum is reported faithfully but is not strong evidence); spatial shuffle `0.07262396017554201`; semantic shuffle `0.07183795113643807`.
- Failed gates: `reachability_at_least_25pct`, `beats_prior_e_by_3pp`, `placebo_sensitivity`. Passed gates: `finite_reproducible_provenance`, `novel_base_gap_at_most_10pp`.
- The primary result and both placebos are effectively indistinguishable; primary reachability is far below `0.25` and below prior E plus 3 percentage points. The frozen source hypothesis therefore lacks the required causal sensitivity and coverage.
- Approach A / Query Allocation Quality Flow is closed for the current source family. Plan 2 is not executed.
- Not authorized: any production module or config, proxy/raw/full training, scale rescue, or alternate evidence level, threshold, seed, or subset. The physical-GPU restriction remains GPU 8 and GPU 9 only, but this closed branch performs no training.

## 2026-08-03 — persistent experiment continuation started

- 恢复 `task_plan.md`、`findings.md`、`progress.md`，确认 E24 strict raw 与 D11/D13-N/QAF 负结果仍是当前证据基础。
- 按最新用户指令将资源范围更新为物理 GPU 0–9、值守至少 18 小时；安全边界保持不变：不删除文件、不杀现有任务、不覆盖产物。
- 已启动一个只读论文整体性审阅子代理；主代理继续核查 GPU、进程、今日实验记录、环境与候选队列。
- 首次运行态审计确认 GPU 0–9 全部空闲，无 compute app；未发现 live `train.py`/`torchrun`/eval 进程。保留所有历史 tmux session，尚未做任何启动、终止或删除。
- `rtk conda env list` 与 `rtk find -newermt` 分别因 wrapper 解析/子命令能力失败；已记录错误，将改用已知环境绝对路径和 GNU find 绝对路径，不重复失败方式。
- 使用 `/usr/bin/find` 核到今天新增的 OMQ source 与 M1 matched 10-GPU 产物。下一步改为聚焦读取 OMQ source report、candidate/control 日志和 audit JSON；在判定 OMQ 状态前不另起重复实验。
- OMQ-M0 source gate 通过且 placebo separation 强；OMQ-M1 E1–E10 matched proxy 仅有最大 `+0.0006`、E10 `-0.0003`，尚不够晋升。两臂在无错误信息下停于未完成 E11；准备先审计 config/commit/启动命令，再从共同 E10 安全成对续到冻结 E12。
- 更正上一条的暂定判断：新发现的冻结 actionability design 明确 M1 在用户指示下以十个完整验证 epoch 关闭，未完成 E11 不构成需要恢复的 endpoint。已将 M0/M1 与下一 actionability gate 追加到 `.lab/results.tsv`、`.lab/log.md`、`.lab/branches.md`；所有 checkpoint/log 保留。
- 在隔离 `research/omq-m0` worktree 上复核 M0/M1 基线：`31 passed`；Task 2 纯 actionability 数学核心已交独立实现代理按 TDD 执行，GPU 0–9 继续空闲等待 capture 阶段。
- 已按 OpenRSD worklog 规范追加 start 记录并定位今日 CSER Phase-1 资产。首次广域 `find` 误纳入 `data_local` 导致输出过大；已停止该扫描，改为只读明确文件，不影响任何实验产物。
- 已核验 OpenRSD CSER control/candidate `0.7313/0.7314`、delta `+0.0001` 与 `HOLD` 决策；该口径是 DOTA1 S2 `2497/2500` + visual 8-shot + RTMDet/NMS，不并入 OV-CapFlow raw DOTA-v2 mAP 排名，只作为压缩证据机制的跨项目负证据。
- 找回昨夜 ODQ full-E6 的真实产物并完成首轮对账：E6 candidate `0.540336192` 明显低于冻结 T7 E6 control `0.553674161`，mAP delta=`-0.013337970`，未过 `+0.010` 晋级门；8-GPU 运行本身健康完成。ODQ 关闭，不续 E12/E24，待把正式负结果补入 `.lab`。
- 论文一致性子代理发现 frozen S1 存在原始场景/像素泄漏；主代理已核对 subset builder 确实只按 tile stem 切分。立即暂停 actionability capture：Task 2 纯数学实现仍可完成，但 Task 3 与 GPU 启动必须先有 scene-disjoint、zero-pixel-overlap split 和复算 manifest。
- 主代理独立复算泄漏计数与 SHA 后，将 OMQ actionability 正式记为 `STOP_CONTRACT_FAIL`：当前冻结协议按其自身规则关闭，Task3/M2 均取消，不用 post-hoc scene-disjoint rerun 救援。ODQ full-E6 负结果和 OMQ split 合同失败已追加到 `.lab/results.tsv/.lab/log.md/.lab/branches.md`，未删除或覆盖任何训练产物。
- 下一阶段完成只读可行性收敛：Swin-B 能在当前环境构建（232.909M 参数），可作为“先冲 raw mAP”的非创新容量锚点；真正 ICLR 方法候选收敛为 strict base-only 下的 incomplete-label negative-safe fixed-set prediction。正在写两阶段冻结设计，按 brainstorming 约束获批前不下载权重、不写实现、不启动 GPU。
- OMQ Task2 收尾完成：独立实现 commit `06d070a` 只含两份预期代码/测试，主代理 fresh 重跑 `69 passed` 并通过编译/diff 检查；已追加 `.lab` 为 `keep-infrastructure-only`。Task3/M2 仍严格关闭。
- B0 + negative-safe 两阶段设计已写入 spec 并经论文一致性子代理三轮修订后最终 PASS。当前授权边界仍要求用户显式批准：尚未下载官方 B 权重、未实现 config/test、未构造数据、未启动 GPU；GPU0–9 最后快照仍各 17 MiB 空闲。
- 用户明确授权后续自行审批并整夜真实实验。已创建隔离 worktree/branch `research/swinb-b0`，写入执行计划，委派 config+test TDD；官方 895 MiB GroundingDINO-B checkpoint 正从 GitHub release 下载到全新 `.part`，完成哈希前不会改名或用于模型。
- Task1 静态审计发现原 world10×batch1/acc3 对 47,294 exposures 必然产生空 rank；已在任何训练前自批并冻结为 world10×batch2/acc1/update_multiple1/with_cp=True，保持全量无 pad/drop/意外重复。配置代理已收到修订，后续真实 Stage0 决定是否可运行。
- 官方 B 权重下载、哈希与 payload 检查完成并安全改名；实测 SHA `46270f7a…ac6ab7` 已回传配置代理。batch2 amendment 计划在补齐三步十卡/global14 tail smoke 后获论文守门最终 PASS。
- B0 Task1/Task2 完成：配置TDD commit `9e6c504`，主代理 fresh `76 passed`；官方 B compatible/provenance 生成并严格 allowlist 审计通过，`.lab` 已登记 D168。下一步进入单卡真实 sparse/dense/GT>600 显存与十卡连续3步 DDP Stage0。
- 单卡真实 batch2 forward/backward/Q600-predict 已通过 `3 passed`。其 dataloader 写出的 world1 epoch0 audit 被完整保留；为防正式 world10 no-clobber 冲突，TDD commit `448a153` 将 formal audit/compatible/provenance 迁到全新 r2 路径。实际GT-aware world10 plan复算为2366 steps、global17–20（非synthetic的14）、local1–2、零遗漏/重复；`.lab` D169追加更正。
- 用户授权常规步骤自行审批、整夜值守后，重新核验 GPU0--9 均空闲、B0 formal work_dir 与 epoch0 audit 均不存在；source/compatible/formal-provenance SHA256 分别复算为 `46270f7a…ac6ab7`、`86a7d0b9…e7056`、`f84c476e…64e1e`。论文守门再次 PASS，并将当前/B0 条件/strict-OV 方法三层结论边界固化到 `findings.md`。
- B0 单卡真实 Stage-0 commit `6144179` 经主代理 fresh 回归后在物理 GPU0 完成：三类注册 case 连续更新和 Q600×5 推理均 PASS，最坏 reserved=`37476 MiB`、余量=`18.65%`；不可变报告 SHA256=`01d42543…c502`。GPU0 已释放，下一项是十卡连续三 update DDP 烟测。
- B0 十卡 r2 烟测在首个 backward fail-closed：10/10 ranks 同时报 parameter index325 ready-twice，未发布 PASS、未启动 formal；日志 SHA=`6a87e69e…f9c9`，GPU0--9 已释放。按 systematic-debugging 只授权 `static_graph=True` 的最小 runtime compatibility 修订并重复同一烟测，不改模型/数据/优化器/门控。
- 单变量 `static_graph=True` 修订、b0r3 provenance 重封与最终 config 同哈希单卡/十卡复验全部 PASS。正式 r3 已在 tmux `b0_swinb_r3_formal_20260803` 启动，独立 monitor 每60秒记录 GPU/进程/fatal；epoch0 audit SHA=`9057d97d…d263c`，E1 已到 iter40/2366、loss=`50.0203`、无 fatal，继续到 exact mAP `0.4000` 门。
- 正式 r3 已到 E1 iter320/2366，launcher PID 391764 仍存活，十卡持续参与且 monitor fatal-null。一次 `nvidia-smi` 代理采样显示 GPU5 `42816/46068 MiB`；因真实 preflight 的 `torch.max_memory_reserved` 门已通过且没有 OOM/fatal，论文守门裁定 CONTINUE，并把“不得声称正式全程 >=10% headroom”加入披露。
- commit `eb2bde8` 的自动门控脚本首轮独立复审给出 5 项 Important；已按 TDD 修复 watch 终态/liveness、严格 journal 状态机、no-follow authority、durable-before-action 单 owner 与 send-point identity TOCTOU。修复完成并再次复审前不启动 `--enforce --watch`，独立只读 monitor 继续整夜值守。
- 已将中英双语论文结论骨架、B0 五门条件句、N0/N1/sealed-novel4 证据责任和禁止性边界固化到 `docs/superpowers/specs/2026-08-03-ov-capflow-icl-paper-conclusion.md`，SHA256=`7d96bbc8…b1a17b9`。当前中心 takeaway 是闭集容量不能替代独立、无泄漏的 open-vocabulary 监督机制验证。
- 守门器修复 commit `6fcd0982` 已完成双重验证：主代理 fresh `45 passed in 3.17s`、编译/diff PASS，独立复审 0 Critical/Important。enforce watch 已启动于 `b0_swinb_r3_gate_supervisor_20260803`（tmux `$42/%52`，pane PID403395，Python PID403398），精确绑定正式 `$40/%50`、PID391764 与同一 config/log/metrics inode；启动时决策 JSONL 不存在，receipt SHA=`6a2f3d2b…1db3a`。
- Task6 scene-only spec 经论文审查 PASS，M0 builder 最终 commit `6822133b` 由主代理 fresh `28 passed` 并获独立复审 PASS。真实 metadata-only M0 在2.6秒内完成、输出11MB：`COMMITTED.json` SHA=`2ae10b3a…8140d`，manifest=`0c2382ed…d13aa`，47,294行inventory=`bafb17b7…09ba8`；没有打开图像内容/header/annotation/prediction/metric，未影响live B0。M1 65GB内容hash延后到B0释放共享路径。
- B0 formal r3 E1 完整训练/验证闭环：checkpoint `epoch_1.pth` 为2,800,650,093 bytes、SHA=`e55d1e4d…95388`；完整 13,833-image × Q600 验证给出 exact mAP=`0.44958066940307617`、AP50=`0.4500`，通过预注册 `0.4000` 门。独立 dataset-mouth/sampler audit 均 PASS，supervisor 原子记录不停止并进入 E2。已生成 E1 authority SHA=`5f2873ac…d493b`，更新论文结论骨架 SHA=`ff0bac0e…a470`，只允许“继续至 E6”句式。
- 论文一致性代理对 E1 authority/结论做独立只读复审：0 Critical、0 Important、2 Minor，整体 PASS；独立确认 18 类 det 合计 `8,299,800=13,833×600`、E6 effective threshold/required gain 算术、gate journal 单条 durable PASS 与所有 SHA。两项 Minor 已收紧：novel4 seal 明确定义为对 N0/N1 选择/阈值/控制/判定封存；dataset-mouth JSON 不再被单独描述为完整 runtime call-graph audit。M0 不污染 seal。
- E2 中段 iter1260 再次启用论文一致性代理做只读巡检：PASS，0 Critical/Important/Minor。巡检确认当前仍只有 E1 `0.4495806694/0.4500` 是新增科学证据；E2 loss/grad/throughput/memory 仅作运行健康，不预测 E6；B0 closed-set/non-causal、future N0/N1 novel seal 与 M0 preparation-only 边界完整。monitor fatal-null，未出现超过既有披露的新运行时风险。
- E3 早段 iter280 再次间隔启用论文一致性代理：PASS，0 Critical、1 Important、0 Minor。E2 未在非里程碑处验证/存档而直接进入 E3 符合冻结 cadence；唯一 Important 是 N0/N1 matched controls、scene-disjoint base14/meta-novel、sealed novel4 与 closest-work gate 仍未完成。故 B0 即使最终达到 0.70 也只能是 closed-set anchor，不能支撑 innovation、strict-OV 或 first claim；继续原样到 E6。
- 用户随后明确改令“训完 E3 就停止并总结昨晚”。E3 完成2,366个update，09:21:58出现post-epoch hook；09:22:35仅对精确绑定的正式pane `%50` 发送一次Ctrl-C，launcher一秒内退出，GPU0--9均释放至17 MiB。launcher中零条E4日志，但因20步日志间隔，不能证明绝对零个未记录E4 optimizer step。E3非1/6/12/18/24里程碑，故无E3 checkpoint/mAP；唯一科学指标仍为E1 exact mAP `0.4495806694`。monitor以`all_ranks_dead`、357样本正常收尾，十卡中位利用率均100%；gate的`FATAL target_dead`是用户停训检测事件而非训练崩溃。完整权威记录见 `.lab/workspace/exp-8-swinb-b0-r3/user_stop_after_e3.json`。
- 最终论文一致性复审 PASS（0 Critical/0 Important/2 Minor），两处 Minor 已修复：all18 novel4 只称 closed-set diagnostic；E1-only 停止明确“尚未回答容量上限”，且三 epoch 工程稳定性不能补足 ICLR 方法主张。修订后结论 SHA=`71e97518…e90e408`。
- 已按用户要求把昨夜经验和未完成项目保存到 `.lab/workspace/exp-8-swinb-b0-r3/RESUME_NEXT_GPU_WINDOW.md`（SHA=`781ed2ee…d9d89`）。恢复边界为：唯一 checkpoint 是 E1，必须重做 E2/E3；旧 zero-based audit 00--03 已被 no-replace 占用，故新 run 必须换 audit namespace，并使用 `--resume --cfg-options load_from=None`。P0 B0 E6、后续容量 gates、Task6 M1、N0/N1、sealed novel4、closest-work 与论文收口均已排成续跑队列；本次未启动训练或评测。
- 用户随后授权仅用空闲 GPU8/9 重跑，并将 E6/E12 选择交给代理；选 E6 作为第一个预注册容量门。world2 实采样六轮均零遗漏/重复，batch2×world2×acc5 名义 effective20；E1 checkpoint 经真实 epoch length 重基准到 iter11835，1,077 model 与3,127 optimizer tensor digest 不变。r1 正确恢复但复现 PyTorch1.12 `static_graph+reentrant checkpoint+no_sync accumulation` 的双 rank SystemError，未产生 logger row/checkpoint/科学结果。
- TDD 单变量修复 `AlwaysSyncOptimWrapper` 通过16项测试并提交 `7cecee7`：每个 microstep 做 all-reduce，但仍按1/5缩放、每5步 optimizer.step。r2 在 tmux `$45/%55`、GPU8/9 正确恢复 E1→E2，并通过 `E2 [20/11835]` 短健康门（loss10.5342、grad110.8393、finite、两卡100%、无fatal）。任务现无人值守运行到 `max_epochs=6`，届时自动完成 E6 raw validation/checkpoint 后退出；没有部署长时 monitor、supervisor 或 auto-resume。

## 2026-08-12 — Three-month retrospective and derivative detector study

- **Status:** in_progress
- **Started:** 2026-08-12 12:06 CST
- Actions taken:
  - Loaded the applicable worklog, experiment-summary, planning, brainstorming, idea-optimization, and literature-search skill constraints.
  - Confirmed `/data1/zcy/OpenRSD` and `/data1/zcy/OV-CapFlow` are both available and have substantial user-owned dirty state; no experiment process or training artifact has been changed.
  - Began a date-bounded Git/document/result inventory.
  - Rejected broad artifact-by-artifact enumeration after it surfaced thousands of generated captures; switched to index-first reconstruction with raw-evidence spot checks.
- Files created/modified:
  - `task_plan.md`, `findings.md`, `progress.md` (append-only audit bookkeeping)
  - `/data1/zcy/OpenRSD/CODEX_WORKLOG.md` (task-start record)

### 2026-08-12 12:22 CST — completed

- Reconstructed the 2026-05-12→2026-08-12 OpenRSD→OV-CapFlow timeline using index-first document review and raw-evidence spot checks.
- Separated comparable endpoints, mechanism diagnostics, confounded runs, negative results, and pending implementations.
- Produced a 485-line Chinese rolling review with the main experimental deltas, transition logic, reusable research practices, current evidence gaps, and a staged detector proposal.
- Screened 25 primary-source papers and retained 15 core/adjacent works in Markdown and CSV ledgers; broad HBox consistency, point-layout, mixed-supervision, and rotated-DETR claims were marked occupied.
- Recommended HBox-supervised OBB as the main route and specified provisional HALO-Set components, matched baselines, ablations, metrics, promotion/stop gates, and risks.
- No training code, configuration, checkpoint, dataset, tmux session, process, or GPU workload was modified or launched.
- Deliverables:
  - `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_ovcapflow_review_and_detector_proposal_zh.md`
  - `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/papers.md`
  - `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/papers.csv`
  - `docs/literature-search-20260812-weakly-supervised-oriented-set-detection/search-notes.md`

## 2026-08-12 — CNN-HBox proposal v2 completed

- Reopened the derivative-detector decision after the user selected a CNN base and explicitly requested that fixed-query experience remain as a reference.
- Audited the local H2RBox-v2 implementation: CNN/Rotated-FCOS inheritance, P3–P7 strides, center/range/min-area assignment, background labeling, three-view annotation IDs, and box-ID-level angle aggregation. Also verified the local Rotated RTMDet dynamic soft-label assigner path.
- Performed a focused primary-source search. BGHR was identified as the strongest collision on HBox sample mining; ABBSPO and EIE-Det occupy scale/symmetry and box-equivariance claims; OTA/IQDet/DCFL/RTMDet/TS-Conv occupy generic soft/dynamic assignment claims.
- Replaced the fixed-query HALO-Set recommendation with `OrbitAssign`, a training-only CNN design built on H2RBox-v2 / Rotated FCOS. Fixed-query lessons are explicitly mapped to dense reachability, ownership, local exclusivity, coverage, background safety, and raw/pre-NMS diagnostics.
- Wrote a 577-line Chinese v2 review/proposal and a focused literature package containing 18 selected papers from 26 screened candidates. CSV parsing verified all 18 records and 13 fields.
- No implementation, config, dataset conversion, training, evaluation, process, tmux, checkpoint, or GPU workload was created or changed.
- Deliverables:
  - `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_cnn_hbox_obb_proposal_v2_zh.md`
  - `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.md`
  - `docs/literature-search-20260812-cnn-hbox-oriented-assignment/papers.csv`
  - `docs/literature-search-20260812-cnn-hbox-oriented-assignment/search-notes.md`

## 2026-08-12 — OrbitAssign v3 FMSA fusion completed

- Read the full 319-line FMSA-Point assisted draft and the full 577-line OrbitAssign v2 before revising.
- Performed a private-to-public-safe literature check using only public paper titles and method keywords. Verified MessDet's reported (N=8) group setting, FAA's 2D spatial Fourier formulation, PointOBB-v3's multi-view/instance-weighting boundary, Point2RBox-v2/v3's layout/progressive-pseudo-label boundary, and the metadata/DOI of the TIP 2025 HBox orientation-enhancement near neighbor.
- Selected the minimal fusion route: keep H2RBox-v2 / Rotated FCOS and OrbitAssign; convert FMSA self-awareness into factorized ownership, geometry, and background evidence routing; use a double-angle circular coefficient only as a soft direction-stability signal.
- Rejected point-region warmup, MessDet as the first parent, boundary segmentation, class prototype memory, scalar-(Q) filtering, hard pseudo OBBs, and EMA teacher–student from the core design.
- Created the self-contained v3 design and a traceable incremental literature addendum while preserving v2 unchanged.
- No implementation, configuration, dataset, training, evaluation, checkpoint, tmux session, process, or GPU workload was changed or launched.
- Deliverables:
  - `docs/project_history/exp_20260812_three_month_review/fres_20260512_20260812_openrsd_to_cnn_hbox_obb_proposal_v3_fmsa_fusion_zh.md`
  - `docs/literature-search-20260812-cnn-hbox-oriented-assignment/fmsa-fusion-addendum.md`
