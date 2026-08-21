# Stopping Overhardening Skill Design

## Status

The user approved the conversational design on 2026-08-02. This document is
the written review gate before implementation.

## Purpose

Create a global, discipline-enforcing Codex skill that stops implementation or
research work when defensive engineering, audits, plans, or repeated fixes
grow faster than the core result. The skill must automatically shrink the
scope and resume a minimal end-to-end validation without requiring human
approval.

The skill generalizes across research, coding, automation, and documentation
work. It is not specific to OV-CapFlow.

## Package

Install the skill globally at:

```text
~/.codex/skills/stopping-overhardening/
├── SKILL.md
└── agents/openai.yaml
```

The package contains no scripts, references, assets, README, changelog, or
project-specific history. This is a judgment skill; its complete reusable
workflow belongs in one concise `SKILL.md`.

## Trigger boundary

The frontmatter description starts with `Use when...` and covers these
observable symptoms:

- repeated fix-only or hardening-only commits;
- audit, provenance, planning, or test infrastructure delaying the first real
  result;
- custom infrastructure duplicating an existing framework;
- a growing chain in which each guard reveals another hypothetical guard;
- support work becoming larger than the feature or experiment itself.

The description states only when to load the skill. It does not summarize the
workflow.

## Core law

> Do not add a guardrail, audit layer, or custom support component without an
> observed failure, an explicit requirement, or a high-stakes exception.

Obtain the smallest real result before building generalized infrastructure.
Treat a result as an end-to-end observable: a real batch, runnable behavior,
user-visible output, scientific metric, or demonstrated failure at the actual
integration boundary. Mock-only success, more documentation, and additional
unit coverage do not count as the first real result.

## State and tripwires

Track four facts during applicable work:

1. the single core outcome;
2. whether a minimal real end-to-end run has occurred;
3. the number of active pre-result invariants;
4. the consecutive fix-only/hardening-only streak.

Stop immediately when any tripwire fires:

- three consecutive completed changes produce only fixes, guards, audits, or
  documentation and no new runnable result;
- more than five active invariants block the first real run;
- more than two custom support components are introduced before the first real
  run;
- support/audit tasks outnumber tasks that directly produce the core result;
- an existing framework capability is reimplemented without demonstrated
  evidence that it cannot satisfy the requirement.

Tests that directly specify a core behavior count with that behavior, not as
separate support work. Mandatory system, safety, and governing-skill steps are
not optional hardening, though their implementation must remain minimal.

## Automatic scope reset

On a tripwire, stop new implementation and emit exactly this compact report:

```text
HARDENING STOP
Core outcome:
Triggered by:
Keep (maximum five invariants):
Defer:
Reuse:
Next minimal real run:
```

Then perform the reset without waiting for human approval:

1. keep one outcome and at most five core invariants;
2. defer every guard without observed-failure evidence;
3. replace custom support work with existing framework capabilities where
   possible;
4. compress the active plan to one minimal vertical slice;
5. execute that real validation before adding another guard.

Resume automatically only after all five reset conditions are true. Preserve
existing files, commits, artifacts, and user work: deferral never authorizes
deletion, reset, overwrite, or rollback. A user may voluntarily expand the
scope, but approval is not required for the automatic minimal path.

## Exceptions and precedence

The observed-failure requirement does not delay safeguards needed for:

- security, privacy, authorization, legal, or regulatory obligations;
- destructive actions or irreversible data loss;
- production incidents and high-stakes medical, financial, or safety work;
- explicit user requirements;
- higher-priority system, developer, repository, or selected-skill rules.

Even under an exception, choose the smallest sufficient protection and reuse
the existing framework first.

## Skill structure

`SKILL.md` will be a concise discipline/pattern skill with:

- a two-sentence overview and the core law;
- the objective tripwires;
- the automatic reset procedure and report format;
- a compact quick-reference table;
- one generalized before/after example;
- common rationalizations and explicit counters;
- red flags and the high-stakes exception boundary.

Target fewer than 500 words unless pressure testing demonstrates a necessary
gap. Use imperative language. Do not narrate the originating project.

`agents/openai.yaml` will contain only generated `display_name`,
`short_description`, and `default_prompt` values derived from the final skill.

## Validation

Treat the skill itself with RED-GREEN-REFACTOR.

Run three fresh-agent pressure scenarios without the skill first:

1. a research experiment with sunk cost, three fix-only commits, and pressure
   to add more provenance checks instead of running a real batch;
2. an engineering task where a framework already provides the needed
   mechanism but custom audit infrastructure appears attractive;
3. a destructive or security-sensitive counterexample where required
   protection must not be removed merely to make progress faster.

Record baseline decisions and rationalizations. Initialize the skill only
after observing at least one relevant failure. Rerun equivalent fresh-agent
scenarios with the skill. Passing behavior requires:

- triggering and automatically shrinking scenarios 1 and 2;
- preserving necessary safeguards in scenario 3;
- producing one executable next validation rather than another framework;
- not requesting human approval to resume the minimal path;
- not deleting or rolling back user work.

Close only loopholes demonstrated by the GREEN runs, then rerun until the
behavior is stable. Finally run the official `quick_validate.py`, inspect the
generated UI metadata, verify frontmatter discovery, and confirm that the
skill directory contains only the two approved files.

## Non-goals

- Do not ban testing, provenance, review, or defensive engineering.
- Do not impose arbitrary line-count or wall-clock budgets.
- Do not replace TDD, systematic debugging, or security review.
- Do not optimize one specific model, framework, or repository.
- Do not create a monitoring daemon or automatic file-deletion tool.
