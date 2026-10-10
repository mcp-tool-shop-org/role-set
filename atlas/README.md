# role-set: how it works

Mapped at 2026-10-10 from commit 170b718 by Atlas 1.24.0.

## What this is

role-set plans every slot of a role-shaping batch, has a model on a rented pod write only the writer's parts, and passes each item through aspire-si's gates. (written by a person)

6 parts, mostly Python (9 files). Work enters through 2 doors; the busiest is CI, which reaches 2 parts. People run role-set.

## What changed since 2026-10-10 (9bababb)

- tests now imports src.
- CI (.github/workflows/ci.yml) is a new door. It starts on a pull request to main; on a push to main touching 9 paths; or by hand. It runs tests/.
- role-set (pyproject.toml) is a new command. It runs src/role_set/cli.py.
- README.md is now read by pyproject.toml.
- src/role_set/ is now read by pyproject.toml.
- .github is a new part, drawn from `.github/**`.
- docs is a new part, drawn from `docs/**`.
- roles is a new part, drawn from `roles/**`.
- And 2 more changes to the parts.
- 19 files added, across 6 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 9 paths; or by hand. Runs tests/.
2. **role-set** (a command people run). Runs src/role_set/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (5 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**role-set** (a command people run) runs src/role_set/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, docs/, roles/ and the repository root. Nothing in this repository writes to them.

## Where to start

src/role_set/cli.py → src/role_set/cost.py

Read those in order to follow one run of role-set end to end. This path follows role-set (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 2 reads go to the directory the command is run in, not to this repository.
- 1 read goes to a path its caller passes, not to this repository.
- 1 write goes to the directory the command is run in (plan.json) or a path its caller passes, not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
