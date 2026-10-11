# role-set: how it works

Mapped at 2026-10-11 from commit 4477f59 by Atlas 1.24.0.

## What this is

role-set plans every slot of a role-shaping batch, has a model on a rented pod write only the writer's parts, and passes each item through aspire-si's gates. (written by a person)

7 parts, mostly Python (33 files). Work enters through 2 doors; the busiest is CI, which reaches 2 parts. People run role-set.

## What changed since 2026-10-10 (c366a4e)

- roles/verifier/sources.json is now read by docs/role-set-design.md.
- 21 files added and 12 changed content, across 5 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 9 paths; or by hand. Runs tests/.
2. **role-set** (a command people run). Runs src/role_set/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (19 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**role-set** (a command people run) runs src/role_set/cli.py.

## What breaks what

- **src** is imported only from tests, by 1 part (tests), and sits on the path of 2 doors.

## What tends to change together

No two source files, other than a file and its own test, changed together often enough to name.

1 file changed together with its own test, as expected.

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

People write .github/, assets/, docs/, roles/ and the repository root; 1 write with a path built at run time may land here.

## Where to start

src/role_set/cli.py → src/role_set/harvest.py → src/role_set/assemble.py → src/role_set/plan.py

Read those in order to follow one run of role-set end to end. This path follows role-set (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 3 imports could not be resolved: `src/role_set/cli.py` imports `role_set.readout`, which is no module on its import path and no declared dependency; `src/role_set/write.py` imports `role_set.stuckrule`, which is no module on its import path and no declared dependency, twice.
- 1 write and 3 reads use paths built at run time and are not named here.
- 1 write and 13 reads go to a path their caller passes, not to this repository.
- 6 writes and 3 reads go to the directory the command is run in (plan.json/, roles/ and writers.json) or a path their caller passes, not to this repository.
- 6 reads go to the directory the command is run in, not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
