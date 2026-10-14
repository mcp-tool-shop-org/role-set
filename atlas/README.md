# role-set: how it works

Mapped at 2026-10-14 from commit d38d929 by Atlas 1.24.0.

## What this is

role-set plans every slot of a role-shaping batch, has a loopback model write only the writer's parts, and passes each item through aspire-si's gates. It never rents a pod. (written by a person)

9 parts, mostly Python (68 files), CSS (2), TypeScript (2), shell (2), Astro (1), JavaScript (1) and PowerShell (1). Work enters through 3 doors; the busiest is CI, which reaches 2 parts. It deploys a site to GitHub Pages. People run role-set.

## What changed since 2026-10-14 (5717d53)

- src/role_set/critic.py is now read by src/role_set/generate/t10_variety.py.
- src/role_set/gates.py is now read by src/role_set/generate/t10_variety.py.
- 12 files changed content, across 5 parts.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 9 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **role-set** (a command people run). Runs src/role_set/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches src (33 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**role-set** (a command people run) runs src/role_set/cli.py.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 2 doors.

## What tends to change together

- **src/role_set/readout.py** and **tests/test_paid_run_data.py** changed together in 6 of 9 commits, and the tests part imports the src part.
- **src/role_set/assemble.py** and **src/role_set/prompts.py** changed together in 9 of 17 commits, inside the src part.
- **src/role_set/prompts.py** and **src/role_set/runner.py** changed together in 11 of 21 commits, inside the src part.

7 files changed together with their own tests, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 6 source files reach 10 revisions; the floor rises to 10 when 25 do.

## What no test touches

- **scripts** is imported by no test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **src/role_set/generate/t10_capacity.json** is written by scripts/t10_cell_capacity.py.
- **src/role_set/generate/t10_cells.json** has a block written by scripts/t10_usable_cells.py.
- **src/role_set/generate/t10_record.json** is written by scripts/t10_variety_record.py.

## Hand-authored

People write .github/, assets/, docs/, roles/, the repository root and site/; 1 write with a path built at run time may land here.

## Where to start

src/role_set/cli.py → src/role_set/puzzle_check.py → src/role_set/assemble.py → src/role_set/plan.py

Read those in order to follow one run of role-set end to end. This path follows role-set (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 write and 4 reads use paths built at run time and are not named here.
- 5 writes and 27 reads go to a path their caller passes, not to this repository.
- 7 writes and 4 reads go to the directory the command is run in (plan.json/, roles/ and writers.json) or a path their caller passes, not to this repository.
- 9 reads go to the directory the command is run in, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
