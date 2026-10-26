# role-set: how it works

Mapped at 2026-10-26 from commit cf27f4a by Atlas 1.24.0.

## What this is

role-set plans every slot of a role-shaping batch, has a loopback model write only the writer's parts, and passes each item through aspire-si's gates. It never rents a pod. (written by a person)

9 parts, mostly Python (115 files), shell (5), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and PowerShell (1). Work enters through 3 doors; the busiest is CI, which reaches 3 parts. It deploys a site to GitHub Pages. People run role-set.

## What changed since 2026-10-26 (2b26840)

Nothing structural changed since 2026-10-26; 1 file added and 4 changed content.

## What comes in

1. **CI.** On a pull request; on a push to main touching 10 paths; or by hand. Runs tests/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **role-set** (a command people run). Runs src/role_set/cli.py.

## What happens through CI

1. The workflow runs tests/ in tests.
2. That reaches scripts (9 files) and src (43 files).
3. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**role-set** (a command people run) runs src/role_set/cli.py.

## What breaks what

- **src** is imported by 1 part (scripts), and by 1 more only from tests; it sits on the path of 2 doors.
- **tests** is imported by 1 part (scripts) and sits on the path of 1 door.
- **scripts** is imported only from tests, by 1 part (tests), and sits on the path of 1 door.
- **roles/domains/naics-2022.json** is written by scripts and read by scripts and src; a hand edit reaches every reader.

## What tends to change together

- **src/role_set/generate/build.py** and **src/role_set/generate/recipes.py** changed together in 6 of 10 commits, inside the src part.
- **src/role_set/mock.py** and **tests/test_runner.py** changed together in 14 of 26 commits, and the tests part imports the src part.
- **src/role_set/readout.py** and **tests/test_paid_run_data.py** changed together in 7 of 13 commits, and the tests part imports the src part.

6 files changed together with their own tests, as expected.

Confidence is low: fewer than 25 source files reach 10 revisions in the window.

Window: 180 days; a pair counts from 3 shared commits, since 20 source files reach 10 revisions and the floor had fallen; it rises back to 10 when 25 do.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

Every written place has a reader.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

- **roles/domains/naics-2022.json** is written by scripts/naics_table.py when run from the repository root, and committed.
- **src/role_set/generate/t10_capacity.json** is written by scripts/t10_cell_capacity.py.
- **src/role_set/generate/t10_cells.json** has a block written by scripts/t10_usable_cells.py.
- **src/role_set/generate/t10_record.json** is written by scripts/t10_variety_record.py.
- **src/role_set/generate/t11_catalogue.json** has a block written by scripts/t11_catalogue.py.
- **src/role_set/generate/t12_catalogue.json** has a block written by scripts/t12_catalogue.py.

## Hand-authored

People write .github/, assets/, docs/, the repository root and site/; 8 writes with paths built at run time may land here.

## Where to start

src/role_set/cli.py → src/role_set/cards.py → src/role_set/guards.py

Read those in order to follow one run of role-set end to end. This path follows role-set (a command people run) from its entry, since CI runs only tests.

## What this map cannot see

- 1 import could not be resolved: `scripts/colab/pin_window.py` imports `stub_llama`, which is no module on its import path and no declared dependency.
- 8 writes and 33 reads use paths built at run time and are not named here.
- 56 writes and 110 reads go to a path their caller passes, not to this repository.
- 15 reads go to the directory the command is run in, not to this repository.
- 9 writes and 5 reads go to the directory the command is run in or a path their caller passes, not to this repository.
- Statistics confidence is low: fewer than 25 source files reach 10 revisions in the window.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
