---
title: The T10 generator
description: How role-set builds and proves T10 puzzles in code, spreads them over cells, holds the variety floors, and gates them in CI.
sidebar:
  order: 4
---

Kind B puzzles used to be written freehand by a model and checked afterwards. The check caught broken puzzles, but it could not make a good one. The generator turns that around: code builds the puzzle and proves it, and a model only words it. The wording is parsed back and must map to the same model.

The generator is the `role_set.generate` package. It has no `role-set` command yet. This page follows T10, the lesson on strict size words ("exactly", "at least", "at most"), from one build to the CI gate. The agreed design, with R&D's rulings, is `docs/generator-design.md` in the repository.

## One build

`build_t10(pack, seed, question)` makes one puzzle for one domain pack. The steps are fixed:

1. **Sample a solution.** From the pack (entities, one attribute, its values), seeded, sample a full hidden assignment.
2. **Add clues until the answer is set.** Draw clues that are true in the hidden solution, from the solver's rule forms, until the question has one answer. The trait clue, a count clue, is kept on the deciding path by construction.
3. **Reduce.** Delete clues in tiers (`is`, then `is_not`, then `if`, with the count clue last) while the answer stays set. What is left is minimal.
4. **Prove the trait.** Run the four T10 readings on the trait clue alone. Keep the puzzle only if one of them changes the answer.
5. **Render.** Template sentences word the model. Each sentence round-trips through `check_text`, so the wording names exactly what the rule names.
6. **Receipt.** Record the seed, tries, the trait clue, every reading and its answer, three difficulty metrics, and the skeleton, pack and solver hashes.

The same seed rebuilds the same puzzle.

### The four readings

R&D registered four lazy readings for T10 on 2026-10-11. They are data in `generate/shortcuts.py` (`REGISTRY`), shared with `puzzle-check`:

| Reading | What a careless reader does |
|---|---|
| `T10.exact_read_as_floor` | reads "exactly k" as "at least k" |
| `T10.exact_read_as_ceiling` | reads "exactly k" as "at most k" |
| `T10.floor_read_as_exact` | reads "at least k" as "exactly k" |
| `T10.ceiling_read_as_exact` | reads "at most k" as "exactly k" |

A reading that leads to a contradiction proves nothing. Each must give a definite, different answer.

### `trait_not_tested`

A T10 puzzle carries one count clue, and the readings are applied to that clue. Before this rule, a puzzle could pass because a second "exactly" clue changed under a reading while the trait's own reading left the answer alone. The item then tested "exactly", not the word it claimed to teach. That defect class is `trait_not_tested`, and the builder refuses it.

### Why claims use "exactly" only

A claim asks whether one named entity has a value. Its answer (supported, unsupported, cannot tell) is true of every assignment that still fits. Every assignment that fits "exactly k" also fits "at least k" and "at most k". Reading the wider word as "exactly" keeps a subset of the assignments, and the claim's answer stays true of that subset. So "at least" and "at most" cannot change a claim's verdict, and claim cells use "exactly" only. A set question ("which groups could have the value?") does change when groups of other sizes drop out, so set cells keep all three words.

## The cell map

Variety comes from where each seed is sent, not from luck. Each pack, question and seed maps to a structural **cell**:

| Field | Values |
|---|---|
| `n` | entity count, 3 to 5 |
| `op` | count word: `=`, `>=` or `<=` (claims: `=` only) |
| `k` | 1 to n−1 |
| `ifs` | number of `if` clues, 0 to 2 |
| `if_shape` | claims only, below |

`scripts/t10_usable_cells.py` probes every cell on seed 0, which is neither a tuning seed nor a locked seed, and keeps a cell only when it built within a tenth of the try budget (`MAX_TRIES // 10`). The result is committed as `src/role_set/generate/t10_cells.json`.

### `if_shape`

A claim cell adds the shape of its `if` clues. There are seven shapes:

| `ifs` | Shape | Meaning |
|---|---|---|
| 0 | `none` | no `if` clue |
| 1 | `pos` | the consequent is positive ("then … is v") |
| 1 | `neg` | the consequent is negative ("then … is not v") |
| 2 | `chain` | three entities; one `if`'s consequent entity is the other's antecedent entity |
| 2 | `shared_consequent` | three entities; both `if`s end on the same entity |
| 2 | `shared_antecedent` | three entities; both `if`s start from the same entity |
| 2 | `disjoint` | four different entities |

For one `if`, whether the clue mentions the claimed entity is not a shape. The canonical skeleton records the facts and not the claim, so that split could never be told apart in a skeleton. With it merged, every shape is decided from the reduced clue list alone, and two different shapes never share a skeleton.

The builder only offers `if` clues that can form the cell's shape. A build whose reduced puzzle loses its shape retries inside the same seed. A finished build is never thrown away for repeating a skeleton.

## The allocator

Packs with the same number of values share one skeleton space: a lamps puzzle and a doors puzzle are the same skeleton when their clues have the same shape. So the cells are assigned for a whole stratum at once.

For one question and one value count, a file of 20 seeds is assigned before anything is built, by a min-cost flow (`assign_cells` in `generate/t10_variety.py`):

- each pack's 20 seeds take 20 different cells from that pack's usable list;
- a cell with no `if` clue is used at most twice in a file (the **zero-if cap**), and costs more than any `if` cell, so it is used only when a pack runs short of `if` cells;
- other visits favour cells that gave more distinct skeletons on the tuning seeds 1, 21 and 41. Those weights live in `t10_capacity.json`, kept apart for two-value and three-value packs, and a cell's weight rises above 1 only when its tuning probe had no failed build;
- inside a file, each pack walks its cells by a step coprime to 20 from a different start, and the next file starts further along.

`cell_for(pack, question, seed)` reads the assignment. Seed 1 to 20 is file 0, 21 to 40 is file 1, and so on.

## The variety floors

R&D's floors hold in every stratum (set and claim, two and three values):

- at least 75% distinct canonical skeletons;
- no skeleton more than 3 times;
- no skeleton twice in one pack's file.

They are judged on **locked seeds**, which no tuning step reads. On the locked verdict seeds 101 to 120, three-value claims reach 118 of 120 distinct, with a top repeat of 2 and no zero-`if` visit. Before `if_shape` the same stratum was 112 of 120 with a top repeat of 4, which missed the floor.

## The record gate

Rebuilding every puzzle is too slow for every push. So the measurement is committed, and CI checks the commitment.

`src/role_set/generate/t10_record.json` holds 1,280 builds: seeds 1–20, 61–80, 81–100 and 101–120, for every pack and both questions. Each entry records `built`, the cell, `if_shape`, tries and `canonical_skeleton`. The record carries a `sha256` over the 15 files that decide those builds: the solver and generator source, `t10_cells.json`, `t10_capacity.json` and the pack data.

A test holds that file list **closed under role-set's imports**: if a hashed module imports another role-set module, that module must be hashed too. A change anywhere that could move a build changes the hash.

In the normal CI `test` job (`pytest -m "not sweep"`, 15-minute cap), `tests/test_t10_variety.py`:

- recomputes the hash, and fails with the regenerate command on a mismatch;
- asserts every entry built a proven puzzle;
- computes the floors per stratum from the stored skeletons;
- keeps one real build per cell shape under coverage.

That takes minutes.

## The dispatch sweep

The `sweep` job in `.github/workflows/ci.yml` runs only on `workflow_dispatch`, with a 60-minute cap:

```text
uv run pytest -m sweep --no-cov -n auto -s
```

It rebuilds every recorded puzzle and fails on any difference from the committed record. A pull request that changes the record needs a green sweep on its head commit before R&D passes it. A docs-only change does not touch the record and does not need one.

## The capacity gate and the headroom

```text
uv run python scripts/t10_usable_cells.py --check
```

The gate reads the committed table. It does not probe and does not run inside a build. For each stratum it reports supply (zero-`if` cells times the cap), demand (the zero-`if` visits the packs need), headroom, and each pack's `if`-cell count. When a file cannot be placed it fails with those numbers and names the next step.

With `if_shape`, three-value packs have 61 usable claim cells between them (52 with an `if`), against 27 before. Every pack has more than 20 `if` cells, so no pack needs a zero-`if` visit, and the zero-if cap is 2 in every stratum. A test places twelve three-value packs (the six real ones plus six clones, 240 builds a file) with no repeat inside a file.

## Adding a pack

1. Add the pack, with its `sense` block, to `PACKS` in `src/role_set/generate/recipes.py`.
2. Regenerate the cell table and run the capacity gate:

   ```text
   uv run python scripts/t10_usable_cells.py
   uv run python scripts/t10_usable_cells.py --check
   ```

3. Regenerate the tuning-seed capacity table:

   ```text
   uv run python scripts/t10_cell_capacity.py
   ```

4. Regenerate the committed record, and check it:

   ```text
   uv run python scripts/t10_variety_record.py
   uv run python scripts/t10_variety_record.py --check
   ```

5. Run `uv run pytest -m "not sweep"`, then dispatch the `sweep` job on the pull request's head commit.

The same steps apply after any change to a hashed generator file. The `--check` form of the record script prints the locked-seed figures per stratum, which is what R&D reads.

## What the generator does not do

- It does not decide that a puzzle is training data. Generated items still go through R&D's key-check.
- It does not paraphrase with a model yet. Template wording is the baseline.
- T12 (which edition was in force on a date) has a recipe and a selector. T11 has none yet.
