# B3c: behaviour spec for the T10–T12 generator rewrite

**Status:** draft for R&D's review, 2026-10-16. **The rewrite is on hold** until the maintainer rules whether Grok's generator logic and Argon's pipeline code must be rewritten or only their text replaced. This spec is the record of the behaviour either way.
**Clean room:** this spec is written only from Claude-written tests, `docs/generator-design.md` outside lines 291–383, and the R&D and maintainer rulings in the PR threads (#48–#55, #63, #64). No Grok- or Argon-written code, data, diff or doc text was read for it. The source of each behaviour is in [behaviour-sources.md](behaviour-sources.md). [inventory.md](inventory.md) lists every line from #49, #53, #54 and #55.

## 1. What is kept and what goes

- **Deleted with their code:**
  - every line attributed to #49, #53, #54 and #55;
  - their whole files: `generate/t10_variety.py`, `t10_cells.json`, `t10_capacity.json`, `t10_record.json`, the three `scripts/t10_*.py`, and `tests/test_t10_variety.py`;
  - the 25 tests they wrote ([inventory.md](inventory.md), "whole").
- **Rewritten where their lines sit:** `generate/recipes.py` (156 of 729 lines), `render.py` (26), `build.py` (19), `select.py` (16), the `ci.yml` sweep job, the `pyproject.toml` marker, and `docs/generator-design.md` lines 291–383.
- **Kept:**
  - the Claude-written parts of `build.py`, `select.py`, `shortcuts.py` and `recipes.py`, including the T11 and T12 recipes and the catalogue scripts `t11_catalogue.py` and `t12_catalogue.py`;
  - their committed catalogues;
  - the Claude-written tests, which are the oracle.
  - The four Claude-written tests that #49, #53 and #55 edited keep Claude's lines. Their edited lines are rewritten against this spec.
- **Packs and templates.** Grok's five packs and 31 templates are replaced, not edited.
  - T10–T12 draw from the B3a pack set, within the same cap of at most 4 uses per pack per batch.
  - **There are no prose templates.** In B3b the writer renders the puzzle's prose from the locked model, as in B3a, so `render.py`'s templates are dropped. The Claude-written gloss in `solvefirst.gloss` covers the forms T10–T12 use.

## 2. The packs, extended

Each of the 42 B3a packs gains two fields, both Claude-written row inputs with a provenance record:
- **`state`:** one attribute with 3 values that can't overlap in everyday sense (a slot is booked, a walk-in or cancelled; a bed is damp, dry or waterlogged). T10 counts over it, and T11 claims over it. A two-value use takes its first two values.
- **`editions`:** 4 named editions of one dated thing in the pack's domain (the spring, summer, autumn and winter rota). T12 asks which one is in force.

Pack names stay whole words that no other name in the pack contains. This repeats the rule of Grok's deleted test, so a fresh test covers it.

## 3. T10: the size or count trait

**Kept as ruled (sources T10 #1–8, #10, #20–21):**
- **Questions:** open-set ("which could be …") or claim only. A single-value question is refused before any search, and the question argument has no default.
- **The four size readings:** `=` read as `>=`, `=` read as `<=`, `>=` read as `=`, and `<=` read as `=`.
  - A reading with nothing to rewrite is skipped.
  - A reading that ends in a contradiction is recorded as "no assignment fits", with `differs: false`.
  - Divergence is judged by answer type.
- **The size-word test, a hard assertion:**
  - each puzzle has exactly one `count` clue, on the deciding path;
  - the item rests on the reading of its own trait operator, and that reading must differ;
  - `count` is out of the auxiliary clue pool.
- **Claims are "exactly" only.** A claim cell with another operator is refused up front. Set cells use all three size words. The boundary rule is reported, not gated.
- **Receipts and fixtures:** receipt metrics, the regression fixtures, and the same-seed rebuild.

**Proposed fresh, because no allowed source states them:**
1. **The cell table is computed, not committed data.** A cell is (n, operator, k, ifs): n entities 3–5, operator `=`, `>=` or `<=` (`=` only for claims), k 1 to n−1, ifs 0–2 (`MAX_IF` = 2). The table is every cell `prove` can build at least once in 400 tries, found by a seeded probe at test time and cached in a fixture. The 20 seeds of one pack's file take 20 different cells.
2. **Deletion order in `reduce`:** first plain unit facts (`is_not`, `not_in`), then `in`, then `if` clues, never the `count` clue. Within a tier the order is seeded. If the reduced puzzle loses its `if` target, it retries inside the seed. While adding, `if` clues are offered before unit clues.
3. **`if_shape` for claim cells:** one `if` is a positive or negative consequent; two `if`s are a chain, a shared consequent, a shared antecedent, or disjoint (4 entities). A shape that won't build at an n is left out of that n's probe list.
4. **The zero-`if` cap:** at most 2 zero-`if` cells per file, in every stratum.
5. **Capacity per stratum** (set two, set three, claim two, claim three, never pooled): the test reports supply, demand and headroom. It fails loudly with the numbers when a stratum can't be placed for every pack at 20 seeds, and a 42-pack allocation must pass. **It's a check on the table, never a raise inside the cell lookup.**
6. **`SET_MAX` = 4:** an open-set answer lists at most 4 entities.
7. **The hidden solution** is drawn uniformly from the assignments where the count clue is true.
8. **The variety floors per stratum:** at least 75% distinct skeletons, none on more than 3 builds, and no repeat within one pack's file. They're computed from stored skeletons in a fast test, with the full sweep behind the `sweep` marker. **No committed record file:** the fast test regenerates what it checks.

**Open (R&D to rule):** two-value claims have no inequality cell, which is gap T10 #23. I propose they stay "exactly" only, and the capacity check reports the stratum.

## 4. T11 and T12

**T11 and T12 stay as Claude wrote them** (behaviour sources T11 #1–11 and T12 #1–12), with two changes: they draw packs from §2, and their receipts carry `file_index`. The T12 `file_index` (R&D's non-blocking note on #63) is added, with a test.

## 5. The selector and shortcuts

**Unchanged:**
- file size 5;
- each item rests on exactly one differing reading;
- at least 2 distinct readings, none on more than half;
- `rests_on` counting, the newest floor as a share, file receipts, and `shortcuts.prove` with code-only wrong options.

**Rewritten:** #54's 16 lines in `select.py`, the T10 claim consequence. Every T10 claim rests on a reading that loosens "exactly", so at least half of a T10 file are set items with at-least or at-most. It's restated here, and a fresh test covers it.

**Fresh tests for behaviours no allowed test covers:**
- `Floor.distinct_skeletons`;
- puzzle-check re-running the readings from the sidecar with the same pinned shortcut code;
- the sweep's `contrast_verdict` split.

## 6. Out of scope here

- **B3b, the wiring into `role-set write`:** T10–T12 B slots get `b_path: generator`, and the writer renders the prose from the locked model.
- **B4, the cumulative state.**

## 7. Acceptance

1. Every Claude-written generator test passes unchanged.
2. A fresh test covers each behaviour this spec restates (§2, §3 items 1–8, §5).
3. No line attributed to #49, #53, #54 or #55 remains: `scripts/b3c_blame_check.py` re-runs the inventory and must report zero.
4. `atlas check` is green, coverage holds at 90 or above, and the identity scan is clean.
