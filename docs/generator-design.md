# role-set puzzle generator: design draft

Status: agreed design (R&D-approved with the additions below, 2026-10-11 and 2026-10-12). Drafted by the Publisher. Slice 1 is in progress: the `contains` span predicate is the first piece.

## Why

Kind B puzzles today are written freehand by a model (Grok seats, the 235B writer), then checked by `puzzle-check`. The check catches broken puzzles, but it can't make a good one. Batch 6 shows the cost:
- Three rounds of review on T10, T11 and T12, and T12-B-01..05 still failed.
- Round 3 passed the tool by encoding the answer into the puzzle.
- Every new failure class has needed a new check, written after the fact (skeleton repeats, is-not folds, Question names the target, grouped keys).

The maintainer's direction: a puzzle generator we can build on over time, instead of hoping for luck.

**The change:** code builds the puzzle and proves it, and a model only words it. The wording is parsed back and must map to the same model.

## Research grounding

1. **Solver-proven uniqueness and minimality.** ZebraLogic (Lin et al., 2025, ICML, arXiv:2502.01100) generates logic-grid puzzles from clue templates with a solver, and controls difficulty by search-space size. O'Sullivan and Horan (2007, AAAI demo, "Generating and Solving Logic Puzzles Through Constraint Satisfaction") guarantee a unique solution, graded difficulty, and solvability by inference alone.
   **Implication:** start from a sampled solution, add typed clues until exactly one solution remains, then delete each clue that isn't needed. At ≤6 entities and ≤3 attributes, each check takes milliseconds, and `puzzle.py`'s existing solver can do it (200,000-assignment cap).
2. **Difficulty from solver traces, not clue counts.** Pelánek (2011, FLAIRS; 2014, arXiv:1403.7373) rates Sudoku difficulty from a technique-based solver's step costs and step dependencies, at r = 0.95 against human solve times on about 2,000 puzzles. Bogaerts, Gamba, Claes and Guns (2020, ECAI, arXiv:2006.06343) explain logic-grid solving step by step, scoring each step by how hard it is to verify; each step is a minimal unsatisfiable subset.
   **Implication:** a propagation solver with ranked, named moves records a trace (steps, the costliest move, the longest chain, whether case splits were needed). That trace is the difficulty receipt. The cost table is a config value.
3. **A named shortcut must give a definite wrong answer.** HANS (McCoy, Pavlick and Linzen, 2019, ACL, arXiv:1902.01007) builds items from templates so that a named heuristic gets them wrong by construction. Contrast sets (Gardner et al., 2020, arXiv:2004.02709) and CheckList's directional tests (Ribeiro et al., 2020, arXiv:2005.04118) assert that a perturbation changes the answer. Partial-input baselines (Poliak et al., 2018, arXiv:1805.01042) solve under a relaxed input to show an item can be answered without the required step. No published system combines these into "solve under the shortcut reading and prove the answer differs" for puzzles, so that part is ours.
   **Implication:** each trait gets an executable lazy reading. A puzzle is accepted only if `solve(model) != solve(lazy(model))` (or the lazy reading has no unique answer, where the trait allows it). The search runs until both hold.
4. **Distractors tied to named errors.** In automatic item generation (Gierl, Lai and Turner, 2012, Medical Education; Embretson, 1998, Psychological Methods), wrong options come from specific altered steps in a cognitive model. Gierl et al. (2017, Review of Educational Research) find that few item sets actually guarantee "one distractor, one named error".
   **Implication:** every wrong option is the output of one named lazy reading, tagged with that reading. An option no named reading produces is refused.
5. **Code builds the structure, the model words it, and the wording is checked by parsing.** Reasoning Gym (Stojanovski et al., 2025, arXiv:2505.24760), SynLogic (Liu et al., 2025, arXiv:2505.19641) and Enigmata (Chen et al., 2025, arXiv:2505.19914) pair seeded generators with rule verifiers and difficulty settings, so the same seed rebuilds the same item. Logic-LM (Pan et al., 2023, arXiv:2305.12295) and LINC (Olausson et al., 2023, arXiv:2310.15164) show that errors enter at the language-to-logic translation step. GSM-Symbolic (Mirzadeh et al., 2024, arXiv:2410.05229) shows surface changes alone expose brittleness.
   **Implication:** the generator is seeded and reproducible. The model's wording is parsed back (`check_text` and `fact_lines` already exist) and must match the source model exactly. Surface (names, domains, phrasing, line order) varies on purpose while the model stays fixed.

Source note: Reasoning Gym, SynLogic, Enigmata, ZebraLogic, O'Sullivan and Horan, Pelánek and Bogaerts et al. were confirmed by web search, at abstract level. The rest are from the agents' memory, so confirm the IDs before this goes public.

## Design

### Pipeline

1. **Sample a solution.** From a domain pack (entities, attributes, values, units, dates), seeded, sample a full assignment within the size limits.
2. **Pick the trait recipe.** Each lesson's trait has a recipe with:
   - the clue forms it needs;
   - its **lazy reading**, a function that rewrites the model the way a reader skipping the step would. Examples: a strict size word read as non-strict, dates matched within one edition, a stopping problem that runs past the settled line;
   - the decision rule: the lazy answer must differ, or be ambiguous where the trait allows it.
3. **Add clues until unique** (generate-then-reduce), drawing from the recipe's forms plus general forms. The recipe forces at least one trait clue onto the deciding path.
4. **Reduce:** delete each clue whose removal keeps the puzzle unique. The result is minimal.
5. **Prove the trait step.** Solve the lazy model. Accept only if its answer differs from the key (or the lazy model is ambiguous, for traits that allow it). Record both answers.
6. **Grade:** run the trace solver and record the steps, the costliest move, the longest chain and any case splits. Reject puzzles outside the batch's difficulty band.
7. **Deduplicate** on the canonical skeleton (the existing `puzzle_check.skeleton` with its folds). Cap the count per skeleton family in each batch.
8. **Build the options.** The key, plus one option per named lazy reading (and per general error such as a negation ignored), each tagged. Options must be distinct, with exactly one equal to the key.
9. **Render.** Templates give a deterministic baseline wording. An optional model pass paraphrases it, and the paraphrase is parsed back. A mismatch refuses that paraphrase, and the template wording stands.
10. **Emit** the item row, the model sidecar and a receipt: seed, generator version, recipe, both answers, the trace, the skeleton hash and the domain pack hash.

### What stays with the model or the writer

- The thinking and reply turns, written against the proven model, with the trace as the step outline.
- Kinds A, C, D and E are unchanged for now.
- `puzzle-check` still runs on everything, including generated items.

### Configuration (the maintainer's "configurable wins" rule)

All of these are config values with recorded defaults on the receipt:
- size limits;
- difficulty band;
- move cost table;
- skeleton cap;
- whether a model paraphrase is allowed;
- the lazy-reading set for each recipe.

Defaults are chosen by the Publisher, and R&D reviews them.

### Where it lives

`role-set`, as `role_set/generate/`:
- `domains/` (data packs);
- `recipes/` (one module per trait, each with its lazy reading and tests);
- `build.py` (sample, add, reduce, prove);
- `trace.py` (the human-style solver);
- `render.py`.

It reuses `puzzle.py` (model, holds, solve) and `puzzle_check` (skeleton, folds). The new CLI is `role-set generate --lesson T12 --kind B --n 8 --seed S`.

## First slice (proposed)

- `build.py` with generate-then-reduce and a uniqueness and minimality proof, over the existing rule forms.
- Three recipes, for the traits that failed in batch 6: T10 (strict size words), T11 (falsifier first, claim-keyed), T12 (date matching across editions, using R&D's two approved shapes).
- Template rendering only; the model paraphrase comes later.
- Receipts, skeleton dedupe, and tests for each recipe: the lazy answer differs, a minimal clue set, the same seed rebuilds the same puzzle, and the template wording round-trips through `check_text`.
- **Acceptance:** regenerate T10, T11 and T12's B sets for batch 6 from the generator. They pass `puzzle-check`, and R&D's key-check finds no "trait step doesn't decide" defects.

Later slices: the trace solver and difficulty bands; model paraphrase with parse-back; recipes for every remaining trait; options for kind E built from the same lazy readings.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Seeded generator; the receipt pins the generator version, recipe, domain pack hash and config. The model paraphrase (later slice) pins the model and prompt. |
| ANDON_AUTHORITY | 2 | Every stage refuses: no unique solution, a clue not minimal, a lazy answer equal to the key, out of the difficulty band, a skeleton over its cap, a failed paraphrase parse-back. Nothing partial is emitted. |
| NAMED_COMPENSATORS | n/a | Writes local files only. Merges go through the PR flow. |
| DECOMPOSE_BY_SECRETS | 2 | Domains, recipes, building, tracing and rendering are separate modules, each changing for a different reason. |
| UNCERTAINTY_GATED_HUMANS | 2 | Generated items still go through R&D's key-check. Humans review what the code can't prove (naturalness, domain sense), not uniqueness. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. The solver is the verifier for the structure. |

## Questions for review

1. Is the lazy-reading list right for T10, T11 and T12? Is "ambiguous under the lazy reading" acceptable for any of them, or must it always give a different unique answer?
2. Is generated material still "authored" enough for the training goals, or do kind B items need human-like texture the templates won't give before the paraphrase slice lands?
3. Owner: the Dataset Specialist (role-set) or a Grok seat (Seat C is free) for the first slice, with R&D reviewing.

## R&D review (2026-10-11): approved, with additions

Recorded from R&D's review (gen-review.md in R&D's scratchpad, to be recorded in rnd). These additions are part of the design.

**Comparing lazy readings with the key.** Render the lazy answer in the item's own format.
- *Claim-keyed:* an ambiguous lazy model becomes cannot_tell, which counts as a different answer. Accept.
- *Set-keyed:* the lazy answer is its full set of possible answers, and that set must differ from the key.
- *Single-value:* an ambiguous lazy model is accepted only if its possible answers exclude the key.

**Recipes for slice 1.**
- *T10:* `=`→`>=`, `=`→`<=`, `>=`→`=`, `<=`→`=`, and strict↔non-strict. At most half of a lesson's B items rely on loosening `=`.
- *T11:* the lazy reading is "consistent, so supported", using only the lines that mention the claim's subject.
  - Find side: the key must differ, and no single line refutes the claim on its own.
  - Confirm side: proved by a recorded contrast, a one-clue change that flips the claim to unsupported, named in the thinking.
- *T12:* the latest or only edition; the month name without the span's ends; the boundary day given to the wrong edition; the only fee printed. **Required:** typed spans and dates with containment in `puzzle.py`.
- *Regression tests for every recipe:* the defects review caught must be refused: T10 round-2 B-01..04, T11 round-3 B-03 and B-05, T12 round-3 B-01..05.

**Template text (slice 1) has guards against becoming a stamp of its own.**
- at least 3 templates per clue form;
- shuffled line order;
- at least 4 domain packs per lesson, each with sense rules;
- the skeleton cap per batch;
- no Stage 1 freeze on template-only wording until a batch-wide stamp check passes (its share threshold a config value).

**Ownership.**
- The Dataset Specialist owns `build.py`, the recipes, the lazy readings and the typed span.
- Seat C may take domain packs and templates, under the Specialist's review.
- R&D reviews each lazy reading before it's used.

**Also.**
- Acceptance runs on two seeds.
- After reducing the clues, check the trait clue is still on the deciding path.
- Confirm Pelánek 2014 and Bogaerts 2020 before the doc goes public.
- R&D reviews the config defaults once slice 1 has receipts.

## Later-slice candidate (Publisher, 2026-10-11; from an outside summary the maintainer shared, not yet reviewed by R&D)

**Counterfactual rule cards.** A puzzle states a local rule that overrides the familiar default, for example "on this route the late fare applies from 16:00, not 18:00", or a Knights-and-Knaves-style rule with an exception that applies here. The **default reading** becomes one more named lazy reading: solve with the familiar rule, and the answer must differ from the key. This stresses reading the page over pattern memory, the same failure KOR-Bench (Ma et al., 2024, arXiv:2410.06526) targets with novel in-prompt rules. It fits T11 and T12 in particular. It's cheap to add once recipes exist, since it's one more lazy reading per recipe.

Already covered by the design: renaming and identity masking (domain packs plus skeleton dedupe), and programmatic generation with a verifier. Out of scope for Stage 1: grid-transformation (ARC-style) and interactive-environment puzzles. They test induction and agentic exploration, not the role traits these lessons teach.

## Decisions after Kimi's consult (Publisher, 2026-10-11)

**The engine question: one generator that grows, not a generator per dataset, and no general engine in slice 1.**
- Slice 1 builds the puzzle generator as itself, with two deliberately cheap seams:
  1. **The model is the interface.** Any future constructor that emits a valid model, its wording and named shortcuts inherits the solver, minimality, puzzle-check, the sidecar and the readout untouched. It's a data contract, not an engine API.
  2. **The shortcut registry is data:** a table of named readings, with the divergence rule for each answer type.
- **Revisit trigger:** the day a second *kind* (not a new lesson) needs construction by code, for example code-built planted-flaw C items, extract the shared spine with two concrete instances in hand.
- **Why:** the solver itself grew rule by rule from real review cases (#29 → #44); an engine with no named second tenant generalizes in the wrong places; and an engine round would spend the Specialist's slice time on an abstraction nobody has exercised.

**Five pins before code:**
1. **Difficulty metric, recorded on every receipt:** assignments explored to the unique solution, number of load-bearing facts, and lookup distance (lines between a fact and the line it's combined with). R&D audits calibration per batch. Bands are config values; the metric set is fixed here.
2. **Wrong-option values come from code. Hard rule.** The shortcut run computes each wrong value, and the model only words it. A writer-chosen distractor value is refused.
3. **The shortcut floor applies per review file:** at least 2 distinct shortcuts exercised per lesson per file, and no shortcut on more than half of a file's B items. This replaces the lesson-wide 50% cap, which barely binds at 5 puzzles.
4. **Dates are an ordered integer-day attribute over the named days only,** rendered as calendar dates in the wording, with no date library. *(Amended per R&D, 2026-10-11:)* containment uses one new solver predicate, **`contains`**: a span is a start attribute and an end attribute over the named days, inclusive at both ends. `contains` can be asked or claimed, and is never a fact line. R&D's conditions:
   - the owner of a shared day is stated in a material line the model cites;
   - a day no span holds is refused as a question, and in a claim it is unsupported or cannot_tell;
   - at least one span fact comes from a clue;
   - day records are typed (label, month, ordinal);
   - `contains` is part of the skeleton key.
5. **Scope fence:** slice 1 is the three batch-6 lessons (T10, T11, T12). It's done when regenerated puzzles pass R&D's review on two seeds. Not other lessons, not other kinds, not the Seat C pipeline.

**Independent re-check (the Specialist's addition), endorsed:** puzzle-check re-runs each item's shortcut readings from the sidecar. Generation and review use **the same pinned shortcut code** (the gates-package pattern), so the generator pre-checks with exactly what the reviewer re-checks with. This extends the existing rule that no component vouches for itself.

**Fixtures and sequencing:**
- The T10 regression fixtures are built against post-role-set#45 answer keys only.
- The Specialist's rebuilt T12-B-01..05 are the first **must-pass** fixtures for the date forms. The batch-6 faulty puzzles are **must-refuse** fixtures. Both sit around the same lessons.
- Each new defect class gets a catalog row (`defects.jsonl`, moving from brief to gate once checkable), and each faulty puzzle gets a fixture.
- This doc moves into role-set (`docs/generator-design.md`) in the Specialist's first slice-1 PR.

The ARC grids, masked strategy boards and interactive sandboxes from the outside summary stay out of scope. The counterfactual-rule candidate above stays as a later slice.

## R&D confirmation of the decisions (2026-10-11)

**Dates (confirmed, with four conditions).**
1. The day values are only the days the item names (span ends and asked dates), never a whole calendar.
2. Containment is inclusive at both ends. If the ordered forms are strict only, add one non-strict form or a stated boundary rule, and never shift days in the renderer.
3. The day attribute is not distinct. A shared boundary day must be expressible, with an ownership rule.
4. Containment is computed by the solver, never stated as a fact line. Each T12 shortcut is a rewrite on these forms, and round 3's placeholder span is a must-refuse fixture. *(Superseded in part: containment uses the new `contains` predicate, not the existing ordered forms. See pin 4.)*

If any condition can't be met with the existing forms, say so before code.

**Per-file floor (confirmed).**
- With 5 B puzzles in a file, at most 2 may rest on any one shortcut.
- T11 confirm-side items count under the contrast that proves them, so lookups can't fill the floor.

**Also.**
- Pin the solver version on every receipt, since "assignments explored" depends on search order. Set no difficulty bands on that metric until a batch shows it tracks something.
- One pinned shortcut code for generation and review gives consistency but can't catch a bug in a shortcut itself. The independent check is R&D's review of each shortcut reading before it's registered.
- Counterfactual rule cards: no ruling yet. Bring them with the lesson and the shortcut they serve. R&D's concern: they could teach distrust of correct defaults, unless the card's rule is stated in the item's own material.

## The span representation as built (R&D, 2026-10-12)

Slice 1 implements `contains` with span *values* rather than separate start and end attributes:
- The span attribute is ordered. Each of its values is a span declared with its start and end day, from typed day records (label, day, month, ordinal, and a `year` on a wrap).
- The clues map entities to spans with the existing forms (`offset`, `before`, `is`). "Next" and "before" mean start order.
- Clued span ends need a cross-attribute form. That's a later, separately reviewed addition.

R&D's conditions for it, each with a test in `tests/test_span.py`:
1. Ordinals follow the calendar, and a year wrap needs its marker.
2. Spans are well formed, strictly ordered by start, and labelled by their bounds.
3. A day two spans hold goes to the stated owner rule, or else every holder is the answer.
4. Only declared days can be asked or claimed.
5. The fixtures: the five T12 rebuilds pass, and round 3's placeholder puzzles stay refused.
6. The skeleton key records the span, `contains` and the owner rule.

## The clue-form API (slice 1, as built)

This is the contract between the builder and the wording. Seat C's domain packs and templates work to it.

**The model is the contract.** `build` emits an ordinary role-set model:
- every fact is a rule in the solver's forms;
- the facts are numbered from line 2, and line 1 is the setting line;
- the solver, puzzle-check, the sidecar and the readout take it unchanged.

**Clue forms the builder uses** (`generate/build.py`, `CLUE_FORMS`): `is`, `is_not`, `in`, `not_in`, `before`, `after`, `adjacent`, `not_adjacent`, `offset` (k = 1 or 2), `count` (`=`, `>=`, `<=`), and `if` (an `is` antecedent, with an `is` or `is_not` consequent). A recipe chooses its subset.

**Templates** (`generate/render.py`, `TEMPLATES`) map each form key to sentences with named slots:
- **Form keys:** `is`, `is_not`, `in`, `not_in`, `before`, `after`, `adjacent`, `not_adjacent`, `offset1`, `offset2`, `count=`, `count>=`, `count<=` and `if`. `CLAUSES` holds `is` and `is_not` as clause bodies for use inside `if`.
- **Slots:** `{X}`/`{x}` and `{Y}`/`{y}` are entities (capitalised or not), `{v}`, `{v1}` and `{v2}` are values, `{k}` is a number word, `{be}` is "is" or "are" agreeing with `{k}`, `{plural}` is the pack's plural noun, and `{p}`/`{q}` are clauses.

**Every sentence must round-trip through `check_text`:**
- it names exactly what its rule names;
- a negative rule reads negative (not, never, neither, nor, n't);
- "before" or "after" appears only on the matching rule, in argument order;
- an `offset` sentence avoids both words ("next on from", "two places on from").

Rendering refuses a sentence that breaks a tie.

**Domain packs** (`generate/recipes.py`, `PACKS`; a slice-1 baseline): `plural`, `names` (entities as written, each a whole word no other name contains), `attribute`, and `values`. Before a freeze, Seat C's packs add:
- sense rules;
- at least 4 packs per lesson;
- at least 3 templates per form.

**Shortcut registry** (`generate/shortcuts.py`, `REGISTRY`): named readings stored as data, which the generator proves and puzzle-check will re-run.
- T10 registers R&D's four size readings.
- A reading that leads to a contradiction proves nothing: each reading must give a definite wrong answer.
- A single-value question can't carry a T10 reading, because loosening only widens the possibilities. T10 therefore uses open-set ("which could be …?") and claim questions.

**Receipts** (`build.receipt`) carry:
- the seed, tries and recipe;
- the trait clue;
- every reading with its answer;
- the three difficulty metrics;
- the skeleton hash, the pack hash, and the solver hash.

## T12 recipe (slice 1, as built)

**Shape.**
- 3 or 4 editions over contiguous whole-month spans. The asked date falls inside one span, and is now and then that span's closing day.
- The clue forms are `is_not`, `in`, `not_in`, `before`, `after`, `adjacent`, `not_adjacent` and `offset`. There is no plain `is`, so the mapping always comes from clues.
- The solver matches the date through `contains`.

**Readings** (answer readings in `REGISTRY`): each names the edition a careless reader would, from the same clues.
- `T12.newest`: the edition in force last.
- `T12.highest_number`: the edition with the highest number. It applies only when every name carries a number.
- `T12.boundary_next`: a span's closing day handed to the following span, or its opening day to the one before.
- `T12.month_alone`: every span whose start or end names the asked month.

A puzzle is kept only when at least one reading names a different edition.

**Wording.** `T12_TEMPLATES` overrides the generic forms with span phrasing ("was not in force from …", "came into force next on from …"). The setting line lists the spans in calendar order.

## The selector (slice 1, as built)

`role_set.generate.select` chooses a lesson's review file from generated receipts. No generated file is written without it.

**Shortcut floor (design rule 3).** Each chosen item rests on one reading that differs from its key:
- at least 2 distinct readings rested on;
- no reading rested on by more than half the file (2 of 5);
- at most half the file rests on a reading that loosens `=`.

`Floor.count` sets what a reading's share counts:
- `"rests_on"` (the default): each item once, under the reading it rests on;
- `"differs"`: every reading that differs on it. This is stricter, and for T12 it can't be met, because `T12.newest` differs whenever the answer isn't the newest edition. Holding newest to 2 of 5 would need three newest answers, and the position floor allows two.

**Position floor (R&D on role-set#50, class `filter_induced_shortcut`).** The keep filter rejected more newest-edition answers than others: about 12 in 160 builds, against about 46 expected. That made "never the newest" a working shortcut. Three changes answer it:
- each receipt records `position` (`earliest`, `between`, `newest`; None without a dated question);
- the selector holds each position to half the file, and asks for at least one newest-edition answer;
- the builder takes `position=` to build the answers a batch is short of.

The T12 calendar also asks a span's opening day as well as its closing day, and asks the last span on its opening day more often (0.6 against 0.3). In a pack without numbered names, that is the only way a newest-edition answer has a reading to differ.

A pack without numbered names (cards, sheets) has two T12 readings, so it can't fill a file of 5 alone: two per reading caps it at 4. Review files mix packs. `month_alone` fires only after the mid-month boundaries follow-up, which R&D approved on 2026-10-13.

Every bound is a `Floor` field (the maintainer's "configurable wins" rule). R&D's rulings set the defaults.

## T10 variety (as built, role-set#53, 2026-10-14)

**The cell map.**
- Each (pack, question, seed) maps to a structural cell: entity count n (3-5), a count word, k (1 to n-1), and a target of 0-2 `if` clues. Set cells use exactly, at least, and at most. Claim cells use exactly only.
- The usable cells are probed once and committed as `src/role_set/generate/t10_cells.json`, by `scripts/t10_usable_cells.py`.
- Seeds index the table so that one pack's 20 seeds land in 20 different cells. #53 gave each pack its own offset in that list. Packs of one kind share one skeleton space, and the assignment that covers them is the one under "T10 variety across packs" below.
- A build whose reduced puzzle loses its cell's `if` target retries inside that seed. Finished builds are never discarded for repeating a skeleton.

**What carries the variety.** The cell map does. Deletion now runs in tiers (`is`, then `is_not`, then `if`, with the count clue last), but on the locked seeds the share of puzzles keeping an `if` didn't rise. So it isn't credited.

**What each question tests (2026-10-14).**

A claim asks whether one named entity has a value. The answer is supported, unsupported, or cannot tell, and that answer is true of every assignment that still fits.

Every assignment that fits "exactly k" also fits "at least k". The same inclusion runs from "exactly k" into "at most k". Reading the wider word as "exactly k" keeps a subset of the assignments the real clue allows. The claim's answer stays true of that subset, so it does not change. An empty subset is "no assignment fits", and that does not count as a different answer. A cannot-tell key cannot carry the trait either: adding a clue only removes assignments, so if the answer is cannot tell with the trait, it was already cannot tell without it, and reduction drops the trait. The same subset covers any claim about one assignment, including a claim about two entities or about a count.

A set question asks which groups could have the value. The answer is those groups. Reading "at least k" or "at most k" as "exactly k" drops groups of other sizes, so the set changes. Set items test all three size words.

An "exactly" claim can change. Reading it as "at least k" or "at most k" lets more assignments through, and the verdict can move.

Two values do not open a way around this. An "exactly" clue there fixes both counts, so it makes an "at least" or "at most" trait redundant. That is why those claim cells never built.

Claim cells therefore use "exactly" only. Set cells keep all three size words. The four readings stay the ones registered on 2026-10-11. An off-by-one reading, "at least k" taken as "at least k−1", would change the lesson so the generator could build it, and it is not used.

The builder accepts a puzzle only when one of those readings, applied to the trait clue alone, changes the answer. The puzzle has that one count clue. The receipt still stores `prove`'s reading of every count clue. The selector rests the item on a reading of the trait clue's operator.

On #53's locked seeds this had not held. Nine three-value claim items, and one set item, passed on a second "exactly" count clue while the trait's own reading gave the key's answer. Those claim inequalities also had the hidden count equal to k. The draw was not pinned to k. Claims no longer use those cells.

## T10 variety across packs (as built, role-set#49, 2026-10-14)

Packs with the same number of values share a skeleton space. The canonical skeleton renames entities and values by first appearance, so a lamps puzzle and a doors puzzle are the same skeleton when the clues have the same shape. An offset per pack still lands several packs on one cell. The repeats on #49's earlier seeds sat on cells with no `if` clue.

**The assignment.** For one question and one value-count, a file of 20 seeds is assigned once, before anything is built. Each pack's 20 seeds take 20 different cells from that pack's usable list. A cell with no `if` fills a pack that has fewer than 20 `if` cells. Those cells are capped at 2 visits in the file. Three-value claims are the exception R&D set on 2026-10-14: the cap there is 3. Nine zero-`if` claim cells at a cap of 2 supply 18 visits, and the six packs need 26, so that file cannot be placed. At 3 the supply is 27. Every other stratum stays at 2. The other visits go to `if` cells, more of them to cells that produced more distinct skeletons on the tuning seeds 1, 21 and 41 (`t10_capacity.json`, from `scripts/t10_cell_capacity.py`). Inside a file those 20 cells are walked by a step coprime to 20, and each pack starts at a different place on the walk, so adjacent seeds take cells that sit far apart in the spread order. The next file starts the walk further along. A finished puzzle is kept. The builder still retries inside one seed when reduction loses the cell's `if` target. The skeleton's distinct count is a weight on that choice. It is not a second cap, and a seen skeleton does not discard the build.

**The capacity gate.** `scripts/t10_usable_cells.py --check`, and the same function in `tests/test_t10_variety.py`, read the committed table. For each stratum they report supply (zero-`if` cells times the cap), demand (the zero-`if` visits the packs need), and headroom. Headroom is two numbers. One is how many further packs fit when each new pack is limited to the cells every current pack already lists. The other is how many fit when a new pack may list any cell anyone lists. None means those cells include a whole file of `if` cells, so the zero-`if` cap does not run out: an `if` cell's visits grow with the number of packs. The report also lists each pack's `if`-cell count and its zero-`if` demand. When a stratum cannot be placed, the gate stops with those numbers and names the next step, `if_shape` for claim cells. The gate checks the table. It does not run inside a build, and it does not raise the cap inside `cell_for`. A seventh three-value pack fails it. The union has 18 `if` cells, so that pack still needs 2 zero-`if` visits, and the slack after the present 26 is 1.

`if_shape` is not built in this change. It is the next step if a claim stratum misses a floor on the locked seeds 61-80, or before another three-value pack is added. `polarity_mode` and a claim-role axis stay out.

The floors are unchanged: at least 75% distinct canonical skeletons in a stratum, no skeleton more than 3 times, and no skeleton twice in one pack's file. The verdict on this assignment is the locked seeds 61-80.

**How CI gates the variety floors.**
- **The committed record.** `src/role_set/generate/t10_record.json` stores the full measurement across seeds 1–20 and the locked seeds 61–80 for every pack and both `set` and `claim` (640 builds). For each build it records `built`, `cell`, `tries`, and `canonical_skeleton`.
- **The hash and fast check.** The record carries a `sha256` over the listed files that decide those builds (the solver and generator source, `t10_cells.json`, `t10_capacity.json`, and the pack data in `recipes.py`). In the normal 15-minute CI `test` job (`pytest -m "not sweep"`), `tests/test_t10_variety.py` recomputes that hash and fails on a mismatch with the regenerate command, asserts every entry built a proven puzzle, and computes the variety floors per stratum directly from the stored `canonical_skeleton` entries (distinct ≥ 75%, top repeat ≤ 3, no repeat within a pack's file). One fixed seed per cell shape from seeds 1–20 stays under coverage as a real build.
- **The dispatch sweep job.** In `.github/workflows/ci.yml`, the `sweep` job runs on `workflow_dispatch` (`ubuntu-latest`, `timeout-minutes: 60`, `uv run pytest -m sweep --no-cov -n auto -s`). It rebuilds all 640 puzzles (seeds 1–20 in `test_every_pack_and_seed_renders_a_proven_puzzle` and locked seeds 61–80 by stratum in `test_locked_seeds_clear_the_variety_floors_in_every_stratum`) and fails on any difference from the committed record. Any PR that changes the committed record requires a green `workflow_dispatch` sweep run on its head SHA before R&D passes it.
- **How to regenerate.** After changing any hashed generator file, cell table, capacity table, or pack data, run:
  ```
  uv run python scripts/t10_variety_record.py
  ```
  Use `uv run python scripts/t10_variety_record.py --check` to verify the committed record and print the locked-seed stratum figures.

## T10 Phase 1: the `if_shape` claim axis (as built, 2026-10-14)

**Why.** Claim questions test "exactly" only, which left three-value claims with 27 structural cells (9 with no `if` clue, and 14–16 `if` cells per pack). Six three-value packs demanded 26 zero-`if` visits across 9 zero-`if` cells, requiring a stopgap zero-`if` cap of 3 (`claim three` sat at 108/120 distinct, top repeat 3 on locked seeds 61–80) and leaving 0 headroom for a seventh three-value pack.

**The `if_shape` axis (`IF_SHAPES`, `clues_if_shape`).**
- A claim cell is `(n, op, k, ifs, if_shape)` (`{"n", "op", "k", "ifs", "if_shape"}`). Set cells are unchanged (`(n, op, k, ifs)`).
- `ifs = 0`: `"none"`.
- `ifs = 1`: `"pos"` (`then … is v`) and `"neg"` (`then … is not v`). Antecedents stay positive (`is`).
  - *Merged shapes (`touches` and `not`):* the candidate split of `ifs = 1` by whether the `if` clue mentions the claimed entity (`touches` vs `not`) is merged into `"pos"` and `"neg"`. `canonical_skeleton(model)` records `n`, `attributes`, and `clues` (`model["facts"]`), not `model["claim"]`. Consequently, `pos_touches` and `pos_not` (and `neg_touches` and `neg_not`) produce identical canonical skeletons and `touches` vs `not` cannot be decided from the skeleton or record entry alone. With `touches` and `not` merged, every shape in `IF_SHAPES` is decided directly from the reduced clue list (`clues_if_shape(clues)`), and distinct shapes never share a canonical skeleton.
- `ifs = 2`, defined over the entity pairs `(p1, q1)` and `(p2, q2)` of the two `if` clues:
  - `"chain"`: 3 distinct entities (`len({p1, q1, p2, q2}) == 3`) where one `if`'s consequent entity is the other's antecedent entity (`q1 == p2` or `q2 == p1`);
  - `"shared_consequent"`: 3 distinct entities with the same consequent entity (`q1 == q2`);
  - `"shared_antecedent"`: 3 distinct entities with the same antecedent entity (`p1 == p2`);
  - `"disjoint"`: 4 distinct entities (`len({p1, q1, p2, q2}) == 4`, so `n >= 4`).
- **Builder (`_addition_order`, `build_t10`):** `_addition_order` offers only `if` clues that can form the cell's `if_shape` (filtering to matching consequent polarity for `ifs = 1`, and to the first shuffled `if` clue and its valid partners for `ifs = 2`). A build whose reduced puzzle loses the cell's `if_shape` (`clues_if_shape(kept) != if_shape`) retries inside the same seed, counting the tries. Finished builds are never discarded for repeating a skeleton.

**Restored zero-`if` cap, per-stratum capacity weights, and headroom.**
- Probing claim cells with `if_shape` on `probe_seed = 0` (`RELIABLE_TRIES = 40`) expands usable claim cells from 23–25 per pack (14–16 `if` cells) to 52–61 per pack:
  - `claim two` (`lamps`, `doors`): 61 usable cells each (52 `if` cells, 9 zero-`if` cells), up from 27 (18 `if` cells).
  - `claim three`: 61 union cells (52 `if` cells, 9 zero-`if` cells) and 32 shared across all six packs (23 `if` cells, 9 zero-`if` cells): `cups` 52 (43 `if`), `pans` 52 (43 `if`), `machines` 56 (47 `if`), `trains` 52 (43 `if`), `beds` 55 (46 `if`), `clinics` 52 (43 `if`).
- Every pack now has more than 20 `if` cells, so `zero_if_demand` is `0` in every stratum. `zero_if_cap("three", "claim")` is restored to `2` (the same as every other stratum), giving `supply = 18` against `demand = 0`.
- `scripts/t10_cell_capacity.py` records distinct skeletons on the tuning seeds (`1, 21, 41`) split by value-count stratum (`two` vs `three`) in `t10_capacity.json`, so two-value and three-value packs never pool weights (fixing a pre-existing cross-stratum weight inflation on `set` cell `4|<=|3|1` where `cups` seed 84 failed `MAX_TRIES = 400` at `787ccf8`).
- Zero-`if` cap headroom (`headroom_shared`, `headroom_any`) is `None` in both claim strata because both the shared 23 `if` cells and union 52 `if` cells exceed a file of 20. `test_twelve_three_value_packs_allocate_with_no_within_file_repeats` verifies that 12 three-value packs (6 real + 6 synthetic clones, 240 builds per file) allocate under `zero_if_cap = 2` with 0 zero-`if` visits and 0 within-file repeats. If headroom ever runs out again, `capacity_report()` names `polarity_mode or in/not_in for claim cells` as the next step.
- The committed record (`src/role_set/generate/t10_record.json`) covers seeds `1–20`, `61–80`, `81–100`, and locked verdict seeds `101–120` (1,280 builds) and records `if_shape` on every entry.

**How to add a new three-value pack (Phase 2).**
1. Add the pack entry (with its `sense` block) to `PACKS` in `src/role_set/generate/recipes.py`.
2. Regenerate the usable cell table and run the capacity gate:
   ```
   uv run python scripts/t10_usable_cells.py
   ```
3. Regenerate the per-stratum tuning-seed capacity table:
   ```
   uv run python scripts/t10_cell_capacity.py
   ```
4. Regenerate the committed variety record:
   ```
   uv run python scripts/t10_variety_record.py
   ```
5. Run `uv run pytest -m "not sweep"` and the `workflow_dispatch` `sweep` job in CI.
