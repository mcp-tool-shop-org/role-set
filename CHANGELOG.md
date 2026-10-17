# Changelog

All notable changes to role-set are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and role-set follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **aspire-si-gates 1.7.4** (aspire-si 1d5a43d): NF1-NF3 re-implemented clean-room after Google's ML clause ruled out Argon's 1.7.3 code (aspire-si#164). The leakage check also reads a row's statement field, and a STaR row refuses think tags. role-set's own behaviour is unchanged.
- **The cumulative exclusion state** (BUILD-SPEC B4). `role-set write --exclusion-state <exclusion-state.json>` loads every row earlier batches accepted (with their kind-B models), and the acceptance checks (`check_against`, the reply-opening check) run against those rows as well as this run's; their framings seed the writer's avoid list. A batch whose batch check is clean writes the state it leaves (its accepted rows added, its plan's skeletons, domains and frames counted, its batch id = the plan's sha256 prefix) beside its rows: inline at the end of the run, in waves from the wave that leaves nothing pending; a systemic halt, a refused batch check, halted cells or nothing accepted leaves none. The state is hashed and a damaged or edited file is refused; its sha256 is a batch pin, so every wave of a batch uses the same one; a plan already in the state is refused before any call. `role-set plan --exclusion-state` counts earlier skeletons in the B3a skeleton floor.
- **Solve-first kind B** (BUILD-SPEC B3a, R&D's rulings of 2026-10-16). For five of the eight puzzle structures the model and the answer are made in code before the writer sees the slot (`role_set.solvefirst`): a proving generator per family (order and interval: scheduling_with_exclusions; resource allocation: constraint_grid; dependency DAG: tree_with_pruning; elimination: elimination_from_a_set; underdetermined: underdetermined_with_explicit_set) picks a hidden assignment, draws true facts until the solver leaves the intended answer (or a set of two or three), prunes every fact that changes nothing, and retries a lookup; `solve` is the oracle. The writer gets the locked model, a plain gloss of each line and the fixed key, and returns only `puzzle`, `thinking` and `reply`; the model and key are injected and `check_puzzle` runs unchanged. The new `ANSWER_EARLY` gate refuses thinking that pairs the asked subject with the answer before its final third, or that doesn't cite every fact line as 'line N'. `role-set plan --domains` gives every B slot a pack (42 Claude-written packs in `b_packs.json`, one NAICS code each, distinct within a lesson, at most 4 uses per batch) and a seed; the pack's code is the slot's domain. Receipts carry `b`: the path (`solve_first` or `freehand`), family, pack, seed, tries and the model's sha256. stopping_problem, falsifier_first and when_to_stop stay freehand until the full run. The plan builds every solve-first puzzle at plan time and re-seeds a slot whose fact skeleton (puzzle-check's) another slot of its structure already has, recording `b_skeleton` and `b_reseeds`; per structure, across the batch and any earlier batches' state, it refuses fewer than 75% distinct skeletons or one used more than 3 times (R&D on #73).
- **A domain and an opening frame per slot** (BUILD-SPEC B2, R&D's rulings of 2026-10-16). `role-set plan --domains <table>` sets every slot that carries material or a passage (A, C, D, verdict E, and the C and D apply slots) in a NAICS 2022 domain, by seed, with no model call: 3-digit subsectors, distinct within a lesson and dealt evenly across a batch; an ESCALATE slot draws from the codes mapped to its escalation domain (medical 621-623, legal 5411, money 52 and 5412, safety 23, 2211, 325, 481, 482, 5413 and 9221). Every framing (C, D, verdict E) gets a frame (`ask`, `question`, `situation_then_ask`, `line_question`), balanced within a lesson and rotated across lessons. The layout of the user turn is unchanged: a frame shapes only the framing before `Material:`, and the new `FRAME` gate checks its structure (sentence count, final punctuation; `line_question` may hold no digit and no four words in a row from the material). The writer brief names the domain and the frame; the receipt's slot records the code, title and frame. `scripts/naics_table.py` builds the table from the pinned census.gov source (sha256, URL, registry id `census-naics-2022`), titles verbatim; `domains.load` refuses a table without those pins. A real `write` refuses a plan without domains.
- **Wave mode in `role-set write`** (BUILD-SPEC B1). `--wave write|critique|rewrite` splits a batch so the writer and the critic each run alone, on whatever host holds them, with `wave-drafts.jsonl` between waves (`--drafts` names a wave's input). The write wave runs every mechanical check up to and including role words and retries a mechanical refusal at once; the critique wave re-runs those checks on the stored draft, then the critic and the acceptance checks against every accepted row of the batch; the rewrite wave continues each refused draft's conversation. One attempt budget (3) per slot across all waves; the andon's counts and halts carry across waves; a wave refuses a drafts file under other batch pins, or, in a rewrite, other writer pins. Each wave seals its drafts file with `wave-drafts.jsonl.sha256` and refuses an input that doesn't match its seal (the hand-off crosses hosts); each wave records its input's sha256, and every critique receipt names the drafts file it came from and the writer's host. The runner's slot loop is split into `write_step` and `critique_step`, so inline and waves share one path. CI checks that mock waves accept the same rows and receipts as one inline run, with and without critic refusals. The mock writer's drafts now depend on the slot and attempt, not the order of calls.
- **The critic and the writer are read on `content` only;** a thinking model's `reasoning_content` never reaches a gate (R&D on #70). The Ollama transport (`ollama_chat`, `unload`, `require_ollama_digest`) is removed.
- **The proven list and its pins** (BUILD-SPEC B6, the Stage 1 regeneration). `writers.json` now lists only models whose maker's own documentation allows training on their outputs, each with `family`, `licence`, `proof_url`, `proof_clause`, `proof_date`, `checked_by` and `registry_id` (its record in rnd's licence registry, carried onto every receipt); the list is refused when it loads if any entry lacks them (`role_set.proven`). The writers are Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B, Nemotron 3.5 Bolt 74B (maker QAT), Muse-Glimmer-30B and Nemotron 3.5 Lightning; the critics are Muse-Glimmer and Lightning. The Apache-only entries (Qwen3-235B, qwen3:8b, gemma4:31b, granite4.1:30b) are removed. Models are served by `llama-server` (CUDA 13.4, Q8_0): a model is refused until its revision, GGUF file and GGUF sha256 are pinned, and a run refuses a GGUF or `llama-server` binary whose measured sha256 differs from the pin. A critic of the writer's own family is refused. `write` and `critic-control` take `--writer-gguf-sha256`, `--critic-gguf-sha256`, `--llama-server-sha256`, `--machine` (the CUDA machine line) and, for `write`, `--partition` (`production` or `trial:<name>`); every receipt records them, with the wave (`inline`) and the list's own hash. The grant and heartbeat are required unless the machine line names a Colab VM or a pod (fail closed; a rig line always needs them). The Ollama writer and critic paths are gone from `write` and `critic-control`; `pilot-auto` is refused until the wave scripts replace it.
- **`CONTRIBUTORS.md`** credits the AI seats that wrote the code: Grok ("Written with Grok", xAI), Argon and Claude. The README links it.
- **A cap on unsupported T12 claims** (Argon's review, settled by R&D): `Floor.max_unsupported_share = 0.6` within a file's claims, so claims sit in the same 40/60 band as T11, counted as floor(0.4n) to ceil(0.6n) unsupported claims (`claims_band`) so the band can be met at every size: one claim either way, one or two of three, two or three of five. Five unsupported of five is refused.
- **Apply items, the writer-independent part** (run sheet B6; aspire-si#127's schema, R&D's rulings on #128). `role-set plan --apply` plans five C and five D per lesson, ids `<lesson>-<kind>-AP-NN`, `split: "apply"`, one turn each. Assembly carries the split, refuses a second turn, and requires each T10 apply item's keyed companion claim, `r10_twin: {statement, verdict}`, with a verdict that differs from the item's and one Statement line for the apply sitting to replace. T09 apply items carry `order_free` (true only where reversing the material's lines can't change the answer; never on changelogs or timetables), for R9's reversed-material twin (aspire-si#131). T01, T04 and T14 apply items carry `silent_without_deciding` (true only where removing the deciding lines leaves the material silent; never on a cannot_tell item), for the twin that drops the deciding lines (R&D's ruling on Argon's review; aspire-si#137's SILENT_NEEDED). `write` refuses an apply plan without `--training`, the whole training set: apply items are checked by the gates against it (models included) and refused (`SPENT`) when an 8-word span of their user turn appears in a trained user turn; the training files' sha256 go on the run's pins. Apply rows are never shown as exemplars. The writer prompts and the run wait on who writes the items.
- **T12 from a skeleton catalogue, with claim questions** (R&D's T12 claim ruling; the variety follow-up to #52). `scripts/t12_catalogue.py` enumerates every minimal 2- or 3-clue T12 set and keeps one proven row per skeleton, for which-edition questions and for claims ("<edition> was in force on <date>", keyed supported or unsupported). Each pack and seed of a file takes its own row; a build is kept only when a T12 reading names another edition or flips the claim. Claims record T11's verdict as `t11_reading` (a record only), position None, and hold `Floor.min_unsupported_share` (0.4) within a file; the skeleton key records the form. Combined sweep at 3 spans: 60 of 60 distinct, against R&D's floor of 45 (it was 39 of 80 with one skeleton on 7 builds).
- **The T11 recipe** (#52). Claims decided only by reading across lines, built from a catalogue of proven skeletons (`scripts/t11_catalogue.py`, `t11_catalogue.json`), so every pack and seed of a file takes its own skeleton: 60 of 60 distinct per side on the sweep, against R&D's floor of 45 with none on more than 3. Find side: the key is unsupported, R&D's `T11.consistent_so_supported` reading (only the lines that name the claim's subject) says supported, and no single line refutes it. Confirm side: the key is supported, with a contrast on a line that doesn't name the subject that moves the verdict off supported (`contrast_verdict`; Q3 as amended by R&D on #52). No count clues, since a count is T10's trait (`D020 sibling_trait_item`); round 3's T11-B-03 and B-05 are refused.
- **The selector keeps a file's skeletons distinct** (`Floor.distinct_skeletons`, on by default; R&D's floor on #52). One T10 file of the 18 probed changes: doors/set swaps an item that repeated another's skeleton.
- **`scripts/variety_sweep.py`** reports skeleton variety per recipe against the floor: T11 per side, T10 from its record, T12 per span count.
- **File receipts** (R&D on #51). `file_receipt` records each chosen item's rests-on reading with the floor it was chosen under.
- **Defect classes D041 and D042** (R&D on aspire-si#114): `gate_tokenizer_mismatch`, a gate that claims shared machinery but tokenizes differently, and `provenance_conflated`, wording learned from the dataset treated as model-originated. Both are handled by aspire-si-gates 1.7.0's `check_not_from_traces`.

### Changed

- **The wave pilot, from BUILD-SPEC §9.2 and Argon's §9.4 review, findings 7 and 19 (composition ruled by R&D).**
  - **`role-set plan --pilot`:** 20 T02 slots (A 4; B 5, one per solve-first structure; C 4 and D 4 with one two-turn D and no half of a contrast pair; E 3: 2 written, 1 verdict). The floors are reported, not enforced. A pilot write needs a `trial:` partition, refuses `--exemplar-*` (B7's receipt check isn't needed until batch 2), and never enters the exclusion state.
  - **`role-set plan --pilot-twins`:** the pilot's 2 C and 2 D decided T04 apply items, a batch of their own, written after the pilot with its accepted rows as `--training`, so the leakage check runs for real (R&D: "the pilot can't stand in for the full run").
  - **T04's sides** join `VERIFIER_SIDES`: R&D's ruling, Claude-written prompt text; the trait is lesson 4's own opening, the counterweight R&D-ruled.
  - **Lookup cap 0 for solve-first slots** (finding 7): their generator retries on any lookup, so a lookup there is a defect. Freehand keeps `LOOKUP_CAP`.
  - **A rewrite wave rebuilds the writer's spent list** from the drafts and the exclusion state (finding 7), as an inline retry sees it.
- **Apply items can be written.** Found by the pilot's mock run:
  - no prompt asked for `silent_without_deciding` (T01, T04, T14) or `order_free` (T09), which assembly requires, so every such apply item dropped. Apply briefs now ask for them;
  - the gates' `check_item` refuses apply rows (never trained), and the runner called it on every row. Gates are pinned to **aspire-si-gates 1.7.5** (aspire-si#165, `39a84a8`), and the runner calls `check_apply_item` for `split: apply` slots.
- **The mock writer** names any lesson's trait in its warm-up, and sets the twin flag when the brief asks for it.

- **Run guards, from Argon's BUILD-SPEC §9.4 review, findings 1, 2, 6 and 15.**
  - `--no-critic` is refused on partition `production` and in a critique wave: every counted row has its second reader.
  - A critique wave looks up the drafts' writer in `writers.json`. It refuses one that isn't listed and pins that differ (a blank family, a re-pinned GGUF), and it takes the family that bars a same-family critic from the list, never from the header.
  - A critique or rewrite wave refuses drafts whose slots don't match the plan's, in order: a truncated, reordered or padded file is refused even when resealed.
  - A run refuses an `--exclusion-state` other than the one its plan was made against. The plan records `exclusion_state_sha256`, and a batch is replanned against the state it runs with.
- **Wave outputs and pack uses, from Argon's BUILD-SPEC §9.4 review, findings 8 and 10.**
  - **One folder holds the batch.** The wave that leaves nothing pending writes the whole batch's `accepted.jsonl`, `receipts.jsonl` and `accepted.models.jsonl`, across all waves, in acceptance order, and reports the batch's accepted and dropped counts. Before this, a rewrite round left 37 rows in one wave folder and 3 in another.
  - **`accepted.models.jsonl` is written** for every accepted kind B row, inline and in waves, as `puzzle-check --stamp` writes it (id, structure, model, row_sha256 of the line as written). An apply batch's `--training` reads it through the gates' `load_models`.
  - **Pack uses count across batches.** The exclusion state records `packs`, the plan passes them to `solvefirst.assign`, the least-used packs are dealt first, and the plan refuses any pack over 4 uses across the regeneration. The state file gains a `packs` field; no regeneration state exists yet, so nothing needs migrating.
- **`tree_with_pruning` draws order and neighbour facts** (`before`, `after`, `adjacent`, `not_adjacent`), from Argon's BUILD-SPEC §9.4 review, finding 9. Before/after alone folded to 3 distinct gate skeletons in 84 seeds, too few for the cumulative skeleton floor across the regeneration. It now gives 70 of 84, at most 5 tries (it was 100 or more). No fact of any form names both claim entities. Only 1 of the 70 matches a `scheduling_with_exclusions` skeleton. Every count uses `puzzle_check.skeleton`, the gates' function. #73's 36 of 84 came from an earlier generator state and can't be reproduced on main. A new test runs the floor over 11 batches with the counts carried forward (about 22 uses per structure, B4's cumulative count).
- **Text checks tightened, from Argon's BUILD-SPEC §9.4 review, findings 11–13.**
  - **Solve-first puzzles are checked strictly** (`check_puzzle(..., strict=True)` when the model is locked):
    - a fact line names no entity or value its rule doesn't use;
    - a `not_in` line rules out every value it lists ("neither A nor B", or "not A or B", never "A, not B");
    - an `entity_with` question names the value it asks after.
    
    Freehand B keeps its rules: there the writer encodes its own lines, and complementary values such as "the rest are empty" are legitimate.
  - **Both direction words on one line** ("earlier … after") are refused unless the line holds both a before and an after rule, for every B item.
  - **`ANSWER_EARLY` reads clause by clause.** A negation rules out only its own clause, so "X is not A, so X is B" up front is refused. A clause that still names a rival answer ("A or B", "whether Ama or Bodil") is weighing candidates and passes.
  - **`line_question` points at no line:**
    - no ordinal or position ("the first line", "the last item");
    - no three-word run from the material that carries a content word;
    - no two of the material's content words, not counting the subject's own words (the content kind and the NAICS domain title, R&D on #79);
    - no word the framing shares only with the deciding line(s), the real leak.
  - **The T10 record's hash** moves with the `puzzle.py` edit. Regenerating the record left every build unchanged.
- **Gates pinned to aspire-si-gates 1.7.3** (aspire-si 96acdaa), from 1.7.2. Leakage compares the statement field as well as the user turn, and a STaR row refuses think tags.
- **T09 apply slots draw only unordered content kinds.** `plan --apply` no longer assigns a changelog or a timetable to T09. Those kinds have a line order that carries meaning, and assembly refuses `order_free: true` on them.
- **The T10 record's hash** moves with those plan and assembly edits. Regenerating the record left all 1,280 builds unchanged.
- **Gates pinned to aspire-si-gates 1.7.2** (aspire-si#138, 034a3d2), from 1.7.1. It refuses apply and `-AP-` rows in training data and validates star rows separately.
- **Gates pinned to aspire-si-gates 1.7.1** (aspire-si#116, 2e7d038), from 1.4.0. It brings TRACE_REUSED, distractor_margin 0.20, and `check_against(..., models=)` signing puzzles through the shared `puzzle_sig`.
- **Puzzles reach the gates with their models** (R&D, 2026-10-15). The runner keeps each accepted kind B row's sidecar model and passes `models=` to `check_against` and `check_batch`. A kind B row with no model raises `ModelsMissing` instead of being signed by the regex fallback, and tests prove both.
- **T11 receipts name their file** (R&D on #52): `file_index`, the file of seeds the catalogue row was drawn for.
- **The newest-edition floor is a share** (R&D on #51): `min_newest_share = 0.2`, rounded up, in place of `min_newest`.
- **The T10 record's hash** moves with the T11 edits to shared generator modules. All 1,280 recorded T10 builds are unchanged.


## [0.2.0] - 2026-10-14

Prototype. Interfaces and data shapes can change without notice. Nothing this
version writes is training data until a person checks it. The headline is the
T10 generator: puzzles on the strict size words are now built in code, proven,
spread over structural cells, and held to R&D's variety floors by a record CI
checks.

### Added

- **The puzzle generator** (`role_set.generate`). Code builds a kind B puzzle and proves it, and a model only words it. The builder samples a hidden solution, adds clues that are true in it until the answer is fixed, deletes every clue the answer does not need, and proves that a named lazy reading of the trait gives a different answer. The readings are kept as data in a shortcut registry that `puzzle-check` shares. Template wording round-trips through `check_text`. Each receipt records the seed, tries, readings, three difficulty metrics, and the skeleton, pack and solver hashes. The design and R&D's rulings are in `docs/generator-design.md`.
- **The T10 cell table** (#53). Each pack, question and seed maps to a structural cell: entity count (3 to 5), count word, k, and 0 to 2 `if` clues. The usable cells are probed once and committed as `t10_cells.json` by `scripts/t10_usable_cells.py`. Clues are now deleted in tiers (`is`, then `is_not`, then `if`). The variety comes from the cell table; tiered deletion is not credited with it.
- **The trait-clue test** (#54). A T10 puzzle carries one count clue, and it is kept only when one of R&D's four readings ("exactly" as "at least", "exactly" as "at most", "at least" as "exactly", "at most" as "exactly"), applied to that clue alone, changes the answer. Claim cells use "exactly" only, because the other two words cannot change a claim's verdict. Set cells keep all three.
- **The allocator** (#49). For one question and one value count, a file of 20 seeds is assigned once, before anything is built, by a min-cost flow. Each pack's 20 seeds take 20 different cells. A cell with no `if` clue is used at most twice in a file. Other visits favour cells that gave more distinct skeletons on the tuning seeds (`t10_capacity.json`, from `scripts/t10_cell_capacity.py`).
- **The capacity gate.** `scripts/t10_usable_cells.py --check` reports supply, demand and headroom for each stratum, with each pack's `if`-cell count, and fails with those numbers when a file cannot be placed. It reads the table and does not run inside a build.
- **The committed record and the dispatch sweep.** `t10_record.json` holds every recorded build with its cell, tries and canonical skeleton, and a sha256 over the files that decide those builds. The normal CI test job checks the hash and computes the variety floors from the record in minutes. The exhaustive rebuild runs as the `sweep` job on `workflow_dispatch` only, and fails on any difference from the record. `scripts/t10_variety_record.py` regenerates it, and `--check` verifies it.
- **The `if_shape` claim axis** (#55). A claim cell also names the shape of its `if` clues: `none`; `pos` or `neg` for one `if`; `chain`, `shared_consequent`, `shared_antecedent` or `disjoint` for two. Three-value packs go from 27 usable claim cells to 61 between them, no pack needs a cell without an `if`, and a test places twelve three-value packs with no repeat inside a file. On the locked seeds 101 to 120, three-value claims reach 118 of 120 distinct skeletons with a top repeat of 2. The record now covers seeds 1–20, 61–80, 81–100 and 101–120 (1,280 builds) and stores `if_shape` on every entry.
- **The hash closure.** A test holds the record's hashed file list closed under role-set's own imports, so a change to any module a hashed file imports also changes the hash.
- The T12 recipe: which edition was in force on a date, solved on `contains`, with R&D's four T12 readings (the newest edition, the highest number, a boundary day handed to the next span, the month alone) registered as answer readings.
- The selector (`role_set.generate.select`): chooses a lesson's review file from receipts under the per-file shortcut floor and a position floor for dated lessons. Receipts record the answer `position`.
- Clue wording: at least three sentences for each clue form, and five more domain packs (a kitchen's pans, a workshop's machines, a train timetable, a garden's beds, a clinic rota) beside lamps, cups and doors, each with a sense block.
- `contains`: the solver works out which edition was in force on a date from typed day records and spans. A date or clock time used as an entity is refused. The skeleton key records the span.
- `roles/verifier/defects.jsonl`: one row per defect class the reviews found, with how role-set handles it. `readout` ends with the open classes (`--catalog` names another file).
- `write --closed-form`: the gates' closed-form settings for written kind-E variants, resolved through the pinned gates. Every receipt records the settings.
- `puzzle-check` lists, without refusing, each puzzle that asks which entity holds a value while its Question line names neither the value nor a stem of it.
- `docs/generator-design.md`: the agreed design of the generator, with R&D's rulings.

### Changed

- The README, the handbook and the landing page describe 0.2.0: the two role packs (the verifier, and the Advocate, approved and being built on #56), the T10 generator, the commands and the coverage floor. The handbook adds two pages, "The T10 generator" and "Role packs". `SECURITY.md` supports 0.2.x.
- Capacity weights are kept apart for two-value and three-value packs, and a cell's weight rises above 1 only when its tuning-seed probe had no failed build (#55).
- The zero-if cap is 2 in every stratum. It had been raised to 3 for three-value claims as a stopgap before `if_shape` (#55).
- An underdetermined puzzle's Question asks what could be ("could", "can", "might", "may", "possible").
- `puzzle-check` reads a two-value attribute's clues by what they say when it compares skeletons ("is not lit" is "is dark"; "at least 2 of 3 lit" is "at most 1 of 3 dark").
- Clinic sessions are morning, afternoon or evening, and garden beds are damp, dry or waterlogged, so no state in a pack can be read as another.
- aspire-si-gates is pinned at 1.4.0 (aspire-si 388a6bc). The closed-form carry rule is the gates' own.
- `docs/role-set-design.md` matches the program.
- The README, handbook and landing page state the coverage floor instead of a test count.

### Fixed

- Three-value claim items with an "at least" or "at most" trait passed on a second "exactly" count clue, so they tested "exactly" (class `trait_not_tested`). The trait-clue test refuses them (#54).
- Two-value and three-value packs pooled capacity weights, which over-weighted a set cell and made `cups` seed 84 fail its try budget at `787ccf8` (#55).
- Three-value claims missed the top-repeat floor (a skeleton 4 times on the locked seeds) under the zero-if cap of 3. `if_shape` clears it (#55).
- An underdetermined key that lists possible groups writes each group with "and" and separates groups with semicolons, so a flattened key no longer passes (aspire-si#106 round 3).
- The B solver refuses a puzzle whose Question line does not name the entity the model asks about (aspire-si#106).
- `puzzle-check` now refuses three copies that had passed under a negated or flipped two-value clue (T10-B-02 and T11-B-05 against T10-B-01, and T06-B-02 against T02-B-02).

## [0.1.0] - 2026-10-11

First release. Prototype. Interfaces and data shapes can change without notice.
Nothing this version writes is in a training set.

### Added

- `plan` fixes every slot for a batch from a seed. It does not call a model.
- `cost` estimates pod hours and dollars for a plan (`two-card` or `frontier`) before anything is launched. The figure is an estimate.
- `pack sync` and `pack verify` take the role pack from a local aspire-si clone at a commit, and check the copy on disk.
- `harvest` copies licence-checked excerpts into a local pool.
- `write` is the runner. `--mock` runs it with no model and no spend. A real run talks only to `http://127.0.0.1`.
- `readout` reports one run per lesson and kind.
- `critic-control` measures the critic on reviewed items. The trait and answer legs are advisory until they pass their bars.
- `ingest` and `store` record receipts. The firewall refuses sealed material, interview material, and DEV material during a stage. There is no read path for held-out material.
- `pilot-auto` waits for a loopback lane, writes, and can relaunch after a lost-call stop.
- `puzzle-check` runs the kind B solver over a reviewed data file and its models sidecar. It covers finite assignment puzzles in its closed set of rule forms, at most 6 entities, 3 attributes, and 200,000 assignments.
- `puzzle-check` refuses a puzzle whose fact skeleton repeats an earlier one across the files checked together. Entities, attributes and values are renamed, and line order and clue direction are set aside, so a renamed or mirrored copy is caught.
- The writer pin is Qwen3-235B-A22B-Instruct-2507, AWQ build, Apache-2.0. The critic pin is `gemma4:31b`, by digest, Apache-2.0 in `writers.json`.
- Verdict and evidence checks refuse. Trait, answer, and flaw checks are recorded and do not refuse, until their controls pass.
- Every accepted receipt carries the plan hash and the pack hash. When the run used exemplars, the receipt also carries the exemplar commit and the exclude-file hash. When it used the stuck rule, the receipt carries that rule's hash. Every writer call is kept in `drafts.jsonl`.
