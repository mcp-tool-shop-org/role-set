# Changelog

All notable changes to role-set are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and role-set follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The Advocate's batch-1 build. `plan --role advocate` loads the in-repo pack by its hash and records that hash on every slot. `pack sync` refuses the advocate role, because that pack is committed in the repo and is not fetched from aspire-si. The decision table, the synthetic ledger, the kind checks, the variety record, the hidden-key critic and the controls live in `role_set.advocate`. No model is called.
- `roles/advocate/`: the Advocate's batch-1 role pack, with one verdict vocabulary (`hold`, `cannot_tell`, `pilot_first`, `proceed`). A paid cloud model API is always `hold`. A paid pod run is `proceed` when the local option misses a required check, a free local pilot has passed, the data is clean, and the printed cost is within the cap, and `pilot_first` when no pilot has passed yet.
- The Advocate's phase-1 fix-up. A ledger has one studio rendering, shared by the writer, the critic, and the length and copy checks. A figure plant that crosses a stated floor or cost cap flips the flag the table reads, so the hidden verdict follows. `plan --role advocate --pilot` builds the sixteen slots, and `pilot --role advocate --mock` runs all twenty-six texts without loading a model. The pack stays `handled: none` until that pilot is read; the revision, with the handled fields, waits for the Director's re-approval with the batch go.
- The Advocate's second fix-up. A record prints the studio lines its shape carries, and a failed check carries its evidence on that line. A relay record is the message, with out-of-scope points phrased as the notes the recipient drops. The live path returns the completion's text. A prompt-completeness test holds every pilot text to the items its reference output needs. No model is loaded.
- The Advocate's third fix-up. A live pilot checks the writer and critic pins in `writers.json` before any model call. The critic asks for JSON with thinking off, and a fenced or empty reply is still a payload. Immaterial figures print as `Note:` lines. Kind B names acts by category. The writer and the critic share one decision table. `pilot --report` writes the human read from a run record. No model is loaded.
- The Advocate's second writer is `granite3.3:8b`. `llama3.1:8b` stays listed, with a note that it is not for training items. No model is loaded.
- The Advocate's fourth fix-up. The pilot sizes `num_ctx` from the longest prompt, caps `num_predict`, and refuses a truncated reply. It appends `pilot.jsonl` as it goes, and `pilot --report` can read that file alone. `pilot --preflight` measures each model's VRAM at that context. No model is loaded.

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
