# B3c behaviour sources: T10, T11, T12 generator rewrite

Clean-room list. Allowed sources only: `docs/generator-design.md` lines 1-290 and 384-end ("design" below, by line); R&D and maintainer comments on role-set PRs #48-55, #63, #64 ("R&D #N", "Pub #N" for Publisher briefs relaying R&D rulings); Claude-written tests. No Grok/Argon code, data or diff was read.
Coverage key: **C** = Claude-written test (named). **G/A** = only a Grok/Argon test (not readable here). **none** = no allowed test.

## T10 (size/count trait)

| # | Behaviour | Source | Coverage |
|---|---|---|---|
| 1 | Code builds the puzzle and proves it; same seed rebuilds the same puzzle; a different seed differs. | design 7-12, 58-70 | C `test_the_same_seed_rebuilds_the_same_puzzle` |
| 2 | T10 questions are open-set ("which could be ...") or claim only. A single-value question is refused at once, before any search, with the reason "no single-value questions"; the question argument has no default and an unknown value is refused naming set and claim. | R&D #48 ruling 2 + required item 1 (2026-10-12) | C `test_t10_refuses_a_single_value_question_at_once` |
| 3 | Four size readings: `=` read as `>=` (floor), `=` read as `<=` (ceiling), `>=` read as `=`, `<=` read as `=`. Each is a rewrite of the count clue. | design 101 (R&D review), 258 | C `test_a_reading_with_nothing_to_rewrite_is_skipped`, `test_loosening_eq_rests_on_at_most_half` |
| 4 | A reading with nothing to rewrite is skipped, not an error. | tests | C `test_a_reading_with_nothing_to_rewrite_is_skipped` |
| 5 | A reading that ends in a contradiction is recorded on the receipt as "no assignment fits", `differs: false`, and proves nothing. | R&D #48 ruling 1 | C `test_a_contradiction_is_recorded_but_proves_nothing` |
| 6 | Divergence by answer type: claim differs if the lazy verdict is another verdict or cannot_tell; set differs if the set of answers differs; single-value differs only if the possible answers exclude the key; "no assignment fits" never differs. | design 100-110 (R&D review) | C `test_the_divergence_rules_by_answer_type` |
| 7 | The count clue is on the deciding path: removing it changes the open-set answer; a size reading changes the set. | design 79-80 (step 3), 118 | C `test_a_t10_open_set_puzzle_is_proven_and_its_wording_round_trips` |
| 8 | Size-word test (hard assertion): the trait clue's own operator is rewritten alone, and that reading must differ. A second count clue's reading no longer counts. `count` is out of T10's auxiliary clue pool, so each puzzle has exactly one count clue. An item rests on a reading of its trait operator. | R&D #53 correction (2026-10-14), Pub #53, R&D #54 PASS | C only partly: `test_a_count_item_rests_on_its_trait_operator` is on the excluded list. Else G/A. |
| 9 | Claim cells are "exactly" only; a claim cell with another operator is refused up front. Set cells use all three size words. | R&D #54 PASS | G/A |
| 10 | Boundary rule: when an at-least or at-most clue decides one entity directly, the true count equals k. Claim items therefore cannot separate "at least" from "exactly"; set items carry that lesson. The boundary share on sets is report-only, not a gate. | R&D #53 ruling 1, R&D #54 ruling 1 | none |
| 11 | The hidden solution is drawn uniformly from assignments where the count clue is true, not pinned to k. | R&D #53 PASS | G/A |
| 12 | Cell table: a cell is (n entities, operator, k, number of `if` clues), n in 3-5; a claim cell adds nothing else unless `if_shape` is ruled in (#55, passed by R&D). A seed maps to a cell; the 20 seeds of a pack in one file take 20 different cells. | Pub #49 amend 1-2 (R&D), R&D #53, R&D #55 | G/A |
| 13 | Tiered deletion in `reduce` and the open-set build; retry inside the seed when the reduced puzzle loses its `if` target. The tier order itself is not stated in any allowed source. Tiered deletion did not raise the `if` share; variety comes from the cell map. | R&D #53 PASS + ruling 3 | G/A (**order unknown**) |
| 14 | `if` clues are offered before unit clues while adding. | R&D #53 PASS (accepted) | G/A |
| 15 | Zero-`if` cells are visited at most 2 times per file, in every stratum (cap 3 for three-value claims was ruled 2026-10-14 then restored to 2 once `if_shape` landed). | Pub #49 amend 2; R&D #55 PASS ("cap of 2 for three-value claims") | G/A. **Conflict, resolved by order:** amend 1 (if_shape) -> amend 2 (cap 3) -> #55 (if_shape plus cap 2). #55 is last and passed. |
| 16 | `if_shape` (claim cells): one `if` is positive or negative consequent; two `if`s are chain, shared consequent, shared antecedent or disjoint (4 entities). A shape that will not build at an n is absent from the probe list. | Pub #49 amend 1 (R&D, superseded), R&D #55 PASS ("trait_not_tested 0 for every shape") | G/A |
| 17 | Capacity per stratum (set two, set three, claim two, claim three): supply, demand and headroom reported; every pack has at least 20 usable cells; it fails loudly, with the numbers, when a stratum cannot be placed; it is a check on the table, not a raise in `cell_for`. Weights are per stratum (two-value and three-value never pooled). A seventh three-value pack must not fail it (12-pack scaling test). | Pub #49 amend 1-2, R&D #55 PASS | G/A |
| 18 | `MAX_TRIES` 400, `MAX_IF` 2, `SET_MAX` 4. No try cap, no dropped costly cells. (Meaning of `SET_MAX` not stated in allowed sources.) | R&D #49, R&D #55 | G/A |
| 19 | Variety floors on every stratum: distinct skeletons >= 75%, top skeleton <= 3 builds, no repeat within one pack's file. Locked seeds, committed record with sha256 over every file in the import closure, fast test recomputes floors from stored skeletons, full sweep behind a `sweep` marker in `ci.yml` dispatch. | R&D #49 (2026-10-13/14), R&D #55 PASS | G/A |
| 20 | Metrics on the receipt: `assignments_explored` is values^entities, `load_bearing_facts` is the fact count, `lookup_distance` >= 0. | design 275-280; test | C `test_the_receipt_metrics_and_the_solver_pin` |
| 21 | Regression fixtures: round-2 T10 B-02..04 have no deciding size reading (must-refuse); round-2 B-01 is refused by puzzle-check for its flat key; round-5 B-01 is a must-pass (two counts, each bound read as exact changes the set). | design 133; R&D #48 | C `test_round2_t10_puzzles_have_no_size_reading_that_decides`, `test_round2_t10_b01_is_refused_by_puzzle_check`, `test_round5_t10_b01_carries_a_deciding_size_reading` |
| 22 | Domain packs: whole-word names; values mutually exclusive in everyday sense (clinics `session`: morning/afternoon/evening; beds damp/dry/waterlogged); 3 templates per form; "Whenever {p}, {q}." as the third `if` form; `sense` block shape. | R&D #49, Pub #49 | G/A (pack tests are Grok-written) |
| 23 | Two-value claims have no inequality cell: open gap (prove impossible or probe at full `MAX_TRIES`). | R&D #53 ruling 2 | none |

## T11 (falsifier across lines)

| # | Behaviour | Source | Coverage |
|---|---|---|---|
| 1 | Questions are claims over one unordered state attribute (forms `is`, `is_not`, `in`, `not_in`, `if` with `is` or `is_not` antecedent). A non-claim item is refused ("T11 items are claims"). | design 387 | C `test_each_t11_refusal` |
| 2 | No count clue (D020 sibling_trait_item); round-3 B-03 and B-05 refused; batch-6 B-01 and B-02 pass. | design 395, R&D #52 Q1 | C `test_round_3_count_decided_items_are_refused`, `test_batch_6_find_side_items_pass_and_the_reading_differs` |
| 3 | Reading `T11.consistent_so_supported`: keep only facts naming an entity the claim names; supported if some assignment fits with the claim true. | design 389, R&D 2026-10-11 | C `test_each_t11_refusal` (subject_reading), find-side tests |
| 4 | Find side: key unsupported, reading says supported, no single line refutes the claim alone; two subject lines that refute together are refused ("already refute"). | design 391; R&D #52 | C `test_each_t11_refusal`, `test_a_t11_puzzle_is_proven...` |
| 5 | Confirm side: key supported, no single line supports it; nothing deciding is refused on either side. | design 392; test | C `test_each_t11_refusal` |
| 6 | Contrast (Q3 amended): change one slot of one line that does not name the claim's subject, into a line that does not either, and the verdict must move off supported (cannot_tell or unsupported). Receipt records `{line, form, from, to, contrast_names_subject:false, contrast_verdict}`. No such contrast: item refused. Sweep reports the `contrast_verdict` split per side (all cannot_tell expected). | R&D #52 Q3 ruling (2026-10-15) | C `test_a_t11_puzzle_is_proven...`, `test_a_contrast_only_on_the_subjects_own_lines_is_refused`, `test_an_off_subject_contrast_moves_the_verdict_off_supported`, `test_one_slot_changes`. Split reporting in sweep: none. |
| 7 | Skeleton catalogue: every set of 2 or 3 clues over three entities for 2 and 3 values, one per `puzzle_check.skeleton`, committed JSON, stale catalogue fails a test; `t11_row` gives every pack and seed of a file a different row; build re-proves its row. | design 400-410; R&D #52 | C `test_the_committed_catalogue_is_current`, `test_every_catalogue_row_builds_and_proves` |
| 8 | Variety floor per side: >= 45 distinct of 60, none on more than 3 builds, none shared within a file/stratum. | R&D #52 (2026-10-13) | C `test_the_variety_floor_holds_on_the_sweep`, `test_a_file_never_repeats_a_skeleton_within_a_stratum` |
| 9 | Receipt records `file_index` for catalogue reuse. | R&D #52 note | C `test_a_t11_receipt_names_its_file` |
| 10 | A confirm item rests on `T11.contrast.<form>`; a file of 5 holds at most 2 find items (40/60 split). | design 412; R&D #52 Q2 | C `test_a_t11_file_mixes_the_falsifier_reading_and_contrasts` |
| 11 | Side name validated; same seed same puzzle. | tests | C `test_a_t11_side_is_named`, `test_the_same_seed_rebuilds_the_same_t11_puzzle` |

## T12 (which edition is in force)

| # | Behaviour | Source | Coverage |
|---|---|---|---|
| 1 | Typed day records (label, day, month, ordinal, `year` on a wrap); contiguous whole-month spans; `contains` inclusive at both ends, never a fact line; a day two spans hold goes to a stated owner rule, else every holder is the answer; only declared days can be asked; span facts come from clues (no plain `is`). | design 145-163, 255-262 | C `test_the_calendar_is_contiguous_typed_and_marks_a_wrap`, `test_a_t12_puzzle_is_proven...`, `test_a_position_outside_the_calendar_is_refused` |
| 2 | 3 or 4 spans; the asked date sits inside a span, sometimes its closing or opening day (opening day of last span asked 0.6 vs 0.3). | design 290-300 | C `test_a_batch_can_ask_for_a_newest_edition_answer` |
| 3 | Four readings: `newest`, `highest_number` (None unless every name has a number), `boundary_next` (acts only on a span end or start), `month_alone` (inactive until the mid-month follow-up). Kept only when at least one names another edition. | design 270-283; R&D #50 | C `test_each_t12_reading`, `test_month_alone_names_a_span_that_only_shares_the_month` |
| 4 | Answer-position floor: no position over half a file, `min_newest_share` 0.2 (rounded up); builder takes `position=`; receipt records `earliest/between/newest`, None without a dated question. | R&D #50, #51 | C `test_answer_position_is_none_without_a_dated_question`, `test_a_dated_file_floors_answer_positions` |
| 5 | Skeleton catalogue: all minimal 2- or 3-clue sets (132 per form at 3 spans, 1,001 at 4; capped 480), ask and claim targets, kept only if `newest` or `boundary_next` can differ; `_t12_plan` gives each seed an unused row; committed JSON current. | design 417-430 | C `test_the_committed_t12_catalogue_is_current` |
| 6 | Claim form "<edition> was in force on <date>": date always inside a span, keyed supported/unsupported, rests on a T12 reading that flips (or opens) the verdict; reading answers a claim with a verdict. | R&D #63 terms 1-2 | C `test_a_t12_claim_is_keyed_and_a_reading_flips_it`, `test_a_reading_answers_a_claim_with_a_verdict` |
| 7 | `t11_reading` on the claim receipt as a record only, never a reading to rest on (D020 guard). | R&D #63 term 3 | C same test |
| 8 | Skeleton key records the form (ask vs claim differ). | R&D #63 term 5 | C `test_the_skeleton_records_the_form` |
| 9 | Unsupported share 40/60 band in counts: floor(0.4n) to ceil(0.6n) (n=1 0-1, 2 0-2, 3 1-2, 4 1-3, 5 2-3); 8 of 20 seeds aim unsupported. | R&D #63 term 4, R&D #64 | C `test_claims_keep_their_unsupported_share_in_a_file`, `test_the_claims_band_is_in_counts_so_every_size_can_meet_it` |
| 10 | Variety: 60/60 combined, 80/80 per form; sweep `--check` fails below the floor. | R&D #63 term 6 | C `test_the_t12_variety_floor_holds_per_form_and_combined` |
| 11 | Form validated; same seed same puzzle, packs differ; must-pass fixtures carry a differing reading. | tests | C `test_a_t12_form_is_named`, `test_the_same_seed_rebuilds...`, `test_the_t12_rebuilds_carry_a_reading_that_names_another_edition` |
| 12 | `file_index` on T12 receipts (R&D asked, non-blocking). | R&D #63 note | none |

## Selector

| # | Behaviour | Source | Coverage |
|---|---|---|---|
| 1 | File size 5; each item rests on exactly one reading that differs on it; >= 2 distinct readings rested on; none on more than half (2 of 5); counting mode `rests_on` default, `differs` stricter and unsatisfiable for T12. | design 308-315; R&D #51 | C `test_each_item_rests_on_a_reading_that_differs_on_it`, `test_counting_every_differing_reading_is_stricter`, `test_counting_every_differing_reading_cannot_hold_for_t12`, `test_check_file_names_what_a_file_breaks` |
| 2 | Loosening `=` decides at most half; a pool that cannot fill is refused saying what is short ("generate more seeds"). | R&D #48; tests | C `test_loosening_eq_rests_on_at_most_half`, `test_a_pool_with_one_reading_is_refused_and_says_so` |
| 3 | One lesson per file; floor scales with file size; deterministic in pool order. | tests; R&D #51 | C `test_a_file_is_chosen_per_lesson_and_the_count_is_known`, `test_the_floor_scales_with_the_file` |
| 4 | Newest floor is a share (ceil(0.2 x size)); the newest slot is kept for a late newest answer. | R&D #51 | C `test_the_newest_floor_is_a_share`, `test_the_newest_slot_is_kept_for_a_late_newest_answer` |
| 5 | Distinct skeletons within a file for every lesson (`Floor.distinct_skeletons`, on by default). | design 414; R&D #52 | none named (**no allowed test**) |
| 6 | File receipt records each item's `rests_on`, problems, the floor, positions. | R&D #51, #52 | C `test_a_t11_file_mixes_the_falsifier_reading_and_contrasts` |
| 7 | Consequence: every T10 claim rests on a reading that loosens "exactly", so at least half of a T10 file must be set items with at-least or at-most. | R&D #54 ruling 2 | none |

## Shortcuts.prove and receipts

- `prove` runs named readings, returns answer and `differs` per reading; wrong-option values come from code only. Design 190-195, Tests: `test_the_divergence_rules_by_answer_type`, `test_a_contradiction_is_recorded...`. **C**.
- Trait shortcut readings must answer differently: for T10 the trait operator's own reading (item 8 above); for T12 any reading; for T11 find `consistent_so_supported`. Same pinned shortcut code for generation and puzzle-check re-run (design 191-196). Re-run by puzzle-check: **none**.
- Receipt carries seed, tries, recipe, trait clue, every reading with its answer, three difficulty metrics, skeleton hash, pack hash, solver hash (design 244-250). **C** `test_a_receipt_carries_the_seed_readings_metrics_and_pins`, `test_the_receipt_metrics_and_the_solver_pin`.
- Pin the solver version on every receipt; set no difficulty bands on assignments explored yet (design 237).

## Conflicts and thin spots

- T10 claims: design 101 lists `=`<->inequality readings for claims; R&D #54 later makes claims "exactly" only and the readings loosen "exactly" only. Treat #54 as final.
- Zero-`if` cap: 2 -> 3 -> 2 across Pub #49 amendments and R&D #55; final is 2 with `if_shape`.
- T11 Q3: first "flip to unsupported", amended to "off supported". Final is the amendment.
- Behaviours resting only on Grok/Argon tests: T10 #9, 11-19, 22 (cell table, capacity, if_shape, variety record, packs). T10 tier order and `SET_MAX` meaning are unspecified in allowed sources.
