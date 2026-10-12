# Changelog

All notable changes to role-set are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and role-set follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `contains`: which edition was in force on a date, worked out by the solver. A model declares a span attribute whose values are spans with start and end days, drawn from typed day records over only the named days, and the clues map entities to spans. A question asks `{"contains": day}`, and a claim states `["contains", entity, day]`. A fact line never may. A day two spans hold goes to the owner rule the item states (earlier or later by start), or else both holders stand. A question about a day no span holds is refused, and so is an owner rule over spans that never share a day. A date or clock time ("14:00", "2.40 pm", "noon"; a bare "2.40" is not one) used as an entity is refused (aspire-si#106 round 3's placeholder). The skeleton key records the span. Fixtures: the five rebuilt T12 puzzles must pass, and round 3's five must refuse.
- `docs/generator-design.md`: the agreed design of the puzzle generator, with R&D's rulings.
- `roles/verifier/defects.jsonl`: one row per class of defect the reviews found, with how role-set handles it (gate, puzzle-check, brief, advisory critic, or none). `readout` ends with the open classes (`--catalog` names another file).
- `write --closed-form`: the gates' closed-form settings for written kind-E variants (keyed carry verbatim, paraphrase or either; share; distractor margin; span), resolved through the pinned gates, whose defaults apply when it is unset. Every receipt records the settings.

### Changed

- An underdetermined puzzle's Question asks what could be ("could", "can", "might", "may", "possible"). A key that lists every possibility doesn't answer "Which stoves are hot?" (aspire-si#106 round 4). A uniqueness question keyed "undetermined" is unchanged.
- An underdetermined puzzle that asks which entities hold a value on an attribute that may repeat keys its possible answers as groups: each group is joined with "and" and the groups are separated by semicolons ("red and blue; red and green", and "elm; oak; yew" for groups of one). The form follows from the question, so a set of one-name groups never reads as one group of three. The key used to be one flattened comma list, which a key that lost the grouping also passed (aspire-si#106 round 3). `closed_answer`, the critic's answer form and `names_not_in_puzzle` read the grouped form.
- The B solver refuses a puzzle whose Question line doesn't name the entity the model asks about (`asks.of`). In aspire-si#106 the models asked the fee of an edition while the text asked the fee on a date, so no edition line was needed.
- `puzzle-check` lists, without refusing, each puzzle that asks which entity holds a value while its Question line names neither the value nor a stem of it. It's a pointer for the key-check.
- The README, handbook and landing page state the coverage floor instead of a test count that changes with every PR.
- `puzzle-check` reads a two-value attribute's clues by what they say when it compares skeletons: "is not lit" and "not (is lit)" are "is dark", and "at least 2 of 3 lit" is "at most 1 of 3 dark". Three copies that had passed now refuse: T10-B-02 and T11-B-05 against T10-B-01 in aspire-si#106, and T06-B-02 against T02-B-02 in batch 4.
- aspire-si-gates is pinned at 1.4.0 (aspire-si 388a6bc). The closed-form carry rule is the gates' own: role-set no longer checks word-for-word carry itself, and the writer's brief states the run's setting.

- `docs/role-set-design.md` matches the program. The critic's trait, answer and flaw checks are advisory until their controls pass, a pod is offrig's, and the lookup and skeleton rules are the current ones. The README and handbook no longer carry a caveat about the draft.

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
