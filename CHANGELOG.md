# Changelog

All notable changes to role-set are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and role-set follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `roles/verifier/defects.jsonl`: one row per class of defect the reviews found, with how role-set handles it (gate, puzzle-check, brief, advisory critic, or none). `readout` ends with the open classes (`--catalog` names another file).
- `write --closed-form`: the gates' closed-form settings for written kind-E variants (keyed carry verbatim, paraphrase or either; share; distractor margin; span), resolved through the pinned gates, whose defaults apply when it is unset. Every receipt records the settings.

### Changed

- The README, handbook and landing page state the coverage floor instead of a test count that changes with every PR.
- `puzzle-check` reads a negative clue on a two-value attribute as the other value ("is not lit" is "is dark") when it compares skeletons. Two copies that had passed now refuse: T10-B-01 and T11-B-05 in aspire-si#106, and T06-B-02 against T02-B-02 in batch 4.
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
