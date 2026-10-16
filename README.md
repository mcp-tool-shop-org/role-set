<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/role-set/readme.png" alt="role-set" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/role-set"><img src="https://codecov.io/gh/mcp-tool-shop-org/role-set/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/role-set/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Prototype, developed in the open.** Advisory and experimental: interfaces and data shapes change without notice. Nothing role-set writes is training data until a person checks it. Every item goes through R&D's key-check and a second read before it nears a training set.

role-set plans role-shaping items, writes the parts a writer should write, and records which gates each item passed. It also builds kind B puzzles in code and proves them before a model sees them. It is a generator, not a judge. What a good role-holder is lives in a role pack, which people write.

The writer and the critic are reached only at `http://127.0.0.1` with an explicit port. Anything else is refused before a byte is sent. role-set never rents a machine and never touches a pod. When a run uses a pod, offrig rents it, and role-set only sees the loopback tunnel.

The designs are [docs/role-set-design.md](docs/role-set-design.md) and [docs/generator-design.md](docs/generator-design.md), kept in line with the program. The run notes are in [docs/runs/](docs/runs/).

## Install

Python 3.12, managed by uv. One runtime dependency: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si) 1.4.0, pinned to aspire-si commit `388a6bc`. The gates are imported, not copied.

```text
uv sync --frozen
```

## The role packs

A role pack is the text people wrote for one role. role-set serves it in slices and adds nothing to it. Its hash is on every accepted receipt.

- **The verifier** (`roles/verifier`). `pack sync` copies it from a local aspire-si clone at a commit: the lessons, the 18 traits, the kind table, the amendments and the rules. `pack verify` checks the copy against its manifest. A fresh clone holds only the source list, the exclude list and `defects.jsonl`, the catalog of defect classes the reviews found. Sync fills the rest.
- **The Advocate** (`roles/advocate`). A pack a person wrote in this repository, with no model text in it. The Director approved the pack on 2026-10-14 (commit `ffb9546` on pull request #56), and batch 1 is being built there; it is not on `main` until #56 merges. Batch 1 has three traits: `failed-gate-first`, `exact-figures` and `faithful-relay`. No Advocate item is training data until it passes the pilot read.

## The T10 generator

Kind B puzzles used to be written freehand and then checked. Now code builds the puzzle and proves it, and a model only words it. The generator is the `role_set.generate` package. T10, the lesson on strict size words ("exactly", "at least", "at most"), is the part that changed in 0.2.0. A T12 recipe (which edition was in force on a date) is also built. T11 is not.

**Proven by construction.** The builder samples a hidden solution, adds clues that are true in it until the answer is fixed, and deletes every clue the answer does not need. The solver checks the model. Then the builder runs R&D's four T10 readings: "exactly" read as "at least", "exactly" read as "at most", "at least" read as "exactly", and "at most" read as "exactly". A puzzle is kept only when one of those readings, applied to the trait clue alone, changes the answer. A puzzle that passes on some other count clue is the defect class `trait_not_tested`, and it is refused.

**The cell map.** Each pack, question and seed maps to a structural cell: the entity count `n` (3 to 5), the count word `op`, `k` (1 to n−1), and `ifs`, the number of `if` clues (0 to 2). Set questions use all three count words. Claim questions use "exactly" only, because "at least" and "at most" cannot change a claim's verdict. The usable cells are probed once and committed in `t10_cells.json`.

**The claim axis `if_shape`.** A claim cell also names the shape of its `if` clues. There are seven shapes: `none` for no `if`; `pos` or `neg` for one `if`, by the polarity of its consequent; and `chain`, `shared_consequent`, `shared_antecedent` or `disjoint` for two. A build whose reduced puzzle loses its shape retries inside the same seed.

**The allocator.** For one question and one value count, a file of 20 seeds is assigned once, before anything is built, by a min-cost flow. Each pack's 20 seeds take 20 different cells. A cell with no `if` clue is used at most twice in a file (the zero-if cap). Other visits favour cells that gave more distinct skeletons on the tuning seeds. Those weights are kept apart for two-value and three-value packs.

**The variety floors.** In every stratum (set and claim, two and three values): at least 75% distinct canonical skeletons, no skeleton more than 3 times, and no skeleton twice in one pack's file. The floors are judged on locked seeds. On the locked seeds 101 to 120, three-value claims reach 118 of 120 distinct, with a top repeat of 2.

**The committed record.** `t10_record.json` holds 1,280 builds (seeds 1–20, 61–80, 81–100 and 101–120, every pack, both questions), each with its cell, `if_shape`, tries and canonical skeleton. It carries a sha256 over the generator source, the cell table, the capacity table and the pack data. A test holds that file list closed under role-set's own imports. The normal CI test job checks the hash and the floors from the record in minutes. The exhaustive rebuild of every puzzle runs as the `sweep` job, on `workflow_dispatch` only, and fails on any difference from the record.

**The capacity gate and the headroom.** `scripts/t10_usable_cells.py --check` reports supply, demand and headroom for each stratum, and fails with the numbers when a file cannot be placed. With `if_shape`, three-value packs have 61 usable claim cells between them, against 27 before. No pack needs a zero-`if` visit, and a test places twelve three-value packs (the six real ones and six clones) with no repeat inside a file.

## Commands

`role-set --help` lists the commands. Each line below is a command that exists today. The flags are in each command's `--help`.

| Command | What it does |
|---|---|
| `plan` | Fixes every slot for a batch from a seed. No model call. The same seed writes the same plan. |
| `cost` | Estimates pod hours and dollars for a plan before anything is launched (`two-card` or `frontier`). It is an estimate. |
| `pack sync` / `pack verify` | Copies the verifier pack from a local aspire-si clone at a commit, and checks the copy against its manifest. |
| `harvest` | Copies licence-checked excerpts from pinned repos into a local pool. The clone directory is not recorded. |
| `write` | The runner. `--mock` runs it with no model and no spend. A real run needs a loopback `--endpoint` and a `--model` from `writers.json`. `--wave write`, `critique` and `rewrite` split a batch so the writer and the critic run one at a time, on different hosts if need be, with `wave-drafts.jsonl` between them. |
| `readout` | Reports one run per lesson and kind, passing or failing, from the directory that holds `batch-report.json`. |
| `critic-control` | Measures a critic on reviewed items, on its llama-server. Needs the critic's GGUF and server hashes and the machine line, plus a grant and a heartbeat file off a rented host. `--trait-leg` and `--answer-leg` run those legs. |
| `ingest` / `store` | Records receipts into the store, with the firewall at the door. `store status` reports counts. `store void` appends a void. |
| `pilot-auto` | The 2026-10-11 pilots' paid launcher. Its script names the pilots' Qwen and Gemma pins, which are off the proven list, so `write` refuses it. |
| `puzzle-check` | Runs the kind B solver over data files and their `.models.jsonl` sidecars, and refuses a repeated fact skeleton. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

The generator has no `role-set` command yet. Its three scripts make the committed tables and the record:

| Script | What it does |
|---|---|
| `scripts/t10_usable_cells.py` | Probes every cell and writes `t10_cells.json`. `--check` runs the capacity gate on the committed table and does not probe. |
| `scripts/t10_cell_capacity.py` | Counts distinct skeletons per cell on the tuning seeds 1, 21 and 41, by stratum, into `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Rebuilds every recorded puzzle into `t10_record.json`. `--check` verifies the committed record and prints the locked-seed figures. |

## Gates

Five kinds:

- **A.** A teaching passage. No verdict. A validator runs, and the trait check is advisory.
- **B.** A finite assignment puzzle. The solver checks the model. The critic's answer is advisory.
- **C.** A checker with a planted flaw, or a checker who was right. Verdict and evidence checks refuse. The flaw check is recorded and does not refuse.
- **D.** Two turns. Verdict and evidence checks refuse. The trait check is advisory.
- **E.** A written quiz, or a verdict quiz. A written E has no verdict check. A verdict-type E checks verdict and evidence.

Verdict and evidence checks refuse. Trait, answer and flaw checks are recorded and do not refuse, until their controls pass. The trait leg measured 40.3% overall against an 85% bar, and the worst trait 12.5% against 70%. The answer leg measured 62.7% against a 90% bar. The 2026-10-11 flaw control had 9 of 24.

The B solver covers finite assignment puzzles in its closed set of rule forms: at most 6 entities, 3 attributes and 200,000 assignments. It does not check the thinking against the model. A verdict-keyed B item over free-text material is outside the solver, and `puzzle-check` counts it as outside, never as passed.

The store's firewall refuses held-out, sealed and interview material at ingest, and DEV material during a stage. There is no read path for that material. Steering is designed and has never run.

## What a run keeps

Every accepted receipt carries the plan hash and the pack hash. When the run used exemplars, it also carries the exemplar commit and the exclude-file hash. When it used the stuck rule, it carries that rule's hash. A generated puzzle's receipt carries its seed, tries, readings, difficulty metrics, and the skeleton, pack and solver hashes. Every writer call is kept in `drafts.jsonl`, accepted or refused.

## Models

`writers.json` is the proven list. A model is on it only when its maker's own documentation allows training on its outputs, and each entry carries that proof: the page's URL, the clause quoted, the date it was read, and its `registry_id` in rnd's licence registry (`licences/registry.json`). An entry without its proof is refused when the list loads. A name not on the list is refused, and so is a name containing `cloud`.

The writers are Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B, Nemotron 3.5 Bolt 74B (NVIDIA's own 24GB quantization-aware build), Muse-Glimmer-30B and Nemotron 3.5 Lightning 30B. The critics are Muse-Glimmer-30B and Nemotron 3.5 Lightning. A critic of the writer's own family is refused. The Qwen, Gemma and Granite entries the 2026-10-11 pilots used are gone: their licences are silent on outputs, which isn't proof.

Models are served by llama.cpp's `llama-server`, built with CUDA 13.4, at Q8_0. A listed model is refused until it's pinned: its repo revision, GGUF file and the file's sha256. The wave script hashes the GGUF and the `llama-server` binary before launch, and the run refuses any drift from the list. Every receipt records the partition (`production` or `trial:<name>`), both models' family, revision and GGUF hash, the server's commit and hash, and the CUDA machine line.

A step runs only inside a grant unless its machine line names a Colab VM or a pod: `--grant` and a heartbeat file modified under 60 seconds ago. A line that names the rig needs them whatever its card says. The heartbeat file is not opened. `--mock` loads nothing.

## Pilots

Three paid pilots are recorded in [docs/runs/](docs/runs/), all on 2026-10-11: $3.52, $3.61 and $2.80. These are the only cost figures this README states. No row from them is in a training set. In pilot 2, R&D's key-check of 49 candidates passed 15 as written, 15 more after a mechanical fix, and failed 19. Pilot 3 wrote nothing.

## Threat model

role-set is a local generator. It is not a server. It does not listen, and it does not collect telemetry.

A run touches the plan, the role pack and the output directory you name. `ingest` and `store` also touch the store file you name. `write`, `pilot-auto` and `critic-control` touch a loopback endpoint when they are not `--mock`. `--mock` sends nothing. The generator and its scripts read and write files inside the repository and call no model.

A run does not touch a cloud model service, a pod, a rental account or a rental API key. It does not read sealed rows, held-out sets or interview sittings. `ingest` reads the twelve public interview questions from the aspire-si clone you name, at the commit you name, only to refuse a row that repeats them. They are not stored. `store status` returns counts, not rows. Report a vulnerability through the contact in [SECURITY.md](SECURITY.md).

## Tests

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

Line coverage has a floor of 90, in `pyproject.toml` and `codecov.yml`, and the run fails under it. It measures about 97. The coverage badge above shows the current figure. CI also runs atlas check, `role-set plan` and `puzzle-check` on a fixture, with no model. `scripts/verify` runs the local checks in one command.

## Contributors

Parts of this repository were Written with Grok (xAI). Grok, Argon and Claude are credited in [CONTRIBUTORS.md](CONTRIBUTORS.md).

The handbook is published with the [landing page](https://mcp-tool-shop-org.github.io/role-set/). This README stands on its own.

Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
