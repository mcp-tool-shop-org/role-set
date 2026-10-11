<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/role-set/readme.png" alt="role-set" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/role-set"><img src="https://codecov.io/gh/mcp-tool-shop-org/role-set/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/role-set/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**Prototype, developed in the open.** Advisory and experimental: interfaces and data shapes change without notice.

role-set plans role-shaping items, writes the parts a writer should write, and records which gates each item passed. It is a generator, not a judge. What a good verifier is lives in the role pack, which humans write. Nothing this program writes is in a training set. Every item goes through R&D's key-check and a second read before it nears one.

The writer and the critic are reached only at `http://127.0.0.1` with an explicit port. Anything else is refused before a byte is sent. role-set never rents a machine and never touches a pod. When a run uses a pod, offrig is what rents it, and role-set only sees the loopback tunnel.

The design draft is [docs/role-set-design.md](docs/role-set-design.md). Where that draft still says a trait mismatch is a refusal, or that this program writes on a rented pod, the program is ahead of the draft: the trait check is advisory, and a pod is offrig's. The run notes are in [docs/runs/](docs/runs/).

## Install

Python 3.12, managed by uv. One runtime dependency: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si), pinned to an aspire-si commit. The gates are imported, not copied.

```text
uv sync --frozen
```

## Commands

`role-set --help` lists the commands. Each line below is a command that exists today.

`plan` fixes every slot for a batch from a seed. It does not call a model. The same seed reproduces the plan.

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` estimates pod hours and dollars for that plan before anything is launched. The figure is an estimate. The profiles are `two-card` and `frontier`.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` copies the role pack from a local aspire-si clone at a commit. `pack verify` checks that copy against its manifest. The clone path is not recorded. A fresh clone of this repo is not a synced pack: `roles/verifier` holds the exclude list and the source list, and sync fills the rest.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` copies licence-checked excerpts into a local pool. The clone directory is not recorded.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` is the runner. `--mock` runs the whole pipeline with no model and no spend, and it does not ask for a grant. Mock output is marked, it is not data, and it is written under `<out>/mock/`. A run that calls a model needs `--endpoint` and `--model`, and the endpoint must be loopback. Loading the critic, or a local rehearsal writer, also needs `--grant` and `--heartbeat`. `--no-critic` is a dry run: items pass without the second reader, and the pod writer then does not use the local GPU.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` reports one run per lesson and kind, including a run that accepted nothing. It reads the directory that holds `batch-report.json`. A finished `write` leaves that file. `readout` is the command that turns it into the report.

```text
uv run role-set readout --run out/mock
```

`critic-control` measures the critic on reviewed items. It needs a grant and a heartbeat file. `--trait-leg` and `--answer-leg` run those two measurements.

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` records receipts into the store. The firewall runs at the door. `store status` reports the file. `store void` appends a void for a round. Readers skip a voided round. It does not delete rows.

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` is the paid launcher: one paste, then it waits for the loopback lane, writes, and can relaunch after a lost-call stop. The tests never call it against a live endpoint. CI uses `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` runs the kind B solver over a data file and the `.models.jsonl` sidecar beside it. Give it every data file in the directory at once: a puzzle whose fact skeleton repeats an earlier one, with names, line order and direction set aside, is refused, naming the first.

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## Gates

Five kinds:

- **A.** A teaching passage. No verdict. A validator runs, and the trait check is advisory.
- **B.** A finite assignment puzzle. The solver checks the model. The critic's answer is advisory.
- **C.** A checker with a planted flaw, or a checker who was right. Verdict and evidence checks refuse. The flaw check is recorded and does not refuse.
- **D.** Two turns. Verdict and evidence checks refuse. The trait check is advisory.
- **E.** A written quiz, or a verdict quiz. A written E has no verdict check. A verdict-type E checks verdict and evidence.

Verdict and evidence checks refuse. Trait, answer, and flaw checks are recorded and do not refuse, until their controls pass. The trait leg measured 40.3% overall against an 85% bar, and the worst trait 12.5% against 70%. The answer leg measured 62.7% against a 90% bar. The 2026-10-11 flaw control had 9 of 24. Gating trait or answer on those numbers would refuse items already known to be good.

The B solver covers finite assignment puzzles written in its closed set of rule forms: at most 6 entities, 3 attributes, and 200,000 assignments. It does not check the thinking against the model. A verdict-keyed B item over free-text material is outside the solver. `puzzle-check` counts that item as outside, never as passed.

The store's firewall refuses held-out, sealed, and interview material at ingest, and it refuses DEV material during a stage. There is no read path for held-out, sealed, or interview material. The store, ingest, and the firewall are built. Steering is designed and has never run.

## What a run keeps

Every accepted receipt carries the plan hash and the pack hash. When the run used exemplars, the receipt also carries the exemplar commit and the exclude-file hash. When it used the stuck rule, the receipt carries that rule's hash. A run that accepts nothing has no receipts. Every writer call is kept in `drafts.jsonl`, accepted or refused. A finished write leaves `batch-report.json`, and `readout` reports that directory.

## Models

The writer pin is Qwen3-235B-A22B-Instruct-2507, the AWQ build, Apache-2.0. `writers.json` serves it as `qwen3-235b-instruct-2507` at revision `96548f657d772187f16e9f04bc50aec78a9f9cdd`.

The critic pin is `gemma4:31b`, digest `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, quant Q4_K_M, Apache-2.0. Its endpoint in `writers.json` is `http://127.0.0.1:11434`.

`writers.json` also lists a rehearsal-only writer, `qwen3:8b`, and a comparison critic, `granite4.1:30b` (IBM Granite 4.1, Apache-2.0). `write` refuses the comparison critic as the run's critic. A name that is not in the file is refused. A name containing `cloud` is refused.

The local GPU is used for the critic, and for a rehearsal writer, and only inside a grant. Those paths require `--grant` and a heartbeat file whose modified time is under 60 seconds. `critic-control` requires both on every run. The heartbeat file is not opened. When a run loaded the critic, it is unloaded on the way out, including when the run stops with an error. `--mock` loads nothing.

## Pilots

Three paid pilots are recorded in [docs/runs/](docs/runs/). These are the only cost figures this README states. No generator output from them is in a training set.

- **Pilot 1, 2026-10-11, $3.52.** None of the 104 accepted rows is data. R&D held the 43 C and D rows as mis-assembled before any content review. The note is [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md).
- **Pilot 2, 2026-10-11, $3.61.** R&D's key-check of the 49 candidates: 15 passed as written, 15 more needed a mechanical fix, and 19 failed. The note is [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md).
- **Pilot 3, 2026-10-11, $2.80.** It wrote nothing. The writer's engine hung after startup. The note is [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md).

## Threat model

role-set is a local generator. It is not a server. It does not listen, and it does not collect telemetry.

A run touches the plan, the role pack, and the output directory you name. `ingest` and `store` also touch the store file you name. `write`, `pilot-auto`, and `critic-control` touch a loopback endpoint when they are not `--mock`. `--mock` sends nothing.

A run does not touch a cloud model service, a pod, a rental account, or a rental API key. It does not read sealed rows, held-out sets, or interview sittings. `ingest` does read the twelve public interview questions from the aspire-si clone you name, at the commit you name, and only so it can refuse a row that repeats them. Those questions are not stored. `store status` returns counts, not rows. Report a vulnerability through the contact in [SECURITY.md](SECURITY.md).

## Tests

```text
uv sync --frozen
uv run pytest
```

On this tree that is 241 tests and 97.64% coverage. The floor is 90, in `pyproject.toml` and in CI, and the run fails under it. CI also runs `role-set plan` and `puzzle-check` on a fixture, with no model. `scripts/verify` runs the same local checks in one command.

The handbook is published with the [landing page](https://mcp-tool-shop-org.github.io/role-set/). This README stands on its own.

Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
