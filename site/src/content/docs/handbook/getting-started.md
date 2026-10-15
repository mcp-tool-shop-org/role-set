---
title: Getting started
description: Install role-set, plan a batch, estimate the cost, and run it with the mock.
sidebar:
  order: 1
---

Python 3.12 and [uv](https://docs.astral.sh/uv/). The only runtime dependency is aspire-si's gates package, pinned to one commit of [aspire-si](https://github.com/mcp-tool-shop-org/aspire-si). The gates are imported. They are not copied into this repo.

```text
uv sync --frozen
uv run role-set --version
```

`0.2.0` is the prototype. It is still 0.x: interfaces and data shapes can change without notice.

## Plan

`plan` chooses every slot before a model exists in the picture. A slot is one item's lesson, kind, side, and the other fields the writer is not allowed to invent so the floors hold. The same seed reproduces the same file.

```text
uv run role-set plan --role verifier --lessons T02,T03 --seed 0 --out plan.json
```

`--role` defaults to `verifier`. `--seed` defaults to 0. `--out` defaults to `plan.json`. `--lessons` is required, as a comma-separated list such as `T02,T03,T06`. A lesson id the planner does not know is a refusal, and the command exits 2.

Open the file before you spend anything. It is the batch R&D can read with no model call in it.

## Cost

`cost` turns that plan into pod hours and dollars. It does not launch anything, and the figure is an estimate. Live prices, when a pod is actually rented, are offrig's business at launch time.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`--profile` is required. The choices are `two-card` and `frontier`. `--attempts` defaults to 1.5 and scales the estimate. `--critic-on-pod` counts the critic on the pod instead of on the local GPU. This handbook does not publish the hourly rates or the token rates baked into the estimate. The only dollar figures recorded from real runs are the three pilots on the landing page.

## The pack, then a mock run

A fresh clone is not a synced role pack. `roles/verifier` in the repo holds the exclude list and the source list. `write` checks the pack before it does anything else, including `--mock`, so sync first:

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha> --out roles/verifier
uv run role-set pack verify --pack roles/verifier
```

The clone path is an argument. It is not written into the pack manifest.

`--mock` then runs the pipeline with a stand-in writer and critic. It does not ask for a grant, it sends nothing, and the output is marked so it cannot be mistaken for data. With `--out out`, the files land in `out/mock/`.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
```

## Tests

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

That is the suite CI's `test` job runs. Line coverage has a floor of 90, set in `pyproject.toml` and matched by the Codecov targets in `codecov.yml`, and the run fails under it. It measures about 97. The suite includes the T10 record check: it recomputes the record's hash and the variety floors from the committed record in minutes. The exhaustive rebuild is the `sweep` marker, which CI runs only on `workflow_dispatch`. See [The T10 generator](./generator/).

CI also runs atlas check, `plan` on lessons `T02,T03` and `puzzle-check` on `tests/fixtures/puzzle-check/sample.jsonl`. No step loads a model.

`scripts/verify.ps1` (and `scripts/verify.sh`) runs a local set in one command: sync, pytest, one plan, one puzzle-check. Its pytest call follows the `addopts` in `pyproject.toml`, which skip the sweep, so it runs the same suite as CI's `test` job.

## Try the generator

The generator has no `role-set` command. Its checks run on the committed files and call no model:

```text
uv run python scripts/t10_usable_cells.py --check
uv run python scripts/t10_variety_record.py --check
```

The first prints supply, demand and headroom for each stratum. The second verifies the committed record and prints the locked-seed figures.

## What you have not done yet

A mock run does not exercise the grant, the heartbeat, or a loopback model. Those are in [Usage](./usage/). The pin file is in [Configuration](./configuration/), and the packs are in [Role packs](./roles/).
