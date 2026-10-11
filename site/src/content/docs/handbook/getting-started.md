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

`0.1.0` is the prototype. Do not treat a later tag as this release until the maintainer cuts one.

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
uv run pytest
```

On this tree that is 241 tests and 97.64% line coverage. The floor is 90, set in `pyproject.toml`, and CI fails the job under it. CI also runs `plan` on lessons `T02,T03` and `puzzle-check` on `tests/fixtures/puzzle-check/sample.jsonl`. No step loads a model.

`scripts/verify.ps1` (and `scripts/verify.sh`) runs that same local set: sync, pytest, one plan, one puzzle-check.

## What you have not done yet

A mock run does not exercise the grant, the heartbeat, or a loopback model. Those are in [Usage](./usage/). The pack's lessons and the pin file are in [Configuration](./configuration/).
