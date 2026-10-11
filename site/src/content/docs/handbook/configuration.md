---
title: Configuration
description: The writer pins, the critic pin, and the role pack a run is allowed to serve.
sidebar:
  order: 4
---

Two files decide what a run is allowed to call, and what text it is allowed to serve. Both are in the repository, and neither is a secret.

## writers.json

The file lists licence-checked writers and critics. A model name that is not in it is refused. A name that contains `cloud` is refused even if someone adds it.

The writer a paid batch is pinned to:

| Field | Value |
|---|---|
| Served as | `qwen3-235b-instruct-2507` |
| Model | Qwen3-235B-A22B-Instruct-2507, AWQ build |
| Revision | `96548f657d772187f16e9f04bc50aec78a9f9cdd` |
| Licence | Apache-2.0 |

The critic a paid batch is pinned to:

| Field | Value |
|---|---|
| Served as | `gemma4:31b` |
| Digest | `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` |
| Quant | Q4_K_M |
| Licence | Apache-2.0 |
| Endpoint | `http://127.0.0.1:11434` |

A critic whose local digest differs from that pin is refused. Re-pulling weights under the same tag does not silently become the pin.

Two other rows are not substitutes:

- `qwen3:8b` is rehearsal-only. It is pinned by digest, it runs on the local GPU, and it needs the same grant and heartbeat as the critic. Its output is written under a `rehearsal/` directory, and a request to write it anywhere else is refused.
- `granite4.1:30b` is IBM Granite 4.1, Apache-2.0, listed as a comparison critic. `write` refuses it as the run's critic.

`--writers` defaults to `writers.json` in the working directory. Point it elsewhere only when you mean to.

## The grant

The local GPU is used for the critic and for a rehearsal writer. `write` on those paths, and `critic-control` on every path, require:

- `--grant`, a grant id for the batch window
- `--heartbeat`, a file whose modified time is under 60 seconds

The heartbeat file is not opened. There is no flag that turns the check off. `--mock` never reaches it, because the mock loads nothing.

## The pack

`pack sync` copies the humans' text out of a local aspire-si clone at one commit.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha> --role verifier --out roles/verifier
uv run role-set pack verify --pack roles/verifier
```

`--role` defaults to `verifier` and `--out` defaults to `roles/verifier`. The clone path is not stored. The commit is. `pack verify` recomputes the hash and checks it against the manifest sync wrote.

What sync is expected to fill: the role lessons, one file per trait, the kind table, amendments, rules, an exclusion list, the puzzle-structure menu, pressure kinds, injection kinds, and exemplars taken from merged batches. This repository does not ship that text. `roles/verifier` here has `sources.json`, `exemplar-exclude.json`, and a README that names the layout. Run sync before a write that should see the lessons.

`write` does not scan the pack for operator identity. Scan a pack before you publish one.

## What lands on an accepted receipt

An accepted receipt spreads the run's pins. Always: the plan hash (`plan_sha256`) and the pack hash (`pack_sha256`), plus the pack commit. When you passed exemplars: `exemplar_commit`, and `exemplar_exclude_sha256` when you also passed an exclude file. When the stuck rule was loaded: `stuck_rule_sha256`. The exemplar text itself is not hashed onto the receipt. The commit is the pin for which text was read.

`drafts.jsonl` is every writer call. It does not carry those pins on each line. `batch-report.json` is the run summary. It is not a second copy of the pins. A run with no accepted row has no `receipts.jsonl`.

The gates package version and its hash are on the receipt too, so a later reader can see which checks the row was built against.
