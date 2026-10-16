---
title: Configuration
description: The proven model list, the pins a run must match, and the role pack a run is allowed to serve.
sidebar:
  order: 6
---

Two files decide what a run is allowed to call, and what text it is allowed to serve. Both are in the repository, and neither is a secret.

## writers.json

The file is the proven list. A model is on it only when its maker's own documentation allows training on its outputs. A name that is not on it is refused, and a name that contains `cloud` is refused even if someone adds it.

Every entry needs its proof, and the list is refused when it loads if any entry lacks one:

| Field | What it holds |
|---|---|
| `family` | The model family, for the critic check |
| `licence` | The licence the maker's card names |
| `proof_url` | The maker's live page (`https://`) |
| `proof_clause` | The clause, quoted |
| `proof_date` | The day it was read, `YYYY-MM-DD` |
| `checked_by` | Who read it |
| `registry_id` | Its record in rnd's licence registry (`licences/registry.json`), carried onto every receipt |

| Writers | Critics |
|---|---|
| `nemotron-3-ultra-550b-a55b`, `deepseek-r1-0528`, `nemotron-3.5-super-vl-120b-a12b`, `nemotron-3.5-bolt-74b-a9b`, `muse-glimmer-30b`, `nemotron-3.5-lightning-30b-a3b` | `muse-glimmer-30b`, `nemotron-3.5-lightning-30b-a3b` |

A critic of the writer's own family is refused, so Muse-Glimmer reads the Nemotron and DeepSeek writers, and Lightning reads Muse-Glimmer.

**The pins.** Models are served by `llama-server`, built with CUDA 13.4, at Q8_0. A listed model is refused until it's pinned: `repo`, `revision`, `gguf_file` and `gguf_sha256`. `llama_server` pins the binary by `commit` and `sha256`. The wave script hashes the GGUF and the binary before launch and passes them in (`--writer-gguf-sha256`, `--critic-gguf-sha256`, `--llama-server-sha256`). Any drift refuses the run before a byte is sent.

**The run's context.** `--partition` is `production` or `trial:<name>`. `--machine` is the wave script's CUDA machine line, as JSON: `driver`, `compat` (true or false), `runtime`, `card`, and where it ran (`colab_release_tag`, `pod_id` or `rig`).

An entry with `rehearsal_only: true` writes under a `rehearsal/` directory, and a request to write it anywhere else is refused. An entry with `role: critic-comparison` runs in `critic-control` only, never as a run's critic.

`--writers` defaults to `writers.json` in the working directory. Point it elsewhere only when you mean to.

## The grant

A step runs only inside a grant unless its machine line names a rented host (`colab_release_tag` or `pod_id`). A line that names the `rig` needs the grant whatever its card says. Off a rented host, `write` and `critic-control` require:

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

The pack's files, the Advocate draft, and how the hash is built are on [Role packs](./roles/).

## What lands on an accepted receipt

An accepted receipt spreads the run's pins. Always: the plan hash (`plan_sha256`) and the pack hash (`pack_sha256`), plus the pack commit. On a real run: the `partition`, the `wave`, the writer's and critic's family, repo, revision, GGUF file and hash, licence and proof URL, the `llama-server` commit and hash, the `machine` line, and the hash of `writers.json` itself. When you passed exemplars: `exemplar_commit`, and `exemplar_exclude_sha256` when you also passed an exclude file. When the stuck rule was loaded: `stuck_rule_sha256`. The exemplar text itself is not hashed onto the receipt. The commit is the pin for which text was read.

`drafts.jsonl` is every writer call. It does not carry those pins on each line. `batch-report.json` is the run summary. It is not a second copy of the pins. A run with no accepted row has no `receipts.jsonl`.

The gates package version and its hash are on the receipt too, so a later reader can see which checks the row was built against.
