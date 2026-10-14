---
title: Role packs
description: The verifier pack and the Advocate pack, their files, and how a pack is served, hashed and verified.
sidebar:
  order: 5
---

A role pack is the text people wrote for one role: what the role is, its traits, and the rules a writer must keep. role-set serves it to the writer in slices and adds nothing to it. The program never decides what a good role-holder is. The pack does.

There are two packs. One is in use. The other is approved and being built on a pull request.

## The verifier

`roles/verifier` is the pack every run so far has used. Its text lives in aspire-si, and `pack sync` copies it from a local aspire-si clone at one commit.

What sync fills:

```text
role.md            lessons 1–3 (the role, how it thinks, what is expected)
traits/T01..T18.md one lesson each, with the counterweight named
kinds.md           the per-kind required-parts table and content rules
amendments/        1 quoting, 2 silent ≠ contradicted / escalate / corrections, 3 role-neutral prompts
rules.md           instant agreement, three lines, time words, length limits, provenance, no identity
exclusions.txt     repos and sets the writer must never draw from
structures.yaml    the B-puzzle structure menu, with one worked example each
pressures.yaml     T13 pressure kinds
injections.yaml    T17 injection kinds and placements
exemplars/         approved items, by trait and kind, from merged batches
pack.json          the manifest sync writes
```

What the repository itself holds, before any sync:

| File | What it is |
|---|---|
| `README.md` | the layout above and the defect catalog's rules |
| `sources.json` | the allow-list for `harvest`: permissively licensed repos, each pinned to a commit |
| `exemplar-exclude.json` | reviewed ids that must never become exemplars |
| `defects.jsonl` | one row per defect class the reviews found, and how role-set handles it |

`defects.jsonl` is not synced and is not served to the writer. Each row has `id`, `name`, `description`, `first_source`, `handled` and `anchor`. `handled` is one of `gate` (a refusal in code, or a refusing critic check), `puzzle-check`, `brief` (a writer instruction only), `critic` (advisory, never refusing) or `none` (nothing yet, with `anchor` naming the candidate check). `readout` ends with every `none` row, so the open gaps are in front of every run. When a class becomes checkable, it moves off `brief` or `none` in the same change as the check.

## The Advocate (on #56)

`roles/advocate` is a pack a person wrote, with no model text in it. The Director approved it on 2026-10-14 (commit `ffb9546` on pull request #56), and batch 1 is being built there. It is not on `main` until #56 merges, and no Advocate item is training data until it passes the pilot read.

The Advocate is the person in an AI organisation who turns technical results into a clear decision, carries needs and updates in both directions, upholds ethics and governance, and brings tools in safely. Batch 1 is the daily part of that job: a faithful, decision-ready brief from a record, and a faithful relay. It has three traits, built and judged one at a time:

| Trait | Lesson | Counterweight | Kinds |
|---|---|---|---|
| `failed-gate-first` | a failed check or a tripwire is the headline, before any figure | when everything passes, say so plainly | A, C |
| `exact-figures` | every figure as its receipt has it | no figure the receipt does not have | A, C |
| `faithful-relay` | carry every point the recipient needs, add no instruction | escalate a decision the relay cannot make | B, D |

The pack's files:

```text
README.md              the role, the studio shape, the three traits, the standing rules
behaviours.jsonl       one row per defect or tripwire a planted negative can carry
sources.json           the nine synthetic record templates; no real records
exemplar-exclude.json  the verifier's exclusion list, carried over unchanged
```

`behaviours.jsonl` has the same fields as the verifier's `defects.jsonl`, and every row is `handled: none` until its check is built. Batch 1 uses synthetic records only, and names people by role title only.

The Advocate pack is not synced from aspire-si and has no `pack.json` yet. How it is hashed and served is part of the Advocate build on #56, not of this release.

## How a pack is served

The writer never gets "write a verification exercise". For each slot, the prompt carries:

1. the lesson for the slot's trait, and the kind's required parts from the kind table;
2. the slot's fixed fields and the material;
3. two or three exemplars of the same kind from a *different* trait, so form transfers and text cannot;
4. a short avoid-list of framing lines and frequent phrases already used.

When a draft is refused for reusing a span, a phrase, an opening or a framing, that text joins the avoid-list for later slots (the most recent 60 go to the writer).

## How a pack is hashed

`pack sync` reads each source file from the clone with `git show <commit>:<path>`, splits the trait lessons into one file per trait, and writes `pack.json`:

```json
{
 "role": "verifier",
 "commit": "<aspire-si commit>",
 "files": { "<name>": "<sha256>", "...": "..." },
 "pack_sha256": "<one hash over the whole pack>"
}
```

The commit is recorded. The clone path is not. Every accepted receipt carries `pack_sha256` and the commit, so a later reader knows exactly which text the writer was served.

## How a pack is verified

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha> --role verifier --out roles/verifier
uv run role-set pack verify --pack roles/verifier
```

`pack verify` re-hashes every file the manifest lists. If the whole-pack hash differs, it refuses and names the files that changed. `write` runs the same check before it does anything else, including `--mock`, so a drifted pack never reaches a writer.

`write` does not scan a pack for operator identity. Scan a pack before you publish one.
