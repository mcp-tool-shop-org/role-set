# The verifier role pack

This directory holds the text the humans wrote, served to the writer in slices. role-set adds nothing to it.
It is filled by `role-set pack sync` from a local aspire-si clone at a pinned commit (the lessons, the 18 traits, the kind
table, the amendments and the rules), and the pack's hash is recorded in every accepted receipt. The layout is:

```
role.md            lessons 1–3 (the role, how it thinks, what is expected)
traits/T01..T18.md one lesson each, with the counterweight named
kinds.md           the per-kind required-parts table and content rules
amendments/        1 quoting, 2 silent ≠ contradicted / escalate / corrections, 3 role-neutral prompts
rules.md           instant agreement, three lines, time words, length limits, provenance, no identity
exclusions.txt     repos and sets the writer must never draw from
structures.yaml    the B-puzzle structure menu with one worked example each
pressures.yaml     T13 pressure kinds
injections.yaml    T17 injection kinds and placements
exemplars/         approved items, by trait and kind, from merged batches
```

`write` does not scan the pack for operator identity. Scan a pack before you publish one.

## The defect catalog

`defects.jsonl` is not synced and is not served to the writer. It lists one row per class of defect a review found in
the hand- or Grok-written batches and the pilots: `id`, `name`, `description`, `first_source` (the PR and comment where
it was first found), `handled`, and `anchor`. `handled` says how role-set deals with it today:

- `gate`: a mechanical refusal, either a check in code or a critic check that refuses (verdict, evidence);
- `puzzle-check`: the B solver over reviewed data;
- `brief`: a writer instruction only;
- `critic`: an advisory critic check, logged and never refusing;
- `none`: nothing yet, and `anchor` names the candidate check.

When a review finds a new class, add a row. When a class becomes checkable mechanically, move it from `brief` to
`gate` or `puzzle-check` in the same PR as the check. `role-set readout` lists every `none` row, and a test holds it to
that.
