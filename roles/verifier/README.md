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
