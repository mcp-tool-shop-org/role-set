---
title: Store and firewall
description: How ingest refuses held-out material, and what store status does not return.
sidebar:
  order: 7
---

The store is a SQLite file you name. It is not inside this repo and not inside the aspire-si clone. `ingest` is the only command that writes rows into it. `store` reports counts, or appends a void. Neither command prints a stored row back out.

Steering, the loop that would read the store and change a later plan, is designed and has never run. Building the file is not the same as running that loop.

## ingest

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
```

The receipts directory has to be shaped `stage1-train/<round>/<arm>/<seed>/receipts`. A path that names a held-out set is refused before any row is read. The round, arm, and seed in the path have to match the same fields on every row.

`--repo` and `--commit` are read only for the twelve public interview questions at that commit. `ingest` hashes them and refuses a row whose text shares an 8-word span with a question. The questions are not written into the store. `--questions-sha` defaults to the prefix `b59927e1`. A different hash is a refusal, so a silently edited question list does not become the firewall.

Exit 2 means the command was refused, or at least one row was refused. Exit 0 means every row entered. Refused rows are listed. They are not inserted.

## What the firewall refuses

- A row with a `q` or `sitting` field. That is the shape of an interview sitting.
- An id with a sealed prefix (`sx-`) or a DEV prefix (`pl-` or `rv-`).
- Text that shares an 8-word span with the twelve interview questions.
- A source outside the closed list. `dev_between_stages` is in that list only when you also pass `--dev-between-stages` and `--amendment` with a dated id. During a stage, DEV material is refused.
- An arm other than `ROLE` or `NO-ROLE`.
- A row whose round, arm, or seed does not match its directory.
- A diagnostics field that contains an absolute path.

There is no flag that admits sealed rows, held-out sets, or interview sittings. The amendment flag admits DEV aggregates between stages, and only those, and only with the amendment id recorded on the ingest.

## status and void

```text
uv run role-set store status --store store.sqlite
uv run role-set store void --store store.sqlite --round 3 --reason "replaced by the next batch"
```

`status` prints counts: quiz rows and diagnostic rows by round, and how many ingests the file has seen. It does not print the rows.

`void` appends a row to the voids table with the round, the reason, and the time. Readers are expected to skip a voided round. The quiz rows and the diagnostic rows stay in the file. Void is not a delete.

## What this is not

The store is not a training set, and ingest is not a promotion into one. A receipt can enter the file and still be waiting on R&D's key-check and a second read. Pilot 2's 49 candidates are the picture: the runner had accepted them, and the key-check then scored 15 pass, 15 in need of a mechanical fix, and 19 fail. The firewall does not know that score. It only knows the shapes above.
