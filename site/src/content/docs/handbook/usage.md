---
title: Usage
description: The runner, the paid launcher, and the readout you run afterward.
sidebar:
  order: 2
---

`write` is the runner. `pilot-auto` is the one-paste launcher around it for a paid sitting. `readout` is a separate command you run on the directory the runner left behind. None of the three is a training-set export.

## write

A mock was the getting-started path. A run that calls a model adds `--endpoint` and `--model`. The endpoint has to be plain `http` on `127.0.0.1` with a port. `https`, a name, or a missing port is refused before a request is built.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out \
  --endpoint http://127.0.0.1:8000/v1 --model qwen3-235b-instruct-2507 \
  --grant <grant> --heartbeat <heartbeat file>
```

The grant and the heartbeat are required when the run loads the critic, which is the default, or when the model is a local rehearsal writer. `--no-critic` skips the second reader and, with the pod writer, does not use the local GPU. That flag is a dry run. Items can pass it without the check a real batch is supposed to have.

The heartbeat file is judged by its modified time. It has to be under 60 seconds old. The file is not opened. When the run loaded the critic, the process unloads it on the way out, including when the run stops with an error.

`--concurrency` defaults to 1. `--stop-after-minutes` starts no new slot after the budget. Slots already running finish. `--pool` points at a harvested excerpt file. Exemplar and stuck-rule flags are optional and are recorded on the receipts when you pass them. The flag list is `role-set write --help`.

### Where the files go

| You passed | Directory |
|---|---|
| `--mock` | `<out>/mock/` |
| a rehearsal-only writer | `<out>/rehearsal/` |
| anything else | `<out>/` |

A rehearsal-only writer is refused if that directory is not named `rehearsal`. Mock output and rehearsal output are kept off the path a later step might treat as a real batch.

Inside the directory, a finished run has `batch-report.json`. Accepted rows, when there are any, are in `accepted.jsonl`, with one receipt per row in `receipts.jsonl` and a review copy in `review.jsonl`. Every writer call, accepted or refused, is appended to `drafts.jsonl`. Refusals are also in `refusals.jsonl`. A run that accepts nothing has no receipts. Pilot 3 is that case: the engine hung, and nothing was written.

Exit 2 means the run was refused before it started, or it halted. Exit 0 means it finished without a halt. A halt is still a finished directory: the batch report is written on the way out.

## readout

`readout` needs `batch-report.json` in the directory you name. It writes `readout.json` and `readout.md` there, or to `--out` if you pass one. `--compare` takes earlier run directories.

```text
uv run role-set readout --run out/mock
uv run role-set readout --run out --compare out/earlier
```

The report is per lesson and kind, for a run that passed and for a run that failed. It does not decide that a row is training data. That decision is R&D's key-check and a second read.

## pilot-auto

This is the paid path. The tests do not call it against a live endpoint. CI uses `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

The four arguments are the lane's port on the loopback, the critic's grant, the pod's hard deadline as `HH:MM` UTC, and a work directory that already holds `plan.json`, `pack/`, and `pool.jsonl`. Attempts are written under `attempt-N/`. `--serve-by`, also `HH:MM` UTC, gives up if the lane has not started serving by then.

The command waits for the lane, writes, and can relaunch after a lost-call stop. It does not rent the pod and it does not shut the pod down. If the engine hangs, the pod is still offrig's to stop. Pilot 3 was the first run of this command. It cost $2.80 and wrote nothing.

Exit 0 means the attempts list is non-empty. Exit 3 means it is empty.

## harvest

`harvest` builds the local excerpt pool `write --pool` reads. You pass an allow-list JSON file (`{"repos": [{"name", "commit"}]}`) and a directory of clones named by those repos. The clone directory is not recorded in the pool.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory> --out out/pool.jsonl
```

Exit 0 means the pool has excerpts. Exit 2 means it does not.

## puzzle-check

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
uv run role-set puzzle-check --json path/to/batch.jsonl
```

Each data file `<name>.jsonl` is read beside `<name>.models.jsonl`. `--stamp` writes `row_sha256` onto sidecar lines from the rows as they stand. Use that once, on purpose, when you are creating the sidecar. After that, a changed row no longer matches, which is the point. Exit 1 means some check failed. The limits are on the [kinds page](./kinds/).
