---
title: role-set
description: What the prototype does, what it refuses, and where to read next.
sidebar:
  order: 0
---

role-set is a local generator for role-shaping items. It plans a batch, asks a model for the parts a writer should write, and records which gates each item passed. For kind B, it can also build the puzzle in code and prove it before a model words it.

**Prototype, developed in the open.** Advisory and experimental: interfaces and data shapes change without notice. Version 0.2.0 adds the T10 generator's cell map, `if_shape` axis, allocator and record gate. Nothing role-set writes is training data until a person checks it. Every item goes through R&D's key-check and a second read before it nears a training set.

The program is not a judge. A role pack, which people write, says what a good role-holder is. role-set serves that pack in slices and runs structural checks. A refusal is a failed check, or a known-answer mismatch on verdict and evidence. The critic agreeing is not a pass.

## What a session is

A session is a directory of files, not a server. You name the plan, the pack, the output directory, and, when you use the store, the database file. `write --mock` does the rest of the pipeline with stand-ins, so CI never calls a model.

A real writer or critic has to be `http://127.0.0.1` with an explicit port. The call is refused before anything is sent if the host is anything else. role-set does not rent a machine and does not talk to a pod. offrig is the program that rents one, when a run needs one. role-set only sees the tunnel that lands on the loopback.

## Two packs, one generator

- **The verifier pack** is synced from aspire-si at a commit and is the pack every run so far has used.
- **The Advocate pack** is a draft a person wrote, on pull request #56, waiting for the Director's approval. It is not on `main`.
- **The T10 generator** builds puzzles on the strict size words, proves each one with the solver and R&D's four readings, spreads them over structural cells, and holds R&D's variety floors through a record that CI checks on every pull request.

## Where the numbers come from

Three paid pilots are written up under `docs/runs/` in the repository. Pilot 1 cost $3.52, and none of its 104 accepted rows is data. Pilot 2 cost $3.61, and R&D's key-check of 49 candidates was 15 pass, 15 in need of a mechanical fix, and 19 fail. Pilot 3 cost $2.80 and wrote nothing, because the writer's engine hung after startup. Those three notes are the only cost record this handbook uses. The `cost` command is an estimate for a plan you have not launched.

The generator's figures (cells, distinct skeletons, headroom) come from the committed tables and record in `src/role_set/generate/`, and the scripts that make them.

## Read next

- [Getting started](./getting-started/) installs the tool and runs a plan, a cost estimate, and a mock write.
- [Usage](./usage/) is the real runner, `pilot-auto`, and `readout`.
- [Kinds and gates](./kinds/) is A through E, and the limits of the B solver.
- [The T10 generator](./generator/) is cells, `if_shape`, the allocator, the floors, and the record gate.
- [Role packs](./roles/) is the verifier and the Advocate draft, and how a pack is hashed and verified.
- [Configuration](./configuration/) is `writers.json`, the pins, and the grant.
- [Store and firewall](./store/) is ingest, voids, and what is refused at the door.
- [Security](./security/) is what a run touches.

The designs are `docs/role-set-design.md` and `docs/generator-design.md` in the repository, kept in line with the program.
