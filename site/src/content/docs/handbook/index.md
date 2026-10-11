---
title: role-set
description: What the prototype does, what it refuses, and where to read next.
sidebar:
  order: 0
---

role-set is a local generator for role-shaping items. It plans a batch, asks a model for the parts a writer should write, and records which gates each item passed.

**Prototype, developed in the open.** Advisory and experimental: interfaces and data shapes change without notice. Version 0.1.0 is the first release of that prototype. Nothing it writes is in a training set. Every item goes through R&D's key-check and a second read before it nears one.

The program is not a judge. The role pack, which humans write, is what a good verifier is. role-set serves that pack in slices and runs structural checks. A refusal is a failed check, or a known-answer mismatch on verdict and evidence. The critic agreeing is not a pass.

## What a session is

A session is a directory of files, not a server. You name the plan, the pack, the output directory, and, when you use the store, the database file. `write --mock` does the rest of the pipeline with stand-ins, so CI never calls a model.

A real writer or critic has to be `http://127.0.0.1` with an explicit port. The call is refused before anything is sent if the host is anything else. role-set does not rent a machine and does not talk to a pod. offrig is the program that rents one, when a run needs one. role-set only sees the tunnel that lands on the loopback.

## Where the numbers come from

Three paid pilots are written up under `docs/runs/` in the repository. Pilot 1 cost $3.52, and none of its 104 accepted rows is data. Pilot 2 cost $3.61, and R&D's key-check of 49 candidates was 15 pass, 15 in need of a mechanical fix, and 19 fail. Pilot 3 cost $2.80 and wrote nothing, because the writer's engine hung after startup. Those three notes are the only cost record this handbook uses. The `cost` command is an estimate for a plan you have not launched.

## Read next

- [Getting started](./getting-started/) installs the tool and runs a plan, a cost estimate, and a mock write.
- [Usage](./usage/) is the real runner, `pilot-auto`, and `readout`.
- [Kinds and gates](./kinds/) is A through E, and the limits of the B solver.
- [Configuration](./configuration/) is `writers.json`, the pins, and the pack.
- [Store and firewall](./store/) is ingest, voids, and what is refused at the door.
- [Security](./security/) is what a run touches.

The design draft is `docs/role-set-design.md` in the repository. A few early sentences there still say that a trait mismatch refuses, and that the model writes on a rented pod. The program has moved on: the trait check is advisory, and a pod is offrig's.
