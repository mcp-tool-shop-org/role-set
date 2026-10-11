---
title: Security
description: What a role-set run touches, what it refuses, and how to report a vulnerability.
sidebar:
  order: 6
---

role-set is a local generator. It is not a server, it does not listen, and it does not collect telemetry. The threat model in the README is the short form. This page is the operator's version. Reporting contacts and the supported-version table are in `SECURITY.md` at the repository root.

## The boundary

A writer call and a critic call are HTTP to `127.0.0.1` with an explicit port, and only `http`. The check runs before a byte is sent. A name that contains `cloud` is refused, and a name absent from `writers.json` is refused. A critic tag whose digest does not match the pin is refused.

role-set never rents a machine, never reads a rental API key, and never shuts a pod down. If a launch was made with offrig and this command then stops with an error, the pod is still running until offrig stops it. Pilot 3 is the case to remember: the engine hung, this program wrote nothing, and the pod was offrig's to shut down.

## The local GPU

The critic, and a rehearsal writer, use the local GPU. Those paths require a grant id and a heartbeat file newer than 60 seconds by modified time. The file is not opened, so the check does not depend on what the file contains. On the way out, a model the run loaded is unloaded, including when the run errors.

`--mock` does not load a model and does not ask for a grant. `--no-critic` with the pod writer does not use the local GPU. That is a dry run, not a quieter form of the real check.

## The store

`ingest` will not take a receipts path that names a held-out set, and it will not take a row that looks like an interview sitting, a sealed id, or a DEV id. DEV aggregates between stages need an explicit amendment flag and a dated id. The twelve public interview questions are read from the clone you name, at the commit you name, so a repeating span can be refused. They are not stored.

`store status` returns counts. There is no command that selects a stored row, and no command that opens sealed rows, held-out sets, or interview sittings.

`store void` appends a void. It does not delete.

## What you name, it can read

There is no sandbox in this release. The paths you pass are the paths it uses: the plan, the pack, the output directory, the pool, the store file, the clones for `pack sync`, `harvest`, and `ingest`, and the heartbeat file's modified time. Do not point those flags at a directory you do not mean to expose to a local process.

`write` does not scan the pack for operator identity. A pack you publish is yours to scan first.

## Reporting

Use GitHub's private security advisories on this repository, or the contact in `SECURITY.md`. Do not open a public issue for a vulnerability. The supported line is 0.1.x, as a prototype. Interfaces can change without notice. That includes the refusal text and the exit codes, which are per command and are not a single 0/1/2/3 scheme.
