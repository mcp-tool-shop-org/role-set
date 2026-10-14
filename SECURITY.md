# Security

## Reporting

Report a vulnerability privately through GitHub's security advisories on this
repository, or by email to 64996768+mcp-tool-shop@users.noreply.github.com. Please do
not open a public issue for it.

We aim to acknowledge a report within 48 hours, judge its severity within 7 days,
and ship a fix within 30 days.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes, as a prototype |
| 0.1.x   | No |

0.2.0 is a prototype. Interfaces and data shapes can change without notice.

## Scope

role-set is a local generator. It is not a server. It does not listen, and it
does not collect telemetry.

It calls only loopback endpoints: `http://127.0.0.1` with an explicit port.
Any other host, or an address without a port, is refused before a byte is sent.
A model name that contains `cloud` is refused. A writer that is not in
`writers.json` is refused.

role-set never rents a machine and never talks to a pod. A pod, when one is
used, is offrig's. role-set only sees the tunnel offrig opened on the loopback.

The local GPU is used for the critic, and only inside a grant. `write` and
`critic-control` require `--grant` and a heartbeat file whose modified time is
under 60 seconds. The heartbeat file is not opened. On the way out, the critic
is unloaded.

The store's firewall refuses sealed material, interview material, and DEV
material during a stage, at ingest. `ingest` reads the twelve public interview
questions from the aspire-si clone you name, at the commit you name, and only
to refuse a row that repeats them. Those questions are not stored. `store
status` returns counts, not rows. There is no command that reads a stored row
back out, and none that reads sealed rows, held-out sets, or interview sittings.

## What a run touches

- The plan, the role pack, and the output directory you name.
- The store file you name, when you run `ingest` or `store`.
- The twelve public interview questions in the aspire-si clone you name, when you run `ingest`. They are read so a repeating row can be refused. They are not stored.
- A loopback HTTP endpoint, when you run `write`, `pilot-auto`, or `critic-control` without `--mock`.

`--mock` uses no model and sends nothing.

## What a run does not touch

- A cloud model service.
- A pod, a rental account, or a rental API key.
- Sealed rows, held-out sets, or interview sittings. The twelve public
  interview questions are the exception under "What a run touches": `ingest`
  reads them to refuse overlaps, and does not store them.
