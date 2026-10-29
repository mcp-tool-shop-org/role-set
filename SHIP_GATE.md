# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

**This release:** 0.3.0 prototype, 2026-10-29: the first PyPI release. Lines re-run for this release carry 2026-10-29. A skipped line says why. Earlier lines keep the date they were checked.

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) — executed by `npx @mcptoolshop/shipcheck security-docs` (A1: present + reporting contact, not an empty stub) (2026-10-14, re-run: passed; SECURITY.md supports 0.2.x)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) — executed by `npx @mcptoolshop/shipcheck security-docs` (A2: trust/threat-model section present + non-empty; *quality* is not machine-checkable) (2026-10-14, re-run: passed)
- [ ] `[all]` SKIP: `shipcheck secrets` exited 0 as a skip ("no publishable packages found"). It scans npm tarballs, and this repo is Python. Ticking the box would record a scan that did not run. The identity scan of a git archive is the scan this release runs before every push.
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-10-11, README and SECURITY.md)

### Default safety posture

- [ ] `[cli|mcp|desktop]` SKIP: no kill, delete, or restart command. `store void` appends a void and does not delete rows. Shutting a pod down is offrig's command, not this program's.
- [ ] `[cli|mcp|desktop]` SKIP: the operator names every path this program reads or writes. This release adds no sandbox.
- [ ] `[mcp]` SKIP: not an MCP server. There is no network server in this repo.
- [ ] `[mcp]` SKIP: not an MCP server.

## B. Error Handling

- [ ] `[all]` SKIP: a refusal prints a `REFUSED:` line and the command's exit code. A `code` / `message` / `hint` object would change the program. This release does not.
- [ ] `[cli]` SKIP: exit codes are per command, not 0 ok / 1 user error / 2 runtime / 3 partial. `plan`, `cost`, `pack`, `readout`, and `store` return 2 on a refusal and 0 otherwise. `ingest` returns 2 when the firewall refused any row. `write` returns 2 on a refusal or a halt. `critic-control` returns 2 when a leg misses its bar. `harvest` returns 2 when it wrote no excerpts. `puzzle-check` returns 1 when a check fails. `pilot-auto` returns 3 when `attempts` is empty.
- [ ] `[cli]` SKIP: there is no `--debug` flag. A caught refusal prints one line. An unexpected exception still prints a Python traceback.
- [ ] `[mcp]` SKIP: not an MCP server.
- [ ] `[mcp]` SKIP: not an MCP server.
- [ ] `[desktop]` SKIP: not a desktop app.
- [ ] `[vscode]` SKIP: not a VS Code extension.

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-10-14, Python 3.12; rewritten for the two role packs, the T10 generator and the commands as they are)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-10-14, [0.2.0] grouped Added / Changed / Fixed, new [Unreleased] heading)
- [x] `[all]` LICENSE file present and repo states support status (2026-10-14, MIT; SECURITY.md supports 0.2.x as a prototype)
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-10-14, the README command table and examples checked against `role-set --help` and each subcommand; the generator has no subcommand and the README says so)
- [ ] `[cli|mcp|desktop]` SKIP: no log-level flag. Run records go to `drafts.jsonl`, `refusals.jsonl`, and `batch-report.json`. There is no silent / normal / verbose / debug switch.
- [ ] `[mcp]` SKIP: not an MCP server.
- [ ] `[complex]` SKIP: the operator handbook is the Starlight handbook under `site/`, not a root `HANDBOOK.md`.

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-10-11, `scripts/verify.ps1` and `scripts/verify.sh`: `uv sync --frozen`, pytest, `plan`, `puzzle-check`. No model. 2026-10-14: its pytest call does not exclude the `sweep` marker, so it is slower than CI's job; noted, not changed in this release)
- [x] `[all]` Version in manifest matches git tag — executed by `npx @mcptoolshop/shipcheck manifest` (D2: manifest version not behind the newest released tag; `--expect <ver>` for a strict release-time match) (2026-10-14, version 0.2.0; `shipcheck manifest --expect 0.2.0` passed; newest tag v0.1.0, v0.2.0 is the Publisher's after merge)
- [x] `[all]` Dependency scanning runs in CI (ecosystem-appropriate) — executed by `npx @mcptoolshop/shipcheck ci` (2026-10-29; `npm audit --audit-level=high` on the site tree in ci.yml. The Python side has one pinned runtime dependency, aspire-si-gates 1.7.6, checked by `uv.lock` and the identity hash.)
- [x] `[all]` No known high/critical vulnerabilities in any dependency tree — executed by `npx @mcptoolshop/shipcheck deps` (2026-10-29; the site tree had one high advisory (undici) and `npm audit fix` cleared it, 0 vulnerabilities now. Dependabot alerting was not verified from this environment: no token in env.)
- [ ] `[all]` SKIP: no auto-PR update bot. The org rule restricts that bot. The vulnerability outcome is the `deps` line above.
- [ ] `[npm]` SKIP: not an npm package.
- [ ] `[npm]` SKIP: not an npm package.
- [x] `[npm]` `engines.node` set · `[pypi]` `python_requires` set — executed by `npx @mcptoolshop/shipcheck manifest` (D6, checked per publishable package) (2026-10-29, `requires-python = ">=3.13"`)
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build — lockfile executed by `npx @mcptoolshop/shipcheck manifest` (D7); the pypi wheel/sdist build was executed 2026-10-29: `uv build` makes both, and the wheel installs into a clean Python 3.15 venv and runs `role-set plan`; `uv.lock` is committed.)
- [ ] `[vsix]` SKIP: not a VS Code extension.
- [ ] `[desktop]` SKIP: not a desktop app.

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-10-14, the brand URL returns 200; brand pull request 102 merged 2026-10-11)
- [x] `[all]` Translations (polyglot-mcp, 8 languages): the Publisher ran polyglot (TranslateGemma 27B, local) on the final 0.2.0 README and committed `README.*.md` to this branch (checked 2026-10-14)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) (2026-10-14, `site` build wrote `dist/index.html`, `dist/handbook/index.html`, and `dist/pagefind/pagefind.js`; nine handbook pages, two new)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-10-14, `gh repo view` shows a description, the Pages homepage and three topics; refreshed strings for 0.2.0 are proposed to the Publisher, who runs `gh repo edit`)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Executed vs attested.** `shipcheck audit` only *counts these checkboxes* — it does not read your repo, so a box can be green while the fact is false. The lines that say **"executed by `npx @mcptoolshop/shipcheck <gate>`"** are backed by a command that reads the real artifact and exits 1 on the real defect. Run those gates (they are wired into shipcheck's own `verify`); don't just tick their boxes. Executed today: **A1/A2** (`security-docs`), **A3** (`secrets`), **D2/D6/D7** (`manifest`), **D3-config + OIDC/provenance** (`ci`), **real vulnerabilities + alerting** (`deps`), **D5** (`pack`), plus front-door (`front-door`) and dogfood freshness (`dogfood`). Every other line is still an attestation you are vouching for. Note the two dependency layers: `ci` proves a scanner is *configured*; `deps` proves there are *no known vulnerabilities* — a repo can pass the first while failing the second.

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
