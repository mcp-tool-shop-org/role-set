# Scorecard

**Repo:** role-set
**Date:** 2026-10-11
**Type tags:** `[all]` `[pypi]` `[cli]`

## Pre-remediation assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 3/10 | No SECURITY.md worth shipping, and the README had no threat model |
| B. Error handling | 4/10 | Refusals are `REFUSED:` lines and per-command exit codes. No structured error object, no `--debug`, no log levels |
| C. Operator docs | 3/10 | The README still said `write` was refused. No CHANGELOG, no handbook |
| D. Shipping hygiene | 5/10 | Version 0.1.0, `uv.lock` committed, coverage floor 90 already in CI. No verify script, no dependency scanner |
| E. Identity (soft) | 2/10 | A local wordmark only. No landing page, no handbook, no translations, no repo homepage |
| **Overall** | **17/50** | |

## Key gaps

1. The public README described a generator that had not shipped `write`, and called the output training data.
2. No security policy, no changelog, no handbook.
3. Exit codes and errors do not match the shipcheck shape. This release does not change them.
4. CI has no dependency scanner shipcheck recognises.

## Remediation (this treatment)

1. SECURITY.md, a threat-model section, and a CHANGELOG for 0.1.0.
2. README rewritten to the commands that exist, the critic's advisory status, and the three pilots' key-check results.
3. `scripts/verify.ps1` and `scripts/verify.sh`.
4. Landing page, seven handbook pages, and a Pages workflow. The logo is in brand pull request 102, which is not merged, so the README image URL is not live yet.
5. Generator behavior was not changed. The error shape and the missing CI scanner stay open on purpose.

## Post-remediation

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 8/10 | SECURITY.md and the threat model are in place. `shipcheck secrets` skipped, because it scans npm tarballs and this repo is Python |
| B. Error handling | 4/10 | Unchanged, and documented as such |
| C. Operator docs | 8/10 | README, CHANGELOG, and the handbook. No log-level flag |
| D. Shipping hygiene | 6/10 | Verify script, lockfile, `requires-python`, coverage floor. No dependency scanner. `shipcheck deps` did not audit `uv.lock` |
| E. Identity (soft) | 5/10 | Landing page and handbook are in this PR. Translations and `gh repo edit` are the Publisher's. The logo URL waits on the brand PR |
| **Overall** | **31/50** | Gate E is not done. This is a prototype release, not a 1.0 |

## Post-treatment assessment (0.2.0, 2026-10-14)

Re-treatment for the T10 generator. Scores from `npx @mcptoolshop/shipcheck audit` and the executed gates on the `release/0.2.0` branch, before the Publisher's translation commit.

Audit: 15 checked, 23 skipped with a reason, 1 unchecked (translations), pass rate 94%, exit 1. Sections A–D: every line checked or `SKIP:`. Executed gates: `security-docs` passed, `manifest --expect 0.2.0` passed, `ci` failed on `dependency-scan` (no scanner, unchanged and recorded as a skip), `front-door` skipped (site-theme front-door not installed). Tests: 432 passed with the `sweep` marker deselected, line coverage 97.19% against the floor of 90.

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 8/10 | SECURITY.md supports 0.2.x, the threat model covers the generator, the identity scan is `RESULT CLEAN`. `shipcheck secrets` still skips a Python repo |
| B. Error handling | 4/10 | Unchanged: `REFUSED:` lines and per-command exit codes. No structured error object, no `--debug` |
| C. Operator docs | 9/10 | README, CHANGELOG [0.2.0], and nine handbook pages, two new (the generator end to end, the role packs). No log-level flag |
| D. Shipping hygiene | 6/10 | Version 0.2.0 in `pyproject.toml`, `__version__` and `uv.lock`; coverage floor 90 at 97%; the record gate in CI. Still no dependency scanner, and `scripts/verify` runs the sweep |
| E. Identity (soft) | 10/10 | Logo URL returns 200, landing page and handbook build with search, repo metadata in place, README translated into seven languages (polyglot, TranslateGemma 27B, local) |
| **Overall** | **37/50** | Up from 31. A prototype release, 0.x by the version rule, not a 1.0 |

After the Publisher's translation commit (`ea24d70`, merged as `891b364`), the audit on `main` reads 16 checked, 23 skipped with a reason, 0 unchecked, pass rate 100%, exit 0 (re-run 2026-10-14).