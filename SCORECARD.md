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
