#!/usr/bin/env bash
# One local check: install, tests, a plan, and the puzzle-check fixture.
# No model is loaded. A real write is not part of this script.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --frozen
uv run pytest --cov-report=term-missing

plan="$(mktemp)"
trap 'rm -f "$plan"' EXIT
uv run role-set plan --lessons T02,T03 --seed 0 --out "$plan"
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
