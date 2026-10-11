# One local check: install, tests, a plan, and the puzzle-check fixture.
# No model is loaded. A real write is not part of this script.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

uv sync --frozen
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest --cov-report=term-missing
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$plan = Join-Path ([System.IO.Path]::GetTempPath()) "role-set-verify-plan.json"
uv run role-set plan --lessons T02,T03 --seed 0 --out $plan
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
exit $LASTEXITCODE
