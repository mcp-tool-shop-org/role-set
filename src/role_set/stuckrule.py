"""The frozen Stage 1 stuck rule, read from aspire-si at a pinned commit, never copied.

`stuck.py` and `stage1-stuck-rule.json` were frozen before round 1 (eval plan amendments a7b336b and 4bf6384). The
runner records the rule's measures on every writer draft (hedges, bursts, re-reads, the stuck flags) so a paid run
returns diagnosable data whether or not the writer succeeds. Both files are fetched with `git show` at the pinned
commit and must match their frozen sha256 before anything runs; the code is the repository's own, at that commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import types
from pathlib import Path

STUCK_PY = "docs/runs/stage1-train/stuck.py"
RULE_JSON = "docs/runs/stage1-stuck-rule.json"
# The frozen hashes (aspire-si 4b857ff and every commit since): a different file is refused, not run.
STUCK_PY_SHA256 = "8097b0398ade5e57989ee78f4d9a74a134f2aaefef358c19be0daaad8df13afa"
RULE_SHA256 = "716dc251965973c61b74287e077dc8ae68c7d8c2a1bfccc87bde183d2ff1c1a1"


class StuckRuleError(RuntimeError):
    pass


def _show(repo: Path, commit: str, path: str) -> bytes:
    try:
        return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as e:
        raise StuckRuleError(f"REFUSED: cannot read {path} at {commit}") from e


def rule_for_judge(frozen: dict) -> dict:
    """stuck.judge reads p95_by_tier, reread_rate_p95 and burst {window, k}; the frozen file names them
    p95_thinking_tokens_by_tier and burst_parameters {window_tokens, k}. The same values, under judge's names."""

    try:
        return {"p95_by_tier": frozen["p95_thinking_tokens_by_tier"], "reread_rate_p95": frozen["reread_rate_p95"],
                "burst": {"window": frozen["burst_parameters"]["window_tokens"], "k": frozen["burst_parameters"]["k"]}}
    except KeyError as e:
        raise StuckRuleError(f"REFUSED: the frozen rule has no {e.args[0]}") from e


def load(repo: Path, commit: str):
    """A measure(thinking, material, tier, tokens) -> dict callable over the frozen rule."""

    src, rule_bytes = _show(repo, commit, STUCK_PY), _show(repo, commit, RULE_JSON)
    for name, data, want in ((STUCK_PY, src, STUCK_PY_SHA256), (RULE_JSON, rule_bytes, RULE_SHA256)):
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise StuckRuleError(f"REFUSED: {name} at {commit} is {got[:12]}, not the frozen {want[:12]}")
    module = types.ModuleType("stage1_stuck")
    module.__file__ = STUCK_PY
    exec(compile(src.decode("utf-8"), STUCK_PY, "exec"), module.__dict__)  # noqa: S102 - pinned, hash-checked repo code
    frozen = json.loads(rule_bytes.decode("utf-8"))
    rule = rule_for_judge(frozen)

    def measure(thinking: str, material: str, tier: int, tokens: int) -> dict:
        # Words stand in for the student's tokens (the rule's own fallback); the writer has no target length, so
        # "long" is judged against the tier's p95 alone. Everything is a record here, never a refusal.
        words = len(thinking.split())
        out = module.judge(thinking, material, tier, 0, words, tokens, rule)
        out["hedge_profile"] = module.hedge_profile(thinking, material, words)
        out["thinking_words"] = words
        return out

    measure.rule_sha256 = RULE_SHA256  # type: ignore[attr-defined]
    return measure
