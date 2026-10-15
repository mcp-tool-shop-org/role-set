"""The Advocate's decision table. The rules apply in order, and the first match wins."""

from __future__ import annotations

PAID_SPEND = "B004"
VERDICTS = ("hold", "cannot_tell", "pilot_first", "proceed")


def proposed_failures(ledger) -> list:
    """Failed checks on the proposed run. A baseline miss is not one of them."""

    return [c for c in ledger.checks if c.result == "fail" and c.on == "proposed"]


def holding_tripwires(ledger) -> list[str]:
    """Tripwires other than paid spend. Paid spend alone does not force hold."""

    return [t for t in ledger.tripwires if t != PAID_SPEND]


def decide(ledger) -> str:
    """One verdict. Baseline failures are ignored. Paid spend alone is not rule one."""

    if proposed_failures(ledger) or holding_tripwires(ledger):
        return "hold"
    if ledger.over_cap or ledger.paid_model_api:
        return "hold"
    if ledger.missing:
        return "cannot_tell"
    if (ledger.paid_run or ledger.rollout) and ledger.pilot_passed is not True:
        return "pilot_first"
    return "proceed"
