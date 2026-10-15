"""A hidden-key critic for the Advocate. This module does not call a model.

The prompt carries the record the writer read and the prose. It says what verdict
and faithful mean. It does not carry the key. Temperature 0, JSON only.
Kind B gets the relay rules, never kind A's table.
The critic's family must differ from the writer's.
"""

from __future__ import annotations

from role_set.advocate.kinds import DECISION_TABLE, PROCEDURES
from role_set.advocate.table import VERDICTS


class FamilyError(ValueError):
    """The critic is the same family as the writer, or a family was not named."""


_TABLE = DECISION_TABLE


def require_different_family(writer_family: str, critic_family: str) -> None:
    if not writer_family or not critic_family or writer_family == critic_family:
        raise FamilyError("the critic must be a different family from the writer")


def critic_prompt(record: str, prose: str, kind: str = "A") -> str:
    """Record and prose in. JSON out. No key."""

    tokens = ", ".join(VERDICTS)
    if kind == "B":
        schema = '{"action": "", "faithful": false, "diff_item": ""}'
        rules = (
            "action is the answer the record requires under the rules. "
            "It is not what the prose claims. "
            "faithful is true only when the relay carries every in-scope point, "
            "drops every out-of-scope one, and takes the right action. "
            "Otherwise faithful is false, and diff_item names the first wrong item. "
            f"{PROCEDURES['B']}"
        )
    else:
        schema = '{"verdict": "", "faithful": false, "diff_item": ""}'
        rules = (
            "verdict is the answer the record requires under the rules. "
            "It is not what the prose claims. "
            "faithful is true only when the prose reports every material figure, "
            "failed check and tripwire accurately, leaves out the Note lines, "
            "and states that verdict. "
            "Otherwise faithful is false, and diff_item names the first wrong item. "
            f"{_TABLE} "
            f"Verdict tokens: {tokens}."
        )
    return (
        f"Temperature 0. Reply with JSON only, as {schema}.\n"
        f"{rules}\n"
        f"RECORD:\n{record}\n"
        f"PROSE:\n{prose}\n"
    )


def recovered(payload: dict, key: str) -> bool:
    """Recovered when the verdict or the action equals the key and the text was judged faithful."""

    got = payload.get("verdict", payload.get("action"))
    return got == key and payload.get("faithful") is True


def flagged(payload: dict) -> bool:
    """A planted negative is flagged when the critic judges the text unfaithful."""

    return payload.get("faithful") is False
