"""A hidden-key critic for the Advocate. This module does not call a model.

The prompt is the prose alone, at temperature 0, with JSON-constrained output.
The key is not a field on the prompt. The critic's family must differ from the writer's.
"""

from __future__ import annotations


class FamilyError(ValueError):
    """The critic is the same family as the writer, or a family was not named."""


def require_different_family(writer_family: str, critic_family: str) -> None:
    if not writer_family or not critic_family or writer_family == critic_family:
        raise FamilyError("the critic must be a different family from the writer")


def critic_prompt(prose: str) -> str:
    """Prose in, JSON out. No key field is added."""

    return (
        "Judge the prose. Temperature 0. Reply with JSON only, "
        'as {"verdict": "", "defective": false}.\n'
        f"PROSE:\n{prose}\n"
    )


def recovered(payload: dict, key: str) -> bool:
    """A positive key is recovered when the verdict matches and the text is not marked defective."""

    return payload.get("verdict") == key and payload.get("defective") is False


def flagged(payload: dict) -> bool:
    """A planted negative is flagged when the critic marks the text defective."""

    return payload.get("defective") is True
