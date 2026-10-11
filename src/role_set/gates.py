"""The gates come from aspire-si's package, pinned by commit. They are never copied here.

`write` refuses to start when the package isn't installed, and every item records
the gates' identity (version and source sha256) so a reviewer knows exactly which
checks it passed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


class GatesUnavailable(RuntimeError):
    """aspire-si-gates isn't installed at the pinned commit. Nothing is written."""


@dataclass(frozen=True)
class Gates:
    module: object
    version: str
    sha256: str

    def check_item(self, row: dict) -> None:
        self.module.check_item(row)  # type: ignore[attr-defined]

    def check_against(self, row: dict, accepted: list[dict]) -> list:
        return list(self.module.check_against(row, accepted) or [])  # type: ignore[attr-defined]

    def check_batch(self, rows: list[dict]) -> None:
        self.module.check_batch(rows)  # type: ignore[attr-defined]

    def trait_words(self, lesson: str) -> list[str]:
        """The words the gates accept as naming a trait in a kind-A passage (constants.TRAIT_NAMES), or []."""
        names = getattr(getattr(self.module, "constants", None), "TRAIT_NAMES", None) or {}
        return list(names.get(lesson, []))

    def parse_answer(self, reply: str) -> dict:
        return self.module.answer_format.parse_answer(reply)  # type: ignore[attr-defined]

    def role_words_in_prompt(self, row: dict) -> list[str]:
        return list(self.module.role_words_in_prompt(row))  # type: ignore[attr-defined]

    def error_type(self) -> type[Exception]:
        return self.module.DataError  # type: ignore[attr-defined]


REQUIRED_RULES = ("norm", "critic_mismatch")


def load_gates(importer=importlib.import_module) -> Gates:
    try:
        module = importer("aspire_si_gates")
        importer("aspire_si_gates.answer_format")
    except ImportError as e:
        raise GatesUnavailable(
            "REFUSED: aspire-si-gates is not installed; pin it from aspire-si at the merged commit before `write`"
        ) from e
    ident = module.gates_identity()
    # The critic compares quotes with the scorer's `norm` and evidence with `critic_mismatch` (gates 1.1.0). A package
    # without them is refused here, before anything is sent, rather than falling back to a weaker rule.
    missing = [name for name in REQUIRED_RULES if not callable(getattr(module, name, None))]
    if missing:
        raise GatesUnavailable(f"REFUSED: aspire-si-gates {ident['version']} has no {', '.join(missing)}; pin 1.1.0 or later")
    return Gates(module=module, version=str(ident["version"]), sha256=str(ident["sha256"]))
