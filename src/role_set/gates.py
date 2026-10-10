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

    def check_against(self, row: dict, accepted: list[dict]) -> None:
        self.module.check_against(row, accepted)  # type: ignore[attr-defined]

    def check_batch(self, rows: list[dict]) -> None:
        self.module.check_batch(rows)  # type: ignore[attr-defined]

    def parse_answer(self, reply: str) -> dict:
        return self.module.answer_format.parse_answer(reply)  # type: ignore[attr-defined]

    def role_words_in_prompt(self, row: dict) -> list[str]:
        return list(self.module.role_words_in_prompt(row))  # type: ignore[attr-defined]

    def error_type(self) -> type[Exception]:
        return self.module.DataError  # type: ignore[attr-defined]


def load_gates(importer=importlib.import_module) -> Gates:
    try:
        module = importer("aspire_si_gates")
        importer("aspire_si_gates.answer_format")
    except ImportError as e:
        raise GatesUnavailable(
            "REFUSED: aspire-si-gates is not installed; pin it from aspire-si at the merged commit before `write`"
        ) from e
    ident = module.gates_identity()
    return Gates(module=module, version=str(ident["version"]), sha256=str(ident["sha256"]))
