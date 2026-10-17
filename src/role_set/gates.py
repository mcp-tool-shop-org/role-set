"""The gates come from aspire-si's package, pinned by commit. They are never copied here.

`write` refuses to start when the package isn't installed, and every item records
the gates' identity (version and source sha256) so a reviewer knows exactly which
checks it passed.

The closed-form rule for written kind-E variants (keyed carry verbatim, paraphrase or either; the content-word
share; the distractor margin and near-copy guard; the span) is the gates' own, as settings: `load_gates` resolves the
run's settings through the pinned package's `closed_form_settings`, so its defaults apply unless the run names
others, and every check and every receipt carries the resolved settings. role-set keeps no copy of the rule.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


class GatesUnavailable(RuntimeError):
    """aspire-si-gates isn't installed at the pinned commit. Nothing is written."""


class ModelsMissing(RuntimeError):
    """A kind B row reached the gates with no sidecar model. The gates would sign it by the regex fallback instead of
    its model, so the run stops rather than check it the weaker way (R&D, 2026-10-15)."""


def _need_models(rows: list[dict], models: dict) -> None:
    missing = sorted(str(r.get("id")) for r in rows if r.get("kind") == "B" and r.get("id") not in models)
    if missing:
        raise ModelsMissing(f"no sidecar model for kind B row(s) {', '.join(missing)}; pass models={{id: model}} for every "
                            "B row so puzzles are signed by their model (gates.puzzle_sig)")


@dataclass(frozen=True)
class Gates:
    module: object
    version: str
    sha256: str
    closed_form: dict | None = None  # resolved by the package's closed_form_settings; None only for a stand-in

    @property
    def closed_form_carry(self) -> str | None:
        return (self.closed_form or {}).get("carry")

    def check_item(self, row: dict) -> None:
        if self.closed_form is None:
            self.module.check_item(row)  # type: ignore[attr-defined]
        else:
            self.module.check_item(row, closed_form=self.closed_form)  # type: ignore[attr-defined]

    def check_apply_item(self, row: dict) -> None:
        """An apply item's checks (gates 1.7.5): check_item's content and shape checks, without its never-trained
        refusal, plus the apply rules (split apply and an -AP- id, C or D, one turn, the lesson's twin field)."""
        if self.closed_form is None:
            self.module.check_apply_item(row)  # type: ignore[attr-defined]
        else:
            self.module.check_apply_item(row, closed_form=self.closed_form)  # type: ignore[attr-defined]

    def check_against(self, row: dict, accepted: list[dict], *, models: dict) -> list:
        """The gates' one-at-a-time check. `models` is the {id: model} map of sidecar models for the row and the
        accepted rows: every kind B row needs one (ModelsMissing otherwise)."""
        _need_models([row, *accepted], models)
        return list(self.module.check_against(row, accepted, models=models) or [])  # type: ignore[attr-defined]

    def check_batch(self, rows: list[dict], *, models: dict) -> None:
        _need_models(rows, models)
        if self.closed_form is None:
            self.module.check_batch(rows, models=models)  # type: ignore[attr-defined]
        else:
            self.module.check_batch(rows, closed_form=self.closed_form, models=models)  # type: ignore[attr-defined]

    def trait_words(self, lesson: str) -> list[str]:
        """The words the gates accept as naming a trait in a kind-A passage (constants.TRAIT_NAMES), or []."""
        names = getattr(getattr(self.module, "constants", None), "TRAIT_NAMES", None) or {}
        return list(names.get(lesson, []))

    def parse_answer(self, reply: str) -> dict:
        return self.module.answer_format.parse_answer(reply)  # type: ignore[attr-defined]

    def role_words_in_prompt(self, row: dict) -> list[str]:
        return list(self.module.role_words_in_prompt(row))  # type: ignore[attr-defined]

    def load_models(self, data_path) -> dict:
        """{id: model} from a data file's .models.jsonl sidecar, each line's row_sha256 checked (the gates' own loader)."""
        return dict(self.module.load_models(data_path))  # type: ignore[attr-defined]

    def error_type(self) -> type[Exception]:
        return self.module.DataError  # type: ignore[attr-defined]


REQUIRED_RULES = ("norm", "critic_mismatch", "closed_form_settings", "key_point_share")


def load_gates(importer=importlib.import_module, closed_form: dict | None = None) -> Gates:
    """The pinned package, with the run's closed-form settings resolved through it (`closed_form` overrides the
    package's defaults; None takes them as they are). Settings the package rejects refuse the run."""
    try:
        module = importer("aspire_si_gates")
        importer("aspire_si_gates.answer_format")
    except ImportError as e:
        raise GatesUnavailable(
            "REFUSED: aspire-si-gates is not installed; pin it from aspire-si at the merged commit before `write`"
        ) from e
    ident = module.gates_identity()
    # The critic compares quotes with the scorer's `norm` and evidence with `critic_mismatch` (gates 1.1.0); the
    # closed-form rule is the package's settings (gates 1.4.0). A package without them is refused here, before anything
    # is sent, rather than falling back to a weaker rule.
    missing = [name for name in REQUIRED_RULES if not callable(getattr(module, name, None))]
    if missing:
        raise GatesUnavailable(f"REFUSED: aspire-si-gates {ident['version']} has no {', '.join(missing)}; pin 1.4.0 or later")
    try:
        settings = dict(module.closed_form_settings(closed_form))
    except ValueError as e:
        raise GatesUnavailable(f"REFUSED: closed-form settings the gates reject: {e}") from e
    return Gates(module=module, version=str(ident["version"]), sha256=str(ident["sha256"]), closed_form=settings)
