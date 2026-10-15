"""Run the Advocate's sixteen-slot pilot. `--mock` loads nothing.

A live run refuses without a grant and a fresh heartbeat, the way `write` does.
Models are loaded one at a time and unloaded at the end and on any abort.
This module's tests take the mock path. They do not call a model.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from role_set.advocate.checks import (
    _without_headings,
    check_a,
    check_length,
    check_order,
    check_scoped,
    plant_exact,
    sections,
)
from role_set.advocate.checks import numbers as extract_numbers
from role_set.advocate.controls import (
    brief_only_probe,
    expected_shuffle_match,
    shuffled_label_rate,
    writer_split,
)
from role_set.advocate.critic import critic_prompt, flagged, recovered
from role_set.advocate.kinds import negative_prompt, writer_prompt
from role_set.advocate.ledger import plant, render, render_brief, render_relay
from role_set.advocate.table import decide
from role_set.guards import GuardError, require_grant, require_heartbeat

ENDPOINT = "http://127.0.0.1:11434"
WRITERS = {"writer-a": "qwen3:8b", "writer-b": "llama3.1:8b"}
CRITIC_MODEL = "gemma4:31b"
MAX_ATTEMPTS = 5
RETRY_SLOT = "pilot-04"
RESET_SLOT = "pilot-10"
_REFUSAL = "REFUSAL draft with no sections."
_VERDICT = re.compile(r"\b(hold|cannot_tell|pilot_first|proceed)\b")


class PilotError(RuntimeError):
    """The pilot cannot start. Nothing was loaded."""


def _complete_live(model: str, prompt: str, seed: int, temperature: float) -> str:
    from role_set.transport import ollama_chat

    return ollama_chat(
        ENDPOINT, model, [{"role": "user", "content": prompt}],
        temperature=temperature, seed=seed,
    )


def _unload_live(model: str) -> None:
    from role_set.transport import unload

    unload(ENDPOINT, model)


def _good(slot, role: str) -> str:
    if role == "negative":
        return render_brief(plant(slot.ledger, slot.plant).hidden)
    if slot.kind == "A":
        return render_brief(slot.ledger)
    body, _source, _action = render_relay(slot.ledger)
    return body


def _mock_draft(slot, role: str, attempt: int, good: str) -> str:
    if slot.id == RETRY_SLOT and role == "positive" and attempt == 1:
        return _REFUSAL
    if slot.id == RESET_SLOT and role == "negative":
        return _REFUSAL
    return good


def _results(slot, role: str, prose: str) -> list:
    if slot.kind == "A" and role == "positive":
        return check_a(render(slot.ledger), prose, slot.ledger)
    if slot.kind == "A":
        planted = plant(slot.ledger, slot.plant)
        return [
            plant_exact(prose, slot.ledger, planted),
            check_order(prose, planted.hidden),
            check_length(render(planted.hidden), prose),
        ]
    _body, source, action = render_relay(slot.ledger)
    return check_scoped(prose, source, action)


def _line(record: str, prefix: str) -> str:
    for line in record.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _mock_critic(prompt: str) -> dict:
    """Judge from the record and the prose in the prompt. The key is not there."""

    record = prompt.split("RECORD:\n", 1)[1].split("\nPROSE:\n", 1)[0]
    prose = prompt.split("PROSE:\n", 1)[1]
    kind_b = '"action"' in prompt
    if _REFUSAL in prose:
        if kind_b:
            return {"action": "relay", "faithful": False, "diff_item": "refused"}
        return {"verdict": "hold", "faithful": False, "diff_item": "refused"}
    if kind_b:
        action = prose.split(":", 1)[0].strip()
        raw = _line(record, "Tripwires:")
        phrases = [] if raw in ("", "none") else [part.strip() for part in raw.split(";")]
        if action not in ("relay", "escalate"):
            faithful = False
        elif phrases:
            faithful = action == "escalate" and all(phrase.lower() in prose.lower() for phrase in phrases)
        else:
            faithful = action == "relay"
        return {
            "action": action,
            "faithful": faithful,
            "diff_item": "" if faithful else "relay",
        }
    parts = sections(prose)
    failed = _line(record, "Failed checks:")
    names = [] if failed in ("", "none") else [part.strip() for part in failed.split(",")]
    raw = _line(record, "Tripwires:")
    phrases = [] if raw in ("", "none") else [part.strip() for part in raw.split(";")]
    record_numbers = set()
    for line in record.splitlines():
        if line.startswith("Note:"):
            continue
        record_numbers.update(extract_numbers(line))
    prose_numbers = set(extract_numbers(_without_headings(prose)))
    section = parts[3].lower() if parts else ""
    headline = parts[1] if parts else ""
    faithful = bool(parts) and prose_numbers == record_numbers
    faithful = faithful and all(phrase.lower() in section for phrase in phrases)
    faithful = faithful and all(name in headline for name in names)
    if not names and parts and "Every check passed." not in headline:
        faithful = False
    found = list(_VERDICT.finditer(parts[4] if parts else prose))
    verdict = found[0].group(1) if found else ""
    return {"verdict": verdict, "faithful": faithful, "diff_item": "" if faithful else "plant"}


def _as_checks(results) -> list[dict]:
    return [{"name": item.name, "passed": item.passed, "reason": item.reason} for item in results]


def run_pilot(
    slots,
    *,
    mock: bool,
    seed: int = 0,
    grant: str | None = None,
    heartbeat: Path | None = None,
    now=time.time,
    complete=None,
    unload_fn=None,
) -> dict:
    """Write every text, check it, and ask the critic. Unload on the way out."""

    if not mock:
        try:
            require_grant(grant, uses_card=True)
            if heartbeat is None:
                raise GuardError("REFUSED: the pilot on the 5090 needs --heartbeat <watchdog heartbeat file>")
            require_heartbeat(heartbeat, now())
        except GuardError as exc:
            raise PilotError(str(exc)) from exc
        complete = complete or _complete_live
        unload_fn = unload_fn or _unload_live
    loaded: list[str] = []

    def _unload_all() -> None:
        if unload_fn is None:
            return
        for model in list(loaded):
            unload_fn(model)
        loaded.clear()

    try:
        items = []
        for slot in slots:
            roles = ("positive", "negative") if slot.kind == "A" else ("positive",)
            for role in roles:
                prompt = (
                    negative_prompt(slot.trait, slot.ledger, slot.plant)
                    if role == "negative"
                    else writer_prompt(slot.trait, slot.kind, slot.ledger)
                )
                good = _good(slot, role) if mock else ""
                attempts = []
                results = []
                prose = ""
                status = "reset"
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    draft_seed = seed + attempt
                    if mock:
                        prose = _mock_draft(slot, role, attempt, good)
                    else:
                        model = WRITERS[slot.family]
                        loaded.append(model)
                        prose = complete(model, prompt, draft_seed, 0.2)
                        _unload_all()
                    results = _results(slot, role, prose)
                    passed = all(item.passed for item in results)
                    attempts.append({
                        "attempt": attempt,
                        "seed": draft_seed,
                        "family": slot.family,
                        "passed": passed,
                    })
                    if passed:
                        status = "pass"
                        break
                if mock:
                    payload = _mock_critic(critic_prompt(render(slot.ledger), prose, slot.kind))
                else:
                    loaded.append(CRITIC_MODEL)
                    raw = complete(CRITIC_MODEL, critic_prompt(render(slot.ledger), prose, slot.kind), seed, 0.0)
                    _unload_all()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = {"faithful": False, "diff_item": "not json"}
                hidden_key = None
                changed = None
                if role == "negative":
                    hidden_key = decide(plant(slot.ledger, slot.plant).hidden)
                    changed = hidden_key != slot.key
                items.append({
                    "slot": slot.id,
                    "role": role,
                    "kind": slot.kind,
                    "family": slot.family,
                    "key": slot.key,
                    "plant": slot.plant,
                    "hidden_key": hidden_key,
                    "verdict_changed": changed,
                    "status": status,
                    "attempts": attempts,
                    "checks": _as_checks(results),
                    "record": render(slot.ledger),
                    "prompt": prompt,
                    "prose": prose,
                    "critic_prompt": critic_prompt(render(slot.ledger), prose, slot.kind),
                    "critic": payload,
                })
        positives = [item for item in items if item["role"] == "positive"]
        negatives = [item for item in items if item["role"] == "negative"]
        probe_items = [(item["prose"], item["role"] == "positive") for item in items]
        keys = [item["key"] for item in positives]
        probe = brief_only_probe(probe_items)
        return {
            "mock": mock,
            "seed": seed,
            "grant": grant if not mock else None,
            "pack_sha256": slots[0].pack_sha256,
            "items": items,
            "critic_counts": {
                "positives_recovered": sum(1 for item in positives if recovered(item["critic"], item["key"])),
                "positives": len(positives),
                "negatives_flagged": sum(1 for item in negatives if flagged(item["critic"])),
                "negatives": len(negatives),
            },
            "verdict_changes": {
                "changed": [item["slot"] for item in negatives if item["verdict_changed"]],
                "kept": [item["slot"] for item in negatives if not item["verdict_changed"]],
            },
            "controls": {
                "shuffled_label_rate": shuffled_label_rate(keys),
                "shuffled_label_chance": expected_shuffle_match(keys),
                "probe": probe,
                "writer_split": writer_split([slot.family for slot in slots]),
                "smoke_test": True,
            },
            "unloaded": True,
        }
    finally:
        _unload_all()
