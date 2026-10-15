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
    check_a,
    check_length,
    check_order,
    check_scoped,
    plant_exact,
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
from role_set.advocate.ledger import _REASONS, plant, render, render_relay
from role_set.advocate.table import decide
from role_set.guards import (
    GuardError,
    require_grant,
    require_heartbeat,
    require_model_allowed,
    require_ollama_digest,
)

ENDPOINT = "http://127.0.0.1:11434"
WRITERS = {"writer-a": "qwen3:8b", "writer-b": "granite3.3:8b"}
CRITIC_MODEL = "gemma4:31b"
MAX_ATTEMPTS = 5
RETRY_SLOT = "pilot-04"
RESET_SLOT = "pilot-10"
_REFUSAL = "REFUSAL draft with no sections."
_VERDICT = re.compile(r"\b(hold|cannot_tell|pilot_first|proceed)\b")


class PilotError(RuntimeError):
    """The pilot cannot start. Nothing was loaded."""


def _complete_live(
    model: str,
    prompt: str,
    seed: int,
    temperature: float,
    format: str | None = None,
) -> str:
    from role_set.transport import ollama_chat

    reply = ollama_chat(
        ENDPOINT, model, [{"role": "user", "content": prompt}],
        temperature=temperature, seed=seed, think=False, format=format,
    )
    return reply.text


def _default_writers() -> Path:
    cwd = Path("writers.json")
    if cwd.is_file():
        return cwd
    return Path(__file__).resolve().parents[3] / "writers.json"


def _guard_pinned_models(writers_file: Path | None, opener) -> None:
    """Both writers and the critic, before any model is loaded."""

    from role_set.write import allowed_critics, allowed_writers

    path = writers_file if writers_file is not None else _default_writers()
    listed = allowed_writers(path)
    listed.update(allowed_critics(path))
    allowed = sorted(listed)
    for name in (WRITERS["writer-a"], WRITERS["writer-b"], CRITIC_MODEL):
        require_model_allowed(name, allowed)
        expected = listed.get(name, {}).get("digest")
        if not expected:
            raise GuardError(f"REFUSED: {name} has no digest pin")
        require_ollama_digest(ENDPOINT, name, expected, opener=opener)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```") and len(stripped) >= 6):
        return stripped
    body = stripped[3:-3]
    if body.lower().startswith("json"):
        body = body[4:]
    return body.strip()


def _critic_payload(raw: str) -> dict:
    """JSON from the critic. A fence is stripped. An empty reply is a refusal."""

    text = _strip_fence(raw or "")
    if not text:
        return {"faithful": False, "diff_item": "empty critic reply"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"faithful": False, "diff_item": "not json"}
    if not isinstance(payload, dict):
        return {"faithful": False, "diff_item": "not json"}
    return payload


def _unload_live(model: str) -> None:
    from role_set.transport import unload

    unload(ENDPOINT, model)


def _ledger_text(prompt: str) -> str:
    if "Ledger.\n" not in prompt:
        return ""
    return prompt.split("Ledger.\n", 1)[1].strip()


# Category stems, not the pilot's point strings. A line matches when a stem is a word in it.
_CATEGORY_STEMS = (
    ("a public or irreversible act", ("force-push", "publish", "tag", "release", "rewriting", "settings")),
    ("a floor missed", ("skip",)),
    ("paid spend", ("paid",)),
    ("a second-hand approval", ("peer", "second-hand")),
    ("personal data", ("scan", "mailbox", "identity")),
    ("the Director's own words in a file", ("quote",)),
    ("device use without a grant", ("card",)),
    ("a merge without review", ("merge",)),
    ("a frozen analysis changed", ("frozen",)),
)


def _act_in(line: str, act: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(act)}(?![A-Za-z])", line, re.I) is not None


def _is_note_line(line: str) -> bool:
    return line.lower().startswith(("budget note:", "cost note:", "runner note:", "queue note:"))


def _parsed_record(record: str) -> dict:
    """Read a studio record from prompt text. The ledger object is not available here."""

    lines = [line.strip() for line in record.splitlines() if line.strip()]
    body = lines[1:] if lines else []
    failed = []
    asks = []
    phrases = []
    missing = []
    figures = []
    compute = ""
    pilot = ""
    for line in body:
        if line.startswith("check ") and ": fail" in line and "baseline" not in line.lower():
            name, _, rest = line.partition(": fail")
            name = name.removeprefix("check ").strip()
            evidence = ""
            if "(" in rest and rest.rstrip().endswith(")"):
                evidence = rest[rest.find("(") + 1:rest.rfind(")")]
            failed.append((name, evidence))
        elif line.startswith("Asks:"):
            asks = [part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()]
        elif line.startswith("Tripwires:"):
            raw = line.split(":", 1)[1]
            phrases = [part.strip() for part in raw.split(";") if part.strip()]
        elif line.endswith(": pending"):
            missing.append(line[: -len(": pending")])
        elif line.startswith("compute:"):
            compute = line
        elif line.startswith("local pilot:"):
            pilot = line.split(":", 1)[1].strip()
        elif line.startswith((
            "Local baseline:", "Release approved:", "Provenance:", "Published file:",
            "Head:", "Grant:", "Card after:", "Stopping", "From ",
        )):
            continue
        elif line.lower().startswith("note:"):
            continue
        elif line.lower().startswith(("runner ", "retries ", "queue ")):
            continue
        elif not re.search(r"\d", line):
            continue
        else:
            figures.append(line)
    return {
        "failed": failed,
        "asks": asks,
        "phrases": phrases,
        "missing": missing,
        "figures": figures,
        "compute": compute,
        "pilot": pilot,
        "lines": body,
    }


def _over_cap(lines: list[str]) -> bool:
    cost = None
    cap = None
    for line in lines:
        amounts = extract_numbers(line)
        if not amounts or "$" not in line:
            continue
        if "cap" in line.lower():
            cap = float(amounts[0])
        elif "cost" in line.lower():
            cost = float(amounts[0])
    return cost is not None and cap is not None and cost > cap


def _verdict_from_parsed(parsed: dict) -> str:
    holding = [phrase for phrase in parsed["phrases"] if phrase != "paid spend"]
    if parsed["failed"] or holding:
        return "hold"
    if _over_cap(parsed["lines"]) or "paid cloud model API" in parsed["compute"]:
        return "hold"
    if parsed["missing"]:
        return "cannot_tell"
    if "rented pod" in parsed["compute"] and parsed["pilot"] != "passed":
        return "pilot_first"
    return "proceed"


def _brief_from_record(record: str) -> str:
    parsed = _parsed_record(record)
    verdict = _verdict_from_parsed(parsed)
    asks = "\n".join(parsed["asks"]) if parsed["asks"] else "No ask is on the record."
    parts = []
    for name, evidence in parsed["failed"]:
        parts.append(f"{name} failed ({evidence})." if evidence else f"{name} failed.")
    parts.extend(f"{phrase}." for phrase in parsed["phrases"])
    headline = " ".join(parts) if parts else "Every check passed."
    figures = "\n".join(f"Shows {text}." for text in parsed["figures"])
    trips = "\n".join(parsed["phrases"]) if parsed["phrases"] else "none"
    if verdict == "cannot_tell":
        missing = ", ".join(parsed["missing"])
        recommendation = f"cannot_tell: the record is missing {missing}."
    else:
        alternative = {"hold": "proceed", "proceed": "hold", "pilot_first": "proceed"}[verdict]
        reason = _REASONS["a"][verdict]
        recommendation = (
            f"{verdict}: you'd expect {alternative}; the record supports {verdict} "
            f"because {reason}."
        )
    blocks = (
        f"1. What's needed of you\n{asks}",
        f"2. The headline\n{headline}",
        f"3. The figures\n{figures}",
        f"4. Tripwires\n{trips}",
        f"5. The recommendation\n{recommendation}",
    )
    return "\n\n".join(blocks)


def _phrases_for(line: str) -> list[str]:
    """The category phrases a point line names. The phrase itself counts too."""

    found = []
    folded = line.lower()
    for phrase, stems in _CATEGORY_STEMS:
        if phrase.lower() in folded or any(_act_in(line, stem) for stem in stems):
            found.append(phrase)
    return found


def _relay_from_lines(record: str, _prompt: str) -> str:
    """Match the record's lines to category stems. The prompt is not parsed for quoted acts."""
    kept = []
    phrases = []
    for line in record.splitlines():
        line = line.strip()
        if not line or line.startswith("From ") or _is_note_line(line) or line.lower().startswith("note:"):
            continue
        kept.append(line)
        for phrase in _phrases_for(line):
            if phrase not in phrases:
                phrases.append(phrase)
    if phrases:
        return "escalate: " + ". ".join(phrases) + "."
    return "relay: " + " — ".join(kept)


def _relay_from_prompt(prompt: str) -> str:
    return _relay_from_lines(_ledger_text(prompt), prompt)


def _mock_draft(prompt: str, slot_id: str, role: str, attempt: int, kind: str) -> str:
    """Build the draft from the prompt text alone. The slot ledger is not read."""

    if slot_id == RETRY_SLOT and role == "positive" and attempt == 1:
        return _REFUSAL
    if slot_id == RESET_SLOT and role == "negative":
        return _REFUSAL
    if "Ledger." not in prompt:
        return _REFUSAL
    if kind == "B":
        return _relay_from_prompt(prompt)
    return _brief_from_record(_ledger_text(prompt))


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
    return check_scoped(prose, source, action, slot.ledger)


def _record_and_prose(prompt: str) -> tuple[str, str]:
    record = prompt.split("RECORD:\n", 1)[1].split("\nPROSE:\n", 1)[0].strip("\n")
    prose = prompt.split("PROSE:\n", 1)[1]
    return record, prose


def _diff_item(record: str, prose: str, parsed: dict) -> str:
    """The first item the prose does not report as the record has it."""

    if _REFUSAL in prose:
        return "refused"
    for text in parsed["figures"]:
        if f"Shows {text}." not in prose:
            return text
    for name, _evidence in parsed["failed"]:
        if name not in prose:
            return name
    for phrase in parsed["phrases"]:
        if phrase.lower() not in prose.lower():
            return phrase
    for line in record.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("note:") and stripped in prose:
            return stripped
    found = list(_VERDICT.finditer(prose))
    verdict = _verdict_from_parsed(parsed)
    if not found or found[0].group(1) != verdict:
        return verdict
    return "prose"


def _mock_critic(prompt: str) -> dict:
    """Judge from the record in the prompt. The verdict is the record's, not the prose's."""

    record, prose = _record_and_prose(prompt)
    kind_b = '"action"' in prompt
    if kind_b:
        expected = _relay_from_lines(record, prompt)
        action = expected.split(":", 1)[0].strip()
        faithful = prose.strip() == expected.strip()
        diff = "" if faithful else ("refused" if _REFUSAL in prose else "relay")
        return {"action": action, "faithful": faithful, "diff_item": diff}
    parsed = _parsed_record(record)
    verdict = _verdict_from_parsed(parsed)
    expected = _brief_from_record(record)
    faithful = prose.strip() == expected.strip()
    return {
        "verdict": verdict,
        "faithful": faithful,
        "diff_item": "" if faithful else _diff_item(record, prose, parsed),
    }


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
    writers_file: Path | None = None,
    digest_opener=None,
) -> dict:
    """Write every text, check it, and ask the critic. Unload on the way out."""

    if not mock:
        try:
            require_grant(grant, uses_card=True)
            if heartbeat is None:
                raise GuardError("REFUSED: the pilot on the 5090 needs --heartbeat <watchdog heartbeat file>")
            require_heartbeat(heartbeat, now())
            _guard_pinned_models(writers_file, digest_opener)
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
                attempts = []
                results = []
                prose = ""
                status = "reset"
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    draft_seed = seed + attempt
                    if mock:
                        prose = _mock_draft(prompt, slot.id, role, attempt, slot.kind)
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
                    raw = complete(
                        CRITIC_MODEL,
                        critic_prompt(render(slot.ledger), prose, slot.kind),
                        seed,
                        0.0,
                        format="json",
                    )
                    _unload_all()
                    payload = _critic_payload(raw)
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
        probe = brief_only_probe(probe_items, [item["slot"] for item in items])
        kind_a = [item for item in items if item["kind"] == "A"]
        if len(kind_a) >= 2:
            probe_a = brief_only_probe(
                [(item["prose"], item["role"] == "positive") for item in kind_a],
                [item["slot"] for item in kind_a],
            )
            probe["kind_a"] = {
                "accuracy": probe_a["accuracy"],
                "null_mean": probe_a["null_mean"],
                "null_p95": probe_a["null_p95"],
                "n": probe_a["n"],
            }
            probe["message"] = (
                f"brief-only probe accuracy {probe['accuracy']:.2f} beside a 200-permutation null "
                f"(mean {probe['null_mean']:.2f}, 95th percentile {probe['null_p95']:.2f}). "
                f"On the {probe_a['n']} kind A texts, accuracy {probe_a['accuracy']:.2f} "
                f"beside a 200-permutation null "
                f"(mean {probe_a['null_mean']:.2f}, 95th percentile {probe_a['null_p95']:.2f}). "
                f"On {probe['n']} texts this is a smoke test of the machinery, not a verdict. "
                "Kind B has no negatives."
            )
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


def write_read_packet(record: dict, directory: Path) -> Path:
    """One Markdown file, one section per text, in slot order. The live read uses this too."""

    directory.mkdir(parents=True, exist_ok=True)
    parts = ["# Pilot read", ""]
    items = sorted(
        record.get("items") or [],
        key=lambda item: (item.get("slot", ""), 0 if item.get("role") == "positive" else 1),
    )
    for item in items:
        changed = item.get("verdict_changed")
        if changed is True:
            changed_text = "yes"
        elif changed is False:
            changed_text = "no"
        else:
            changed_text = "not a negative"
        plant = item.get("plant") or "none"
        check_lines = []
        for row in item.get("checks") or []:
            mark = "pass" if row.get("passed") else "fail"
            reason = f" ({row['reason']})" if row.get("reason") else ""
            check_lines.append(f"- {row.get('name')}: {mark}{reason}")
        if not check_lines:
            check_lines.append("- none")
        attempt_lines = []
        for row in item.get("attempts") or []:
            mark = "pass" if row.get("passed") else "fail"
            attempt_lines.append(
                f"- attempt {row.get('attempt')}, family {row.get('family')}, seed {row.get('seed')}: {mark}"
            )
        if not attempt_lines:
            attempt_lines.append("- none")
        critic = json.dumps(item.get("critic"), indent=2, ensure_ascii=False)
        parts.extend([
            f"## {item.get('slot')} {item.get('role')}",
            "",
            "### Record",
            "",
            item.get("record") or "",
            "",
            "### Prose",
            "",
            item.get("prose") or "",
            "",
            "### Key or planted class",
            "",
            f"Key: {item.get('key')}",
            f"Planted class: {plant}",
            f"Verdict changed: {changed_text}",
            "",
            "### Checks",
            "",
            "\n".join(check_lines),
            "",
            "### Attempts and family",
            "",
            f"Family: {item.get('family')}",
            "\n".join(attempt_lines),
            "",
            "### Critic",
            "",
            "```json",
            critic,
            "```",
            "",
        ])
    path = directory / "read.md"
    path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return path
