"""Each writer's card values: the sampling and chat-template settings its maker documents, sent on every call.

The file is rnd's `experiments/local-builders/card-values.json` (DESIGN amendments C, D and E), read from the path
passed with `--cards`. J1 (this pipeline) and J2 (the local-builders harness) read the same file, so the same model
runs at the same settings in both. A writer is selected by its `card` id in writers.json, or by its served name.
Every sampling key the card names goes on every call, including the disabled values: llama-server's defaults
(top_k 40, min_p 0.05) would otherwise reach the cards that name only temperature and top_p.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from role_set.guards import GuardError

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty")
# `reattach_thinking` re-attaches a turn's logged reasoning in J2's multi-turn harness. role-set reads only a reply's
# content and keeps no reasoning to re-attach, so here it is recorded as not applied.
TRANSFORMS = ("fold_system", "reattach_thinking")
APPLIED_HERE = ("fold_system",)


def load(path: Path) -> tuple[dict[str, dict], str]:
    """{candidate id: entry} and the file's sha256. An entry missing a sampling key, or naming an unknown transform,
    refuses the whole file."""

    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    out: dict[str, dict] = {}
    for entry in data.get("candidates", []):
        cid = entry.get("candidate")
        if not cid or cid in out:
            raise GuardError(f"REFUSED: card values need one entry per candidate id ({cid!r})")
        missing = [k for k in SAMPLING_KEYS if k not in (entry.get("sampling") or {})]
        if missing:
            raise GuardError(f"REFUSED: {cid}'s card values have no {', '.join(missing)}; every key is sent on every call")
        unknown = [t for t in entry.get("transforms") or [] if t not in TRANSFORMS]
        if unknown:
            raise GuardError(f"REFUSED: {cid} names unknown transforms {unknown}")
        out[cid] = entry
    return out, hashlib.sha256(raw).hexdigest()


def select(cards: dict[str, dict], listed_entry: dict, model: str) -> dict:
    """The card for a listed writer: its `card` id, else its served name."""

    cid = (listed_entry.get("card") or model).strip()
    if cid not in cards:
        raise GuardError(f"REFUSED: no card values for {model} (candidate {cid!r}); a writer runs only at its card's settings")
    return cards[cid]


def pins(card: dict, sha256: str) -> dict:
    """What a receipt records: the file, the candidate, and exactly what is sent."""

    return {"cards_sha256": sha256, "card_candidate": card["candidate"], "card_revision": card.get("revision"),
            "card_sampling": dict(card["sampling"]), "card_chat_template_kwargs": dict(card.get("chat_template_kwargs") or {}),
            "card_transforms": list(card.get("transforms") or []),
            "card_transforms_applied": [t for t in card.get("transforms") or [] if t in APPLIED_HERE]}


# The seven keys a writer's card records (pins() above): what a chain from an earlier drafts file must restore whole.
PIN_KEYS = frozenset(pins({"candidate": "", "sampling": {}}, ""))


def fold_system(messages: list[dict]) -> list[dict]:
    """R1-Distill's card asks for no system prompt (amendment E2). Leading system messages go to the head of the first
    user message, later ones to the tail of the user message before them, each separated by a blank line."""

    out: list[dict] = []
    head: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            users = [i for i, x in enumerate(out) if x.get("role") == "user"]
            if users:
                i = users[-1]
                out[i] = {**out[i], "content": f"{out[i]['content']}\n\n{m['content']}"}
            else:
                head.append(m["content"])
            continue
        if head and m.get("role") == "user":
            m = {**m, "content": "\n\n".join(head + [m["content"]])}
            head = []
        out.append(dict(m))
    if head:  # system text and no user message at all
        out.insert(0, {"role": "user", "content": "\n\n".join(head)})
    if any(m.get("role") == "system" for m in out):
        raise GuardError("REFUSED: fold_system left a system message")
    return out


def apply(card: dict, messages: list[dict]) -> list[dict]:
    if "fold_system" in (card.get("transforms") or []):
        return fold_system(messages)
    return messages
