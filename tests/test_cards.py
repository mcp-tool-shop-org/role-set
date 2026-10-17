import json
from pathlib import Path

import pytest

from role_set import cards
from role_set.guards import GuardError

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0}


# Every served name the write and wave tests list (test_proven.entry).
TEST_MODELS = ("big-writer", "small-writer", "other-critic", "same-critic", "compare-critic", "w", "c")


def card(cid: str, **extra) -> dict:
    return {"candidate": cid, "revision": "r1", "sampling": dict(SAMPLING), "chat_template_kwargs": {"enable_thinking": True},
            "transforms": [], **extra}


def cards_file(tmp_path: Path, entries=None) -> Path:
    path = tmp_path / "card-values.json"
    path.write_text(json.dumps({"candidates": entries if entries is not None else [card(n) for n in TEST_MODELS]}), encoding="utf-8")
    return path


def test_a_card_needs_every_sampling_key_and_known_transforms(tmp_path: Path):
    loaded, sha = cards.load(cards_file(tmp_path))
    assert set(loaded) == set(TEST_MODELS) and len(sha) == 64
    thin = card("w")
    del thin["sampling"]["min_p"]
    with pytest.raises(GuardError, match="no min_p"):
        cards.load(cards_file(tmp_path, [thin]))
    with pytest.raises(GuardError, match="unknown transforms"):
        cards.load(cards_file(tmp_path, [card("w", transforms=["shout"])]))
    with pytest.raises(GuardError, match="one entry per candidate"):
        cards.load(cards_file(tmp_path, [card("w"), card("w")]))


def test_a_writer_is_selected_by_its_card_id_or_its_served_name(tmp_path: Path):
    loaded, sha = cards.load(cards_file(tmp_path, [card("big-writer"), card("bolt-24gb")]))
    assert cards.select(loaded, {}, "big-writer")["candidate"] == "big-writer"
    assert cards.select(loaded, {"card": "bolt-24gb"}, "bolt")["candidate"] == "bolt-24gb"
    with pytest.raises(GuardError, match="no card values"):
        cards.select(loaded, {}, "unlisted")
    p = cards.pins(card("r1", transforms=["fold_system", "reattach_thinking"]), sha)
    assert p["card_sampling"] == SAMPLING and p["card_chat_template_kwargs"] == {"enable_thinking": True}
    assert p["card_transforms_applied"] == ["fold_system"] and p["cards_sha256"] == sha


def test_fold_system_leaves_no_system_message():
    """Amendment E2: leading system text heads the first user turn; a later one (Aider's reminder, or role-set's
    own) tails the user turn before it."""

    msgs = [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}, {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}, {"role": "system", "content": "R"}]
    out = cards.fold_system(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"] == "S1\n\nS2\n\nU1" and out[2]["content"] == "U2\n\nR"
    assert msgs[2]["content"] == "U1"  # the caller's messages are untouched
    assert cards.fold_system([{"role": "system", "content": "only"}]) == [{"role": "user", "content": "only"}]
    assert cards.apply(card("w"), msgs) is msgs
    assert cards.apply(card("w", transforms=["fold_system"]), msgs) == out
