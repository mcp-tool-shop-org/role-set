"""The call journal and the call-level resume (R&D, 2026-10-17): a wave cut off partway and resumed from its journal
writes what one uninterrupted wave writes, asks the model only for the calls that never finished, and loses no slot."""

import json
from functools import partial
from pathlib import Path

import pytest
from test_waves import _cfg, _jsonl, pack_dir, plan_file  # noqa: F401 - fixtures

from role_set import callcache, waves
from role_set.mock import MockCritic, MockWriter
from role_set.transport import Completion
from role_set.write import run


class _Killed(BaseException):
    """The VM going away: nothing in the wave catches it."""


def _dying(cls, after: int):
    calls = {"n": 0}

    class Dying(cls):
        def __call__(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] > after:
                raise _Killed()
            return super().__call__(*a, **kw)

    return Dying


def _drafts(out: Path) -> list[dict]:
    return waves.read(out / "mock" / waves.DRAFTS_FILE)[1]


def _journal(out: Path) -> list[dict]:
    return _jsonl(out / "mock" / callcache.JOURNAL_FILE)


def _slot_calls(journal: list[dict]) -> int:
    return sum(1 for d in journal if not d["reused"])


def test_a_write_wave_resumed_from_its_journal_matches_an_uninterrupted_one(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    whole = run(_cfg(plan_file, pack_dir, tmp_path / "whole", wave="write"))
    total = whole["calls"]["called"]
    assert total > 10 and whole["calls"]["reused"] == 0

    cut = total // 2
    monkeypatch.setattr(w, "MockWriter", _dying(MockWriter, cut))
    with pytest.raises(_Killed):
        run(_cfg(plan_file, pack_dir, tmp_path / "cut", wave="write"))
    assert len(_journal(tmp_path / "cut")) == cut  # every call that returned is on disk, none that didn't

    monkeypatch.setattr(w, "MockWriter", MockWriter)
    resumed = run(_cfg(plan_file, pack_dir, tmp_path / "resumed", wave="write",
                       reuse_calls=tmp_path / "cut" / "mock" / callcache.JOURNAL_FILE))
    assert resumed["calls"] == {"journal": callcache.JOURNAL_FILE, "reused": cut, "called": total - cut, "unused_from_reuse": 0}
    # No slot lost and none run twice: the same drafts, in the same states, from the same number of model calls.
    assert _drafts(tmp_path / "resumed") == _drafts(tmp_path / "whole")
    assert resumed["states"] == whole["states"]
    assert _slot_calls(_journal(tmp_path / "cut")) + _slot_calls(_journal(tmp_path / "resumed")) == total
    # The resumed journal holds every call, so a second cut can resume from it in turn.
    assert len(_journal(tmp_path / "resumed")) == total


def test_a_critique_wave_resumed_from_its_journal_matches_an_uninterrupted_one(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    whole = run(_cfg(plan_file, pack_dir, tmp_path / "whole", wave="critique", drafts=drafts))
    total = whole["calls"]["called"]
    assert total > 10

    cut = total // 3
    monkeypatch.setattr(w, "MockCritic", _dying(MockCritic, cut))
    with pytest.raises(_Killed):
        run(_cfg(plan_file, pack_dir, tmp_path / "cut", wave="critique", drafts=drafts))
    monkeypatch.setattr(w, "MockCritic", MockCritic)
    resumed = run(_cfg(plan_file, pack_dir, tmp_path / "resumed", wave="critique", drafts=drafts,
                       reuse_calls=tmp_path / "cut" / "mock" / callcache.JOURNAL_FILE))
    assert (resumed["calls"]["reused"], resumed["calls"]["called"]) == (cut, total - cut)
    assert resumed["states"] == whole["states"] and resumed["accepted"] == whole["accepted"]
    assert _drafts(tmp_path / "resumed") == _drafts(tmp_path / "whole")
    rows = lambda d: _jsonl(d / "mock" / "accepted.jsonl")  # noqa: E731
    assert rows(tmp_path / "resumed") == rows(tmp_path / "whole")


def test_a_resume_with_critic_refusals_keeps_the_refusals(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    first = [json.loads(d["candidate"]["text"])["statement"] for d in waves.read(drafts)[1] if d["slot"]["kind"] == "D"][:3]
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_once=set(first)))
    whole = run(_cfg(plan_file, pack_dir, tmp_path / "whole", wave="critique", drafts=drafts))
    assert whole["states"]["needs_rewrite"] >= 2
    cut = whole["calls"]["called"] // 2
    # The mock critic remembers what it has refused (a real model doesn't); one set across the cut and the resume
    # stands in for a stateless critic, so a statement refused before the cut isn't refused a second time after it.
    once = set(first)
    monkeypatch.setattr(w, "MockCritic", partial(_dying(MockCritic, cut), disagree_once=once))
    with pytest.raises(_Killed):
        run(_cfg(plan_file, pack_dir, tmp_path / "cut", wave="critique", drafts=drafts))
    assert once != set(first)  # some refusals happened before the cut: they come back from the journal
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_once=once))
    resumed = run(_cfg(plan_file, pack_dir, tmp_path / "resumed", wave="critique", drafts=drafts,
                       reuse_calls=tmp_path / "cut" / "mock" / callcache.JOURNAL_FILE))
    assert resumed["states"] == whole["states"]
    assert _drafts(tmp_path / "resumed") == _drafts(tmp_path / "whole")


def test_a_truncated_call_replays_as_truncated_and_a_torn_line_is_asked_again(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    asked = []

    def model(messages, seed=0):
        asked.append(seed)
        return Completion(text="cut", prompt_tokens=5, completion_tokens=16384, model="m", seconds=1.0,
                          finish_reason="length" if seed == 1 else "stop")

    j = callcache.CallJournal(path)
    f = j.wrap("writer", model)
    f([{"role": "user", "content": "a"}], seed=1)
    f([{"role": "user", "content": "b"}], seed=2)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "torn", "role": "writer", "compl')  # the VM went mid-write

    again = callcache.CallJournal(tmp_path / "k.jsonl", callcache.load(path))
    g = again.wrap("writer", model)
    c = g([{"role": "user", "content": "a"}], seed=1)
    assert c.finish_reason == "length" and c.completion_tokens == 16384  # kept as its outcome, not re-asked
    g([{"role": "user", "content": "b"}], seed=2)
    g([{"role": "user", "content": "c"}], seed=3)
    assert asked == [1, 2, 3] and again.summary()["reused"] == 2 and again.summary()["called"] == 1


def test_the_same_request_twice_replays_in_journal_order(tmp_path: Path):
    path = tmp_path / "j.jsonl"
    texts = iter(["first", "second", "third"])
    j = callcache.CallJournal(path)
    f = j.wrap("critic", lambda messages, **kw: Completion(next(texts), 1, 1, "m", 0.1, "stop"))
    msg = [{"role": "user", "content": "same"}]
    assert [f(msg).text, f(msg).text] == ["first", "second"]
    again = callcache.CallJournal(tmp_path / "k.jsonl", callcache.load(path)).wrap("critic", lambda messages, **kw: Completion("new", 1, 1, "m", 0.1, "stop"))
    assert [again(msg).text, again(msg).text, again(msg).text] == ["first", "second", "new"]


def test_the_wrapper_passes_on_the_keyed_mock_flag(tmp_path: Path):
    j = callcache.CallJournal(tmp_path / "j.jsonl")
    assert j.wrap("writer", MockWriter(keyed_order={"x": 0})).takes_slot_id is True
    assert j.wrap("writer", MockWriter()).takes_slot_id is False
    assert j.wrap("critic", None) is None


def test_journal_row_carries_messages_sha256_slot_and_attempt_with_unchanged_key_and_replays_old_journal(
    pack_dir: Path, plan_file: Path, tmp_path: Path  # noqa: F811
):
    import hashlib

    msg = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    expected_msg_sha = hashlib.sha256(
        json.dumps(msg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    old_way_writer_key = callcache.request_key("writer", (msg,), {"seed": 2})
    old_way_critic_key = callcache.request_key("critic", (msg,), {"seed": 1, "temperature": 0.2})

    j_path = tmp_path / "direct.jsonl"
    j = callcache.CallJournal(j_path)
    seen_kwargs: list[dict] = []

    def fake_model(messages, **kw):
        seen_kwargs.append(dict(kw))
        return Completion(text="ok", prompt_tokens=3, completion_tokens=4, model="m", seconds=0.1, finish_reason="stop")

    w_fn = j.wrap("writer", fake_model)
    c_fn = j.wrap("critic", fake_model)
    w_fn(msg, seed=2, slot="T10-B-01", attempt=2)
    c_fn(msg, seed=1, temperature=0.2, check="trait", slot="T10-B-01", attempt=1)

    # slot and attempt are popped before calling fn and before computing request_key
    assert seen_kwargs == [{"seed": 2}, {"seed": 1, "temperature": 0.2}]
    rows = [json.loads(line) for line in j_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["key"] == old_way_writer_key
    assert rows[0]["messages_sha256"] == expected_msg_sha
    assert rows[0]["slot"] == "T10-B-01"
    assert rows[0]["attempt"] == 2
    assert rows[1]["key"] == old_way_critic_key
    assert rows[1]["messages_sha256"] == expected_msg_sha
    assert rows[1]["slot"] == "T10-B-01"
    assert rows[1]["attempt"] == 1
    assert rows[1]["check"] == "trait"

    # Without slot/attempt passed explicitly, key is byte-identical
    j2_path = tmp_path / "without-fields.jsonl"
    j2 = callcache.CallJournal(j2_path)
    j2.wrap("writer", fake_model)(msg, seed=2)
    j2.wrap("critic", fake_model)(msg, seed=1, temperature=0.2, check="trait")
    rows_without = [json.loads(line) for line in j2_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["key"] for r in rows] == [r["key"] for r in rows_without]

    # End-to-end inline run: every row in calls-journal.jsonl carries messages_sha256, slot, and attempt
    run(_cfg(plan_file, pack_dir, tmp_path / "inline-run", wave="inline"))
    inline_journal = _journal(tmp_path / "inline-run")
    assert len(inline_journal) > 10
    for r in inline_journal:
        assert isinstance(r.get("messages_sha256"), str) and len(r["messages_sha256"]) == 64
        assert isinstance(r.get("slot"), str) and r["slot"]
        assert isinstance(r.get("attempt"), int)

    # Strip messages_sha256, slot, and attempt to simulate an old journal without the new fields, and replay it
    old_journal_path = tmp_path / "old-journal.jsonl"
    stripped_lines = []
    for r in inline_journal:
        r_old = {k: v for k, v in r.items() if k not in ("messages_sha256", "slot", "attempt")}
        assert "messages_sha256" not in r_old and "slot" not in r_old and "attempt" not in r_old
        stripped_lines.append(json.dumps(r_old, ensure_ascii=False))
    old_journal_path.write_text("\n".join(stripped_lines) + "\n", encoding="utf-8")

    replayed = run(_cfg(plan_file, pack_dir, tmp_path / "replayed-run", wave="inline", reuse_calls=old_journal_path))
    assert replayed["calls"]["called"] == 0
    assert replayed["calls"]["reused"] == len(inline_journal)
    assert replayed["calls"]["unused_from_reuse"] == 0
    replayed_journal = _journal(tmp_path / "replayed-run")
    assert all(
        r["reused"] is True
        and isinstance(r.get("messages_sha256"), str)
        and isinstance(r.get("slot"), str)
        and isinstance(r.get("attempt"), int)
        for r in replayed_journal
    )


def test_old_journal_before_retry_seed_change_replays_first_ask_and_calls_retry_fresh(tmp_path: Path):
    """Journals written before the critic retry seed change had no seed in kwargs on verdict/twin_verdict/answer asks:
    replaying such a journal replays the first ask (ask 0) and calls any VOID retry (ask 1) fresh at seed=1."""
    msg = [{"role": "system", "content": "verdict sys"}, {"role": "user", "content": "Material:\nm\nStatement: s"}]
    legacy_key = callcache.request_key("critic", (msg,), {})
    old_journal_path = tmp_path / "pre-seed-change.jsonl"
    # Old journal has two entries under legacy_key (ask 0 VOID + ask 1 retry, both recorded without seed in kwargs)
    old_rows = [
        {"key": legacy_key, "role": "critic", "reused": False, "check": "verdict",
         "completion": {"text": "not a verdict", "prompt_tokens": 5, "completion_tokens": 3, "model": "m", "seconds": 0.1, "finish_reason": "stop"}},
        {"key": legacy_key, "role": "critic", "reused": False, "check": "verdict",
         "completion": {"text": "stale unseeded retry", "prompt_tokens": 5, "completion_tokens": 3, "model": "m", "seconds": 0.1, "finish_reason": "stop"}},
    ]
    old_journal_path.write_text("\n".join(json.dumps(r) for r in old_rows) + "\n", encoding="utf-8")

    live_seeds: list[int] = []

    def live_critic(messages, seed: int = 0, **_):
        live_seeds.append(seed)
        return Completion(text="VERDICT: supported\nDECIDING: m\nESCALATE: no", prompt_tokens=5, completion_tokens=8, model="m", seconds=0.1, finish_reason="stop")

    new_journal_path = tmp_path / "new-journal.jsonl"
    journal = callcache.CallJournal(new_journal_path, callcache.load(old_journal_path))
    wrapped = journal.wrap("critic", live_critic)

    c0 = wrapped(msg, check="verdict", slot="T10-D-01", attempt=0, seed=0)
    c1 = wrapped(msg, check="verdict", slot="T10-D-01", attempt=1, seed=1)

    assert c0.text == "not a verdict"  # replayed from old journal's first ask
    assert c1.text.startswith("VERDICT: supported")  # called fresh at seed=1, not replaying old ask 1
    assert live_seeds == [1]
    assert journal.summary()["reused"] == 1 and journal.summary()["called"] == 1

