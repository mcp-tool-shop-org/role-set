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
