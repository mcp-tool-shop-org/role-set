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
    replaying such a journal replays the first ask (ask 0, which sends no seed kwarg) directly without any fallback
    and calls any VOID retry (ask 1) fresh at retry_seed(0, 1) == 10001."""
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

    live_seeds: list[int | None] = []

    def live_critic(messages, **kw):
        live_seeds.append(kw.get("seed"))
        return Completion(text="VERDICT: supported\nDECIDING: m\nESCALATE: no", prompt_tokens=5, completion_tokens=8, model="m", seconds=0.1, finish_reason="stop")

    new_journal_path = tmp_path / "new-journal.jsonl"
    journal = callcache.CallJournal(new_journal_path, callcache.load(old_journal_path))
    wrapped = journal.wrap("critic", live_critic)

    c0 = wrapped(msg, check="verdict", slot="T10-D-01", attempt=0)
    c1 = wrapped(msg, check="verdict", slot="T10-D-01", attempt=1, seed=callcache.retry_seed(0, 1))

    assert c0.text == "not a verdict"  # replayed from old journal's first ask by exact key match
    assert c1.text.startswith("VERDICT: supported")  # called fresh at seed=10001, not replaying old ask 1
    assert live_seeds == [10001]
    assert journal.summary()["reused"] == 1 and journal.summary()["called"] == 1


def test_retry_seed_bounds_and_first_ask_keys_equal_8df62c0(tmp_path: Path):
    """Item 1: retry_seed refuses out-of-range first_seed or ask loudly, computes 10_000 + first_seed * 100 + ask
    inside [10_001, 19_999], and first-ask request keys across runner, critic_eval, and critic_control are
    byte-identical to 8df62c0's."""
    assert callcache.retry_seed(0, 1) == 10_001
    assert callcache.retry_seed(None, 1) == 10_001
    assert callcache.retry_seed(5, 1) == 10_501
    assert callcache.retry_seed(99, 99) == 19_999

    for bad_first in (-1, 100, 101, True, False, "0"):
        with pytest.raises(ValueError, match="REFUSED: retry_seed requires 0 <= first_seed < 100"):
            callcache.retry_seed(bad_first, 1)  # type: ignore[arg-type]

    for bad_ask in (0, -1, 100, 101, True, False, "1"):
        with pytest.raises(ValueError, match="REFUSED: retry_seed requires 1 <= ask < 100"):
            callcache.retry_seed(0, bad_ask)  # type: ignore[arg-type]

    # First-ask keys at 8df62c0:
    # - Runner verdict, twin_verdict, flaw, B-answer; critic_control answer_leg -> no seed kwarg at all: kwargs == {}
    # - Runner trait votes -> kwargs == {"temperature": 0.7, "seed": s}
    # - critic_eval verdict & 1-vote flaw -> kwargs == {"seed": critic_seed}; 3-vote flaw -> kwargs == {"seed": vote_seed}
    msg = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    j_path = tmp_path / "keys-8df62c0.jsonl"
    j = callcache.CallJournal(j_path)
    critic_fn = j.wrap("critic", lambda messages, **kw: Completion("ok", 1, 1, "m", 0.01, "stop"))

    # Simulate first asks as sent by runner, critic_eval, and critic_control.answer_leg
    critic_fn(msg, check="verdict", slot="T02-D-01", attempt=0)
    critic_fn(msg, check="twin_verdict", slot="T10-D-01", attempt=0)
    critic_fn(msg, check="flaw", slot="T02-C-01", attempt=0)
    critic_fn(msg, check="answer", slot="T02-B-01", attempt=0)
    critic_fn(msg, check="trait", slot="T02-A-01", attempt=2, temperature=0.7, seed=2)
    critic_fn(msg, slot="T02-D-01:d", attempt=0, seed=5)

    rows = [json.loads(ln) for ln in j_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    key_8df62c0_no_seed = callcache.request_key("critic", (msg,), {})
    key_8df62c0_trait_2 = callcache.request_key("critic", (msg,), {"temperature": 0.7, "seed": 2})
    key_8df62c0_eval_5 = callcache.request_key("critic", (msg,), {"seed": 5})

    assert rows[0]["key"] == key_8df62c0_no_seed
    assert rows[1]["key"] == key_8df62c0_no_seed
    assert rows[2]["key"] == key_8df62c0_no_seed
    assert rows[3]["key"] == key_8df62c0_no_seed
    assert rows[4]["key"] == key_8df62c0_trait_2
    assert rows[5]["key"] == key_8df62c0_eval_5


def test_window_key_set_across_runner_critic_eval_and_critic_control_has_no_unintended_duplicates(
    pack_dir: Path, plan_file: Path, tmp_path: Path  # noqa: F811
):
    """F3: run each role on the stub (runner with forced VOID retries, critic_eval across critic seeds 0, 1, 2
    with VOID retries and 3 flaw votes, critic_control) and assert the window's full key set has no unintended
    duplicates."""
    from role_set.critic_control import answer_leg, cases_from_rows, plant_errors, run_control, trait_leg
    from role_set.critic_eval import CriticEvalConfig, run_critic_eval
    from role_set.plan import plan_apply, plan_lesson
    from role_set.trait_steps import TRAIT_STEPS
    from role_set.write import WriteConfig, trait_names_from_pack
    from test_runner import PACK, SIDES, _gates, _runner

    # 1. Runner with forced VOID retries on verdict, twin_verdict, and answer
    slots = plan_lesson("T02", SIDES, seed=0)
    d_slot = next(s for s in slots if s.kind == "D" and not s.two_turn)
    c_slot = next(s for s in slots if s.kind == "C")
    b_slot = next(s for s in slots if s.kind == "B")
    apply_slot = plan_apply("T10", SIDES, seed=0)[0]
    order = {s.id: i for i, s in enumerate([d_slot, c_slot, b_slot, apply_slot])}

    runner_journal_path = tmp_path / "runner-journal.jsonl"
    runner_journal = callcache.CallJournal(runner_journal_path)
    runner, base_writer, base_critic = _runner(tmp_path / "runner-out", answer_gate=True)
    runner.pack = {**runner.pack, "traits/T10.md": PACK["traits/T02.md"]}
    runner.trait_names = {**runner.trait_names, "T10": "goes to the source"}
    base_writer.keyed_order = order
    verdict_calls = {"n": 0}
    answer_calls = {"n": 0}

    def void_stub_critic(messages, **kwargs):
        sys_text = messages[0]["content"]
        if "VERDICT:" in sys_text:
            verdict_calls["n"] += 1
            # d_slot verdict ask 0 (call 1) and apply_slot twin_verdict ask 0 (call 5) return VOID
            if verdict_calls["n"] in (1, 5):
                return Completion("not a verdict", 5, 3, "mock", 0.01, "stop")
        elif "Solve the puzzle" in sys_text:
            answer_calls["n"] += 1
            if answer_calls["n"] == 1:
                return Completion("missing answer line", 5, 3, "mock", 0.01, "stop")
        return base_critic(messages, **kwargs)

    runner.writer = runner_journal.wrap("writer", base_writer)
    runner.critic = runner_journal.wrap("critic", void_stub_critic)
    assert runner.warmup("T02")
    assert runner.run_slot(d_slot).accepted
    assert runner.run_slot(c_slot).accepted
    assert runner.run_slot(b_slot).accepted
    base_critic.trait = "T10"
    assert runner.run_slot(apply_slot).accepted

    runner_rows = [json.loads(ln) for ln in runner_journal_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    runner_keys = [r["key"] for r in runner_rows]
    assert len(runner_keys) == len(set(runner_keys)), f"Runner had duplicate keys: {runner_keys}"

    # 2. critic_eval across critic seeds 0, 1, 2 with VOID retries and 3 flaw votes
    w0_dir = tmp_path / "w0"
    run(WriteConfig(plan=plan_file, pack=pack_dir, out=w0_dir, mock=True, partition="trial:f3", wave="write"))
    h0, d0 = waves.read(w0_dir / "mock" / waves.DRAFTS_FILE)
    kept_drafts = [
        next(d for d in d0 if d["slot"]["kind"] == "C"),
        next(d for d in d0 if d["slot"]["kind"] == "D" and not d["slot"].get("two_turn")),
    ]
    eval_items = []
    for idx, d in enumerate(kept_drafts):
        slot = d["slot"]
        cand = json.loads(d["candidate"]["text"])
        row_obj = d.get("row") or {}
        first_line = d["material"]["text"].splitlines()[0]
        verdict = row_obj.get("verdict") or slot.get("verdict") or "supported"
        deciding = list(row_obj.get("deciding") or cand.get("deciding") or ([] if verdict == "cannot_tell" else [first_line]))
        planted_flaw = row_obj.get("planted_flaw") or cand.get("planted_flaw") or ("guess" if slot["kind"] == "C" else "none")
        eval_items.append({
            **d,
            "type": "draft",
            "item_id": f"{slot['id']}:d",
            "source_row_id": slot["id"],
            "perturbation_type": "d",
            "writer_model": "qwen3.8-27b",
            "writer_family": "qwen",
            "key": {"verdict": verdict, "deciding": deciding, "planted_flaw": planted_flaw},
        })
    items_path = tmp_path / "sealed-eval-items.jsonl"
    waves.write(items_path, h0, eval_items)

    eval_base_critic = MockCritic(answers=waves.load_answers(h0))
    eval_rows_by_seed: dict[int, list[dict]] = {}
    root = Path(__file__).resolve().parents[1]
    for c_seed in (0, 1, 2):
        voided_first_verdict: set[int] = set()
        verdict_seq = {"n": 0}

        def eval_critic(messages, _voided=voided_first_verdict, _seq=verdict_seq, **kwargs):
            if "VERDICT:" in messages[0]["content"]:
                _seq["n"] += 1
                # Every odd call (1, 3, ...) is the first ask for that item's verdict; force VOID so retry runs
                if _seq["n"] % 2 == 1:
                    _voided.add(_seq["n"])
                    return Completion("unparseable verdict reply", 10, 5, "mock", 0.01, "stop")
            return eval_base_critic(messages, **kwargs)

        j_seed_path = tmp_path / f"eval-seed-{c_seed}.jsonl"
        run_critic_eval(
            CriticEvalConfig(
                items=items_path,
                pack=pack_dir,
                out=tmp_path / f"eval-out-{c_seed}",
                writers_file=root / "writers.json",
                critic_model="nemotron-3.5-lightning-30b-a3b",
                critic_seed=c_seed,
                flaw_votes=3,
                call_journal=j_seed_path,
                mock=True,
            ),
            critic=eval_critic,
        )
        seed_rows = [json.loads(ln) for ln in j_seed_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        eval_rows_by_seed[c_seed] = seed_rows
        seed_keys = [r["key"] for r in seed_rows]
        assert len(seed_keys) == len(set(seed_keys)), f"critic_eval seed {c_seed} had internal duplicate keys"

    # Across critic_seed=0, 1, 2: every verdict call (2 items * 2 asks * 3 seeds = 12 calls) is a distinct call
    # and must have a unique key; only flaw votes with identical vote_seed=critic_seed+vote_idx share a key.
    verdict_keys_with_origin: list[tuple[str, tuple[int, str, int]]] = []
    flaw_keys_by_call: dict[tuple[str, int], set[str]] = {}
    for c_seed, seed_rows in eval_rows_by_seed.items():
        # For each item: 2 verdict calls (attempt 0 and 1), then (for C) 3 flaw vote calls (attempt 0, 1, 2)
        assert len(seed_rows) == 7  # item 0 (C): 2 verdict + 3 flaw; item 1 (D): 2 verdict
        for r in (seed_rows[0], seed_rows[1], seed_rows[5], seed_rows[6]):
            verdict_keys_with_origin.append((r["key"], (c_seed, r["slot"], r["attempt"])))
        for vote_idx, r in enumerate(seed_rows[2:5]):
            vote_seed = c_seed + vote_idx
            flaw_keys_by_call.setdefault((r["slot"], vote_seed), set()).add(r["key"])

    seen_verdict_key: dict[str, tuple[int, str, int]] = {}
    collisions: list[tuple[str, tuple[int, str, int], tuple[int, str, int]]] = []
    for k, origin in verdict_keys_with_origin:
        if k in seen_verdict_key:
            collisions.append((k, seen_verdict_key[k], origin))
        else:
            seen_verdict_key[k] = origin
    assert not collisions, f"Unintended duplicate verdict keys across critic_eval seeds 0, 1, 2: {collisions}"
    assert len(flaw_keys_by_call) == 5 and all(len(ks) == 1 for ks in flaw_keys_by_call.values())
    all_flaw_keys = [next(iter(ks)) for ks in flaw_keys_by_call.values()]
    assert len(all_flaw_keys) == len(set(all_flaw_keys))
    assert not (set(seen_verdict_key) & set(all_flaw_keys))

    # 3. critic_control (run_control, trait_leg, answer_leg) with forced VOID retries
    ctrl_journal_path = tmp_path / "ctrl-journal.jsonl"
    ctrl_journal = callcache.CallJournal(ctrl_journal_path)
    accepted_rows = list(runner.accepted)
    ctrl_cases = cases_from_rows(accepted_rows)
    planted_cases = plant_errors(ctrl_cases, 1, seed=0)
    ctrl_verdict_seen: set[int] = set()
    ctrl_verdict_n = {"n": 0}
    ctrl_answer_n = {"n": 0}

    def ctrl_stub_critic(messages, **kwargs):
        sys_text = messages[0]["content"]
        if "VERDICT:" in sys_text:
            ctrl_verdict_n["n"] += 1
            if ctrl_verdict_n["n"] == 1:
                ctrl_verdict_seen.add(1)
                return Completion("void verdict", 5, 3, "mock", 0.01, "stop")
        elif "Solve the puzzle" in sys_text:
            ctrl_answer_n["n"] += 1
            if ctrl_answer_n["n"] == 1:
                return Completion("void answer", 5, 3, "mock", 0.01, "stop")
        return base_critic(messages, **kwargs)

    journaled_ctrl_critic = ctrl_journal.wrap("critic", ctrl_stub_critic)
    base_critic.trait = "T02"
    rep_ctrl = run_control(_gates(), journaled_ctrl_critic, ctrl_cases, planted=planted_cases, seed=0)
    assert rep_ctrl.voids == 1
    t_leg = trait_leg(
        journaled_ctrl_critic,
        [r for r in accepted_rows if r.get("lesson") == "T02"],
        trait_names_from_pack(PACK),
        TRAIT_STEPS,
    )
    assert t_leg["items"] >= 1
    a_leg = answer_leg(journaled_ctrl_critic, accepted_rows)
    assert a_leg["voids"] == 1

    ctrl_rows = [json.loads(ln) for ln in ctrl_journal_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in ctrl_rows:
        assert isinstance(r.get("slot"), str) and r["slot"]
        assert isinstance(r.get("attempt"), int)
        assert isinstance(r.get("messages_sha256"), str) and len(r["messages_sha256"]) == 64
    # Only the planted case intentionally shares its unplanted source case's verdict call; all other ctrl calls are unique
    assert len(ctrl_rows) - len({r["key"] for r in ctrl_rows}) == len(planted_cases)


def test_fold_system_card_row_hashes_folded_messages_with_unchanged_request_key(tmp_path: Path, monkeypatch):
    """F4: messages_sha256 hashes the messages exactly as sent to transport.chat (after cards.apply, e.g. fold_system),
    while the request key stays computed from the call's unfolded args."""
    import role_set.write as w
    from role_set import cards, transport
    from role_set.write import critic_caller

    raw_messages = [
        {"role": "system", "content": "You are a careful verifier."},
        {"role": "user", "content": "Material:\nLine 1.\nStatement: Line 1."},
    ]
    r1_card = {
        "candidate": "deepseek-r1-distill-qwen-32b",
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        },
        "transforms": ["fold_system"],
    }
    folded = cards.apply(r1_card, raw_messages)
    assert folded != raw_messages
    assert all(m["role"] != "system" for m in folded)

    expected_folded_sha = callcache.messages_sha256(folded)
    unfolded_sha = callcache.messages_sha256(raw_messages)
    assert expected_folded_sha != unfolded_sha

    expected_key = callcache.request_key("critic", (raw_messages,), {"seed": 3})

    sent_bodies: list[dict] = []

    def fake_post(url, body, timeout_s):
        sent_bodies.append(body)
        return {
            "model": "deepseek-r1-distill-qwen-32b",
            "choices": [{"message": {"content": "VERDICT: supported\nDECIDING: Line 1.\nESCALATE: no"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    monkeypatch.setattr(w, "chat", partial(transport.chat, post=fake_post))
    caller = critic_caller("http://127.0.0.1:18080/v1", "deepseek-r1-distill-qwen-32b", r1_card)
    j_path = tmp_path / "folded-journal.jsonl"
    journal = callcache.CallJournal(j_path)
    wrapped = journal.wrap("critic", caller)

    c = wrapped(raw_messages, seed=3, check="verdict", slot="T02-D-01", attempt=0)
    assert c.messages_sha256 == expected_folded_sha
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["messages"] == folded

    rows = [json.loads(ln) for ln in j_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["key"] == expected_key
    assert rows[0]["messages_sha256"] == expected_folded_sha
    assert rows[0]["messages_sha256"] != unfolded_sha
    assert rows[0]["slot"] == "T02-D-01"
    assert rows[0]["attempt"] == 0

    # Replay from journal also records the folded messages_sha256 and keeps the unfolded request key
    j_replay_path = tmp_path / "folded-replay.jsonl"
    replay_journal = callcache.CallJournal(j_replay_path, callcache.load(j_path))
    wrapped_replay = replay_journal.wrap("critic", caller)
    c_replayed = wrapped_replay(raw_messages, seed=3, check="verdict", slot="T02-D-01", attempt=0)
    assert c_replayed.text == c.text
    assert len(sent_bodies) == 1  # no second transport call
    replay_rows = [json.loads(ln) for ln in j_replay_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert replay_rows[0]["reused"] is True
    assert replay_rows[0]["key"] == expected_key
    assert replay_rows[0]["messages_sha256"] == expected_folded_sha



