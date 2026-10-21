import json
import types
from pathlib import Path

import pytest

from role_set.assemble import Material
from role_set.gates import Gates
from role_set.mock import MockCritic, MockWriter, mock_excerpts
from role_set.plan import plan_lesson
from role_set.runner import SUGGESTIONS, RunHalted, Runner, truth_checks

SIDES = ("quotes the source", "stops at the source")
PACK = {"traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n", "kinds.md": "kinds\n", "amendments/1.md": "## Amendment 1\n\nq\n"}
TRAITS = {"T02": "goes to the source", "T03": "precise"}


def _gates(refuse_ids=(), against=None, role_words=()):
    m = types.SimpleNamespace()
    m.DataError = ValueError

    def check_item(row):
        if row["id"] in refuse_ids:
            raise ValueError(f"refused {row['id']}")

    m.check_item = check_item
    m.check_apply_item = check_item  # gates 1.7.5: apply rows have their own check; the fake treats them alike
    m.check_trace_leaks_label = lambda row: None
    m.models_seen = []

    def check_against(row, accepted, models=None):
        m.models_seen.append(models)
        return (against or (lambda row, accepted: []))(row, accepted)

    m.check_against = check_against
    m.check_batch = lambda rows, models=None: None
    m.role_words_in_prompt = lambda row: list(role_words) if row["kind"] in "CD" else []
    m.norm = lambda x: " ".join(x.replace("**", "").replace("__", "").split())
    m.critic_mismatch = lambda key, critic, material: not ({" ".join(x.split()) for x in key} & {" ".join(x.split()) for x in critic})
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    return Gates(module=m, version="1.1.0", sha256="g" * 64)


def _runner(tmp_path: Path, **kw):
    answers = {}
    writer = MockWriter(answers=answers, fail_first=kw.pop("fail_first", None))
    critic = MockCritic(answers=answers, disagree=kw.pop("disagree", None), trait="T02")
    excerpts = kw.pop("excerpts", None)
    if excerpts is None:
        base_ex = mock_excerpts()
        excerpts = {
            **base_ex,
            "changelog": [
                Material("## [1.0.0] - 2026-01-01\n- Added the thing.\n", {"repo": "x/y", "commit": "abc", "path": "C.md", "lines": "L1", "licence": "MIT"}),
                *base_ex["changelog"],
            ],
        }
    r = Runner(gates=kw.pop("gates", _gates()), pack=PACK, trait_names=TRAITS, writer=writer, critic=critic, out_dir=tmp_path / "out", excerpts=excerpts, pins={"writer_revision": "deadbeef"}, **kw)
    return r, writer, critic


def test_a_whole_lesson_runs_end_to_end_with_the_mock(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    slots = plan_lesson("T02", SIDES, seed=0)
    outcomes = r.run(slots)
    assert all(o.accepted for o in outcomes), [(o.slot_id, o.refusals) for o in outcomes if not o.accepted]
    rows = [json.loads(l) for l in (tmp_path / "out" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and {x["kind"] for x in rows} == set("ABCDE")
    receipts = [json.loads(l) for l in (tmp_path / "out" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert receipts[0]["gates"]["version"] == "1.1.0" and receipts[0]["writer_revision"] == "deadbeef"
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["accepted"] == 40 and status["dropped"] == 0 and status["critic_agreement"] == 1.0
    assert status["actions"][-1]["action"] == "end" and "shut the pod down" in status["actions"][-1]["reminder"]
    assert (tmp_path / "out" / "warmup-T02.json").exists()
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert len(report["outcomes"]) == 40
    real = [x for x in rows if x["key_basis"] == "source"]
    assert real and real[0]["sources"][0]["repo"] == "x/y"
    assert report["batch_check"] == "clean" and r.batch_check == "clean"


def test_the_finished_set_goes_through_check_batch_and_a_refusal_is_reported(tmp_path: Path):
    gates = _gates()
    seen = []

    def refuse(rows, models=None):
        seen.append(len(rows))
        raise ValueError("T02: 0 of 12 verdict-bearing items have 2-4 DECIDING lines")

    gates.module.check_batch = refuse
    r, _, _ = _runner(tmp_path, gates=gates)
    r.run(plan_lesson("T02", SIDES, seed=0))
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert seen == [40] and report["batch_check"].startswith("REFUSED: T02: 0 of 12")
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert any(a["action"] == "batch_check" and a["result"].startswith("REFUSED") for a in status["actions"])


def test_refusal_becomes_a_suggestion_and_the_retry_is_accepted(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    d = next(s for s in slots if s.kind == "D" and not s.two_turn)
    r, writer, _ = _runner(tmp_path, fail_first={f"{d.lesson}-{d.kind}"})
    assert r.warmup("T02")
    out = r.run_slot(d)
    assert out.accepted and out.attempts == 2
    assert out.refusals[0]["code"] == "PARSE" and "JSON object" in out.refusals[0]["suggestion"]
    assert any("Refused:" in s for s in writer.seen)


def test_gate_refusals_critic_mismatch_and_role_words(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    d = next(s for s in slots if s.kind == "D" and not s.two_turn)
    r, _, critic = _runner(tmp_path, gates=_gates(refuse_ids={d.id}))
    out = r.run_slot(d)
    assert not out.accepted and out.attempts == 3 and {x["code"] for x in out.refusals} == {"GATE"}
    assert "validator refused" in out.refusals[0]["suggestion"]

    r2, _, critic2 = _runner(tmp_path / "b")
    critic2.disagree_all = True
    d2 = next(s for s in slots if s.kind == "D" and s.verdict == "supported" and not s.two_turn)
    out2 = r2.run_slot(d2)
    assert not out2.accepted and out2.refusals[0]["code"] == "VERDICT_MISMATCH" and "second reader" in out2.refusals[0]["suggestion"]

    r3, _, _ = _runner(tmp_path / "c", gates=_gates(role_words=("verifier",)))
    out3 = r3.run_slot(d)
    assert out3.refusals[0]["code"] == "ROLE_WORDS" and "amendment 3" in out3.refusals[0]["suggestion"]

    r4, _, _ = _runner(tmp_path / "d", gates=_gates(against=lambda row, acc: [("FRAMING_REUSED", "the framing is already used")]))
    out4 = r4.run_slot(d)
    assert out4.refusals[0]["code"] == "AGAINST" and "FRAMING_REUSED" in out4.refusals[0]["detail"]


def _two_lessons(r):
    # T03 borrows T02's lesson text so its warm-up passes with the mock writer; the mock critic only knows T02, and these
    # tests are about the andon, so they run without it.
    r.pack = {**r.pack, "traits/T03.md": PACK["traits/T02.md"]}
    r.critic = None
    r.trait_names = {**r.trait_names, "T03": "goes to the source", "T06": "goes to the source"}
    t2, t3 = plan_lesson("T02", SIDES, seed=0), plan_lesson("T03", SIDES, seed=0)
    return t2, t3, {"T02": SIDES, "T03": SIDES}


def test_a_cell_halt_ends_that_lessons_kind_and_the_rest_goes_on(tmp_path: Path):
    # The 235B pilot: kind B failed first in every lesson and its halt ended the whole lesson. Now three drops end only
    # that lesson's kind; its other kinds and the other lessons run.
    r, _, _ = _runner(tmp_path)
    t2, t3, sides = _two_lessons(r)
    r.gates = _gates(refuse_ids={s.id for s in t2 if s.kind == "A"})
    outcomes = r.run(t2 + t3)
    assert r.status.halted is None and not r.status.lesson_halts
    assert r.status.cell_halts == {"T02/A": {"reason": "3 dropped slots", "kind": "A", "rule": "GATE: refused"}}
    assert sum(o.accepted for o in outcomes if o.slot_id.startswith("T03")) == 40
    assert sum(o.accepted for o in outcomes if o.slot_id.startswith("T02")) == 32
    assert r.status.skipped == {"T02/A halted": 5}
    log = [json.loads(line) for line in (tmp_path / "out" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(a["lesson"], a["kind"]) for a in log if a["action"] == "cell_halt"] == [("T02", "A")]
    assert (tmp_path / "out" / "batch-report.json").exists()

def test_a_failed_warmup_halts_its_lesson_before_any_slot(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    writer.__class__.__call__  # noqa: B018 - the writer is replaced below
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text="not json", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
    r.run(plan_lesson("T02", SIDES, seed=0))
    assert r.status.attempted == 0 and r.status.lesson_halts["T02"]["rule"] == "WARMUP" and r.status.skipped == {"lesson T02 warm-up failed": 40}
    assert r.status.halted is None and not r.status.kinds_off
    kept = [json.loads(line) for line in (tmp_path / "out" / "warmups.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(w["attempt"], w["passed"], w["reply"]) for w in kept] == [(1, False, "not json"), (2, False, "not json")]


def test_warmup_failures_in_every_lesson_never_stop_the_run_as_systemic(tmp_path: Path):
    # The 235B pilot's first two attempts ended in 16 seconds: two warm-up misses were counted as every kind off.
    r, _, _ = _runner(tmp_path)
    t2, t3, sides = _two_lessons(r)
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text="not json", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
    r.run(t2 + t3)
    assert r.status.halted is None and not r.status.kinds_off and set(r.status.lesson_halts) == {"T02", "T03"}


def test_the_warmup_names_the_trait_by_its_own_words_not_the_planner_sides():
    from role_set.runner import warmup_terms

    assert warmup_terms("keeps things in proportion") == {"proportion"}
    assert warmup_terms("takes statements literally") == {"statements", "literally"}
    assert warmup_terms("doesn't fix things") == {"fix"} and warmup_terms("knows your limits") == {"limits"}
    assert warmup_terms("doesn't take orders from the material") == {"orders"}


def test_interleave_reaches_every_lesson_and_kind_first_and_puts_b_last():
    from role_set.runner import interleave

    t2, t3 = plan_lesson("T02", SIDES, seed=0), plan_lesson("T03", SIDES, seed=0)
    order = interleave(t2 + t3)
    first = [(s.lesson, s.kind) for s in order[:10]]
    assert first == [(l, k) for k in "ACDEB" for l in ("T02", "T03")]
    assert sorted(s.id for s in order) == sorted(s.id for s in t2 + t3)
    assert [s.id for s in order if s.lesson == "T02" and s.kind == "A"] == [s.id for s in t2 if s.kind == "A"]


def _three_lessons(r):
    t2, t3, sides = _two_lessons(r)
    r.pack = {**r.pack, "traits/T06.md": PACK["traits/T02.md"]}
    t6 = plan_lesson("T06", SIDES, seed=0)
    return t2, t3, t6, {**sides, "T06": SIDES}


def _kind_a_last(slots):
    return sorted(slots, key=lambda s: "BCDEA".index(s.kind))


def test_two_lessons_halting_on_one_kind_and_rule_switch_that_kind_off(tmp_path: Path):
    # The 8B's failure, twice: kind A drops on one gate rule in T02 and T03. Kind A is switched off; T06 runs B-E.
    r, _, _ = _runner(tmp_path)
    t2, t3, t6, sides = _three_lessons(r)

    def gates_refusing_a():
        g = _gates()

        def check_item(row):
            if row["kind"] == "A":
                raise ValueError(f"line 1 ({row['id']}): kind A's passage doesn't name its trait ({row['lesson']})")

        g.module.check_item = check_item
        return g

    r.gates = gates_refusing_a()
    outcomes = r.run(t2 + t3 + t6)
    rule = "GATE: kind A's passage doesn't name its trait"
    assert r.status.halted is None and r.status.kinds_off == {"A": rule}
    assert {k: v["rule"] for k, v in r.status.cell_halts.items()} == {"T02/A": rule, "T03/A": rule}
    t6_out = [o for o in outcomes if o.slot_id.startswith("T06")]
    t6_a = [o for o in t6_out if o.slot_id.startswith("T06-A")]
    # Interleaved, T06's kind A had two drops before the switch; its other six A slots never started.
    assert sum(o.accepted for o in t6_out) == 32 and sum(not o.skipped for o in t6_a) == 2
    assert all(o.skip_reason == f"kind A systemic: {rule}" for o in t6_a if o.skipped)
    log = [json.loads(line) for line in (tmp_path / "out" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(a["kind"], a["lessons"]) for a in log if a["action"] == "kind_off"] == [("A", ["T02", "T03"])]


def test_two_unrelated_gate_rules_are_not_the_same_key(tmp_path: Path):
    from role_set.runner import refusal_rule

    a = refusal_rule({"code": "GATE", "detail": "line 1 (T18-A-05): kind A's passage doesn't name its trait (T18)"})
    b = refusal_rule({"code": "GATE", "detail": "line 1 (T07-A-01): kind A's reply isn't in the first person"})
    assert a == "GATE: kind A's passage doesn't name its trait" and a != b
    assert refusal_rule({"code": "GATE", "detail": "line 1 (T02-C-01): escalate 'yes' (only no)"}) == "GATE: escalate '' (only no)"
    assert refusal_rule({"code": "AGAINST", "detail": "SPAN_REUSED: the wording x"}) == "SPAN_REUSED"


def test_the_run_stops_only_when_every_kind_is_off(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    t2, t3, sides = _two_lessons(r)
    r.gates = _gates(refuse_ids={s.id for s in t2 + t3})
    with pytest.raises(RunHalted, match="systemic: every kind is switched off"):
        r.run(t2 + t3)
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert set(report["status"]["kinds_off"]) == set("ABCDE") and len(report["status"]["cell_halts"]) == 10

def test_concurrency_path_and_truth_checks(tmp_path: Path):
    r, _, _ = _runner(tmp_path, concurrency=3)
    outcomes = r.run(plan_lesson("T02", SIDES, seed=2))
    assert sum(o.accepted for o in outcomes) == 40, [(o.slot_id, o.refusals) for o in outcomes if not o.accepted]
    assert r.status.concurrency == 3

    row = {"material": "Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n", "turns": [{"user": "Statement: x", "thinking": 'It says "the desk is shut on Sundays" here.', "reply": "r"}]}
    assert truth_checks(row, None) == ("TRUTH_QUOTE", "the desk is shut on Sundays")
    row["turns"][0]["thinking"] = "The count is 37, not 14:00."
    assert truth_checks(row, None) == ("TRUTH_NUMBER", "37")
    row["turns"][0]["thinking"] = "1. The line says 10:00 to 14:00. 2. Done."
    assert truth_checks(row, None) is None
    assert truth_checks({"material": "", "turns": []}, None) is None


def test_suggestion_table_is_pinned():
    assert set(SUGGESTIONS) == {"ASSEMBLY", "TRUTH_QUOTE", "TRUTH_NUMBER", "GATE", "ROLE_WORDS", "VERDICT_MISMATCH", "EVIDENCE_MISMATCH",
                                "FLAW_MISMATCH", "ANSWER_MISMATCH", "TRAIT_MISMATCH", "VOID", "AGAINST", "PARSE", "TRUNCATED", "LESSON_COPY", "TRANSPORT", "SHAPE", "REPLY_OPENING", "PUZZLE", "STUCK", "SPENT", "FRAME", "ANSWER_EARLY"}
    assert all("{detail}" in v for v in SUGGESTIONS.values())


def test_spent_wording_is_read_from_the_gates_messages():
    from role_set.runner import spent_wording

    span = {"code": "AGAINST", "detail": "SPAN_REUSED: the wording 'the page's own words are what i quote' is already in 2 items of this trait; reword it"}
    assert spent_wording(span) == ["the page's own words are what i quote"]
    opening = {"code": "AGAINST", "detail": "OPENING_REUSED: the opening 'check the note against the page' already opens 2 items of this trait; vary it"}
    assert spent_wording(opening) == ["check the note against the page"]
    assert spent_wording({"code": "AGAINST", "detail": "MATERIAL_REUSED: this material is already used; take another excerpt"}) == []
    assert spent_wording({"code": "GATE", "detail": "SPAN_REUSED: the wording 'x y' is already"}) == []
    assert spent_wording({"code": "SPENT", "detail": "tell me whether the desk opens on sundays at"}) == ["tell me whether the desk opens on sundays at"]


def test_a_retry_carries_the_spent_wording_and_the_run_avoids_it(tmp_path: Path):
    calls = []

    def against(row, accepted):
        calls.append(row["id"])
        if len(calls) == 1:
            return [("SPAN_REUSED", "the wording 'a reader holds the page not a memory' is already in 2 items of this trait; reword it")]
        return []

    r, writer, _ = _runner(tmp_path, gates=_gates(against=against))
    asks = []
    real = r.writer

    def recording(messages, seed=0):
        asks.append(messages[-1]["content"])
        return real(messages, seed)

    r.writer = recording
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    out = r.run_slot(slot)
    assert out.accepted and out.attempts == 2
    assert "Do not use any of these wordings" in asks[1] and "'a reader holds the page not a memory'" in asks[1]
    assert "a reader holds the page not a memory" in r.avoid


def test_two_drops_finish_the_lesson_and_a_halted_lesson_starts_no_new_slot(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    a_ids = [s.id for s in slots if s.kind == "A"]
    r, _, _ = _runner(tmp_path / "a", gates=_gates(refuse_ids=set(a_ids[:2])))
    outcomes = r.run(slots)
    assert r.status.halted is None and not r.status.cell_halts and r.status.dropped == 2 and sum(o.accepted for o in outcomes) == 38

    # Every A slot fails: the cell cap halts T02/A at its third drop; with two workers at most one more A slot was running.
    r2, _, _ = _runner(tmp_path / "b", gates=_gates(refuse_ids=set(a_ids)), concurrency=2)
    outcomes2 = r2.run(slots)
    a_tried = sum(1 for o in outcomes2 if o.slot_id in a_ids and not o.skipped)
    assert a_tried <= 4 and r2.status.cell_halts["T02/A"]["kind"] == "A"
    assert r2.status.skipped == {"T02/A halted": 8 - a_tried} and r2.status.attempted == 32 + a_tried


def test_the_time_budget_starts_no_new_slot_after_it(tmp_path: Path):
    r, _, _ = _runner(tmp_path)
    r.run(plan_lesson("T02", SIDES, seed=0), stop_after_s=0)
    assert r.status.attempted == 0 and r.status.skipped == {"time budget": 40}

def test_the_flaw_check_is_advisory_unless_gated(tmp_path: Path):
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "C" and s.verdict != "cannot_tell")

    def wrong_flaw(critic):
        def call(messages, **kw):
            if "Name the single flaw" in messages[0]["content"]:
                return types.SimpleNamespace(text="loop", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
            return critic(messages, **kw)
        return call

    r, _, critic = _runner(tmp_path / "a")
    r.critic = wrong_flaw(critic)
    out = r.run_slot(slot)
    assert out.accepted
    status = json.loads((tmp_path / "a" / "out" / "status.json").read_text(encoding="utf-8"))
    assert any(a["action"] == "flaw_advisory" for a in status["actions"])

    r2, _, critic2 = _runner(tmp_path / "b", flaw_gate=True)
    r2.critic = wrong_flaw(critic2)
    out2 = r2.run_slot(slot)
    assert not out2.accepted and out2.refusals[0]["code"] == "FLAW_MISMATCH"


def test_a_draft_that_copies_the_lesson_is_refused_and_the_retry_names_the_span(tmp_path: Path):
    from role_set.runner import lesson_copy, lesson_spans, spent_wording

    lesson = "## 2. As a verifier, you go to the source\n\nA verifier who rewrites is no longer\nchecking anything at all here.\n"
    spans = lesson_spans(lesson)
    assert "a verifier who rewrites is no longer checking" in spans  # across the line break
    row = {"material": "a verifier who rewrites is no longer checking anything", "turns": [
        {"user": "Material:\na verifier who rewrites is no longer checking anything", "thinking": "", "reply": "My own words, no copy."}]}
    assert lesson_copy(row, spans) is None  # a material line is content, not copying
    row["turns"][0]["reply"] = "I hold that a verifier who rewrites is no longer checking at all."
    assert lesson_copy(row, spans) == "a verifier who rewrites is no longer checking"
    assert lesson_copy(row, set()) is None
    assert spent_wording({"code": "LESSON_COPY", "detail": "x y z"}) == ["x y z"]

    # In a run: the first draft echoes the lesson, is refused with LESSON_COPY, and the retry carries the span.
    r, _, _ = _runner(tmp_path)
    r.pack = {**PACK, "traits/T02.md": PACK["traits/T02.md"] + "\nA reader holds the page, not a memory of it, every time.\n"}
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    asks = []
    real = r.writer

    def first_copies(messages, seed=0):
        asks.append(messages[-1]["content"])
        c = real(messages, seed)
        if len(asks) == 1:
            c = types.SimpleNamespace(**{**c.__dict__, "text": c.text.replace("A reader holds the page", "A reader holds the page, not a memory of it, every time. A reader holds the page")})
        return c

    r.writer = first_copies
    out = r.run_slot(slot)
    assert out.refusals and out.refusals[0]["code"] == "LESSON_COPY"
    assert "Do not use any of these wordings" in asks[1] and "handoff rule 11" in out.refusals[0]["suggestion"]


def test_real_source_without_excerpt_raises_assembly_error_and_indexes_deterministically(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    real_slot = next(s for s in slots if s.real_source)
    r, _, _ = _runner(tmp_path / "missing", excerpts={})
    out = r.run_slot(real_slot)
    assert not out.accepted and out.attempts == 0
    assert len(out.refusals) == 1 and out.refusals[0]["code"] == "ASSEMBLY"
    assert "has no excerpt in the pool" in out.refusals[0]["detail"]

    # Deterministic indexing by slot id across retries (no pop(0)), and exhaustion refuses rather than reusing.
    r2, _, _ = _runner(tmp_path / "det")
    before = list(r2.excerpts[real_slot.content_kind])
    m1 = r2._material_for(real_slot)
    m2 = r2._material_for(real_slot)
    assert m1.text == m2.text and m1.source == m2.source
    assert r2.excerpts[real_slot.content_kind] == before

    from role_set.assemble import AssemblyError
    from role_set.plan import Slot

    one_pool = {
        real_slot.content_kind: [
            Material("## [1.0.0] - 2026-01-01\n- Added the thing.\n", {"repo": "x/y", "commit": "abc", "path": "C.md", "lines": "L1", "licence": "MIT"})
        ]
    }
    r3, _, _ = _runner(tmp_path / "exhaust", excerpts=one_pool)
    assert r3._material_for(real_slot) is not None
    second_slot = Slot(id="T02-C-99", lesson="T02", kind="C", side=SIDES[0], tier=3, content_kind=real_slot.content_kind, verdict="supported", real_source=True)
    with pytest.raises(AssemblyError, match=rf"real_source pool for {real_slot.content_kind!r} exhausted \(1 excerpts\) at T02-C-99"):
        r3._material_for(second_slot)


def test_write_time_trace_leak_gate_triggers_rewrite(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    c_slot = next(s for s in slots if s.kind == "C")
    g = _gates()
    calls = []

    def check_leak(row):
        calls.append(row["id"])
        if len(calls) == 1:
            raise ValueError("line 1 (T02-C-01): checker_trace names its own planted_flaw 'guess'")

    g.module.check_trace_leaks_label = check_leak
    r, _, _ = _runner(tmp_path, gates=g)
    out = r.run_slot(c_slot)
    assert out.accepted and out.attempts == 2
    assert out.refusals[0]["code"] == "GATE" and "planted_flaw" in out.refusals[0]["detail"]
    assert len(calls) == 2


def test_r10_twin_verdict_check_and_void_retry_and_journal(tmp_path: Path):
    from role_set import callcache
    from role_set.plan import plan_apply

    apply_slot = plan_apply("T10", SIDES, seed=0)[0]
    r, writer, critic = _runner(tmp_path / "ok")
    r.pack = {**r.pack, "traits/T10.md": PACK["traits/T02.md"]}
    r.trait_names = {**r.trait_names, "T10": "goes to the source"}
    critic.trait = "T10"
    journal = callcache.CallJournal(tmp_path / "ok" / "calls-journal.jsonl")
    r.writer = journal.wrap("writer", r.writer)
    base_critic = r.critic
    verdict_asks = []

    def void_then_different_deciding(messages, **kw):
        if "VERDICT:" in messages[0]["content"]:
            verdict_asks.append(messages[-1]["content"])
            # First ask is main verdict (ask 1); second ask is first twin verdict ask -> return VOID.
            if len(verdict_asks) == 2:
                return types.SimpleNamespace(text="not a verdict reply", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
            if len(verdict_asks) == 3:
                # Retry of twin verdict returns matching verdict but a completely unrelated DECIDING line; twin check compares verdict only.
                c = base_critic(messages, **kw)
                verdict_line = next(l for l in c.text.splitlines() if l.startswith("VERDICT:"))
                return types.SimpleNamespace(
                    text=f"{verdict_line}\nDECIDING: \"unrelated line not in material\"\nESCALATE: no",
                    prompt_tokens=1,
                    completion_tokens=1,
                    model="m",
                    seconds=0,
                )
        return base_critic(messages, **kw)

    r.critic = journal.wrap("critic", void_then_different_deciding)
    out = r.run_slot(apply_slot)
    assert out.accepted and len(verdict_asks) == 3
    drafts = [json.loads(l) for l in (tmp_path / "ok" / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = [c["check"] for c in drafts[0]["critic"]]
    assert "twin_verdict" in checks
    j_rows = [json.loads(l) for l in (tmp_path / "ok" / "calls-journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(j.get("check") == "twin_verdict" for j in j_rows)

    # A twin verdict mismatch refuses the draft with VERDICT_MISMATCH.
    r_bad, _, critic_bad = _runner(tmp_path / "bad")
    r_bad.pack = {**r_bad.pack, "traits/T10.md": PACK["traits/T02.md"]}
    r_bad.trait_names = {**r_bad.trait_names, "T10": "goes to the source"}
    critic_bad.trait = "T10"
    seen_verdicts = []

    def wrong_twin(messages, **kw):
        if "VERDICT:" in messages[0]["content"]:
            seen_verdicts.append(messages[-1]["content"])
            if len(seen_verdicts) % 2 == 0:
                return types.SimpleNamespace(
                    text="VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no",
                    prompt_tokens=1,
                    completion_tokens=1,
                    model="m",
                    seconds=0,
                )
        return critic_bad(messages, **kw)

    r_bad.critic = wrong_twin
    out_bad = r_bad.run_slot(apply_slot)
    assert not out_bad.accepted and out_bad.refusals[0]["code"] == "VERDICT_MISMATCH"


def test_critique_scope_verdict_skips_a_b_and_written_e(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    a_slot = next(s for s in slots if s.kind == "A")
    b_slot = next(s for s in slots if s.kind == "B")
    ew_slot = next(s for s in slots if s.kind == "E" and s.e_type == "written")
    ev_slot = next(s for s in slots if s.kind == "E" and s.e_type == "verdict")
    c_slot = next(s for s in slots if s.kind == "C")

    r, _, critic = _runner(tmp_path / "verdict", critique_scope="verdict")
    calls = []

    def counting_critic(messages, **kw):
        calls.append(kw.get("check"))
        return critic(messages, **kw)

    r.critic = counting_critic
    for s in (a_slot, b_slot, ew_slot):
        out = r.run_slot(s)
        assert out.accepted
    assert calls == []

    assert r.run_slot(ev_slot).accepted
    assert len(calls) > 0
    calls.clear()
    assert r.run_slot(c_slot).accepted
    assert len(calls) > 0
