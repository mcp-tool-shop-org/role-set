"""The maintainer's rule for paid runs: a failed run returns data as readable as a passing one."""

import hashlib
import json
import subprocess
import types
from pathlib import Path

import pytest

from role_set.plan import plan_lesson
from role_set.runner import RunHalted
from role_set.transport import TransportError
from test_runner import SIDES, _gates, _runner


def test_every_writer_call_is_kept_with_tokens_latency_and_the_stuck_measures(tmp_path: Path):
    seen = []

    def stuck(thinking, material, tier, tokens):
        seen.append((tier, tokens))
        return {"stuck": False, "hedges": 0}

    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "C")
    r, _, _ = _runner(tmp_path, gates=_gates(refuse_ids={slot.id}), stuck=stuck)
    r.run_slot(slot)
    calls = [json.loads(line) for line in (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["attempt"] for c in calls] == [1, 2, 3] and {c["outcome"] for c in calls} == {"GATE"}
    assert all(c["draft"] and c["prompt_tokens"] == 100 and c["completion_tokens"] == 50 and c["stuck"] == {"stuck": False, "hedges": 0} for c in calls)
    assert seen and seen[0] == (slot.tier, 50)


def test_a_writer_that_cannot_be_reached_ends_the_slot_not_the_run(tmp_path: Path):
    r, _, _ = _runner(tmp_path)
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")

    def down(messages, seed=0):
        raise TransportError("connection refused")

    r.writer = down
    out = r.run_slot(slot)
    assert not out.accepted and out.refusals[0]["code"] == "TRANSPORT" and r.status.halted is None
    call = json.loads((tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert call["outcome"] == "TRANSPORT" and call["draft"] == ""


def test_more_than_a_fifth_of_twenty_calls_unparsable_is_systemic(tmp_path: Path):
    r, _, _ = _runner(tmp_path)
    slots = [s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A"]
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text="no json here", prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1)
    with pytest.raises(RunHalted, match="systemic: .* of the last 20 writer calls"):
        for s in slots:
            r.run_slot(s)
    assert r.status.halted.startswith("systemic")
    calls = (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 20  # the window fills before it can fire


def test_a_slot_accepted_on_the_call_that_fires_the_halt_is_still_reported(tmp_path: Path):
    """Pilot 2's halt fired on calls whose rows were already accepted: four outcomes were lost from batch-report.json."""

    r, writer, _ = _runner(tmp_path)
    slots = [s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A"][:3]
    calls = []

    def halting(messages, seed=0):
        calls.append(1)
        if len(calls) == 2:  # the warm-up, then the first slot's draft
            r.status.halted = "systemic: 5 of the last 20 writer calls failed to parse or reach the writer"
        return writer(messages, seed=seed)

    r.writer = halting
    with pytest.raises(RunHalted):
        r.run(slots)
    accepted = [json.loads(x)["id"] for x in (tmp_path / "out" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert accepted == [slots[0].id]
    assert [o["slot"] for o in report["outcomes"] if o["accepted"]] == accepted


def test_a_draft_cut_off_at_the_length_limit_is_truncated_and_never_systemic(tmp_path: Path):
    """Pilot 2 stopped at seven minutes on five kind B drafts cut off at max_tokens: a defect of that cell's drafts,
    not of the endpoint. They count toward the cell's drops, never toward the systemic rate."""

    r, _, _ = _runner(tmp_path)
    slots = [s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A"]
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text='{"passage": "unterminated', prompt_tokens=1, completion_tokens=3500,
                                                              model="m", seconds=0.1, finish_reason="length")
    outs = [r.run_slot(s) for s in slots]
    assert r.status.halted is None
    assert all(o.refusals and o.refusals[0]["code"] == "TRUNCATED" for o in outs)
    calls = [json.loads(x) for x in (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(calls) > 20 and {c["outcome"] for c in calls} == {"TRUNCATED"}


def test_the_frozen_rule_is_read_by_git_hash_checked_and_mapped(tmp_path: Path, monkeypatch):
    from role_set import stuckrule

    repo = tmp_path / "aspire"
    (repo / "docs/runs/stage1-train").mkdir(parents=True)
    src = (
        "import json\nimport re\nfrom pathlib import Path\n"
        "def judge(thinking, material, tier, target, tokens, total, rule):\n"
        "    return {'stuck': tokens > rule['p95_by_tier'][str(tier)], 'window': rule['burst']['window']}\n"
        "def hedge_profile(thinking, material='', tokens=0, count=None):\n"
        "    return {'hedges': thinking.lower().count('wait')}\n"
    )
    frozen = {"p95_thinking_tokens_by_tier": {"2": 3, "7+": 9}, "reread_rate_p95": 8.09, "burst_parameters": {"window_tokens": 200, "k": 3}}
    (repo / stuckrule.STUCK_PY).write_text(src, encoding="utf-8", newline="\n")
    (repo / stuckrule.RULE_JSON).write_text(json.dumps(frozen), encoding="utf-8", newline="\n")
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

    # Not the frozen hashes: refused, never run.
    with pytest.raises(stuckrule.StuckRuleError, match="not the frozen"):
        stuckrule.load(repo, sha)
    with pytest.raises(stuckrule.StuckRuleError, match="cannot read"):
        stuckrule.load(repo, "0" * 40)

    monkeypatch.setattr(stuckrule, "STUCK_PY_SHA256", hashlib.sha256(src.encode()).hexdigest())
    monkeypatch.setattr(stuckrule, "RULE_SHA256", hashlib.sha256(json.dumps(frozen).encode()).hexdigest())
    measure = stuckrule.load(repo, sha)
    got = measure("wait wait one two three four", "", 2, 40)
    assert got["stuck"] is True and got["window"] == 200 and got["hedge_profile"] == {"hedges": 2} and got["thinking_words"] == 6
    assert stuckrule.rule_for_judge(frozen) == {"p95_by_tier": {"2": 3, "7+": 9}, "reread_rate_p95": 8.09, "burst": {"window": 200, "k": 3}}
    with pytest.raises(stuckrule.StuckRuleError, match="no burst_parameters"):
        stuckrule.rule_for_judge({"p95_thinking_tokens_by_tier": {}, "reread_rate_p95": 1})


def test_the_readout_reads_a_failing_run_as_well_as_a_passing_one(tmp_path: Path, capsys):
    from role_set.cli import main
    from role_set.readout import readout

    slots = plan_lesson("T02", SIDES, seed=0)
    a_ids = [s.id for s in slots if s.kind == "A"]
    r, _, _ = _runner(tmp_path / "fail", gates=_gates(refuse_ids=set(a_ids[:3])), stuck=lambda *a: {"stuck": True, "bursts": 1, "hedges": 3})
    r.run(slots)
    data, md = readout(tmp_path / "fail" / "out")
    run = data["run"]
    assert run["cell_halts"]["T02/A"]["kind"] == "A" and run["per_kind"]["A"]["dropped"] == 3 and run["warmups"] == {"passed": 1}
    assert run["refusals"] == {"A / GATE: refused": 9} and run["retry_recovery"] == {"refused_slots": 3, "recovered": 0, "rate": 0.0}
    assert run["calls"] == 9 + run["accepted"] and run["stuck"]["stuck"] == run["calls"] and run["seconds_per_call"]["p50"] is not None
    assert "| T02 |" in md and "Retry recovery: 0 of 3" in md
    assert run["per_lesson"]["T02"]["skipped_by_reason"] == {"T02/A halted": 5} and "T02/A halted: 5" in md and "T02/A (GATE: refused)" in md

    # An old-format run (outcomes lost on a halt, no drafts.jsonl) is rebuilt from its action log, and compared.
    old = tmp_path / "old"
    old.mkdir()
    (old / "batch-report.json").write_text(json.dumps({"status": {"accepted": 1, "attempted": 2, "dropped": 1, "tokens": {"prompt": 30, "completion": 6},
        "refused_by_code": {"GATE": 2}, "actions": [{"action": "accepted", "slot": "T18-A-02"}, {"action": "refused", "slot": "T18-A-01", "code": "GATE"},
        {"action": "refused", "slot": "T18-A-01", "code": "GATE"}, {"action": "dropped", "slot": "T18-A-01"}]}, "outcomes": []}), encoding="utf-8")
    both, md2 = readout(tmp_path / "fail" / "out", [old])
    assert both["compare"][0]["per_lesson"]["T18"] == {"accepted": 1, "dropped": 1, "skipped": 0, "accept_rate": 0.5, "skipped_by_reason": {}}
    assert both["compare"][0]["refusals"] == {"GATE": 2} and both["compare"][0]["tokens_per_call"] == {"prompt": 10, "completion": 2}
    assert "| old |" in md2

    assert main(["readout", "--run", str(tmp_path / "fail" / "out"), "--out", str(tmp_path / "ro")]) == 0
    assert (tmp_path / "ro" / "readout.json").exists() and (tmp_path / "ro" / "readout.md").exists()
    assert main(["readout", "--run", str(tmp_path)]) == 2
    assert "no batch-report.json" in capsys.readouterr().err


def test_a_spent_span_holding_the_trait_words_keeps_them_required():
    from role_set.runner import trait_words_note

    words = ["fix things", "fixing", "don't fix", "doesn't fix"]
    note = trait_words_note(words, ["a verifier you don't fix things you stick"])
    assert "which are required: 'fix things', 'fixing', 'don't fix', 'doesn't fix'." in note
    assert "In 'a verifier you don't fix things you stick' only the wording around the trait words is spent" in note
    assert "only the wording around" not in trait_words_note(words, ["an unrelated spent span of eight words"])
    assert trait_words_note([], ["a verifier you don't fix things you stick"]) == ""


def test_the_retry_after_a_spent_trait_span_names_the_trait_words(tmp_path: Path):
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    calls = []

    def against(row, accepted):
        calls.append(1)
        return [("SPAN_REUSED", "the wording 'a reader goes to the source and holds the page' is already in 2 items of this trait; reword it")] if len(calls) == 1 else []

    g = _gates(against=against)
    g.module.constants = types.SimpleNamespace(TRAIT_NAMES={"T02": ["to the source", "goes to the source"]})
    r, _, _ = _runner(tmp_path, gates=g)
    asks = []
    real = r.writer

    def recording(messages, seed=0):
        asks.append(messages[-1]["content"])
        return real(messages, seed)

    r.writer = recording
    assert r.run_slot(slot).accepted
    assert "which are required: 'to the source', 'goes to the source'." in asks[1]
    assert "only the wording around the trait words is spent" in asks[1]


def test_the_kind_a_brief_warns_off_the_stock_opening():
    from role_set.prompts import hard_rules

    a = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    assert any("don't open the passage with 'As a verifier, you' followed by the trait words" in r for r in hard_rules(a, ["to the source"]))


def test_the_readout_states_the_inputs_the_run_was_pinned_to(tmp_path: Path):
    from role_set.readout import readout

    r, _, _ = _runner(tmp_path)
    r.pins = {"writer_model": "w", "writer_revision": "96548f65", "pack_commit": "0801513", "pack_sha256": "7d8e", "exemplar_commit": "a7ec783", "stuck_rule_sha256": "716d"}
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    r.run([slot])
    data, md = readout(tmp_path / "out")
    inputs = data["run"]["inputs"]
    assert inputs["exemplar_commit"] == "a7ec783" and inputs["writer_revision"] == "96548f65" and "other traits only" in inputs["exemplars"]
    assert "Inputs: writer_model w;" in md and "exemplar_commit a7ec783" in md


def test_the_readout_counts_keyed_letters_of_closed_form_variants(tmp_path: Path):
    from role_set.readout import keyed_letters

    def row(lesson, letter):
        return {"lesson": lesson, "quiz": {"type": "written"}, "variant": {"user": "u", "key": {"answer": letter}}}

    rows = [row("T07", "B"), row("T07", "B"), row("T07", "D"), row("T08", "A"), {"lesson": "T07", "quiz": {"type": "verdict"}, "variant": {"key": {"verdict": "supported"}}}]
    assert keyed_letters(rows) == {"T07": {"B": 2, "D": 1}, "T08": {"A": 1}}


def test_the_readout_reports_how_often_the_keyed_option_is_the_longest():
    from role_set.readout import keyed_longest

    def row(lesson, letter, opts):
        user = "Which?\n\n" + "\n".join(f"{k}. {t}" for k, t in zip("ABC", opts)) + "\n\nANSWER: <letter>"
        return {"lesson": lesson, "quiz": {"type": "written"}, "variant": {"user": user, "key": {"answer": letter}}}

    rows = [row("T07", "B", ["short", "the much longer keyed option", "brief"]), row("T07", "A", ["equal one", "equal two", "equal six"]),
            row("T08", "C", ["a long distractor here", "x", "keyed"])]
    assert keyed_longest(rows) == {"T07": {"keyed_longest": 1, "items": 2}, "T08": {"keyed_longest": 0, "items": 1}}
