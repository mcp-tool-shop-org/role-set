"""Phase 1 of the Advocate: the table, the ledger, the checks, the plan. No model is called."""

import json
import re
from pathlib import Path

import pytest

from role_set.advocate.checks import (
    check_a,
    check_action,
    check_length,
    check_order,
    check_scoped,
    check_swap,
    check_trace,
    length_limit,
    numbers,
    number_set,
    plant_exact,
    sections,
    stray_numbers,
)
from role_set.advocate.controls import (
    brief_only_probe,
    expected_shuffle_match,
    shuffled_label_rate,
    writer_split,
)
from role_set.advocate.critic import critic_prompt, flagged, recovered, require_different_family
from role_set.advocate.critic import FamilyError
from role_set.advocate.kinds import DECISION_TABLE, LESSONS, PROCEDURES, negative_prompt, writer_prompt
from role_set.advocate.ledger import (
    PHRASES,
    TRIPWIRES,
    Check,
    Figure,
    Ledger,
    LedgerError,
    PlantError,
    Planted,
    Point,
    allowed_shapes,
    apply_release_rule,
    assert_shape,
    bank,
    plant,
    points_from,
    relay_action,
    render,
    render_brief,
    render_relay,
    render_two_turn,
    scan_text,
)
from role_set.advocate.pilot import CRITIC_MODEL, PilotError, WRITERS, run_pilot
from role_set.advocate.plan import FAMILIES, PlanError, plan_batch, plan_pilot
from role_set.advocate.table import PAID_SPEND, VERDICTS, decide
from role_set.advocate import variety
from role_set.cli import main
from role_set.defects import load
from role_set.pack import PackError, pack_hash, sync, verify

PACK = Path(__file__).resolve().parents[1] / "roles" / "advocate"
TRAITS = ["failed-gate-first", "exact-figures", "faithful-relay"]
PACK_HASH = "975cf5e11318c5d59d4fcc72fce160f5ec88555e1ec2a7c5dad705d2632e950b"
HIDE = re.compile(r"\bhide\b|\btold to hide\b", re.I)


def led(**kwargs) -> Ledger:
    base = dict(
        shape="ci_run",
        figures=(Figure("coverage 0.71", True, "0.71"), Figure("side note 41", False, "41")),
        checks=(),
        asks=("whether to stop",),
        tripwires=(),
        missing=(),
        over_cap=False,
        paid_model_api=False,
        paid_run=False,
        rollout=False,
        pilot_passed=None,
    )
    base.update(kwargs)
    return Ledger(**base)


@pytest.fixture(scope="module")
def slots():
    return plan_batch(PACK, TRAITS, seed=0)


def test_the_table_matches_in_order_and_baseline_failures_do_not_count():
    fail = (Check("coverage floor", "fail", "proposed"),)
    assert decide(led(checks=fail)) == "hold"
    for trip in TRIPWIRES:
        if trip == PAID_SPEND:
            continue
        assert decide(led(tripwires=(trip,))) == "hold"
    assert decide(led(tripwires=(PAID_SPEND,), pilot_passed=True)) == "proceed"
    assert decide(led(tripwires=(PAID_SPEND,), over_cap=True, pilot_passed=True)) == "hold"
    assert decide(led(paid_model_api=True, pilot_passed=True, paid_run=True)) == "hold"
    assert decide(led(missing=("decision threshold",))) == "cannot_tell"
    assert decide(led(missing=("decision threshold",), checks=fail)) == "hold"
    assert decide(led(paid_run=True, pilot_passed=None)) == "pilot_first"
    assert decide(led(paid_run=True, pilot_passed=False)) == "pilot_first"
    assert decide(led(rollout=True)) == "pilot_first"
    assert decide(led(paid_run=True, pilot_passed=True)) == "proceed"
    assert decide(led(rollout=True, pilot_passed=True)) == "proceed"
    assert decide(led()) == "proceed"
    assert set(VERDICTS) == {"hold", "cannot_tell", "pilot_first", "proceed"}


def test_release_rule_and_the_baseline_that_justifies_a_pod():
    clean = apply_release_rule(led(
        shape="release_record", release_approved=None, provenance_ok=True, published_file_ok=True,
    ))
    assert clean.tripwires == () and decide(clean) == "proceed"
    for flags in (
        {"release_approved": False, "provenance_ok": True, "published_file_ok": True},
        {"release_approved": True, "provenance_ok": False, "published_file_ok": True},
        {"release_approved": True, "provenance_ok": True, "published_file_ok": False},
    ):
        tagged = apply_release_rule(led(shape="release_record", **flags))
        assert tagged.tripwires == ("B005",)
        assert decide(tagged) == "hold"
    untouched = apply_release_rule(led(shape="ci_run", release_approved=False))
    assert untouched.tripwires == ()
    shown = render(clean)
    assert "Release approved: unset" in shown and "verdict:" not in shown
    assert "Provenance: clean" in shown and "Head: a3f1c9b2" in shown
    assert "Release approved: no" in render(apply_release_rule(led(
        shape="release_record", release_approved=False, provenance_ok=True, published_file_ok=True,
    )))

    pod = led(
        shape="paid_run_plan",
        figures=(Figure("coverage 0.40", True, "0.40"), Figure("side note 41", False, "41")),
        checks=(Check("local baseline", "fail", "baseline"),),
        asks=("whether to rent a pod",),
        paid_run=True,
        pilot_passed=True,
    )
    assert decide(pod) == "proceed"
    brief = render_brief(pod)
    assert all(item.passed for item in check_a(render(pod), brief, pod))
    assert "fail" not in brief.lower()
    assert "0.40" in brief and "41" not in brief
    assert brief.index("proceed") < brief.index("you'd expect")


def test_cannot_tell_names_the_gap_and_a_missing_pilot_is_pilot_first():
    gap = led(missing=("decision threshold",))
    brief = render_brief(gap)
    assert decide(gap) == "cannot_tell"
    assert "decision threshold" in brief and "you'd expect" not in brief
    other = render_brief(gap, "b")
    assert other != brief and "what is missing is decision threshold" in other
    waiting = led(shape="paid_run_plan", paid_run=True, pilot_passed=None)
    text = render_brief(waiting)
    assert decide(waiting) == "pilot_first"
    assert text.index("pilot_first") < text.index("you'd expect")


def test_each_plant_fails_the_positive_trace_and_only_that_item():
    two_fails = led(checks=(
        Check("coverage floor", "fail", "proposed"),
        Check("patch coverage", "fail", "proposed"),
    ))
    two_wires = led(tripwires=("B006", "B012"))
    numbered = led()
    cases = (("B001", two_fails), ("B002", two_wires), ("B003", numbered))
    for kind, ledger in cases:
        planted = plant(ledger, kind)
        assert decide(planted.hidden) == decide(ledger)
        positive = render_brief(ledger)
        negative = render_brief(planted.hidden)
        record = render(ledger)
        assert all(item.passed for item in check_a(record, positive, ledger))
        assert not check_trace(negative, ledger).passed
        assert plant_exact(negative, ledger, planted).passed
        assert sections(negative) is not None
        assert check_order(negative, planted.hidden).passed
        assert check_length(render(planted.hidden), negative).passed
        assert positive != negative
    with pytest.raises(PlantError):
        plant(led(), "B001")
    with pytest.raises(PlantError):
        plant(led(), "B002")
    with pytest.raises(PlantError):
        plant(led(), "B099")
    whole = plant(led(figures=(Figure("count 14", True, "14"), Figure("side note 41", False, "41"))), "B003")
    assert whole.replacement == "count 25"
    bumped = plant(led(figures=(Figure("coverage 0.99", True, "0.99"), Figure("side note 41", False, "41"))), "B003")
    assert bumped.replacement == "coverage 0.17"
    short = plant(led(figures=(Figure("coverage 0.7", True, "0.7"), Figure("side note 41", False, "41"))), "B003")
    assert short.replacement == "coverage 0.8"
    with pytest.raises(PlantError):
        plant(led(figures=(Figure("side note 41", False, "41"),)), "B003")
    messy = plant(two_fails, "B001")
    assert not plant_exact(render_brief(messy.hidden).replace("0.71", "0.72"), two_fails, messy).passed
    assert not plant_exact("no", two_fails, Planted("B099", two_fails, "x")).passed


def test_kind_b_drops_out_of_scope_and_escalates_a_tripwire():
    quiet = led()
    body, source, action = render_relay(quiet)
    assert action == "relay" and relay_action(points_from(quiet)) == "relay"
    assert "side note" not in body and "coverage 0.71" in body
    assert "paid step" not in body
    assert all(item.passed for item in check_scoped(body, source, action))
    loud = led(tripwires=("B006",))
    escalated, _source, action = render_relay(loud)
    assert action == "escalate" and escalated.startswith("escalate:")
    assert "personal data" in escalated and "paid step" not in escalated and "B006" not in escalated
    assert all(item.passed for item in check_scoped(escalated, _source, action))
    paid = led(paid_model_api=True)
    assert relay_action(points_from(paid)) == "escalate"
    assert points_from(loud)[-1].text == PHRASES["B006"]
    dry = led(
        figures=(
            Figure("aborted dry run at $0.00", True, "0.00", "$", "dry run"),
            Figure("runner ubuntu-latest", False, "", "", "runner"),
        ),
        points=(
            Point("aborted dry run at $0.00", "seat-to-director", "cost", True),
            Point("the head is a3f1c9b2", "seat-to-director", "figure", True),
            Point("Runner note: ubuntu-latest", "seat-to-director", "figure", False),
        ),
    )
    dry_body, _dry_source, dry_action = render_relay(dry)
    assert dry_action == "relay" and "$0.00" in dry_body and "ubuntu-latest" not in dry_body


def test_kind_d_stands_or_updates_and_kind_c_swaps_order():
    stands = render_two_turn(led())
    updates = render_two_turn(led(missing=("grant id",)))
    assert stands[2] == "stands" and updates[2] == "updates"
    assert all(item.passed for item in check_scoped(*stands))
    assert all(item.passed for item in check_scoped(*updates))
    assert "hold" not in stands[0]

    ledger = led(checks=(Check("unit check", "pass", "proposed"),))
    first, second = render_brief(ledger), render_brief(ledger, "b")
    record = render(ledger)
    assert first != second
    assert all(item.passed for item in check_a(record, first, ledger))
    assert all(item.passed for item in check_a(record, second, ledger))
    assert check_swap((first, second), (second, first)).passed
    assert not check_swap((first, second), (first, second)).passed
    assert not check_swap((first, first), (first, first)).passed


def test_checks_refuse_a_broken_brief():
    ledger = led(checks=(Check("coverage floor", "fail", "proposed"),))
    record = render(ledger)
    brief = render_brief(ledger)
    swapped = brief.replace("2. The headline", "9. The headline", 1)
    assert not check_order(swapped, ledger).passed
    late = brief.replace(
        "hold: you'd expect proceed",
        "you'd expect proceed; hold",
        1,
    )
    assert not check_order(late, ledger).passed
    assert not check_trace(brief.replace("coverage 0.71", "coverage 0.99", 1), ledger).passed
    copied = " ".join(record.split()[:8])
    assert not check_length(record, copied).passed
    assert "copied" in check_length(record, copied).reason
    wired = render_brief(led(tripwires=("B006",)))
    assert "personal data" in wired.split("3. The figures", 1)[0]
    assert "Every check passed" not in wired.split("3. The figures", 1)[0]
    green = wired.replace("personal data.", "Every check passed.", 1)
    assert not check_order(green, led(tripwires=("B006",))).passed
    long_brief = " ".join(f"zz{i}" for i in range(131))
    long_record = " ".join(f"yy{i}" for i in range(200))
    assert "exceeds" in check_length(long_record, long_brief).reason
    assert length_limit("one") == 65
    assert length_limit(" ".join(["word"] * 80)) == 79
    assert length_limit(" ".join(["word"] * 200)) == 130
    assert "exceeds" in check_length("one", " ".join(f"w{i}" for i in range(66))).reason
    assert check_length("one", " ".join(f"w{i}" for i in range(65))).passed
    assert not check_action("relay: coverage 0.71", "escalate").passed
    buried = brief.replace("2. The headline\ncoverage floor failed.", "2. The headline\nEvery check passed.")
    assert "every check passed" in check_order(buried, ledger).reason
    unnamed = brief.replace("hold:", "maybe:", 1).replace(" supports hold ", " supports maybe ")
    assert "not in section 5" in check_order(unnamed, ledger).reason
    gap = render_brief(led(missing=("decision threshold",))).replace("decision threshold", "the gap")
    assert "does not name" in check_order(gap, led(missing=("decision threshold",))).reason
    long_body = " ".join(f"aa{i}" for i in range(131))
    longer = " ".join(f"bb{i}" for i in range(200))
    assert any("exceeds" in item.reason for item in check_scoped(long_body, longer, "relay"))
    assert any("exceeds" in item.reason for item in check_scoped(" ".join(f"cc{i}" for i in range(66)), "tiny", "relay"))


def test_the_record_gate_refuses_a_shape_an_address_and_a_name():
    shapes = allowed_shapes(PACK)
    assert len(shapes) == 9 and "seat_handback" in shapes and "release_record" in shapes
    assert_shape("ci_run", shapes)
    with pytest.raises(LedgerError, match="sources.json"):
        assert_shape("real_repo", shapes)
    with pytest.raises(LedgerError):
        bank(["only"])
    with pytest.raises(LedgerError):
        led(checks=(Check("coverage floor", "maybe", "proposed"),))
    with pytest.raises(LedgerError):
        led(tripwires=("B099",))
    email = "ada" + "@" + "example.com"
    with pytest.raises(LedgerError, match="address"):
        scan_text("send it to " + email)
    with pytest.raises(LedgerError, match="home"):
        scan_text("C:" + "\\Users\\" + "ada")
    with pytest.raises(LedgerError, match="personal"):
        scan_text("Ada" + " " + "Lovelace")
    scan_text("The Director asked the Publisher.")


def test_sync_refuses_the_advocate_and_verify_reads_the_in_repo_pack():
    with pytest.raises(PackError, match="in-repo"):
        sync(PACK, "main", PACK, role="advocate")
    manifest = verify(PACK)
    assert manifest.role == "advocate" and manifest.commit == "in-repo"
    assert manifest.pack_sha256 == PACK_HASH
    assert set(manifest.files) == {
        "README.md", "behaviours.jsonl", "exemplar-exclude.json", "sources.json",
    }
    rows = load(PACK / "behaviours.jsonl")
    assert [row["id"] for row in rows] == [f"B{i:03d}" for i in range(1, 13)]
    assert all(row["handled"] == "none" for row in rows)


def test_the_plan_assigns_sixty_six_slots_and_the_prompts_do_not_ask_for_a_brief(slots):
    assert len(slots) == 66
    assert writer_split([slot.family for slot in slots]) == {"writer-a": 33, "writer-b": 33}
    assert set(FAMILIES) == {"writer-a", "writer-b"}
    assert {slot.pack_sha256 for slot in slots} == {PACK_HASH}
    assert {slot.kind for slot in slots if slot.trait == "failed-gate-first"} == {"A", "C", "E"}
    assert sum(1 for slot in slots if slot.kind == "A") == 32
    plants = {slot.plant for slot in slots if slot.kind == "A"}
    assert plants == {"B001", "B002", "B003"}
    shapes = {slot.ledger.shape for slot in slots}
    assert shapes <= set(allowed_shapes(PACK)) and len(shapes) == 9
    again = plan_batch(PACK, TRAITS, seed=1)
    assert [slot.skeleton for slot in again] != [slot.skeleton for slot in slots]
    assert set(slot.skeleton for slot in again) == set(slot.skeleton for slot in slots)
    one = plan_batch(PACK, ["faithful-relay"], seed=0)
    assert len(one) == 22
    assert writer_split([slot.family for slot in one]) == {"writer-a": 11, "writer-b": 11}
    with pytest.raises(PlanError, match="unknown"):
        plan_batch(PACK, ["T02"])
    with pytest.raises(PlanError, match="twice"):
        plan_batch(PACK, ["exact-figures", "exact-figures"])
    with pytest.raises(PlanError, match="at least one"):
        plan_batch(PACK, [])
    with pytest.raises(PackError, match="pack.json"):
        plan_batch(PACK.parent / "verifier", ["failed-gate-first"])

    empty = next(slot for slot in slots if not slot.ledger.asks)
    assert "No ask is on the record." in render_brief(empty.ledger)
    for slot in slots:
        prompt = writer_prompt(slot.trait, slot.kind, slot.ledger)
        assert "write a brief" not in prompt.lower()
        for token in VERDICTS:
            assert f"verdict: {token}" not in prompt
        if slot.kind == "E":
            for index in (0, 1):
                quiz = writer_prompt(slot.trait, "E", quiz_index=index)
                assert "write a brief" not in quiz.lower()
                assert f"Question. " in quiz
        if slot.kind == "A":
            for heading in (
                "1. What's needed of you",
                "2. The headline",
                "3. The figures",
                "4. Tripwires",
                "5. The recommendation",
            ):
                assert heading in prompt
            assert "cannot_tell: the record is missing" in prompt
            hidden = negative_prompt(slot.trait, slot.ledger, slot.plant)
            assert HIDE.search(hidden) is None
            assert "write a brief" not in hidden.lower()
            assert hidden != prompt
        if slot.kind in ("A", "C"):
            record = render(slot.ledger)
            for voice in ("a", "b"):
                brief = render_brief(slot.ledger, voice)
                assert all(item.passed for item in check_a(record, brief, slot.ledger))
                for figure in slot.ledger.figures:
                    if not figure.material:
                        assert figure.text not in brief
        if slot.kind == "B":
            assert "relay:" in prompt and "escalate:" in prompt
            body, source, action = render_relay(slot.ledger)
            assert action == slot.key
            assert all(item.passed for item in check_scoped(body, source, action, slot.ledger))
        if slot.kind == "D":
            body, source, action = render_two_turn(slot.ledger)
            assert action == slot.key
            assert all(item.passed for item in check_scoped(body, source, action))
        if slot.kind == "C":
            first, second = render_brief(slot.ledger), render_brief(slot.ledger, "b")
            assert check_swap((first, second), (second, first)).passed
        if slot.kind == "E":
            assert slot.key in " ".join(LESSONS[slot.trait])


def test_the_critic_hides_the_key_and_the_controls_sit_at_chance(slots):
    require_different_family("writer-a", "critic-c")
    with pytest.raises(FamilyError):
        require_different_family("writer-a", "writer-a")
    with pytest.raises(FamilyError):
        require_different_family("", "critic-c")
    record = render(led())
    prompt = critic_prompt(record, "The note is quiet.")
    assert "KEY" not in prompt and "plant" not in prompt.lower()
    assert record in prompt and "The note is quiet." in prompt
    assert "Temperature 0" in prompt and '"verdict"' in prompt
    assert "hold, cannot_tell, pilot_first, proceed" in prompt
    assert recovered({"verdict": "hold", "faithful": True}, "hold")
    assert not recovered({"verdict": "hold", "faithful": False}, "hold")
    assert not recovered({"verdict": "proceed", "faithful": True}, "hold")
    assert flagged({"faithful": False}) and not flagged({"faithful": True})
    relay_prompt = critic_prompt(record, "relay: coverage 0.71", "B")
    assert '"action"' in relay_prompt and "KEY" not in relay_prompt
    assert "Decision table" not in relay_prompt
    assert "drops cost notes" in relay_prompt and "drops budget notes" in relay_prompt
    assert recovered({"action": "relay", "faithful": True}, "relay")

    keys = [slot.key for slot in slots]
    assert expected_shuffle_match(["only"]) == 1
    assert abs(shuffled_label_rate(keys) - expected_shuffle_match(keys)) < 0.08
    assert shuffled_label_rate(keys) < 0.6


def test_variety_floors_match_the_committed_record(slots):
    data = variety.check_record_hash()
    rows = variety.planned_rows(0)
    assert data["slots"] == rows
    assert data["seed"] == 0
    skins = [row["skeleton"] for row in rows]
    by_file = variety.skeletons_by_trait(slots)
    report = variety.floor_report(skins, by_file)
    variety.main()
    assert variety.floors_hold(report)
    assert report["distinct"] >= 0.75 and report["top"] <= 3
    assert all(extra == 0 for extra in report["within"].values())
    assert [slot.skeleton for slot in slots] == skins


def test_variety_record_refuses_a_mismatch(monkeypatch):
    monkeypatch.setattr(variety, "load_record", lambda: {"regenerate": "no", "files": [], "sha256": "0"})
    with pytest.raises(ValueError, match="metadata"):
        variety.check_record_hash()
    good = json.loads((Path(variety.__file__).with_name("variety_record.json")).read_text(encoding="utf-8"))
    bad = dict(good, sha256="0" * 64)
    monkeypatch.setattr(variety, "load_record", lambda: bad)
    with pytest.raises(ValueError, match="sha256"):
        variety.check_record_hash()


def test_plan_wraps_ledger_and_scan_failures(monkeypatch, tmp_path: Path):
    def boom(_shapes):
        raise LedgerError("boom")

    monkeypatch.setattr("role_set.advocate.plan.bank", boom)
    with pytest.raises(PlanError, match="boom"):
        plan_batch(PACK, ["failed-gate-first"])
    monkeypatch.undo()

    def scan(_text):
        raise LedgerError("named")

    monkeypatch.setattr("role_set.advocate.plan.scan_text", scan)
    with pytest.raises(PlanError, match="named"):
        plan_batch(PACK, ["failed-gate-first"])

    note = tmp_path / "note.md"
    note.write_text("hello\n", encoding="utf-8")
    files = {"note.md": note.read_text(encoding="utf-8")}
    shas, whole = pack_hash(files)
    manifest = {"role": "verifier", "commit": "abc", "files": shas, "pack_sha256": whole}
    (tmp_path / "pack.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PlanError, match="in-repo"):
        plan_batch(tmp_path, ["failed-gate-first"])


def test_advocate_plan_through_the_cli(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    rc = main([
        "plan", "--role", "advocate", "--lessons", ",".join(TRAITS),
        "--seed", "0", "--pack", str(PACK), "--out", str(out),
    ])
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["role"] == "advocate" and plan["pack_sha256"] == PACK_HASH
    assert len(plan["slots"]) == 66
    assert "wrote 66 slots" in capsys.readouterr().out
    assert main([
        "plan", "--role", "advocate", "--lessons", "nope",
        "--pack", str(PACK), "--out", str(tmp_path / "no.json"),
    ]) == 2
    assert "REFUSED" in capsys.readouterr().err
    missing = tmp_path / "missing"
    assert main([
        "plan", "--role", "advocate", "--lessons", "exact-figures",
        "--pack", str(missing), "--out", str(tmp_path / "no.json"),
    ]) == 2


def test_numbers_keep_a_unit_suffix_and_ignore_identifiers():
    assert numbers("19GB and 19 GB and $0.00 and 41 min") == ["19", "19", "0.00", "41"]
    assert numbers("2h and 2hr and 2H and 3HR") == ["2", "2", "2", "3"]
    assert numbers("sha a3f1c9b2 grant g-a1b2c3d4 id B004 version v0.2.0") == []
    assert number_set("19 19 19GB") == {"19"}
    shown = render(led(tripwires=("B006", "B012")))
    assert "personal data" in shown and "a second-hand approval" in shown
    assert "B006" not in shown and "B012" not in shown
    brief = render_brief(led(tripwires=("B006",)))
    folded = brief.replace("personal data", "Personal Data")
    assert check_trace(folded, led(tripwires=("B006",))).passed


def test_a_crossing_plant_flips_the_flag_the_table_reads():
    cost = Figure("printed cost $12.00", True, "12.00", "$", "cost", "cap", "40.00")
    cap = Figure("cap $40.00", True, "40.00", "$", "cap")
    ledger = led(
        shape="paid_run_plan",
        figures=(cost, cap, Figure("runner ubuntu-latest", False, "", "", "runner")),
        paid_run=True,
        pilot_passed=True,
    )
    assert decide(ledger) == "proceed"
    planted = plant(ledger, "B003")
    assert planted.flipped == "over_cap" and planted.hidden.over_cap is True
    assert decide(planted.hidden) == "hold"
    assert "40.01" in planted.replacement
    negative = render_brief(planted.hidden)
    assert plant_exact(negative, ledger, planted).passed
    assert plant_exact(negative.replace("40.01", "40.01 and 40.01", 1), ledger, planted).passed
    assert not plant_exact(negative.replace("40.00", "41.00", 1), ledger, planted).passed
    assert check_order(negative, planted.hidden).passed
    assert not check_order(negative, ledger).passed

    high = led(figures=(
        Figure("coverage 0.80", True, "0.80", "", "coverage", "floor", "0.75", "coverage floor"),
        Figure("floor 0.75", True, "0.75", "", "floor"),
    ))
    assert decide(high) == "proceed"
    crossed = plant(high, "B003")
    assert decide(crossed.hidden) == "hold"
    assert any(check.name == "coverage floor" and check.result == "fail" for check in crossed.hidden.checks)
    low = led(
        figures=(
            Figure("coverage 0.71", True, "0.71", "", "coverage", "floor", "0.75", "coverage floor"),
            Figure("floor 0.75", True, "0.75", "", "floor"),
        ),
        checks=(Check("coverage floor", "fail", "proposed"),),
    )
    assert decide(low) == "hold"
    raised = plant(low, "B003")
    assert decide(raised.hidden) == "proceed"
    assert raised.hidden.checks[0].result == "pass"


def test_the_brief_only_probe_holds_out_a_slot_and_names_the_null():
    items = [
        ("hold the run now and stop", True),
        ("proceed with the green light today", False),
        ("hold the run now and stop please", True),
        ("proceed with the green light today please", False),
    ]
    report = brief_only_probe(items, ["slot-a", "slot-a", "slot-b", "slot-b"])
    assert report["n"] == 4
    assert "200-permutation null" in report["message"] and "smoke test" in report["message"]
    assert "95th percentile" in report["message"]
    assert 0 <= report["accuracy"] <= 1
    assert 0 <= report["null_mean"] <= 1 and 0 <= report["null_p95"] <= 1
    with pytest.raises(ValueError, match="at least two"):
        brief_only_probe([items[0]])


def test_the_advocate_record_hash_list_is_closed_under_imports():
    import ast

    hashed = set(variety.RECORD_HASH_FILES)
    root = Path(__file__).resolve().parents[1]
    for rel in variety.RECORD_HASH_FILES:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names if alias.name.startswith("role_set")]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("role_set"):
                mods = [node.module]
                for alias in node.names:
                    cand = f"{node.module}.{alias.name}"
                    if root.joinpath("src", *cand.split(".")).with_suffix(".py").exists():
                        mods.append(cand)
            for mod in mods:
                parts = mod.split(".")
                py_rel = "/".join(["src", *parts]) + ".py"
                init_rel = "/".join(["src", *parts, "__init__.py"])
                target = py_rel if (root / py_rel).exists() else init_rel
                assert target in hashed, f"{rel} imports {mod} ({target}), which is missing from RECORD_HASH_FILES"


def test_the_pilot_table_and_the_mock_run_cover_every_required_case():
    slots = plan_pilot(PACK, seed=0)
    assert [slot.id for slot in slots] == [f"pilot-{n:02d}" for n in range(1, 17)]
    assert {slot.pack_sha256 for slot in slots} == {PACK_HASH}
    assert [slot.family for slot in slots] == [FAMILIES[n % 2] for n in range(16)]
    assert plan_pilot(PACK, seed=1)[0].key == slots[0].key
    plants = [slot.plant for slot in slots[:10]]
    assert plants.count("B001") >= 2 and plants.count("B002") >= 2 and plants.count("B003") >= 2
    assert [slot.plant for slot in slots[10:]] == [None] * 6
    pod = slots[2]
    assert decide(pod.ledger) == "proceed" and "fail" not in render_brief(pod.ledger).lower()
    assert "0.40" in render_brief(pod.ledger)
    dry = slots[12]
    body, _source, action = render_relay(dry.ledger)
    assert action == "relay" and "$0.00" in body and "paid step" not in body

    result = run_pilot(slots, mock=True, seed=0)
    assert result["mock"] is True and result["grant"] is None and result["unloaded"] is True
    assert result["pack_sha256"] == PACK_HASH
    assert len(result["items"]) == 26
    assert result["verdict_changes"] == {
        "changed": ["pilot-01", "pilot-02", "pilot-03", "pilot-06", "pilot-09"],
        "kept": ["pilot-04", "pilot-05", "pilot-07", "pilot-08", "pilot-10"],
    }
    assert result["critic_counts"] == {
        "positives_recovered": 16, "positives": 16, "negatives_flagged": 10, "negatives": 10,
    }
    probe = result["controls"]["probe"]
    assert probe["n"] == 26 and "smoke test" in probe["message"]
    assert "200-permutation null" in probe["message"] and "95th percentile" in probe["message"]
    assert probe["kind_a"]["n"] == 20 and "kind A" in probe["message"]
    assert "Kind B has no negatives" in probe["message"]
    by_id = {(item["slot"], item["role"]): item for item in result["items"]}
    retry = by_id[("pilot-04", "positive")]
    assert [item["passed"] for item in retry["attempts"]] == [False, True]
    assert len({item["family"] for item in retry["attempts"]}) == 1
    reset = by_id[("pilot-10", "negative")]
    assert reset["status"] == "reset" and len(reset["attempts"]) == 5
    assert len({item["family"] for item in reset["attempts"]}) == 1
    assert reset["verdict_changed"] is False and reset["critic"]["faithful"] is False
    changed = by_id[("pilot-06", "negative")]
    assert changed["verdict_changed"] is True and changed["hidden_key"] == "proceed"
    assert HIDE.search(changed["prompt"]) is None
    assert "KEY" not in changed["critic_prompt"] and "plant" not in changed["critic_prompt"].lower()
    assert "Decision table" in changed["critic_prompt"]
    assert render(slots[5].ledger) in changed["critic_prompt"]
    assert changed["critic"]["verdict"] == "hold"
    relayed = by_id[("pilot-11", "positive")]
    assert relayed["prose"] == "relay: rerun the suite on the patch"
    assert "Budget note: redo the budget" in relayed["record"]
    assert "publish" not in relayed["record"]
    assert "Tripwires:" not in relayed["record"]
    pushed = by_id[("pilot-16", "positive")]
    assert pushed["prose"] == "escalate: a public or irreversible act."
    assert "force-push the branch" in pushed["record"] and "Tripwires:" not in pushed["record"]
    review = by_id[("pilot-09", "positive")]
    assert "required changes" not in review["record"] and review["key"] == "proceed"
    review_bad = by_id[("pilot-09", "negative")]
    assert review_bad["verdict_changed"] is True and review_bad["hidden_key"] == "hold"
    assert "coverage floor" in review_bad["prose"]


class _Tags:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _pinned_opener(digests: dict | None = None):
    """A fake /api/tags. The default digests are the pins in writers.json."""

    if digests is None:
        data = json.loads((PACK.parents[1] / "writers.json").read_text(encoding="utf-8"))
        models = [
            {"name": row["served_as"], "digest": row["digest"]}
            for row in data["writers"] + data["critics"]
            if row.get("digest")
        ]
    else:
        models = [{"name": name, "digest": digest} for name, digest in digests.items()]

    def open_(url, timeout=30):
        return _Tags({"models": models})

    return open_


def test_a_live_pilot_refuses_without_a_grant_and_unloads_on_abort(tmp_path: Path, monkeypatch):
    called = []
    monkeypatch.setattr("role_set.transport.ollama_chat", lambda *a, **k: called.append("chat"))
    monkeypatch.setattr("role_set.transport.unload", lambda *a, **k: called.append("unload"))
    slots = plan_pilot(PACK, seed=0)
    with pytest.raises(PilotError, match="grant"):
        run_pilot(slots, mock=False)
    with pytest.raises(PilotError, match="heartbeat"):
        run_pilot(slots, mock=False, grant="g-a1b2c3d4")
    beat = tmp_path / "beat"
    beat.write_text("x", encoding="utf-8")
    with pytest.raises(PilotError, match="heartbeat"):
        run_pilot(
            slots, mock=False, grant="g-a1b2c3d4", heartbeat=beat,
            now=lambda: beat.stat().st_mtime + 120,
        )
    assert called == []

    unloaded = []

    def explode(model, prompt, seed, temperature, format=None):
        raise RuntimeError("stop before a model")

    with pytest.raises(RuntimeError, match="stop"):
        run_pilot(
            slots[:1], mock=False, grant="g-a1b2c3d4", heartbeat=beat,
            complete=explode, unload_fn=unloaded.append, digest_opener=_pinned_opener(),
        )
    assert unloaded == ["qwen3:8b"]

    slot = slots[12]

    def complete(model, prompt, seed, temperature, format=None):
        if "gemma4" in model:
            assert format == "json"
            return '{"action": "relay", "faithful": true, "diff_item": ""}'
        assert format is None
        body, _source, _action = render_relay(slot.ledger)
        return body

    fresh = []
    result = run_pilot(
        [slot, slot], mock=False, grant="g-a1b2c3d4", heartbeat=beat,
        complete=complete, unload_fn=fresh.append, digest_opener=_pinned_opener(),
    )
    assert fresh == ["qwen3:8b", "gemma4:31b", "qwen3:8b", "gemma4:31b"]
    assert result["grant"] == "g-a1b2c3d4" and result["unloaded"] is True
    assert result["critic_counts"]["positives_recovered"] == 2

    def bad_json(model, prompt, seed, temperature, format=None):
        if "gemma4" in model:
            return "not json"
        body, _source, _action = render_relay(slot.ledger)
        return body

    broken = run_pilot(
        [slot, slot], mock=False, grant="g-a1b2c3d4", heartbeat=beat,
        complete=bad_json, unload_fn=lambda _model: None, digest_opener=_pinned_opener(),
    )
    assert broken["items"][0]["critic"]["faithful"] is False

    from role_set.advocate.pilot import _unload_live

    monkeypatch.setattr("role_set.transport.unload", lambda *a, **k: None)
    _unload_live("qwen3:8b")


def _writer_items(ledger, kind: str) -> list[str]:
    if kind == "B":
        return [point.text for point in points_from(ledger)]
    items = list(ledger.asks)
    items.extend(figure.text for figure in ledger.figures if figure.material)
    for check in ledger.checks:
        if check.result == "fail" and check.on == "proposed":
            items.append(check.name)
            if check.evidence:
                items.append(check.evidence)
    items.extend(PHRASES[trip] for trip in ledger.tripwires)
    items.extend(ledger.missing)
    return items


def test_prompt_completeness_over_the_twenty_six_pilot_texts():
    """Every item a reference output depends on is in the writer prompt, and in the critic prompt."""

    slots = plan_pilot(PACK, seed=0)
    texts = 0
    for slot in slots:
        roles = ("positive", "negative") if slot.kind == "A" else ("positive",)
        for role in roles:
            texts += 1
            seen = plant(slot.ledger, slot.plant).hidden if role == "negative" else slot.ledger
            prompt = (
                negative_prompt(slot.trait, slot.ledger, slot.plant)
                if role == "negative"
                else writer_prompt(slot.trait, slot.kind, slot.ledger)
            )
            for item in _writer_items(seen, slot.kind):
                assert item in prompt, (slot.id, role, item)
            critic_record = render(slot.ledger)
            critic = critic_prompt(critic_record, "prose", slot.kind)
            for item in _writer_items(slot.ledger, slot.kind):
                assert item in critic, (slot.id, role, item)
            assert "faithful" in critic and "diff_item" in critic
            if slot.kind == "B":
                assert "Decision table" not in critic
                for phrase in PHRASES.values():
                    assert phrase in prompt and phrase in critic
            else:
                assert "not what the prose claims" in critic
    assert texts == 26


def test_the_live_path_returns_text_from_a_loopback_server_with_think_false(monkeypatch):
    """The real transport, against a fake Ollama on loopback. No model is loaded.

    The writer is asked with thinking off. The critic is asked the same way, plus JSON,
    and a fenced reply parses while an empty reply is recorded as a refusal.
    """

    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from role_set.advocate.pilot import _complete_live, _critic_payload

    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw.decode("utf-8"))
            seen.append({"path": self.path, "body": body})
            if body["model"] == "gemma4:31b" and sum(1 for item in seen if item["body"]["model"] == "gemma4:31b") == 1:
                content = '```json\n{"faithful": true, "diff_item": ""}\n```'
            elif body["model"] == "gemma4:31b":
                content = "  \n"
            else:
                content = "the brief is text"
            payload = json.dumps({
                "model": body["model"],
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A003 - BaseHTTPRequestHandler's name
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr("role_set.advocate.pilot.ENDPOINT", f"http://127.0.0.1:{port}")
    try:
        text = _complete_live("qwen3:8b", "Say the line.", 0, 0.2)
        fenced = _complete_live("gemma4:31b", "Judge.", 0, 0.0, format="json")
        empty = _complete_live("gemma4:31b", "Judge again.", 1, 0.0, format="json")
    finally:
        server.shutdown()
        server.server_close()
    assert text == "the brief is text"
    assert seen[0]["path"] == "/api/chat"
    assert seen[0]["body"]["think"] is False
    assert "format" not in seen[0]["body"]
    assert seen[0]["body"]["model"] == "qwen3:8b"
    assert seen[1]["body"]["model"] == "gemma4:31b"
    assert seen[1]["body"]["think"] is False and seen[1]["body"]["format"] == "json"
    assert seen[2]["body"]["think"] is False and seen[2]["body"]["format"] == "json"
    assert _critic_payload(fenced) == {"faithful": True, "diff_item": ""}
    assert _critic_payload(empty) == {"faithful": False, "diff_item": "empty critic reply"}
    assert _critic_payload("```json\n{\"verdict\": \"hold\"}\n```")["verdict"] == "hold"


def test_a_live_pilot_refuses_a_cloud_tag_an_unlisted_name_and_a_bad_digest_before_any_model_call(tmp_path, monkeypatch):
    """Three refusals, each before any model call."""

    called = []

    def spy(model, prompt, seed, temperature, format=None):
        called.append(model)
        raise AssertionError("a model call")

    beat = tmp_path / "beat"
    beat.write_text("x", encoding="utf-8")
    slots = plan_pilot(PACK, seed=0)[:1]
    common = dict(mock=False, grant="g-a1b2c3d4", heartbeat=beat, complete=spy, unload_fn=lambda _model: None)

    monkeypatch.setattr("role_set.advocate.pilot.CRITIC_MODEL", "gemma4:31b-cloud")
    with pytest.raises(PilotError, match="cloud"):
        run_pilot(slots, digest_opener=_pinned_opener(), **common)
    assert called == []

    monkeypatch.setattr("role_set.advocate.pilot.CRITIC_MODEL", "gemma4:31b")
    monkeypatch.setattr(
        "role_set.advocate.pilot.WRITERS",
        {"writer-a": "qwen3:8b", "writer-b": "not-in-the-list:8b"},
    )
    with pytest.raises(PilotError, match="not on the licence-checked"):
        run_pilot(slots, digest_opener=_pinned_opener(), **common)
    assert called == []

    monkeypatch.setattr(
        "role_set.advocate.pilot.WRITERS",
        {"writer-a": "qwen3:8b", "writer-b": "granite3.3:8b"},
    )
    data = json.loads((PACK.parents[1] / "writers.json").read_text(encoding="utf-8"))
    digests = {
        row["served_as"]: row["digest"]
        for row in data["writers"] + data["critics"]
        if row.get("digest")
    }
    digests["granite3.3:8b"] = "0" * 64
    with pytest.raises(PilotError, match="differs from the pinned"):
        run_pilot(slots, digest_opener=_pinned_opener(digests), **common)
    assert called == []


_MODEL_FAMILY = {
    "qwen3:8b": "qwen3",
    "granite3.3:8b": "granite",
    "gemma4:31b": "gemma4",
}


def test_the_three_pilot_models_are_three_different_families():
    """The pilot's two writers and the critic are three families. A later swap cannot collapse them."""

    data = json.loads((PACK.parents[1] / "writers.json").read_text(encoding="utf-8"))
    listed = {row["served_as"] for row in data["writers"] + data["critics"]}
    models = (WRITERS["writer-a"], WRITERS["writer-b"], CRITIC_MODEL)
    families = []
    for name in models:
        assert name in listed
        families.append(_MODEL_FAMILY[name])
    assert families == ["qwen3", "granite", "gemma4"]
    assert len(set(families)) == len(models)


def test_no_kind_b_pilot_point_is_verbatim_in_the_procedure_or_the_critic_prompt():
    """The procedure and the critic's rules name categories. They do not quote a slot's point."""

    slots = plan_pilot(PACK, seed=0)
    rules = critic_prompt("RECORD", "PROSE", "B").split("RECORD:", 1)[0]
    points = [
        point.text
        for slot in slots if slot.kind == "B"
        for point in points_from(slot.ledger)
    ]
    assert len(points) >= 9
    for text in points:
        assert text not in PROCEDURES["B"], text
        assert text not in rules, text
    assert "force-pushing" in PROCEDURES["B"]
    assert "a public or irreversible act" in PROCEDURES["B"]
    assert "Skipping a check or a gate is a floor missed." in PROCEDURES["B"]


def test_the_writer_table_and_the_critic_table_are_the_same_text():
    slots = plan_pilot(PACK, seed=0)
    writer = writer_prompt(slots[0].trait, "A", slots[0].ledger)
    critic = critic_prompt(render(slots[0].ledger), "prose", "A")
    assert DECISION_TABLE in PROCEDURES["A"]
    assert DECISION_TABLE in writer and DECISION_TABLE in critic
    assert writer[writer.index(DECISION_TABLE):writer.index(DECISION_TABLE) + len(DECISION_TABLE)] == (
        critic[critic.index(DECISION_TABLE):critic.index(DECISION_TABLE) + len(DECISION_TABLE)]
    )
    assert "cannot_tell: the record is missing <item>." in writer
    assert "Note: runner ubuntu-latest" in writer
    relay = writer_prompt(slots[15].trait, "B", slots[15].ledger)
    assert "force-pushing" in relay
    assert "force-push the branch" not in PROCEDURES["B"]


def test_a_note_line_number_is_not_a_stray_when_the_brief_leaves_it_out():
    ledger = led(figures=(
        Figure("coverage 0.71", True, "0.71", "", "coverage"),
        Figure("retries 0", False, "0", "", "retries"),
    ))
    record = render(ledger)
    brief = render_brief(ledger)
    assert "Note: retries 0" in record
    assert "Note:" not in brief
    assert "0" not in stray_numbers(brief, ledger)
    assert stray_numbers(brief + "\nseen 0", ledger) == ["0"]
    assert all(item.passed for item in check_a(record, brief, ledger))


def test_advocate_pilot_commands(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    assert main([
        "plan", "--role", "advocate", "--pilot", "--seed", "0",
        "--pack", str(PACK), "--out", str(out),
    ]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["pack_sha256"] == PACK_HASH and len(plan["slots"]) == 16
    assert "wrote 16 slots" in capsys.readouterr().out

    run_out = tmp_path / "run.json"
    assert main([
        "pilot", "--role", "advocate", "--mock", "--seed", "0",
        "--pack", str(PACK), "--out", str(run_out),
    ]) == 0
    data = json.loads(run_out.read_text(encoding="utf-8"))
    assert len(data["items"]) == 26 and data["grant"] is None
    assert "smoke test" in capsys.readouterr().out
    assert main(["pilot", "--role", "verifier", "--mock", "--out", str(tmp_path / "no.json")]) == 2
    assert main([
        "pilot", "--role", "advocate", "--pack", str(PACK), "--out", str(tmp_path / "live.json"),
    ]) == 2
    assert "REFUSED" in capsys.readouterr().err

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pilot.json").write_text(run_out.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["pilot", "--role", "advocate", "--report", str(run_dir)]) == 0
    packet = (run_dir / "read.md").read_text(encoding="utf-8")
    assert packet.count("\n## ") == 26
    assert "## pilot-01 positive" in packet and "## pilot-01 negative" in packet
    assert "Note: runner ubuntu-latest" in packet
    assert "### Critic" in packet and "### Checks" in packet
    assert main(["pilot", "--role", "advocate", "--report", str(tmp_path / "missing")]) == 2
