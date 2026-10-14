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
    plant_exact,
    sections,
)
from role_set.advocate.controls import (
    brief_only_rate,
    chance_uniform,
    expected_shuffle_match,
    shuffled_label_rate,
    writer_split,
)
from role_set.advocate.critic import critic_prompt, flagged, recovered, require_different_family
from role_set.advocate.critic import FamilyError
from role_set.advocate.kinds import LESSONS, writer_prompt, negative_prompt
from role_set.advocate.ledger import (
    TRIPWIRES,
    Check,
    Figure,
    Ledger,
    LedgerError,
    PlantError,
    Planted,
    allowed_shapes,
    apply_release_rule,
    assert_shape,
    bank,
    plant,
    points_from,
    relay_action,
    render_brief,
    render_record,
    render_relay,
    render_two_turn,
    render_writer_ledger,
    scan_text,
)
from role_set.advocate.plan import FAMILIES, PlanError, plan_batch
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
    shown = render_writer_ledger(clean)
    assert "release approved: unset" in shown and "verdict:" not in shown
    assert "release approved: no" in render_writer_ledger(apply_release_rule(led(
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
    assert all(item.passed for item in check_a(render_record(pod), brief, pod))
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
        record = render_record(ledger)
        assert all(item.passed for item in check_a(record, positive, ledger))
        assert not check_trace(negative, ledger).passed
        assert plant_exact(negative, ledger, planted).passed
        assert sections(negative) is not None
        assert check_order(negative, planted.hidden).passed
        assert check_length(record, negative).passed
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
    with pytest.raises(PlantError, match="no stable"):
        from role_set.advocate.ledger import stable_plants
        stable_plants(led(figures=(Figure("side note 41", False, "41"),)))
    messy = plant(two_fails, "B001")
    assert not plant_exact(render_brief(messy.hidden).replace("0.71", "0.72"), two_fails, messy).passed
    assert not plant_exact("no", two_fails, Planted("B099", two_fails, "x")).passed


def test_kind_b_drops_out_of_scope_and_escalates_a_tripwire():
    quiet = led()
    body, source, action = render_relay(quiet)
    assert action == "relay" and relay_action(points_from(quiet)) == "relay"
    assert "side note" not in body and "coverage 0.71" in body
    assert all(item.passed for item in check_scoped(body, source, action))
    loud = led(tripwires=("B006",))
    escalated, _source, action = render_relay(loud)
    assert action == "escalate" and escalated.startswith("escalate:")
    assert all(item.passed for item in check_scoped(escalated, _source, action))
    paid = led(paid_model_api=True)
    assert relay_action(points_from(paid)) == "escalate"


def test_kind_d_stands_or_updates_and_kind_c_swaps_order():
    stands = render_two_turn(led())
    updates = render_two_turn(led(missing=("grant id",)))
    assert stands[2] == "stands" and updates[2] == "updates"
    assert all(item.passed for item in check_scoped(*stands))
    assert all(item.passed for item in check_scoped(*updates))
    assert "hold" not in stands[0]

    ledger = led(checks=(Check("unit check", "pass", "proposed"),))
    first, second = render_brief(ledger), render_brief(ledger, "b")
    record = render_record(ledger)
    assert first != second
    assert all(item.passed for item in check_a(record, first, ledger))
    assert all(item.passed for item in check_a(record, second, ledger))
    assert check_swap((first, second), (second, first)).passed
    assert not check_swap((first, second), (first, second)).passed
    assert not check_swap((first, first), (first, first)).passed


def test_checks_refuse_a_broken_brief():
    ledger = led(checks=(Check("coverage floor", "fail", "proposed"),))
    record = render_record(ledger)
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
    copied = " ".join(["amber", "birch", "cedar", "dune", "elm", "flint", "grove", "haze"])
    assert not check_length(record, copied).passed
    assert "copied" in check_length(record, copied).reason
    long_brief = " ".join(f"zz{i}" for i in range(131))
    long_record = " ".join(f"yy{i}" for i in range(200))
    assert "exceeds" in check_length(long_record, long_brief).reason
    assert "shorter" in check_length("one", "one two three").reason
    assert not check_action("relay: coverage 0.71", "escalate").passed
    buried = brief.replace("2. The headline\ncoverage floor failed", "2. The headline\nEvery check passed.")
    assert "not in section 2" in check_order(buried, ledger).reason
    unnamed = brief.replace("hold:", "maybe:", 1).replace(" supports hold ", " supports maybe ")
    assert "not in section 5" in check_order(unnamed, ledger).reason
    gap = render_brief(led(missing=("decision threshold",))).replace("decision threshold", "the gap")
    assert "does not name" in check_order(gap, led(missing=("decision threshold",))).reason
    long_body = " ".join(f"aa{i}" for i in range(131))
    longer = " ".join(f"bb{i}" for i in range(200))
    assert any("exceeds" in item.reason for item in check_scoped(long_body, longer, "relay"))
    assert any("shorter" in item.reason for item in check_scoped("relay: one two three four", "tiny", "relay"))


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
        assert "verdict:" not in prompt
        if slot.kind == "E":
            for index in (0, 1):
                quiz = writer_prompt(slot.trait, "E", quiz_index=index)
                assert "write a brief" not in quiz.lower()
                assert f"Question. " in quiz
        if slot.kind == "A":
            hidden = negative_prompt(slot.trait, slot.ledger, slot.plant)
            assert HIDE.search(hidden) is None
            assert "write a brief" not in hidden.lower()
            assert hidden != prompt
        if slot.kind in ("A", "C"):
            record = render_record(slot.ledger)
            for voice in ("a", "b"):
                brief = render_brief(slot.ledger, voice)
                assert all(item.passed for item in check_a(record, brief, slot.ledger))
                for figure in slot.ledger.figures:
                    if not figure.material:
                        assert figure.text not in brief
        if slot.kind == "B":
            body, source, action = render_relay(slot.ledger)
            assert action == slot.key
            assert all(item.passed for item in check_scoped(body, source, action))
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
    prompt = critic_prompt("The note is quiet.")
    assert "KEY" not in prompt and "hold" not in prompt
    assert "Temperature 0" in prompt and '"verdict"' in prompt
    assert recovered({"verdict": "hold", "defective": False}, "hold")
    assert not recovered({"verdict": "hold", "defective": True}, "hold")
    assert not recovered({"verdict": "proceed", "defective": False}, "hold")
    assert flagged({"defective": True}) and not flagged({"defective": False})

    keys = [slot.key for slot in slots]
    assert expected_shuffle_match(["only"]) == 1
    assert abs(shuffled_label_rate(keys) - expected_shuffle_match(keys)) < 0.08
    assert abs(brief_only_rate(keys) - chance_uniform(keys)) < 0.08
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
