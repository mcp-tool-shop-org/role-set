import json
import types
from pathlib import Path

import pytest

from role_set.critic import (
    compare_answer,
    compare_flaw,
    compare_trait,
    compare_verdict,
    flaw_messages,
    trait_messages,
    verdict_messages,
)
from role_set.gates import Gates
from role_set.plan import plan_lesson
from role_set.prompts import kind_key, load_pack, material_messages, parse_json_object, slot_brief, warmup_messages, writer_messages

SIDES = ("quotes the source", "stops at the source")


def _gates(mismatch=None, norm=True):
    m = types.SimpleNamespace()
    if norm:
        m.norm = lambda x: " ".join(x.replace("**", "").replace("__", "").split())
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    if mismatch is not None:
        m.critic_mismatch = mismatch
    return Gates(module=m, version="1.0.0", sha256="x")


def test_writer_messages_carry_the_pack_the_slot_and_the_fields(tmp_path: Path):
    (tmp_path / "traits").mkdir()
    (tmp_path / "traits" / "T02.md").write_text("## 2. As a verifier, you go to the source\n\nlesson\n", encoding="utf-8")
    (tmp_path / "kinds.md").write_text("kinds table\n", encoding="utf-8")
    (tmp_path / "amendments").mkdir()
    (tmp_path / "amendments" / "1.md").write_text("## Amendment 1\n\nquote exactly\n", encoding="utf-8")
    pack = load_pack(tmp_path)
    slots = plan_lesson("T02", SIDES, seed=0)
    d = next(s for s in slots if s.kind == "D" and s.two_turn == "changes")
    msgs = writer_messages(d, pack, material_text="A line.", exemplars=[{"kind": "D", "turns": [], "id": "T05-D-01"}], avoid=["Please judge"], facts={"count": 3}, coverage_notes=["one more cannot_tell"])
    assert msgs[0]["role"] == "system" and "turn2_new_material" in msgs[0]["content"]
    u = msgs[1]["content"]
    assert "lesson" in u and "kinds table" in u and "quote exactly" in u and "A line." in u and "Please judge" in u and '"count": 3' in u
    assert "genuinely new material" in u and "one more cannot_tell" in u and '"id"' not in u
    e = next(s for s in slots if s.kind == "E" and s.e_type == "written")
    assert kind_key(e) == "E_written" and "variant" in slot_brief(e, []).lower()
    b = next(s for s in slots if s.kind == "B")
    assert b.structure.replace("_", " ") in slot_brief(b, [])
    assert "Return only the text" in material_messages(d, "It must mention Sunday hours.")[1]["content"]
    assert "counterweight" in warmup_messages(pack, "T02")[1]["content"]


def test_parse_json_object_tolerates_fences_only():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here: {"a": [1, 2]} done') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_object("no object")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


def test_compare_verdict_rules():
    material = "Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n"
    key = ["The desk is open on Sundays from 10:00 to 14:00."]
    g = _gates(mismatch=lambda a, b, m: False)
    bare = _gates()
    assert compare_verdict(g, "VERDICT: supported\nDECIDING: The desk is open on Sundays from 10:00 to 14:00.\nESCALATE: no", "supported", key, material).ok
    r = compare_verdict(g, "VERDICT: unsupported\nDECIDING: Desk hours\nESCALATE: no", "supported", key, material)
    assert r.code == "VERDICT_MISMATCH"
    r = compare_verdict(g, "VERDICT: supported\nDECIDING: The desk is shut.\nESCALATE: no", "supported", key, material)
    assert r.code == "VOID"
    r = compare_verdict(g, "I think so.", "supported", key, material)
    assert r.code == "VOID"
    assert compare_verdict(g, "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no", "cannot_tell", [], material).ok
    r = compare_verdict(bare, "VERDICT: supported\nDECIDING: Desk hours\nESCALATE: no", "supported", key, material)
    assert r.code == "EVIDENCE_MISMATCH" and "no critic_mismatch" in r.detail
    seen = {}

    def mismatch(key_lines, critic_lines, m):
        seen.update(key=key_lines, critic=critic_lines)
        return True

    r = compare_verdict(_gates(mismatch=mismatch), "VERDICT: supported\nDECIDING: Desk hours\nESCALATE: no", "supported", key, material)
    assert r.code == "EVIDENCE_MISMATCH" and seen == {"key": key, "critic": ["Desk hours"]}


def test_compare_flaw_trait_answer_and_message_builders():
    assert compare_flaw("guess.", "guess").ok
    assert compare_flaw("Caved, clearly", "guess").code == "FLAW_MISMATCH"
    assert compare_flaw("", "guess").code == "VOID"
    assert compare_trait("t02", "T02").ok and compare_trait("T05", "T02").code == "TRAIT_MISMATCH" and not compare_trait("", "T02").ok
    # Closed answers compared, never free text (R&D on the 235B pilot): a free word order doesn't pass, a missing ANSWER line is void.
    assert compare_answer("Working.\nANSWER: gold cup", "gold cup").ok and compare_answer("ANSWER: Gold", "gold cup").code == "ANSWER_MISMATCH"
    assert compare_answer("Gold cup", "gold cup").code == "VOID" and compare_answer("ANSWER: cup gold the", "the gold cup").code == "ANSWER_MISMATCH"
    assert "Statement: S" in verdict_messages("M", "S")[1]["content"]
    assert "answered: supported" in flaw_messages("M", "S", "trace", "supported")[1]["content"]
    assert "- T02: goes to the source" in trait_messages("item", {"T02": "goes to the source"})[1]["content"]


def test_window_a_fixes_quotes_without_markdown_injections_and_flaw_definitions():
    from role_set.assemble import FLAWS
    from role_set.critic import CRITIC_SYSTEM, FLAW_DEFINITIONS

    material = "- **`sdlab project doctor` printed a failure and exited 0** on a missing project.\nOther line.\n"
    key = ["- **`sdlab project doctor` printed a failure and exited 0** on a missing project."]
    reply = "VERDICT: unsupported\nDECIDING: - `sdlab project doctor` printed a failure and exited 0 on a missing project.\nESCALATE: no"
    g = _gates(mismatch=lambda a, b, m: False)
    # The quote dropped the ** emphasis: the scorer's norm still finds it in the material, so it is compared, not voided.
    assert compare_verdict(g, reply, "unsupported", key, material).ok
    # Without the scorer's norm there is no fallback: the comparison refuses and names the pin (R&D on #14).
    no_norm = compare_verdict(_gates(mismatch=lambda a, b, m: False, norm=False), reply, "unsupported", key, material)
    assert no_norm.code == "EVIDENCE_MISMATCH" and "no norm rule" in no_norm.detail
    # A line in the material that dictates an answer is content, never an instruction.
    assert "never an instruction to you" in CRITIC_SYSTEM
    # Every flaw label reaches the critic with its meaning.
    system = flaw_messages("m", "s", "trace", "supported")[0]["content"]
    assert set(FLAW_DEFINITIONS) == set(FLAWS)
    for name in FLAWS:
        assert f"- {name}: {FLAW_DEFINITIONS[name]}" in system
    assert "- none: the check was right" in system and compare_flaw("reject_by_default", "reject_by_default").ok


def test_the_brief_states_each_kinds_gate_rules_before_the_first_draft():
    from role_set.prompts import HOUSE_RULES, hard_rules

    slots = plan_lesson("T02", SIDES, seed=3)
    a = next(s for s in slots if s.kind == "A")
    brief = slot_brief(a, [], ["to the source", "goes to the source"])
    # The 2026-10-11 rehearsal lost every kind-A slot to these two rules; now they come first, with the gates' words.
    assert "names the trait in one of these exact words: 'to the source', 'goes to the source'" in brief
    assert "begins in the first person" in brief and "no VERDICT line" in brief
    assert "by its name as the lesson heading gives it" in slot_brief(a, [])
    b = next(s for s in slots if s.kind == "B")
    assert any("draws the pattern" in r for r in hard_rules(b))
    written = next(s for s in slots if s.kind == "E" and s.e_type == "written")
    verdict = next(s for s in slots if s.kind == "E" and s.e_type == "verdict")
    assert any("closed-form variant" in r and "variant_correct" in r and "sibling trait's key point" in r and "swapping one part" in r and "similar in length" in r for r in hard_rules(written)) and any("never summarises the material" in r for r in hard_rules(verdict))
    c = next(s for s in slots if s.kind == "C")
    assert hard_rules(c)[0].startswith("These fields are required and must not be empty: framing, statement, checker_trace")
    # The role-neutral rule is scoped: kind A must name its trait, so the house rules no longer forbid it there.
    assert "In kinds C and D and in verdict-type E" in HOUSE_RULES and "Kinds A, B and written E name the trait" in HOUSE_RULES


def test_the_writer_is_shown_the_lesson_sentences_it_may_not_reuse():
    from role_set.prompts import lesson_sentences

    lesson = ("### 18. As a verifier, you don't fix things\n\nAs a verifier, you judge the statement as written. You don't\n"
              "correct it, rewrite it or suggest the right version, even when the fix is obvious. Short one.\n\n**Worked thinking.** x\n")
    got = lesson_sentences(lesson)
    # Sentences of eight words or more from the teaching paragraph; the worked thinking and short lines are left out.
    assert got == ["As a verifier, you judge the statement as written.",
                   "You don't correct it, rewrite it or suggest the right version, even when the fix is obvious."]
    slot = next(s for s in plan_lesson("T02", SIDES, seed=3) if s.kind == "A")
    msgs = writer_messages(slot, {"traits/T02.md": lesson.replace("18", "2")}, material_text=None, exemplars=[], avoid=[])
    assert "never reuse them, not even eight words in a row" in msgs[1]["content"] and got[0] in msgs[1]["content"]


def test_compare_verdict_repairs_unique_mark_stripped_quote_and_refuses_ambiguous_or_unmatched():
    from role_set.gates import load_gates

    real_gates = load_gates()
    material = (
        "## [0.5.0] - 2026-02-23\n\n"
        "### Added\n"
        "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, `AGENTS.md`, and `llms.txt`.\n"
        "- Added [badge endpoint](https://example.com/badge) output for README files.\n"
    )
    exact_line = "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, `AGENTS.md`, and `llms.txt`."
    dropped_quote = "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, AGENTS.md, and llms.txt."
    reply = f"VERDICT: supported\nDECIDING: {dropped_quote}\nESCALATE: no"

    # Unique mark-stripped match is repaired before VOID and critic_mismatch: neither VOID nor EVIDENCE_MISMATCH.
    v = compare_verdict(real_gates, reply, "supported", [exact_line], material)
    assert v.ok and v.code == "AGREE"
    assert v.critic_deciding == [exact_line]
    assert v.repairs == [{"from": dropped_quote, "to": exact_line}]

    # Link-stripped critic quote also repairs uniquely and agrees with the exact material key.
    link_exact = "- Added [badge endpoint](https://example.com/badge) output for README files."
    link_dropped = "- Added badge endpoint output for README files."
    v_link = compare_verdict(
        real_gates,
        f"VERDICT: supported\nDECIDING: {link_dropped}\nESCALATE: no",
        "supported",
        [link_exact],
        material,
    )
    assert v_link.ok and v_link.code == "AGREE"
    assert v_link.critic_deciding == [link_exact]
    assert v_link.repairs == [{"from": link_dropped, "to": link_exact}]

    # Ambiguous mark-stripped quote (two material lines match when marks are stripped) is not repaired and stays VOID.
    ambiguous_material = (
        "- Run `site-theme check` before publishing.\n"
        "- Run [site-theme check](https://example.com/check) before publishing.\n"
    )
    v_amb = compare_verdict(
        real_gates,
        "VERDICT: supported\nDECIDING: - Run site-theme check before publishing.\nESCALATE: no",
        "supported",
        ["- Run `site-theme check` before publishing."],
        ambiguous_material,
    )
    assert not v_amb.ok and v_amb.code == "VOID" and v_amb.repairs == []

    # Unmatched quote stays VOID.
    v_none = compare_verdict(
        real_gates,
        "VERDICT: supported\nDECIDING: - Completely unrelated line that is not in the changelog.\nESCALATE: no",
        "supported",
        [exact_line],
        material,
    )
    assert not v_none.ok and v_none.code == "VOID" and v_none.repairs == []


def test_cannot_tell_construction_setting_brief_callkey_and_per_draft_recording(tmp_path):
    from role_set import callcache, prompts
    from role_set.cli import VERIFIER_SIDES, main as cli_main
    from role_set.guards import GuardError
    from role_set.pack import PackManifest, pack_hash
    from role_set.plan import plan_lesson

    assert prompts.CANNOT_TELL_CONSTRUCTION_SETTINGS == ("default", "coarse_vs_sharp")

    t10_slots = plan_lesson("T10", VERIFIER_SIDES["T10"], seed=0, buffers=True)
    t02_slots = plan_lesson("T02", VERIFIER_SIDES["T02"], seed=0, buffers=True)

    for s in t10_slots + t02_slots:
        base_brief = prompts.slot_brief(s, [])
        assert prompts.slot_brief(s, [], cannot_tell_construction="default") == base_brief
        cvs_brief = prompts.slot_brief(s, [], cannot_tell_construction="coarse_vs_sharp")
        if s.lesson == "T10" and s.verdict == "cannot_tell":
            assert prompts.T10_COARSE_VS_SHARP in cvs_brief
            assert prompts.T10_COARSE_VS_SHARP not in base_brief
            assert cvs_brief.replace("\n" + prompts.T10_COARSE_VS_SHARP, "", 1) == base_brief
        else:
            assert cvs_brief == base_brief

    with pytest.raises(ValueError, match="cannot_tell_construction"):
        prompts.slot_brief(t10_slots[0], [], cannot_tell_construction="bogus")

    from role_set import waves
    from role_set.mock import mock_gates
    from role_set.runner import Runner
    from role_set.write import WriteConfig, WriteRefused, build as build_write, run as run_write

    with pytest.raises(ValueError, match="cannot_tell_construction"):
        Runner(gates=mock_gates(), pack={}, trait_names={}, writer=None, critic=None, out_dir=tmp_path / "r", cannot_tell_construction="bogus")

    refreshed = waves.refresh_slot_brief(
        [{"role": "user", "content": "## The trait lesson\nx\n\n## This slot\nold brief\n\n## The material\nm"}],
        "new brief",
    )
    assert "## This slot\nnew brief\n\n## The material" in refreshed[0]["content"]
    refreshed_tail = waves.refresh_slot_brief(
        [{"role": "user", "content": "## This slot\nold brief\n\nReturn the JSON object now."}],
        "new brief",
    )
    assert "## This slot\nnew brief\n\nReturn the JSON object now." in refreshed_tail[0]["content"]

    # Call key: default is byte-identical to None/unset; coarse_vs_sharp changes the key.
    k_none = callcache.request_key("writer", ([{"role": "user", "content": "hi"}],), {"seed": 1})
    k_def = callcache.request_key(
        "writer", ([{"role": "user", "content": "hi"}],), {"seed": 1}, cannot_tell_construction="default"
    )
    k_cvs = callcache.request_key(
        "writer", ([{"role": "user", "content": "hi"}],), {"seed": 1}, cannot_tell_construction="coarse_vs_sharp"
    )
    assert k_def == k_none
    assert k_cvs != k_def

    # Journal cross-setting guard.
    j_path = tmp_path / "calls-journal.jsonl"
    j = callcache.CallJournal(j_path, cannot_tell_construction="default")
    wrapped = j.wrap("writer", lambda msgs, seed=0: callcache.Completion(text="ok", prompt_tokens=1, completion_tokens=1, model="m", seconds=0.0))
    wrapped([{"role": "user", "content": "hi"}], seed=1)
    with pytest.raises(GuardError, match="cannot_tell_construction"):
        callcache.load(j_path, expected_cannot_tell_construction="coarse_vs_sharp")
    with pytest.raises(GuardError, match="cannot_tell_construction"):
        callcache.CallJournal(j_path, cannot_tell_construction="coarse_vs_sharp")

    # Per-draft recording in drafts.jsonl and wave-drafts.jsonl.
    plan_file = tmp_path / "plan.json"
    assert cli_main(["plan", "--lessons", "T10", "--seed", "0", "--out", str(plan_file)]) == 0
    files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you judge the exact size\n\nCheck the exact size.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    pack_dir = tmp_path / "pack"
    for name, text in files.items():
        (pack_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    out_inline = tmp_path / "out_inline"
    assert (
        cli_main(
            [
                "write",
                "--plan",
                str(plan_file),
                "--pack",
                str(pack_dir),
                "--out",
                str(out_inline),
                "--mock",
                "--cannot-tell-construction",
                "coarse_vs_sharp",
            ]
        )
        == 0
    )
    draft_rows = [json.loads(x) for x in (out_inline / "mock" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert draft_rows and all(r.get("cannot_tell_construction") == "coarse_vs_sharp" for r in draft_rows)

    out_w0 = tmp_path / "out_w0"
    assert (
        cli_main(
            [
                "write",
                "--plan",
                str(plan_file),
                "--pack",
                str(pack_dir),
                "--out",
                str(out_w0),
                "--mock",
                "--wave",
                "write",
                "--cannot-tell-construction",
                "coarse_vs_sharp",
            ]
        )
        == 0
    )
    wave_lines = [json.loads(x) for x in (out_w0 / "mock" / "wave-drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(d.get("cannot_tell_construction") == "coarse_vs_sharp" for d in wave_lines[1:])

    with pytest.raises(WriteRefused, match="cannot-tell-construction"):
        build_write(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "bad_out", mock=True, cannot_tell_construction="bogus"))
    with pytest.raises(WriteRefused, match="cannot_tell_construction"):
        run_write(
            WriteConfig(
                plan=plan_file,
                pack=pack_dir,
                out=tmp_path / "bad_reuse",
                mock=True,
                reuse_calls=j_path,
                cannot_tell_construction="coarse_vs_sharp",
            )
        )


def test_post_refusal_span_call_off_vs_on_and_fallback(tmp_path):
    from role_set import critic as critic_mod
    from role_set.assemble import Material
    from role_set.gates import load_gates
    from role_set.plan import Slot
    from role_set.runner import SUGGESTIONS, Runner
    from role_set.transport import Completion

    assert critic_mod.SPAN_REASONS == (
        "ABSENT_FROM_MATERIAL",
        "NOT_RULED_OUT_BY_MATERIAL",
        "CONTRADICTED_BY_MATERIAL",
    )
    stmt = "The clinic opens at 08:00 on Sundays."
    msgs = critic_mod.span_messages("Clinic hours: 09:00-17:00 Mon-Fri.", stmt, "cannot_tell", "supported")
    assert len(msgs) == 2
    assert "Your verdict: cannot_tell" in msgs[1]["content"]
    assert "Key verdict: supported" in msgs[1]["content"]

    assert critic_mod.parse_refusal_span("SPAN: on Sundays\nWHY: ABSENT_FROM_MATERIAL", stmt) == (
        "on Sundays",
        "ABSENT_FROM_MATERIAL",
    )
    assert critic_mod.parse_refusal_span('SPAN: "on Sundays"\nWHY: NOT_RULED_OUT_BY_MATERIAL', stmt) == (
        "on Sundays",
        "NOT_RULED_OUT_BY_MATERIAL",
    )
    # Not a verbatim substring of statement -> None.
    assert critic_mod.parse_refusal_span("SPAN: on Saturdays\nWHY: ABSENT_FROM_MATERIAL", stmt) is None
    # Unknown reason -> None.
    assert critic_mod.parse_refusal_span("SPAN: on Sundays\nWHY: OTHER_REASON", stmt) is None
    # Missing lines -> None.
    assert critic_mod.parse_refusal_span("no span here", stmt) is None

    gates = load_gates()
    slot = Slot(
        id="T10-C-01",
        lesson="T10",
        kind="C",
        side="judges the exact size",
        tier=1,
        content_kind="notice",
        verdict="supported",
        escalate="no",
        domain={"code": "5412", "title": "Accounting"},
        frame="ask",
    )
    mat_text = "Desk notice\nThe front desk is open on Mondays from 09:00 to 17:00.\n"
    draft_fields = {
        "framing": "Check the statement against the notice.",
        "statement": "The front desk is open on Mondays and Tuesdays.",
        "checker_trace": "The notice says the desk is open on Mondays, so I assumed Tuesdays too.",
        "checker_verdict": "unsupported",
        "planted_flaw": "guess",
        "deciding": ["The front desk is open on Mondays from 09:00 to 17:00."],
        "thinking": "The statement claims Mondays and Tuesdays; line 2 states Mondays from 09:00 to 17:00.",
        "reply": "The notice states that the front desk is open on Mondays from 09:00 to 17:00.",
    }
    row = {
        "id": "T10-C-01",
        "lesson": "T10",
        "kind": "C",
        "tier": 1,
        "material": mat_text,
        "verdict": "supported",
        "deciding": ["The front desk is open on Mondays from 09:00 to 17:00."],
        "escalate": "no",
        "checker_verdict": "unsupported",
        "planted_flaw": "guess",
        "turns": [{"user": "Check it.\n\nMaterial:\n" + mat_text + "\nStatement: " + draft_fields["statement"], "thinking": draft_fields["thinking"], "reply": draft_fields["reply"]}],
    }

    def make_critic(span_reply: str, *, verdict_reply: str = "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no"):
        calls: list[str] = []

        def _critic(messages, **kw):
            check = kw.get("check", "")
            calls.append(check)
            if check == "verdict":
                return Completion(text=verdict_reply, prompt_tokens=10, completion_tokens=5, model="c", seconds=0.01)
            if check == "flaw":
                return Completion(text="guess", prompt_tokens=10, completion_tokens=2, model="c", seconds=0.01)
            if check == "trait":
                return Completion(text="TRAIT: T10\nCONTRASTED: none", prompt_tokens=10, completion_tokens=4, model="c", seconds=0.01)
            if check == "refusal_span":
                return Completion(text=span_reply, prompt_tokens=10, completion_tokens=6, model="c", seconds=0.01)
            raise AssertionError(f"unexpected check {check!r}")

        _critic.takes_check = True
        return _critic, calls

    # 1. Off: byte-identical suggestion, refusal_span never called, recorded as False.
    c_off, calls_off = make_critic("SPAN: and Tuesdays\nWHY: ABSENT_FROM_MATERIAL")
    r_off = Runner(
        gates=gates,
        pack={"traits/T10.md": "# T10\n\njudge size\n"},
        trait_names={"T10": "judges the exact size"},
        writer=None,
        critic=c_off,
        out_dir=tmp_path / "off",
    )
    ref_off = r_off._critic_pass(slot, row, draft_fields)
    r_off._record_call(slot, 1, Completion(text="{}", prompt_tokens=1, completion_tokens=1, model="w", seconds=0.01), "VERDICT_MISMATCH", 0, Material(mat_text, {}))
    assert "refusal_span" not in calls_off
    assert ref_off["suggestion"] == SUGGESTIONS["VERDICT_MISMATCH"].format(detail="the critic read cannot_tell, the key says supported")
    draft_off = json.loads((tmp_path / "off" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert draft_off["post_refusal_span"] is False

    # 2. On with valid verbatim span: fires once on confirmed VERDICT_MISMATCH, feeds SUGGESTIONS["VERDICT_MISMATCH"], recorded per draft.
    c_on, calls_on = make_critic("SPAN: and Tuesdays\nWHY: ABSENT_FROM_MATERIAL")
    r_on = Runner(
        gates=gates,
        pack={"traits/T10.md": "# T10\n\njudge size\n"},
        trait_names={"T10": "judges the exact size"},
        writer=None,
        critic=c_on,
        out_dir=tmp_path / "on",
        post_refusal_span=True,
    )
    ref_on = r_on._critic_pass(slot, row, draft_fields)
    r_on._record_call(slot, 1, Completion(text="{}", prompt_tokens=1, completion_tokens=1, model="w", seconds=0.01), "VERDICT_MISMATCH", 0, Material(mat_text, {}))
    assert calls_on.count("refusal_span") == 1
    assert "span 'and Tuesdays': ABSENT_FROM_MATERIAL" in ref_on["suggestion"]
    draft_on = json.loads((tmp_path / "on" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert draft_on["post_refusal_span"] is True
    assert any(n.get("check") == "refusal_span" and n.get("span") == "and Tuesdays" for n in draft_on["critic"])

    # 3. On with bad (non-substring) span: falls back to today's exact suggestion string.
    c_bad, calls_bad = make_critic("SPAN: on Wednesdays\nWHY: ABSENT_FROM_MATERIAL")
    r_bad = Runner(
        gates=gates,
        pack={"traits/T10.md": "# T10\n\njudge size\n"},
        trait_names={"T10": "judges the exact size"},
        writer=None,
        critic=c_bad,
        out_dir=tmp_path / "bad",
        post_refusal_span=True,
    )
    ref_bad = r_bad._critic_pass(slot, row, draft_fields)
    assert calls_bad.count("refusal_span") == 1
    assert ref_bad["suggestion"] == ref_off["suggestion"]

    # 4. On when verdict agrees: refusal_span does not fire.
    c_agree, calls_agree = make_critic(
        "SPAN: and Tuesdays\nWHY: ABSENT_FROM_MATERIAL",
        verdict_reply="VERDICT: supported\nDECIDING: The front desk is open on Mondays from 09:00 to 17:00.\nESCALATE: no",
    )
    r_agree = Runner(
        gates=gates,
        pack={"traits/T10.md": "# T10\n\njudge size\n"},
        trait_names={"T10": "judges the exact size"},
        writer=None,
        critic=c_agree,
        out_dir=tmp_path / "agree",
        post_refusal_span=True,
    )
    assert r_agree._critic_pass(slot, row, draft_fields) is None
    assert "refusal_span" not in calls_agree

    # 5. MockCritic span branch and CLI --post-refusal-span recording.
    from role_set.cli import main as cli_main
    from role_set.mock import MockCritic
    from role_set.pack import PackManifest, pack_hash

    mc = MockCritic()
    mc_res = mc(msgs)
    assert critic_mod.parse_refusal_span(mc_res.text, stmt) == ("The", "ABSENT_FROM_MATERIAL")

    r_on._note_refusal(slot, 1, ref_on, draft="{}")
    refusal_row = json.loads((tmp_path / "on" / "refusals.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert refusal_row["refusal_span"]["span"] == "and Tuesdays"

    plan_file = tmp_path / "plan_prs.json"
    assert cli_main(["plan", "--lessons", "T10", "--seed", "0", "--out", str(plan_file)]) == 0
    files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you judge the exact size\n\nCheck the exact size.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    pack_dir = tmp_path / "pack_prs"
    for name, text in files.items():
        (pack_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    out_prs = tmp_path / "out_prs"
    assert (
        cli_main(
            [
                "write",
                "--plan",
                str(plan_file),
                "--pack",
                str(pack_dir),
                "--out",
                str(out_prs),
                "--mock",
                "--post-refusal-span",
            ]
        )
        == 0
    )
    prs_rows = [json.loads(x) for x in (out_prs / "mock" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert prs_rows and all(r.get("post_refusal_span") is True for r in prs_rows)


def test_all_default_settings_match_4033ebd_writer_and_critic_requests_over_full_generated_plan(tmp_path):
    """With all settings at default (side_pairing='default', domain_scope='default', cannot_tell_construction='default',
    post_refusal_span=False, verdict_prompt='old'), every writer and critic request over a full generated plan equals
    4033ebd's byte-for-byte."""
    import hashlib
    from pathlib import Path

    from role_set import callcache
    from role_set.cli import main as cli_main
    from role_set.pack import PackManifest, pack_hash
    from role_set.write import WriteConfig, run

    root = Path(__file__).resolve().parents[1]
    naics = root / "roles" / "domains" / "naics-2022.json"
    plan_path = tmp_path / "plan.json"
    assert cli_main(["plan", "--lessons", "T10", "--buffers", "--seed", "0", "--domains", str(naics), "--out", str(plan_path)]) == 0
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "side_pairing" not in plan_data
    assert len(plan_data["slots"]) == 60
    assert all("side_pairing" not in s for s in plan_data["slots"])

    files = {
        "role.md": "# L\n\nr\n",
        "traits/T10.md": "## 10. As a verifier, you read size words strictly\n\nExactly, at least and at most mean what they say.\n",
        "kinds.md": "## 2. What to deliver\n\nk\n",
        "amendments/1.md": "## Amendment 1\n\nq\n",
    }
    pack_dir = tmp_path / "pack"
    for name, text in files.items():
        (pack_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({"id": "T01-A-01", "lesson": "T01", "kind": "A", "split": "train", "turns": [{"user": "u", "thinking": "t", "reply": "r"}]}) + "\n",
        encoding="utf-8",
    )

    res = run(
        WriteConfig(
            plan=plan_path,
            pack=pack_dir,
            out=tmp_path / "out_default",
            mock=True,
            wave="inline",
            training=(train_file,),
        )
    )
    assert res["accepted"] == 60 and res["batch_check"] == "clean"
    assert "cannot_tell_construction" not in res["calls"]
    assert "domain_scope" not in res["calls"]

    journal_path = tmp_path / "out_default" / "mock" / callcache.JOURNAL_FILE
    rows = [json.loads(x) for x in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 349
    writer_rows = [r for r in rows if r["role"] == "writer"]
    critic_rows = [r for r in rows if r["role"] == "critic"]
    assert len(writer_rows) == 91
    assert len(critic_rows) == 258
    assert all(r.get("cannot_tell_construction") == "default" for r in rows)
    assert all(r.get("domain_scope") == "default" for r in rows)
    assert not any(r.get("check") == "refusal_span" for r in rows)

    items = [
        {
            "role": r["role"],
            "slot": r["slot"],
            "attempt": r["attempt"],
            "check": r.get("check"),
            "key": r["key"],
            "messages_sha256": r["messages_sha256"],
        }
        for r in rows
    ]
    all_digest = hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    w_digest = hashlib.sha256(
        json.dumps([x for x in items if x["role"] == "writer"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    c_digest = hashlib.sha256(
        json.dumps([x for x in items if x["role"] == "critic"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    # Exact 4033ebd request digests across all 60 slots (91 writer requests, 258 critic requests).
    assert all_digest == "d6bb48008ff06ed9764d932c57bf61b57684429df890a9dbccc36b5aa4c4249a"
    assert w_digest == "39dd39229aed65464bf0fdca1ba9c25b6f82fe167dbbf36586cf8e693a860c56"
    assert c_digest == "9d28acf09c1d6a368cdaf40411aaa23a58cc3930b0e7e64a81208ced093a290d"


def test_callcache_and_write_guard_side_pairing_and_post_refusal_span_while_keeping_default_bytes_identical(tmp_path):
    """FIX 3: request_key, callcache.load, CallJournal, and write.run include side_pairing and post_refusal_span
    only when non-default, keeping 4033ebd default keys and journal lines byte-identical while refusing cross-setting replay."""
    from role_set import callcache
    from role_set.guards import GuardError
    from role_set.pack import PackManifest, pack_hash
    from role_set.plan import Slot
    from role_set.transport import Completion
    from role_set.write import WriteConfig, WriteRefused, run

    msgs = ([{"role": "user", "content": "hi"}],)
    k_base = callcache.request_key("writer", msgs, {})
    assert callcache.request_key("writer", msgs, {}, side_pairing=None, post_refusal_span=None) == k_base
    assert callcache.request_key("writer", msgs, {}, side_pairing="default", post_refusal_span=False) == k_base
    k_vs = callcache.request_key("writer", msgs, {}, side_pairing="verdict_safe")
    k_prs = callcache.request_key("writer", msgs, {}, post_refusal_span=True)
    assert len({k_base, k_vs, k_prs}) == 3

    # Default journal omits side_pairing and post_refusal_span keys so 4033ebd lines stay byte-identical.
    j_def_path = tmp_path / "j_def.jsonl"
    j_def = callcache.CallJournal(j_def_path, side_pairing="default", post_refusal_span=False)
    w_def = j_def.wrap("writer", lambda m, **kw: Completion("ok", 1, 1, "m", 0.01))
    w_def([{"role": "user", "content": "hi"}], slot="T10-A-01", attempt=0)
    def_line = json.loads(j_def_path.read_text(encoding="utf-8").splitlines()[0])
    assert "side_pairing" not in def_line
    assert "post_refusal_span" not in def_line
    assert "side_pairing" not in j_def.summary()
    assert "post_refusal_span" not in j_def.summary()

    # Loading or appending to a default journal under non-default side_pairing or post_refusal_span is refused.
    assert callcache.load(j_def_path, expected_side_pairing="default", expected_post_refusal_span=False)
    with pytest.raises(GuardError, match=r"side_pairing='default'.*side_pairing='verdict_safe'"):
        callcache.load(j_def_path, expected_side_pairing="verdict_safe")
    with pytest.raises(GuardError, match=r"post_refusal_span=False.*post_refusal_span=True"):
        callcache.load(j_def_path, expected_post_refusal_span=True)
    with pytest.raises(GuardError, match=r"side_pairing='default'.*side_pairing='verdict_safe'"):
        callcache.CallJournal(j_def_path, side_pairing="verdict_safe")
    with pytest.raises(GuardError, match=r"post_refusal_span=False.*post_refusal_span=True"):
        callcache.CallJournal(j_def_path, post_refusal_span=True)

    # Non-default journal records side_pairing and post_refusal_span and refuses default load/append.
    j_nondef_path = tmp_path / "j_nondef.jsonl"
    j_nondef = callcache.CallJournal(j_nondef_path, side_pairing="verdict_safe", post_refusal_span=True)
    w_nondef = j_nondef.wrap("writer", lambda m, **kw: Completion("ok", 1, 1, "m", 0.01))
    w_nondef([{"role": "user", "content": "hi"}], slot="T10-A-01", attempt=0)
    nondef_line = json.loads(j_nondef_path.read_text(encoding="utf-8").splitlines()[0])
    assert nondef_line["side_pairing"] == "verdict_safe"
    assert nondef_line["post_refusal_span"] is True
    assert j_nondef.summary()["side_pairing"] == "verdict_safe"
    assert j_nondef.summary()["post_refusal_span"] is True
    with pytest.raises(GuardError, match=r"side_pairing='verdict_safe'.*side_pairing='default'"):
        callcache.load(j_nondef_path, expected_side_pairing="default")
    with pytest.raises(GuardError, match=r"post_refusal_span=True.*post_refusal_span=False"):
        callcache.load(j_nondef_path, expected_post_refusal_span=False)

    # End-to-end via write.run: replaying a default journal under side_pairing='verdict_safe' or post_refusal_span=True is refused.
    slot_c = Slot(
        id="T10-C-01",
        lesson="T10",
        side="judges the exact size",
        kind="C",
        tier=1,
        verdict="supported",
        domain={"code": "522110", "title": "Commercial Banking"},
    )
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "role": "verifier",
                "seed": 0,
                "pilot": True,
                "sides": {"T10": ["judges the exact size", "an equivalent wording still matches"]},
                "figures": [],
                "slots": [slot_c.as_dict()],
            }
        ),
        encoding="utf-8",
    )
    files = {
        "role.md": "# L\n\nr\n",
        "traits/T10.md": "## 10. As a verifier, you read size words strictly\n\nExactly, at least and at most mean what they say.\n",
        "kinds.md": "## 2. What to deliver\n\nk\n",
        "amendments/1.md": "## Amendment 1\n\nq\n",
    }
    pack_dir = tmp_path / "pack"
    for name, text in files.items():
        (pack_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    out_def = tmp_path / "out_def"
    run(WriteConfig(plan=plan_file, pack=pack_dir, out=out_def, mock=True))
    def_journal = out_def / "mock" / callcache.JOURNAL_FILE

    with pytest.raises(WriteRefused, match=r"side_pairing='default'.*side_pairing='verdict_safe'"):
        run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "out_vs", mock=True, side_pairing="verdict_safe", reuse_calls=def_journal))
    with pytest.raises(WriteRefused, match=r"post_refusal_span=False.*post_refusal_span=True"):
        run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "out_prs", mock=True, post_refusal_span=True, reuse_calls=def_journal))







