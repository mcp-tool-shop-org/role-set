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



