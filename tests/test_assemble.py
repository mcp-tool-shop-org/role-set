import pytest

from role_set.assemble import AssemblyError, Material, assemble, slot_flaw, three_lines, trait_name, verbatim_lines
from role_set.plan import plan_lesson

SIDES = ("quotes the source", "stops at the source")
NOTICE = Material("Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n", {"invented": "notice, written for this item"})
REAL = Material("## [1.2.0] - 2026-01-01\n\n### Added\n- The `--dry-run` flag no longer writes the lock file.\n",
                {"repo": "example/repo", "commit": "abc", "path": "CHANGELOG.md", "lines": "L1-L4", "licence": "Apache-2.0"})


def _slots():
    return {s.kind + ("v" if s.kind == "E" and s.e_type == "verdict" else ""): s for s in reversed(plan_lesson("T02", SIDES, seed=1))}


def test_trait_name_rules():
    assert trait_name("## 1. As a verifier, you are skeptical") == "skeptical"
    assert trait_name("## 2. As a verifier, you go to the source") == "goes to the source"
    assert trait_name("## 6. As a verifier, you break things down") == "breaks things down"
    assert trait_name("### 14. As a verifier, you don't fill gaps") == "doesn't fill gaps"
    with pytest.raises(AssemblyError):
        trait_name("not a heading")


def test_three_lines_and_verbatim():
    assert three_lines("cannot_tell", [], "no").endswith("DECIDING: NONE\nESCALATE: no")
    assert "a | b" in three_lines("supported", ["a", "b"], "yes (medical advice)")
    with pytest.raises(AssemblyError, match="exactly when"):
        three_lines("cannot_tell", ["a"], "no")
    with pytest.raises(AssemblyError, match="four"):
        three_lines("supported", list("abcde"), "no")
    with pytest.raises(AssemblyError, match="verdict"):
        three_lines("maybe", [], "no")
    assert verbatim_lines(["The desk is  open on Sundays from 10:00 to 14:00."], NOTICE.text) == ["The desk is  open on Sundays from 10:00 to 14:00."]
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["The desk is shut."], NOTICE.text)
    with pytest.raises(AssemblyError, match="non-empty"):
        verbatim_lines([""], NOTICE.text)


def test_assemble_a_and_b():
    s = _slots()
    a = assemble(s["A"], "goes to the source", {"passage": "P one. P two.", "question": "Q?", "reply": "I quote it.", "key_points": ["x", "y"]})
    assert a["kind"] == "A" and a["content_kind"] == "teaching" and a["turns"][0]["user"].startswith("P one.")
    with pytest.raises(AssemblyError, match="key_points"):
        assemble(s["A"], "t", {"passage": "P.", "question": "Q?", "reply": "R", "key_points": []})
    b = assemble(s["B"], "goes to the source", {"puzzle": "Three: x, y, z. \nQuestion: which?", "thinking": "1. x\n2. y", "reply": "Answer: y", "answer": "y"})
    assert b["answer"] == "y" and b["structure"] == s["B"].structure and ("non-verification" in b["tags"]) == (not s["B"].verification)
    with pytest.raises(AssemblyError, match="Question"):
        assemble(s["B"], "t", {"puzzle": "no label", "thinking": "t", "reply": "r", "answer": "a"})
    with pytest.raises(AssemblyError, match="left these fields empty"):
        assemble(s["B"], "t", {"puzzle": "Question: ?", "thinking": "", "reply": "r", "answer": "a"})


def test_assemble_c_with_flaw_and_without():
    s = _slots()["C"]
    s.verdict, s.pair = "unsupported", "t02-pair-1"
    fields = {"framing": "Check the statement against the page.", "statement": "The desk is closed on Sundays.", "checker_trace": "They skimmed.",
              "deciding": ["The desk is open on Sundays from 10:00 to 14:00."], "thinking": "The claim asserts closure.", "reply": "They guessed.", "planted_flaw": "guess", "checker_verdict": "supported"}
    row = assemble(s, "goes to the source", fields, NOTICE)
    assert row["planted_flaw"] == "guess" and row["checker_verdict"] == "supported" and row["pair"] == "t02-pair-1"
    assert row["turns"][0]["reply"].endswith("VERDICT: unsupported\nDECIDING: The desk is open on Sundays from 10:00 to 14:00.\nESCALATE: no")
    assert "Checker verdict: supported" in row["turns"][0]["user"] and row["invented"] is True
    right = assemble(s, "t", {**fields, "planted_flaw": "none"}, NOTICE)
    assert right["checker_verdict"] == "unsupported"
    with pytest.raises(AssemblyError, match="planted_flaw"):
        assemble(s, "t", {**fields, "planted_flaw": "lazy"}, NOTICE)
    with pytest.raises(AssemblyError, match="needs material"):
        assemble(s, "t", fields)
    s.verdict = "cannot_tell"
    silent = assemble(s, "t", {**fields, "deciding": []}, Material("Desk\nThe top drawer holds pens.\n", {"invented": "notice"}))
    assert silent["deciding"] == [] and "DECIDING: NONE" in silent["turns"][0]["reply"]


def test_assemble_d_single_two_turn_stands_and_changes():
    s = _slots()["D"]
    s.verdict, s.two_turn, s.escalate = "supported", None, "no"
    base = {"framing": "A notice.", "statement": "The desk is open on Sundays.", "deciding": ["The desk is open on Sundays from 10:00 to 14:00."], "thinking": "T", "reply": "R"}
    one = assemble(s, "t", base, NOTICE)
    assert len(one["turns"]) == 1 and one["verdict"] == "supported" and "turn_keys" not in one

    s.two_turn = "stands"
    stands = assemble(s, "t", {**base, "turn2_user": "A manager insists it is shut.", "turn2_thinking": "No new line.", "turn2_reply": "It stands."}, NOTICE)
    assert stands["turn_keys"]["1"]["verdict"] == "supported" and stands["turns"][1]["reply"].endswith("ESCALATE: no") and "pressure" in stands["tags"]
    with pytest.raises(AssemblyError, match="no new material"):
        assemble(s, "t", {**base, "turn2_user": "x", "turn2_thinking": "t", "turn2_reply": "r", "turn2_new_material": ["new"]}, NOTICE)

    s.two_turn, s.verdict = "changes", "cannot_tell"
    silent = Material("Desk\nThe top drawer holds pens.\n", {"invented": "notice"})
    fields = {**base, "deciding": [], "turn2_user": "A line is pinned under it: The desk is open on Sundays from 10:00 to 14:00.",
              "turn2_new_material": ["The desk is open on Sundays from 10:00 to 14:00."], "turn2_verdict": "supported",
              "turn2_deciding": ["The desk is open on Sundays from 10:00 to 14:00."], "turn2_thinking": "What changed is the new line.", "turn2_reply": "Now supported."}
    changes = assemble(s, "t", fields, silent)
    assert changes["verdict"] == "supported" and changes["turn_keys"]["1"]["verdict"] == "cannot_tell" and "correction" in changes["tags"]
    assert changes["turns"][1]["new_material"] == fields["turn2_new_material"]
    with pytest.raises(AssemblyError, match="different turn2_verdict"):
        assemble(s, "t", {**fields, "turn2_verdict": "cannot_tell"}, silent)
    with pytest.raises(AssemblyError, match="needs turn2_new_material"):
        assemble(s, "t", {**fields, "turn2_new_material": []}, silent)
    with pytest.raises(AssemblyError, match="must appear in turn2_user"):
        assemble(s, "t", {**fields, "turn2_user": "Someone says something else."}, silent)
    with pytest.raises(AssemblyError, match="quote the new line"):
        assemble(s, "t", {**fields, "turn2_deciding": ["Desk"]}, silent)


def test_assemble_e_written_and_verdict():
    s = _slots()
    w = s["E"]
    closed = {"variant_question": "Which words go in the answer?", "variant_answer": "C",
              "variant_options": ["A summary of the page", "What the reader remembers", "The deciding line, quoted exactly"]}
    row = assemble(w, "t", {"question": "What do you quote?", "reply": "The line.", "key_points": ["the deciding line, quoted exactly"], **closed})
    assert row["quiz"]["type"] == "written" and row["variant"]["key"] == {"answer": "C"}
    assert row["variant"]["user"].splitlines()[2:5] == ["A. A summary of the page", "B. What the reader remembers", "C. The deciding line, quoted exactly"]
    assert row["variant"]["user"].rstrip().endswith("ANSWER: <letter>")
    with pytest.raises(AssemblyError, match="new words"):
        assemble(w, "t", {"question": "Which words go in the answer?", "reply": "r", "key_points": ["k"], **closed})
    with pytest.raises(AssemblyError, match="key_points"):
        assemble(w, "t", {"question": "Q?", "reply": "r", "key_points": [], **closed})
    v = s["Ev"]
    v.verdict = "supported"
    row = assemble(v, "t", {"framing": "A changelog.", "statement": "The dry-run flag stopped writing the lock file in 1.2.0.",
                            "deciding": ["## [1.2.0] - 2026-01-01", "- The `--dry-run` flag no longer writes the lock file."], "thinking": "T", "reply": "R", "variant_user": "Re-asked?"}, REAL)
    assert row["quiz"]["type"] == "verdict" and row["key_basis"] == "source" and row["invented"] is False
    assert row["turns"][0]["reply"].endswith("DECIDING: ## [1.2.0] - 2026-01-01 | - The `--dry-run` flag no longer writes the lock file.\nESCALATE: no")


def test_closed_form_variants_key_exactly_one_option_that_carries_a_key_point():
    w = _slots()["E"]
    base = {"question": "What do you quote?", "reply": "The line.", "key_points": ["the deciding line, quoted exactly"],
            "variant_question": "Which words go in the answer?"}
    opts = ["A summary of the page", "What the reader remembers", "The deciding line, quoted exactly"]
    with pytest.raises(AssemblyError, match="keyed option must carry"):
        assemble(w, "t", {**base, "variant_options": opts, "variant_answer": "A"})
    with pytest.raises(AssemblyError, match="must be one of A, B, C"):
        assemble(w, "t", {**base, "variant_options": opts, "variant_answer": "D"})
    with pytest.raises(AssemblyError, match="3 or 4"):
        assemble(w, "t", {**base, "variant_options": opts[:2], "variant_answer": "B"})
    with pytest.raises(AssemblyError, match="the same"):
        assemble(w, "t", {**base, "variant_options": [opts[0], opts[0].upper(), opts[2]], "variant_answer": "C"})
    # A one-word distractor inside a key point also "carries" it under the gates' rule (R&D on aspire-si #91).
    with pytest.raises(AssemblyError, match="also carry a key point"):
        assemble(w, "t", {**base, "variant_options": ["quoted", opts[1], opts[2]], "variant_answer": "C"})
    assert assemble(w, "t", {**base, "variant_options": opts, "variant_answer": "c"})["variant"]["key"] == {"answer": "C"}


def test_slot_flaw_alternates():
    s = _slots()["C"]
    seen = set()
    for n in range(1, 9):
        s.id = f"T02-C-{n:02d}"
        seen.add(slot_flaw(s))
    assert "none" in seen and len(seen) >= 4
