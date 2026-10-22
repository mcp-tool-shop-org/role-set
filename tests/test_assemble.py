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
    closed = {"variant_question": "Which words go in the answer?", "variant_correct": "The deciding line, quoted exactly",
              "variant_distractors": ["A summary of the page", "What the reader remembers"]}
    row = assemble(w, "t", {"question": "What do you quote?", "reply": "The line.", "key_points": ["the deciding line, quoted exactly"], **closed})
    key = row["variant"]["key"]["answer"]
    shown = row["variant"]["user"].splitlines()[2:5]
    assert row["quiz"]["type"] == "written" and sorted(x[:2] for x in shown) == ["A.", "B.", "C."]
    assert next(x for x in shown if x.startswith(key + ".")) == f"{key}. The deciding line, quoted exactly"
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
    right, wrong = "The deciding line, quoted exactly", ["A summary of the page", "What the reader remembers"]
    with pytest.raises(AssemblyError, match="2 or 3"):
        assemble(w, "t", {**base, "variant_correct": right, "variant_distractors": wrong[:1]})
    with pytest.raises(AssemblyError, match="variant_correct must be"):
        assemble(w, "t", {**base, "variant_correct": "", "variant_distractors": wrong})
    with pytest.raises(AssemblyError, match="the same"):
        assemble(w, "t", {**base, "variant_correct": right, "variant_distractors": [wrong[0], wrong[0].upper()]})


def test_assembly_leaves_the_carry_rule_to_the_gates():
    # Whether the keyed option carries a key point (verbatim, paraphrase or either) is the gates' closed-form rule with
    # the run's settings (aspire-si-gates 1.4.0); assembly keeps no copy, so a paraphrased keyed option assembles.
    w = _slots()["E"]
    fields = {"question": "What do you quote?", "reply": "The deciding line, word for word.",
              "key_points": ["the deciding line, quoted exactly"], "variant_question": "Which words go in the answer?",
              "variant_correct": "The line that decides it, copied exactly.",
              "variant_distractors": ["A summary of the page.", "What the reader remembers."]}
    row = assemble(w, "t", fields)
    assert "The line that decides it, copied exactly." in row["variant"]["user"]


def test_a_writers_own_letter_labels_are_stripped_before_lettering():
    from role_set.assemble import closed_form_variant

    v = closed_form_variant("Which words?", "C. The deciding line, quoted exactly", ["A) A summary of the page", "(b) What the reader remembers"],
                            seed="T07-E-01")
    shown = v["user"].splitlines()[2:5]
    assert [line[:3] for line in shown] == ["A. ", "B. ", "C. "]
    assert {line[3:] for line in shown} == {"The deciding line, quoted exactly", "A summary of the page", "What the reader remembers"}


def test_the_keyed_letter_is_shuffled_per_slot_deterministically_and_spread_across_slots():
    # R&D on #26: the writer's position bias can't decide the letter; the slot id seeds the shuffle.
    from role_set.assemble import closed_form_variant

    right, wrong = "The deciding line, quoted exactly", ["A summary of the page", "What the reader remembers", "The heading above it"]
    same = [closed_form_variant("Which words?", right, wrong, seed="T07-E-01")["key"]["answer"] for _ in range(3)]
    assert len(set(same)) == 1
    letters = [closed_form_variant("Which words?", right, wrong, seed=f"T07-E-{i:02d}")["key"]["answer"] for i in range(1, 41)]
    counts = {x: letters.count(x) for x in "ABCD"}
    assert all(counts[x] >= 4 for x in "ABCD"), counts
    v = closed_form_variant("Which words?", right, wrong, seed="T07-E-01")
    key = v["key"]["answer"]
    assert f"{key}. The deciding line, quoted exactly" in v["user"].splitlines()


def test_slot_flaw_alternates():
    s = _slots()["C"]
    seen = set()
    for n in range(1, 9):
        s.id = f"T02-C-{n:02d}"
        seen.add(slot_flaw(s))
    assert "none" in seen and len(seen) >= 4


def test_verbatim_repair_backticks_bold_and_links_and_gates_check_item():
    from role_set.gates import load_gates

    gates = load_gates()
    s = _slots()["D"]
    s.verdict, s.two_turn, s.escalate, s.real_source, s.content_kind = "supported", None, "no", True, "changelog"

    # 1. Backtick-dropping quote of a changelog bullet is repaired and passes gates.check_item.
    repairs: list[dict] = []
    fields = {
        "framing": "Check the release note below against the statement.",
        "statement": "Version 1.2.0 stopped the dry-run flag from writing the lock file.",
        "deciding": ["## [1.2.0] - 2026-01-01", "- The --dry-run flag no longer writes the lock file."],
        "thinking": "Line 1 gives version 1.2.0 and line 4 states that the `--dry-run` flag no longer writes the lock file, which directly settles the claim.",
        "reply": "The changelog entry for 1.2.0 states that `--dry-run` no longer writes the lock file.",
    }
    row = assemble(s, "goes to the source", fields, REAL, repairs=repairs)
    exact_bullet = "- The `--dry-run` flag no longer writes the lock file."
    assert row["deciding"] == ["## [1.2.0] - 2026-01-01", exact_bullet]
    assert f"DECIDING: ## [1.2.0] - 2026-01-01 | {exact_bullet}" in row["turns"][0]["reply"]
    assert repairs == [{"from": "- The --dry-run flag no longer writes the lock file.", "to": exact_bullet}]
    gates.check_item(row)

    # 2. Same for ** / __ emphasis and [text](url) links.
    marked_mat = Material(
        "## [2.0.0] - 2026-02-01\n"
        "- **Breaking**: removed the __legacy__ v1 config loader.\n"
        "- Added [front-door](https://example.com/docs/front-door) checks for repository docs.\n",
        {"repo": "example/repo", "commit": "def4567", "path": "CHANGELOG.md", "lines": "L1-L3", "licence": "MIT"},
    )
    repairs_bold: list[dict] = []
    out_bold = verbatim_lines(
        [
            "- Breaking: removed the legacy v1 config loader.",
            "- Added front-door checks for repository docs.",
        ],
        marked_mat.text,
        repairs=repairs_bold,
    )
    assert list(out_bold) == [
        "- **Breaking**: removed the __legacy__ v1 config loader.",
        "- Added [front-door](https://example.com/docs/front-door) checks for repository docs.",
    ]
    assert len(repairs_bold) == 2
    row_marked = assemble(
        s,
        "goes to the source",
        {
            **fields,
            "statement": "Version 2.0.0 removed the legacy v1 config loader and added front-door checks.",
            "deciding": [
                "## [2.0.0] - 2026-02-01",
                "- Breaking: removed the legacy v1 config loader.",
                "- Added front-door checks for repository docs.",
            ],
            "thinking": "Lines 2 and 3 under 2.0.0 state that the legacy v1 config loader was removed and front-door checks were added.",
            "reply": "Both parts of the claim are stated in the 2.0.0 changelog bullets.",
        },
        marked_mat,
    )
    gates.check_item(row_marked)

    # 3. Substring (>= 20 chars) of a single non-table line with marks dropped is repaired to the exact span,
    # while a table row substring is not matched by the fallback.
    sub_repairs: list[dict] = []
    sub_out = verbatim_lines(
        ["the --dry-run flag no longer writes the lock file."],
        "- Note: the `--dry-run` flag no longer writes the lock file.\n",
        repairs=sub_repairs,
    )
    assert list(sub_out) == ["the `--dry-run` flag no longer writes the lock file."]
    assert sub_repairs == [
        {
            "from": "the --dry-run flag no longer writes the lock file.",
            "to": "the `--dry-run` flag no longer writes the lock file.",
        }
    ]
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(
            ["the --dry-run flag no longer writes the lock file"],
            "| `--dry-run` | the `--dry-run` flag no longer writes the lock file |\n",
        )


def test_verbatim_repair_t12_c_02_window_15e_real_case():
    from role_set.gates import load_gates

    gates = load_gates()
    t12_mat = Material(
        "## [0.5.0] - 2026-02-23\n\n"
        "### Added\n"
        "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, `AGENTS.md`, and `llms.txt`.\n"
        "- `site-theme front-door check` — 15 checks across 3 categories (`FD.README.*`, `FD.AGENTS.*`, `FD.LLMS.*`) with hard/soft/off modes.\n"
        "- `site-theme front-door init` — generates starter `AGENTS.md` and `llms.txt` from repo metadata (`package.json`, `pyproject.toml`, `Cargo.toml`) without overwriting existing files (`--force` to overwrite).\n"
        "- `site-theme front-door badge` — emits shields.io endpoint JSON for README badges.\n",
        {
            "repo": "mcp-tool-shop-org/site-theme",
            "commit": "7fa785194a762656a1d619de2b3e7f43f2289b5b",
            "path": "CHANGELOG.md",
            "lines": "L38-L49",
            "licence": "MIT",
        },
    )
    dropped_quote = "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, AGENTS.md, and llms.txt."
    exact_line = "- **front-door** — a second pillar: the AI-native front-door verifier for a repo's README, `AGENTS.md`, and `llms.txt`."
    s = _slots()["C"]
    s.id = "T12-C-02"
    s.lesson = "T12"
    s.verdict = "supported"
    s.real_source = True
    s.content_kind = "changelog"
    repairs: list[dict] = []
    row = assemble(
        s,
        "consistent",
        {
            "framing": "Review the other reader's work against the 0.5.0 changelog.",
            "statement": "Version 0.5.0 added front-door for a repository's README, AGENTS.md, and llms.txt.",
            "checker_trace": "Line 4 under 0.5.0 lists front-door for a repo's README, AGENTS.md, and llms.txt, so the claim holds.",
            "checker_verdict": "supported",
            "planted_flaw": "none",
            "deciding": ["## [0.5.0] - 2026-02-23", dropped_quote],
            "thinking": "Line 1 gives version 0.5.0 and line 4 states that front-door covers a repo's README, `AGENTS.md`, and `llms.txt`; the other reader quoted that deciding line and reached supported without adding any outside assumption.",
            "reply": "The other reader's verdict and reasoning follow directly from the first Added bullet under 0.5.0.",
        },
        t12_mat,
        repairs=repairs,
    )
    assert row["deciding"] == ["## [0.5.0] - 2026-02-23", exact_line]
    assert f"DECIDING: ## [0.5.0] - 2026-02-23 | {exact_line}" in row["turns"][0]["reply"]
    assert repairs == [{"from": dropped_quote, "to": exact_line}]
    gates.check_item(row)


def test_verbatim_repair_refuses_ambiguous_or_unmatched_and_leaves_strict_untouched(tmp_path):
    import json

    from role_set.mock import MockCritic, mock_gates
    from role_set.runner import Runner, refusal_rule
    from role_set.transport import Completion

    # 1. Strict quote is untouched, with no repair recorded.
    strict_repairs: list[dict] = []
    strict_out = verbatim_lines(
        ["- The `--dry-run` flag no longer writes the lock file."],
        REAL.text,
        repairs=strict_repairs,
    )
    assert list(strict_out) == ["- The `--dry-run` flag no longer writes the lock file."]
    assert strict_repairs == [] and strict_out.repairs == []

    # 2. A quote that matches two material lines with marks ignored refuses.
    ambiguous_material = (
        "- Run `site-theme check` before publishing.\n"
        "- Run **site-theme check** before publishing.\n"
    )
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["- Run site-theme check before publishing."], ambiguous_material)

    # 3. A quote that matches none refuses, with the full line, closest material line, and divergence index.
    long_bad_quote = (
        "- The `--dry-run` flag no longer writes the lock file when invoked from the continuous integration runner."
    )
    with pytest.raises(AssemblyError) as exc_info:
        verbatim_lines([long_bad_quote], REAL.text)
    msg = str(exc_info.value)
    assert long_bad_quote in msg
    assert "- The `--dry-run` flag no longer writes the lock file." in msg
    assert "first differs at char 53" in msg
    assert refusal_rule({"code": "ASSEMBLY", "detail": msg}) == "ASSEMBLY: DECIDING line is not verbatim in the material (first differs at char N)"

    # 4. Runner records verbatim_repairs in drafts.jsonl and receipts.jsonl when repaired, and omits it when strict.
    pack = {
        "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
        "kinds.md": "kinds\n",
        "amendments/1.md": "## Amendment 1\n\nq\n",
    }
    traits = {"T02": "goes to the source"}
    s_repaired = _slots()["D"]
    s_repaired.id = "T02-D-01"
    s_repaired.frame = None
    s_repaired.verdict, s_repaired.two_turn, s_repaired.escalate, s_repaired.real_source, s_repaired.content_kind = (
        "supported",
        None,
        "no",
        True,
        "changelog",
    )
    s_strict = _slots()["D"]
    s_strict.id = "T02-D-02"
    s_strict.frame = None
    s_strict.verdict, s_strict.two_turn, s_strict.escalate, s_strict.real_source, s_strict.content_kind = (
        "supported",
        None,
        "no",
        True,
        "changelog",
    )
    mat_2 = Material(
        "## [1.3.0] - 2026-02-01\n\n### Added\n- The `--strict` flag validates every header.\n",
        {"repo": "example/repo", "commit": "def", "path": "CHANGELOG.md", "lines": "L1-L4", "licence": "Apache-2.0"},
    )

    def writer(messages, seed=0):
        text = messages[-1]["content"]
        if "In your own words: the trait's main point" in text:
            return Completion(
                text=json.dumps(
                    {
                        "main_point": "goes to the source: quote the source",
                        "counterweight": "stop when settled",
                        "deciding_case": "a changelog line",
                    }
                ),
                prompt_tokens=10,
                completion_tokens=10,
                model="mock",
                seconds=0.01,
            )
        if "[1.2.0]" in text:
            return Completion(
                text=json.dumps(
                    {
                        "framing": "Alpha check:",
                        "statement": "1.2.0 stopped dry-run from writing the lock file.",
                        "deciding": ["- The --dry-run flag no longer writes the lock file."],
                        "thinking": "Line 4 settles it.",
                        "reply": "Supported by line 4.",
                    }
                ),
                prompt_tokens=10,
                completion_tokens=10,
                model="mock",
                seconds=0.01,
            )
        return Completion(
            text=json.dumps(
                {
                    "framing": "Beta check:",
                    "statement": "1.3.0 added the strict flag.",
                    "deciding": ["- The `--strict` flag validates every header."],
                    "thinking": "Line 4 settles it.",
                    "reply": "Supported by line 4.",
                }
            ),
            prompt_tokens=10,
            completion_tokens=10,
            model="mock",
            seconds=0.01,
        )

    answers = {
        ("1.2.0 stopped dry-run from writing the lock file.", "## [1.2.0] - 2026-01-01"): (
            "supported",
            ["- The `--dry-run` flag no longer writes the lock file."],
        ),
        ("1.3.0 added the strict flag.", "## [1.3.0] - 2026-02-01"): (
            "supported",
            ["- The `--strict` flag validates every header."],
        ),
    }
    runner = Runner(
        gates=mock_gates(),
        pack=pack,
        trait_names=traits,
        writer=writer,
        critic=MockCritic(answers=answers, trait="T02"),
        out_dir=tmp_path / "out",
        excerpts={"changelog": [mat_2, REAL]},
    )
    outcomes = runner.run([s_repaired, s_strict])
    assert all(o.accepted for o in outcomes)
    drafts = [json.loads(line) for line in (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    receipts = [json.loads(line) for line in (tmp_path / "out" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    d_by_slot = {d["slot"]: d for d in drafts}
    r_by_slot = {r["slot"]["id"]: r for r in receipts}
    expected_repair = [
        {
            "from": "- The --dry-run flag no longer writes the lock file.",
            "to": "- The `--dry-run` flag no longer writes the lock file.",
        }
    ]
    assert d_by_slot["T02-D-01"]["verbatim_repairs"] == expected_repair
    assert r_by_slot["T02-D-01"]["verbatim_repairs"] == expected_repair
    assert "verbatim_repairs" not in d_by_slot["T02-D-02"]
    assert "verbatim_repairs" not in r_by_slot["T02-D-02"]


def test_mark_atom_re_and_resolve_verbatim_line_are_linear_on_adversarial_lines():
    import time

    from role_set.assemble import _MARK_ATOM_RE, resolve_verbatim_line

    mixed_unit = "**bold `code`_x_[link](https://example.com/a)__under `tick` *[unclosed "
    mixed_5000 = (mixed_unit * (5000 // len(mixed_unit) + 1))[:5000]
    assert len(mixed_5000) == 5000

    cases = [
        ("star_backtick", "**" + "`)" * 5000, ")" * 25),
        ("under_backtick", "__" + "`^" * 5000, "^" * 25),
        ("mixed_5000", mixed_5000, "bold code_x_linkunder tick *[unclosed"),
    ]
    for name, adv_line, probe_quote in cases:
        t0 = time.perf_counter()
        atoms = list(_MARK_ATOM_RE.finditer(adv_line))
        dt_finditer = time.perf_counter() - t0
        assert atoms
        assert dt_finditer < 0.5, f"{name} _MARK_ATOM_RE.finditer took {dt_finditer:.3f}s"

        t1 = time.perf_counter()
        _ = resolve_verbatim_line(probe_quote, adv_line)
        dt_resolve = time.perf_counter() - t1
        assert dt_resolve < 0.5, f"{name} resolve_verbatim_line took {dt_resolve:.3f}s"


def test_verbatim_repair_preserves_dunders_and_globs_in_code_record_and_fences():
    from role_set.assemble import _strip_inline_marks, resolve_verbatim_line
    from role_set.critic import compare_verdict

    # 1. A quote `def init(self):` is NOT repaired to `def __init__(self):` in code/record or inside a code fence.
    code_material = "class Loader:\n    def __init__(self):\n        self.ready = True\n"
    assert resolve_verbatim_line("def init(self):", "    def __init__(self):", content_kind="code") is None
    assert resolve_verbatim_line("def init(self):", "    def __init__(self):", content_kind="record") is None
    assert resolve_verbatim_line("def init(self):", "```python\n    def __init__(self):\n```") is None
    assert _strip_inline_marks("    def __init__(self):", in_code_fence=True) == "def __init__(self):"
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["def init(self):"], code_material, content_kind="code")

    fenced_doc = (
        "## Setup\n\n"
        "```python\n"
        "class Loader:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "```\n"
        "- Call `Loader()` after **setup**.\n"
    )
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["def init(self):"], fenced_doc, content_kind="readme")
    # Prose line outside the fence still repairs bold/backtick marks as usual.
    assert list(verbatim_lines(["- Call Loader() after setup."], fenced_doc, content_kind="readme")) == [
        "- Call `Loader()` after **setup**."
    ]
    # Backtick-stripped quote inside a code fence or code material still repairs backticks without stripping dunders.
    tilde_fenced = "~~~\n# run `pytest __init__.py` first\n~~~\n"
    assert list(verbatim_lines(["# run pytest __init__.py first"], tilde_fenced, content_kind="readme")) == [
        "# run `pytest __init__.py` first"
    ]
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["# run pytest init.py first"], tilde_fenced, content_kind="readme")

    # 2. A `docs/**/*.md` glob survives in record/code materials and inside code fences.
    assert (
        _strip_inline_marks("- include: `docs/**/*.md` in the bundle", content_kind="record")
        == "- include: docs/**/*.md in the bundle"
    )
    assert (
        _strip_inline_marks("glob = 'docs/**/*.md'", content_kind="code")
        == "glob = 'docs/**/*.md'"
    )
    glob_material = (
        "- include: `docs/**/*.md` in the bundle\n"
        "paths = \"docs/**/*.md\"\n"
    )
    assert list(verbatim_lines(["- include: docs/**/*.md in the bundle"], glob_material, content_kind="record")) == [
        "- include: `docs/**/*.md` in the bundle"
    ]
    assert list(verbatim_lines(['paths = "docs/**/*.md"'], glob_material, content_kind="record")) == [
        'paths = "docs/**/*.md"'
    ]
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(["- include: docs//*.md in the bundle"], glob_material, content_kind="record")
    with pytest.raises(AssemblyError, match="not verbatim"):
        verbatim_lines(['paths = "docs//*.md"'], glob_material, content_kind="record")

    # 3. End-to-end through `assemble` and `compare_verdict` with content_kind="code".
    from role_set.mock import mock_gates

    s_code = _slots()["D"]
    s_code.two_turn = None
    s_code.verdict = "supported"
    s_code.content_kind = "code"
    with pytest.raises(AssemblyError, match="not verbatim"):
        assemble(
            s_code,
            "goes to the source",
            {
                "framing": "Check loader init:",
                "statement": "Loader sets ready in init.",
                "deciding": ["def init(self):"],
                "thinking": "Line 2 defines __init__.",
                "reply": "Defined on line 2.",
            },
            Material(code_material, {"invented": "code"}),
        )
    cv = compare_verdict(
        mock_gates(),
        "VERDICT: supported\nDECIDING: def init(self):\nESCALATE: no",
        "supported",
        ["    def __init__(self):"],
        code_material,
        content_kind="code",
    )
    assert cv.repairs == []
    assert cv.critic_deciding == ["def init(self):"]



