"""The writer's prompts: the pack in slices, the slot, exemplars from another trait, and a short avoid-list.

The writer is asked for fields, as JSON, never for a whole row and never for the
material. Everything it must not be free to repeat is fixed in the slot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from role_set.frames import FRAMES
from role_set.assemble import FLAW_VERDICTS, SILENT_NEEDED, VERDICTS, slot_flaw
from role_set.critic import CANNOT_TELL_SILENT, FLAW_DEFINITIONS
from role_set.plan import Slot
from role_set.trait_steps import DEFAULTS_NOTE, trait_step

FIELDS = {
    "A": ["passage", "question", "reply", "key_points"],
    "B": ["puzzle", "model", "thinking", "reply", "answer"],
    "B_locked": ["puzzle", "thinking", "reply"],  # solve-first: the model and the answer are made in code (B3)
    "C": ["framing", "statement", "checker_trace", "checker_verdict", "deciding", "thinking", "reply"],
    "D": ["framing", "statement", "deciding", "thinking", "reply", "pressure", "injection",
          "turn2_user", "turn2_new_material", "turn2_verdict", "turn2_deciding", "turn2_thinking", "turn2_reply"],
    "E_written": ["question", "thinking", "reply", "key_points", "variant_question", "variant_correct", "variant_distractors"],
    "E_verdict": ["framing", "statement", "deciding", "thinking", "reply", "variant_user"],
}

HOUSE_RULES = (
    "Write every field in fresh words of your own. Never copy a sentence from the lesson, the rules or an "
    "exemplar. In kinds C and D and in verdict-type E, the user-facing text must not contain the word "
    "'verifier', any trait name, or the words tree, prune, falsifier, elimination, grid; the other party in a "
    "contrasting case is 'the other checker'. Kinds A, B and written E name the trait or the pattern by design. "
    "A prompt frames the task and never states the answer, the verdict, the flaw, or what the material "
    "contains. Quote material lines exactly, marks and all, whole table rows, and the version heading with "
    "its bullet. Thinking states what the claim asserts, the line found, the sweep for anything that "
    "qualifies it, and why this trait decided the step, in this item's own words, without a stock closer."
)


B_MODEL_RULE = (
    "Kind B, the model: every line of the puzzle before 'Question:' is numbered (1., 2., ...), setting lines included, "
    "and 'model' writes the same puzzle as data. model = {\"entities\": [names], \"attributes\": {name: {\"values\": "
    "[...], \"ordered\": true if the values have an order, \"distinct\": false only if values may repeat}}, "
    "\"facts\": [{\"line\": n, \"rule\": [form, ...]}], \"flavour\": [line numbers that state no fact, at most 2], "
    "\"asks\": {\"attribute\": a} or {\"attribute\": a, \"of\": entity} or {\"entity_with\": {a: value}}}. "
    "A reference is an entity name or {attribute: value} (the entity holding that value). The rule forms are: "
    "[\"is\"|\"is_not\", ref, attr, value], [\"in\"|\"not_in\", ref, attr, [values]], [\"same\"|\"different\", ref, ref], "
    "[\"before\"|\"after\"|\"adjacent\"|\"not_adjacent\", ref, ref, attr], [\"offset\", ref, ref, attr, k] (the second "
    "is k places after the first), [\"count\", attr, value, \"=\"|\"<=\"|\">=\", n], [\"not\", rule], [\"and\"|\"or\", "
    "rule, rule, ...], [\"if\", rule, rule]. Each rule encodes only its own line: it names only what that line names, a "
    "line with not or n't gets a negative form, and before or after keep the line's order. Every fact must be needed: a "
    "line that changes nothing is refused. Don't settle the answer with 'is' lines alone. If the puzzle says left, right, "
    "first or last, one line states the order ('seats 1 to 4, left to right'). The code solves the model and refuses "
    "the draft unless its answer is your key. The reply's last line is the answer alone: 'Answer: ...'."
)
# One worked example of clues encoded as rules (R&D and the Publisher, after rehearsal 3: the 8B wrote the solution
# into the model as facts, having never seen a clue encoded). Its domain is in no batch, the pool or the sealed set
# (leakage-checked 2026-10-11), and a test solves it, so the brief never shows a model the code would refuse.
B_EXAMPLE_PUZZLE = ("1. Three canoes, Ash, Briar and Cole, launch in turn 1, turn 2 and turn 3.\n"
                    "2. Ash does not launch in turn 1.\n"
                    "3. Briar launches after Ash.\n"
                    "Question: Which canoe launches in which turn?")
B_EXAMPLE_MODEL = {"entities": ["Ash", "Briar", "Cole"],
                   "attributes": {"turn": {"values": ["turn 1", "turn 2", "turn 3"], "ordered": True}},
                   "facts": [{"line": 2, "rule": ["is_not", "Ash", "turn", "turn 1"]},
                             {"line": 3, "rule": ["after", "Briar", "Ash", "turn"]}],
                   "flavour": [1], "asks": {"attribute": "turn"}}
B_EXAMPLE_KEY = "Ash=turn 2; Briar=turn 3; Cole=turn 1"
B_MODEL_EXAMPLE = (
    "Kind B, how a model encodes clues (an example only; never reuse its names or domain): the puzzle\n"
    + B_EXAMPLE_PUZZLE + "\nhas the model " + json.dumps(B_EXAMPLE_MODEL) + " and the key '" + B_EXAMPLE_KEY + "'. Line 1 "
    "declares the canoes and turns, so it is flavour. Line 2 is a negative line, so its rule is is_not. Line 3 says "
    "after, so its rule is after, with Briar first as the line names it. No rule states where a canoe is: the answer "
    "comes only by elimination (Ash is not in turn 1, and Briar is after Ash, so Ash is in turn 2 and Briar in turn 3). "
    "A model whose facts state the answer is refused. The puzzle states facts and a question only, never how to lay "
    "out the working."
)
# R&D on PR H: what each structure's key is, computed by the code.
B_STRUCTURE_NOTES = {
    "underdetermined_with_explicit_set": (
        "This puzzle is underdetermined: the facts leave two or more answers to one value (asks with \"of\" or "
        "\"entity_with\"), the Question asks what could be ('Which cups could hold the pen?'), and 'answer' lists every "
        "possible one, separated by commas. When a possible answer is a group "
        "of names, join each group with 'and' and separate the groups with semicolons ('red and blue; red and green')."),
    "stopping_problem": (
        "This is a stopping problem: the Question asks after which numbered line the answer is settled, and 'answer' is "
        "that line's number. Lines after it may follow, but each line up to it must be needed."),
    "when_to_stop": (
        "This is a when-to-stop case: either the Question asks after which numbered line the answer is settled ('answer' "
        "is that number), or model.claim holds the rule the Question asks about and 'answer' is its verdict."),
    "tree_with_pruning": (
        "This is a verification case: model.claim is the rule the Question asks about, every name in it appears in the "
        "Question line, and 'answer' is supported if the claim holds in every assignment that fits the facts, "
        "unsupported if in none, cannot_tell otherwise."),
    "falsifier_first": (
        "This is a verification case: model.claim is the rule the Question asks about, every name in it appears in the "
        "Question line, and 'answer' is supported if the claim holds in every assignment that fits the facts, "
        "unsupported if in none, cannot_tell otherwise."),
}


def load_pack(pack_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for p in sorted(pack_dir.rglob("*.md")):
        files[p.relative_to(pack_dir).as_posix()] = p.read_text(encoding="utf-8")
    return files


def kind_key(slot: Slot) -> str:
    if slot.kind == "E":
        return "E_written" if slot.e_type == "written" else "E_verdict"
    if slot.kind == "B" and slot.b_path == "solve_first":
        return "B_locked"
    return slot.kind


def required_fields(slot: Slot) -> list[str]:
    """The fields assembly refuses to find empty for this slot (assemble.py's _require and list checks)."""

    k = kind_key(slot)
    req = {
        "A": ["passage", "question", "reply", "key_points"],
        "B": ["puzzle", "model", "thinking", "reply", "answer"],
        "B_locked": ["puzzle", "thinking", "reply"],
        "C": ["framing", "statement", "checker_trace", "thinking", "reply"],
        "D": ["framing", "statement", "thinking", "reply"],
        "E_written": ["question", "reply", "key_points", "variant_question", "variant_correct", "variant_distractors"],
        "E_verdict": ["framing", "statement", "thinking", "reply", "variant_user"],
    }[k]
    if k in ("C", "D", "E_verdict") and slot.verdict != "cannot_tell":
        req = req + ["deciding"]
    if k == "D" and slot.two_turn:
        req = req + ["turn2_user", "turn2_thinking", "turn2_reply"]
        if slot.two_turn == "changes":
            req = req + ["turn2_new_material", "turn2_verdict", "turn2_deciding"]
    return req


def closed_form_carry_rule(closed_form: dict | None) -> str:
    """The right option's rule, worded from the run's closed-form settings (the gates' keyed carry, share and span).
    With no settings (a stand-in gates module, mock runs only) it names the rule without numbers."""

    if not closed_form:
        return "'variant_correct' is the right option's text and carries one of your key_points as the run's gates require."
    share = f"at least {round(closed_form['share'] * 100)}% of that key point's content words"
    span = (f"Neither the variant's question nor any option repeats a run of {closed_form['span']} words from this item's "
            "question, thinking or reply.")
    paraphrase = (f"says one of your key_points in its own words, keeping {share}. {span} Each distractor stays clearly further "
                  f"from every key point than the right option is, and none is a near-copy of one.")
    carry = closed_form["carry"]
    if carry == "verbatim":
        return ("'variant_correct' is the right option's text and carries one of your key_points word for word, inside a full "
                "sentence. No distractor may contain any key point or sit inside one.")
    if carry == "either":
        return ("'variant_correct' is the right option's text and either carries one of your key_points word for word or "
                + paraphrase + " No distractor may contain a key point.")
    return "'variant_correct' is the right option's text: it " + paraphrase


def hard_rules(slot: Slot, trait_words: list[str] | None = None, closed_form: dict | None = None) -> list[str]:
    """The gates' mechanical rules for this kind, stated before the first draft rather than after a refusal.
    The 2026-10-11 rehearsal lost every kind-A slot to two rules the writer was never told, and the 235B pilot the same
    day lost most of B and E to assembly rules the brief never stated (the Question: line, empty fields, DECIDING)."""

    rules: list[str] = ["These fields are required and must not be empty: " + ", ".join(required_fields(slot)) + "."]
    if slot.kind == "A":
        names = ", ".join(f"'{w}'" for w in (trait_words or []))
        rules.append("Kind A: the passage has at least two full sentences before the question, and it names the trait "
                     + (f"in one of these exact words: {names}" if names else "by its name as the lesson heading gives it")
                     + ", written with a straight apostrophe (don't, not don’t). Put the trait words inside a sentence of your own;"
                     " don't open the passage with 'As a verifier, you' followed by the trait words, which other items of this trait"
                     " already use. The reply begins in the first person (I, I'm, I've or My) and has no VERDICT line.")
        rules.append("Kind A is situational, never a request to restate the lesson: the passage sets a concrete case with "
                     "something to decide (a reviewer pushes, a person is about to act on a check, a claim comes back reworded), "
                     "and the question asks what you do and why. The reply decides it. It doesn't open with 'As a verifier'.")
    if slot.kind == "B" and slot.b_path == "solve_first":
        rules.append("Kind B: the thinking or the reply draws the pattern, as tree glyphs (├ └ │), a grid table of | rows, "
                     "or numbered elimination steps (1. 2. 3.).")
        rules.append("Kind B, locked puzzle: the puzzle and its answer are already made and solved. Write 'puzzle' as the "
                     "numbered lines the locked puzzle lists, each in your own words with its names and values exactly as "
                     "given, then the Question line. Never add, drop or change a fact. 'thinking' works through the lines in "
                     "order, cites each fact line as 'line N', and reaches the answer only in its last third. 'reply' "
                     "ends with 'Answer: ' and the fixed key.")
        if slot.structure in B_STRUCTURE_NOTES:
            rules.append(B_STRUCTURE_NOTES[slot.structure])
    elif slot.kind == "B":
        rules.append("Kind B: the thinking or the reply draws the pattern, as tree glyphs (├ └ │), a grid table of | rows, "
                     "or numbered elimination steps (1. 2. 3.).")
        rules.append("Kind B: the puzzle gives its facts and ends with a line that begins 'Question:'. The puzzle has exactly one "
                     "solution. 'answer' is that solution in closed form and nothing else, with no explanation: a verdict label "
                     "(supported, unsupported or cannot_tell), a single value, an ordered list separated by commas, or "
                     "assignments written 'name=value; name=value'. Every person, item or option in the answer is named "
                     "exactly as the puzzle names it.")
        # Pilot 2: five kind B drafts searched for their own puzzle's solution inside the thinking, found two, and ran
        # past the length limit. The puzzle is built from its answer, so the thinking only has to walk to it.
        rules.append("Kind B: build the puzzle backwards. Fix the solution first, then write facts that each rule out other "
                     "options until only that solution fits, and check that no second solution fits before you write the "
                     "JSON. The thinking walks the elimination to the answer in at most eight steps and never searches, "
                     "backtracks or reconsiders. If your facts leave two solutions, add a fact; don't reason about it.")
        # PR H (R&D's design): the puzzle is also written as data, and the code solves it. Only the solver decides.
        rules.append(B_MODEL_RULE)
        rules.append(B_MODEL_EXAMPLE)
        if slot.structure in B_STRUCTURE_NOTES:
            rules.append(B_STRUCTURE_NOTES[slot.structure])
    if slot.kind == "E" and slot.e_type == "written":
        # Handoff Amendment 5: written variants are closed-form, scored by the letter. How the right option carries a
        # key point is the gates' setting for this run (aspire-si-gates 1.4.0), so the brief states that setting.
        rules.append("Written E, closed-form variant: 'variant_question' asks the same point in new words and stands alone "
                     "(no material). " + closed_form_carry_rule(closed_form) + " 'variant_distractors' is a list of 2 or 3 "
                     "wrong options: plausible near-misses a reader who half-learned the trait would pick, never absurd and "
                     "never 'none of the above'. Build them from this item's own vocabulary or from a sibling trait's key "
                     "point, so familiar wording doesn't single out the right one, and don't build one by swapping one part "
                     "of the right answer. Keep all options similar in length and form, so the right one isn't the longest. "
                     "Every option is a full sentence that opens with a capital and ends with a full stop, all in the same "
                     "person. Give the options without letters; they are lettered and shuffled for you.")
    if slot.kind == "E" and slot.e_type == "verdict":
        rules.append("Verdict E: the variant is shown with the same material and re-asks the same statement in other words; "
                     "it never summarises the material or hints at the verdict.")
        rules.append("Verdict E: neither the prompt nor the variant says what the material does or doesn't contain, and neither "
                     "refers to another item or an earlier case; each stands alone.")
    if slot.kind in ("C", "D") or (slot.kind == "E" and slot.e_type == "verdict"):
        rules.append("'framing' introduces the task in a sentence or two and never contains the material or the statement, "
                     "which are added after it.")
        rules.append("'deciding' is an empty array exactly when the verdict is cannot_tell; otherwise it holds the verbatim "
                     "material lines. The statement uses no undated time words (yesterday, today, tomorrow, last week, "
                     "recently): name the date or leave time out. Each reply is prose only: never write VERDICT, DECIDING or "
                     "ESCALATE lines in it, because the three lines are added from the slot.")
    return rules


def slot_brief(slot: Slot, coverage_notes: list[str], trait_words: list[str] | None = None,
               closed_form: dict | None = None) -> str:
    """What this slot fixes, and what the batch still needs, in plain words."""

    lines = [f"Lesson {slot.lesson}, kind {slot.kind}, side: {slot.side}, tier {slot.tier}.",
             f"The item teaches its side, '{slot.side}': the reply shows it in this item's own words, and a reply that argues the trait's other side fails."]
    step = trait_step(slot.lesson)
    if step:
        lines.append(f"The step this trait decides, which this item is built around: {step}.")
        lines.append(DEFAULTS_NOTE)
    if slot.domain:
        what = "passage" if slot.kind == "A" else ("framing" if slot.kind in ("C", "D") else "material, the statement and the framing")
        lines.append(f"Domain: {slot.domain['title']} (NAICS {slot.domain['code']}). Set the {what} in this field of work, "
                     "with the people, documents and terms it really has; never a famous or implausible subject.")
    if slot.frame:
        lines.append(f"Frame: {slot.frame}. {FRAMES[slot.frame]} The words are yours: never a stock opener.")
    if slot.content_kind:
        lines.append(f"The material is a {slot.content_kind.replace('_', ' ')}; it is given below and must be used verbatim.")
    if slot.verdict:
        lines.append(f"The correct verdict is {slot.verdict}; ESCALATE is {slot.escalate}.")
        if slot.escalate.startswith("yes ("):
            domain = slot.escalate[5:].rstrip(")")
            lines.append(f"The user is about to act on a {domain} question. The reply says what the material shows and that the decision belongs with a qualified person.")
        if slot.verdict == "cannot_tell":
            lines.append(f"The material must be {CANNOT_TELL_SILENT}.")
        elif slot.kind in ("C", "D") and slot.verdict in ("supported", "unsupported"):
            lines.append("The statement names only people, products, places and terms that the material names.")
        if slot.multi_line:
            lines.append("The deciding evidence spans two to four lines of the material (a heading with its bullet, or a whole table row).")
    # The apply twins (aspire-si#137): assembly requires these fields, and no prompt asked for them, so every such apply
    # item dropped (found by the pilot's mock run, 2026-10-17).
    if slot.split == "apply" and slot.lesson in SILENT_NEEDED and slot.verdict != "cannot_tell":
        lines.append("Add silent_without_deciding: true if removing the deciding lines leaves the material silent on the "
                     "statement (neither showing it nor ruling it out), false otherwise. Its twin is this item with those "
                     "lines removed.")
    if slot.split == "apply" and slot.lesson == "T09":
        lines.append("Add order_free: true if reversing the order of the material's lines can't change the verdict, false "
                     "otherwise. Its twin is this item with the lines reversed.")
    if slot.split == "apply" and slot.lesson == "T10":
        lines.append("Add r10_twin: {\"statement\": \"...\", \"verdict\": \"...\"}, a strengthened or weakened claim on the "
                     f"same material whose correct verdict differs from {slot.verdict}.")
    if slot.kind == "C":
        # The pilot's writer was never told which flaw the slot plants, so its trace and the assembled verdict disagreed.
        flaw = slot_flaw(slot)
        if flaw == "none":
            lines.append(f"The other checker was right: checker_verdict is {slot.verdict}, and checker_trace is sound reasoning for it.")
        else:
            reach = [v for v in FLAW_VERDICTS.get(flaw, VERDICTS) if v != slot.verdict]
            lines.append(f"The other checker is wrong through one flaw, {flaw}: it {FLAW_DEFINITIONS[flaw]}. checker_verdict is "
                         f"{' or '.join(reach)}; checker_trace argues for that verdict, names no other, and shows the flaw.")
    if slot.pair:
        lines.append(f"This item is one half of contrast pair {slot.pair}: the same statement is checked against other material in its twin.")
    if slot.two_turn == "stands":
        lines.append("Two turns: the second brings only pressure or repetition, never a new line, and the verdict stands.")
    if slot.two_turn == "changes":
        lines.append("Two turns: the second brings genuinely new material that changes the verdict; the reply names exactly what changed and quotes the new line.")
    if slot.two_turn:
        lines.append("Leave 'pressure' empty: turn 1 holds no pushback and its reply answers none; the pushback is turn2_user alone.")
    if slot.structure:
        lines.append(f"The thinking pattern is: {slot.structure.replace('_', ' ')}. " + ("This is a verification case." if slot.verification else "This is not verification; it is a puzzle where the trait still decides a named step."))
    if slot.kind == "E":
        lines.append("Write a quiz item and a variant that re-asks the same point in new words." if slot.e_type == "written" else "Write a short verdict mini-case and a variant that re-asks it in new words.")
    lines += hard_rules(slot, trait_words, closed_form)
    if coverage_notes:
        lines.append("Still needed in this trait: " + "; ".join(coverage_notes) + ".")
    return "\n".join(lines)


def writer_messages(
    slot: Slot,
    pack: dict[str, str],
    *,
    material_text: str | None,
    exemplars: list[dict],
    avoid: list[str],
    facts: dict | None = None,
    coverage_notes: list[str] | None = None,
    trait_words: list[str] | None = None,
    closed_form: dict | None = None,
    locked_brief: str | None = None,
) -> list[dict]:
    trait = pack.get(f"traits/{slot.lesson}.md", "")
    kinds = pack.get("kinds.md", "")
    amendments = "\n\n".join(v for k, v in sorted(pack.items()) if k.startswith("amendments/"))
    fields = FIELDS[kind_key(slot)]
    system = (
        "You write one training item for a role-shaping curriculum. The role is a verifier. You are given the "
        "trait lesson, the required parts for this kind of item, the amendments, the fixed slot, and the material. "
        "Return a single flat JSON object with exactly these fields and no others: " + ", ".join(fields) + ". "
        "Never return a finished row: no 'kind', 'turns', 'answer' or 'quiz' wrapper unless it is one of these fields. "
        "The exemplars are finished rows, shown for form only; you write the fields they were built from. "
        "Lists are JSON arrays; 'deciding' and 'turn2_deciding' are arrays of verbatim material lines; "
        "'turn2_new_material' is an array of the new lines the second user turn brings. " + HOUSE_RULES
    )
    user_parts = [
        "## The trait lesson\n" + trait.strip(),
        "## Required parts for this kind\n" + kinds.strip(),
        "## Amendments\n" + amendments.strip(),
        "## This slot\n" + slot_brief(slot, coverage_notes or [], trait_words, closed_form),
    ]
    if locked_brief is not None:
        user_parts.append(locked_brief)
    if material_text is not None:
        user_parts.append("## The material (use verbatim; do not alter a character)\n" + material_text.rstrip("\n"))
    if facts:
        user_parts.append("## Computed facts you may state and must not recompute\n" + json.dumps(facts, ensure_ascii=False))
    if exemplars:
        shown = [{k: v for k, v in e.items() if k in ("kind", "turns", "key_points", "answer", "quiz", "variant")} for e in exemplars[:3]]
        user_parts.append("## Exemplars of this kind from a different trait (form only; never reuse their words)\n" + json.dumps(shown, ensure_ascii=False, indent=1))
    if avoid:
        user_parts.append("## Do not use these framings or phrases; they are spent\n" + "\n".join(f"- {a}" for a in avoid))
    taught = lesson_sentences(trait)
    if taught:
        # Rule 11, stated before the first draft: the rehearsal's writer copied the lesson's own sentence into every passage.
        user_parts.append("## The lesson's sentences: never reuse them, not even eight words in a row\n" + "\n".join(f"- {s}" for s in taught))
    user_parts.append("Return the JSON object now.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user_parts)}]


def material_messages(slot: Slot, needs: str) -> list[dict]:
    """When no real excerpt is assigned, the writer drafts the material first, alone, and the generator then fixes it."""

    kind = (slot.content_kind or "notice").replace("_", " ")
    return [
        {"role": "system", "content": "You write a short, realistic piece of everyday material for a verification exercise. Return plain text only, no commentary, no JSON."},
        {"role": "user", "content": f"Write a {kind}, 2 to 8 lines, invented, with no real names, paths, emails or brands. {needs} Do not include any instruction addressed to a reader or reviewer unless asked. Return only the text."},
    ]


def lesson_sentences(trait_text: str) -> list[str]:
    """The sentences of a trait lesson's teaching paragraph (after the heading, before the worked thinking): the
    prose a writer is most tempted to restate."""

    body: list[str] = []
    for line in trait_text.splitlines()[1:]:
        if line.startswith("**") or line.startswith("```"):
            break
        if line.strip():
            body.append(line.strip())
        elif body:
            break
    text = " ".join(body)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 8]


def warmup_messages(pack: dict[str, str], lesson: str) -> list[dict]:
    trait = pack.get(f"traits/{lesson}.md", "")
    return [
        {"role": "system", "content": "You are about to write training items for one trait of a verifier. First show that you have absorbed the lesson. Answer in your own words, briefly, as JSON with the fields main_point, counterweight, deciding_case."},
        {"role": "user", "content": "## The trait lesson\n" + trait.strip() + "\n\nIn your own words: the trait's main point; its counterweight (the opposite error it must also avoid); and one concrete case where this trait, and not general care, decides a step. Return the JSON object."},
    ]


def parse_json_object(text: str) -> dict:
    """The writer's reply must be one JSON object; a fenced block is tolerated, nothing else is."""

    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("the reply is not a JSON object")
    return obj
