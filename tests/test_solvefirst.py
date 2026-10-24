"""BUILD-SPEC B3a: solve-first kind B. The model and the key are made in code; the writer renders them."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import t10_replan_check  # noqa: E402
from role_set import domains, solvefirst  # noqa: E402
from role_set.plan import Slot, plan_batch  # noqa: E402
from role_set.puzzle import PuzzleError, check_puzzle, parse_model, solve  # noqa: E402

PACKS = solvefirst.load_packs()
SIDES = {"T02": ("quotes the source", "stops at the source"), "T03": ("exact", "not fussy")}


def test_the_packs_are_many_and_each_in_its_own_domain():
    assert len(PACKS) >= 42
    assert len({p["naics"] for p in PACKS.values()}) == len(PACKS)
    for p in PACKS.values():
        assert len(p["entities"]) == 6 and len(p["order"]["values"]) == 6 and all(len(k["values"]) == 6 for k in p["kinds"])
        # No value the closed-form answer can't hold, and none the solver reads as an order or a negation.
        for v in p["order"]["values"] + [x for k in p["kinds"] for x in k["values"]]:
            assert ":" not in v and "," not in v, (p["id"], v)


@pytest.mark.parametrize("structure", solvefirst.SOLVE_FIRST)
def test_every_generator_proves_its_puzzles(structure: str):
    for i, pid in enumerate(sorted(PACKS)[:10]):
        locked = solvefirst.generate(structure, PACKS[pid], 500 + i)
        got = solve(parse_model(locked.model), structure, "")
        assert got.answer == locked.key and not got.lookup
        assert locked.family == solvefirst.FAMILIES[structure] and locked.pack == pid and locked.tries >= 1
        # Every fact is needed: the solver refuses a line that changes nothing, so a fresh solve passing proves it.
        assert len(locked.model["facts"]) >= 2
        if structure == "underdetermined_with_explicit_set":
            assert 2 <= len(locked.keyed) <= 3 and "could" in locked.question
        if structure == "tree_with_pruning":
            assert locked.key in ("supported", "unsupported") and locked.subject is None
        if structure == "stopping_problem":
            last = locked.model["facts"][-1]["line"]
            assert locked.key == str(got.stop_at) and last - 2 <= got.stop_at <= last
            assert "After which numbered line" in locked.question
        if structure == "falsifier_first":
            assert locked.key == "unsupported" and locked.subject is None
            assert "claim" in locked.model and "asks" in locked.model
        if structure == "when_to_stop":
            assert locked.key in ("supported", "unsupported") and locked.subject is None
            assert "claim" in locked.model and got.stop_at is None
        # The same seed, the same puzzle.
        again = solvefirst.generate(structure, PACKS[pid], 500 + i)
        assert again.model == locked.model and again.sha256 == locked.sha256


def test_the_mock_rendering_passes_the_unchanged_puzzle_check():
    for structure in solvefirst.SOLVE_FIRST:
        locked = solvefirst.generate(structure, PACKS["clinic-rota"], 7)
        fields = {**solvefirst.render_for_mock(locked), "model": locked.model, "answer": locked.key}
        assert check_puzzle(fields, structure, strict=True).answer == locked.key
        assert solvefirst.answer_early(locked, fields["thinking"]) is None


def test_a_generator_gives_up_and_says_so(monkeypatch):
    monkeypatch.setitem(solvefirst.GENERATORS, "constraint_grid", lambda rng, pack: None)
    with pytest.raises(solvefirst.SolveFirstError, match="no provable constraint_grid"):
        solvefirst.generate("constraint_grid", PACKS["clinic-rota"], 1)
    monkeypatch.delitem(solvefirst.GENERATORS, "when_to_stop")
    with pytest.raises(solvefirst.SolveFirstError, match="stays freehand"):
        solvefirst.generate("when_to_stop", PACKS["clinic-rota"], 1)


def test_answer_early_refuses_the_answer_up_front_and_missing_citations():
    locked = solvefirst.generate("scheduling_with_exclusions", PACKS["clinic-rota"], 11)
    facts = [f["line"] for f in locked.model["facts"]]
    cites = " ".join(f"Then line {n} applies." for n in facts)
    early = f"{locked.subject} has {locked.key}, as I'll show. {cites} " + "Filler to pad the thinking out. " * 6
    assert "before its final third" in solvefirst.answer_early(locked, early)
    # A negative step may name both (it rules a pairing out), and a late pairing is the point.
    late = cites + " " + "Working through it step by step. " * 6 + f"So {locked.subject} has {locked.key}."
    assert solvefirst.answer_early(locked, late) is None
    assert solvefirst.answer_early(locked, f"{locked.subject} does not have {locked.key}. " + late) is None
    uncited = late.replace(f"line {facts[0]} ", "the first fact ")
    assert f"never cites line {facts[0]}" in solvefirst.answer_early(locked, uncited)
    # Any form counts, anywhere in a sentence; a number the puzzle doesn't have is refused.
    mixed = (f"First, (line {facts[0]}) trims it. Reading line {facts[1]} next" + "".join(f", and lines {n} and {n}" for n in facts[2:])
             + ". " + "Working through it step by step. " * 6 + f"So {locked.subject} has {locked.key}.")
    assert solvefirst.answer_early(locked, mixed) is None
    assert "line 99, which the puzzle doesn't have" in solvefirst.answer_early(locked, late + " Line 99 agrees.")
    assert solvefirst.answer_early(locked, " ") == "the thinking is empty"
    # A claim: the verdict word up front is early.
    claim = solvefirst.generate("tree_with_pruning", PACKS["clinic-rota"], 3)
    cites = " ".join(f"Line {f['line']} holds." for f in claim.model["facts"])
    assert "gives the verdict" in solvefirst.answer_early(claim, f"It is {claim.key}. {cites} " + "More steps here. " * 6)
    assert solvefirst.answer_early(claim, cites + " More steps here." * 6 + f" So the claim is {claim.key}.") is None


def test_the_plan_gives_b_slots_packs_seeds_and_paths(domain_table: Path):
    table = domains.load(domain_table)
    slots = plan_batch(SIDES, seed=5)
    domains.assign(slots, table, seed=5)
    solvefirst.assign(slots, PACKS, seed=5, table=table)
    domains.check(slots)
    for lesson in SIDES:
        bs = [s for s in slots if s.lesson == lesson and s.kind == "B"]
        assert len({s.pack for s in bs}) == 8 and all(s.b_seed for s in bs)
        assert sum(s.b_path == "solve_first" for s in bs) == 8 and sum(s.b_path == "freehand" for s in bs) == 0
        assert all(s.domain["code"] == PACKS[s.pack]["naics"] and s.domain["title"] for s in bs)
        # No pack's code repeats another slot's domain in the lesson.
        codes = [s.domain["code"] for s in slots if s.lesson == lesson and s.domain]
        assert len(codes) == len(set(codes))


def test_pack_uses_are_capped_across_a_batch(domain_table: Path):
    slots = [Slot(id=f"T{n:02d}-B-01", lesson=f"T{n:02d}", kind="B", side="x", tier=1, pack="clinic-rota") for n in range(5)]
    with pytest.raises(solvefirst.SolveFirstError, match="more than 4 times"):
        solvefirst.check_pack_uses(slots)
    twice = [Slot(id=f"T01-B-0{n}", lesson="T01", kind="B", side="x", tier=1, pack="clinic-rota") for n in range(2)]
    with pytest.raises(solvefirst.SolveFirstError, match="uses a pack twice"):
        solvefirst.check_pack_uses(twice)
    # A pack whose code the table doesn't hold is refused at plan time.
    small = domains.DomainTable(subsectors=({"code": "999", "title": "x"},), escalate={}, source={}, sha256="x")
    with pytest.raises(solvefirst.SolveFirstError, match="not a subsector"):
        solvefirst.assign(plan_batch({"T02": SIDES["T02"]}, seed=1), PACKS, seed=1, table=small)


def test_duplicate_pack_codes_are_refused(tmp_path: Path):
    data = json.loads(solvefirst.PACKS_FILE.read_text(encoding="utf-8"))
    data["packs"][1]["naics"] = data["packs"][0]["naics"]
    p = tmp_path / "packs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(solvefirst.SolveFirstError, match="share a NAICS code"):
        solvefirst.load_packs(p)


def test_the_writer_never_writes_the_model_or_the_answer(domain_table: Path, tmp_path: Path):
    from role_set.mock import MockCritic, MockWriter, mock_gates
    from role_set.prompts import FIELDS, hard_rules, kind_key
    from role_set.runner import Runner

    table = domains.load(domain_table)
    slots = plan_batch({"T02": SIDES["T02"]}, seed=2)
    domains.assign(slots, table, seed=2)
    solvefirst.assign(slots, PACKS, seed=2, table=table)
    b = next(s for s in slots if s.kind == "B" and s.b_path == "solve_first")
    assert kind_key(b) == "B_locked" and FIELDS["B_locked"] == ["puzzle", "thinking", "reply"]
    assert any("locked puzzle" in r for r in hard_rules(b))
    free = Slot(**{**b.as_dict(), "b_path": "freehand"})
    assert kind_key(free) == "B" and not any("locked puzzle" in r for r in hard_rules(free))

    answers: dict = {}
    r = Runner(gates=mock_gates(), pack={"traits/T02.md": "## 2. As a verifier, you go to the source\n\nx\n"}, trait_names={"T02": "goes to the source"},
               writer=MockWriter(answers=answers), critic=MockCritic(answers=answers), out_dir=tmp_path / "o")
    st = r.open_slot(b)
    assert "## The locked puzzle" in st.messages[-1]["content"]
    locked = r.locked_for(b)
    fields = solvefirst.render_for_mock(locked)
    # A writer's own model and key are overwritten by the locked ones.
    fake = type("C", (), {"text": json.dumps({**fields, "model": {"entities": []}, "answer": "nonsense"}), "finish_reason": "stop", "completion_tokens": 10})()
    refusal, cand, _ = r.mechanical(st, fake)
    assert refusal is None and cand.fields["model"] == locked.model and cand.row["answer"] == locked.key
    # Prose that drops a fact line is refused by the unchanged check.
    lines = fields["puzzle"].splitlines()
    broken = {**fields, "puzzle": "\n".join(lines[:-2] + lines[-1:])}
    fake.text = json.dumps(broken)
    refusal, cand, _ = r.mechanical(st, fake)
    assert refusal["code"] == "PUZZLE" and cand is None
    # Thinking that states the answer up front is refused.
    fake.text = json.dumps({**fields, "thinking": f"{locked.subject} has {locked.key}. " + fields["thinking"]})
    refusal, _, _ = r.mechanical(st, fake)
    assert refusal["code"] == "ANSWER_EARLY"
    # A slot naming a pack the file doesn't hold is refused before any call.
    gone = Slot(**{**b.as_dict(), "id": "T02-B-99", "pack": "nowhere"})
    out = r.open_slot(gone)
    assert out.refusals[0]["code"] == "PUZZLE" and "nowhere" in out.refusals[0]["detail"]


def test_receipts_carry_the_b_path(domain_table: Path, tmp_path: Path):
    from role_set.cli import main
    from role_set.pack import PackManifest, pack_hash

    plan = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(plan), "--domains", str(domain_table)]) == 0
    files = {"role.md": "# L\n\nr\n", "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
             "kinds.md": "## 2. What to deliver\n\nk\n", "amendments/1.md": "## Amendment 1\n\nq\n"}
    pack = tmp_path / "pack"
    for name, text in files.items():
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    assert main(["write", "--plan", str(plan), "--pack", str(pack), "--out", str(tmp_path / "o"), "--mock"]) == 0
    receipts = [json.loads(x) for x in (tmp_path / "o" / "mock" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    bs = [r["b"] for r in receipts if r["slot"]["kind"] == "B"]
    assert len(bs) == 8 and all(b["b_path"] == "solve_first" for b in bs)
    for first in bs:
        assert {"family", "pack", "seed", "tries", "model_sha256"} <= set(first) and len(first["model_sha256"]) == 64


def test_glosses_say_each_form_in_words_the_check_accepts():
    assert solvefirst.gloss(["in", "Ama", "room", ["x", "y"]]) == "Ama's room is x or y"
    assert solvefirst.gloss(["not_in", "Ama", "room", ["x", "y"]]) == "Ama's room is neither x nor y"
    assert solvefirst.gloss(["same", {"room": "x"}, {"task": "y"}]) == "the one with room x is the one with task y"
    with pytest.raises(solvefirst.SolveFirstError, match="no gloss"):
        solvefirst.gloss(["offset", "a", "b", "slot", 2])
    with pytest.raises(PuzzleError):
        parse_model({"entities": []})


def test_the_dag_question_asks_for_the_lines():
    locked = solvefirst.generate("tree_with_pruning", PACKS["clinic-rota"], 3)
    assert "Name the lines that settle it" in locked.question


def test_the_plan_reseeds_repeated_skeletons_and_holds_the_floor(domain_table: Path):
    table = domains.load(domain_table)
    slots = plan_batch(SIDES, seed=9)
    domains.assign(slots, table, seed=9)
    solvefirst.assign(slots, PACKS, seed=9, table=table)
    counts = solvefirst.lock_skeletons(slots, PACKS)
    locked = [s for s in slots if s.b_path == "solve_first"]
    assert all(s.b_skeleton and len(s.b_skeleton) == 16 for s in locked)
    for structure, c in counts.items():
        assert len(c) == sum(c.values())  # two lessons, two of each: re-seeding keeps them distinct
    # The slot's recorded seed is the one its puzzle is built from.
    s = locked[0]
    assert solvefirst.skeleton_of(solvefirst.generate(s.structure, PACKS[s.pack], s.b_seed)) == s.b_skeleton
    # Earlier batches count: a skeleton an earlier batch already used is re-seeded away from.
    prior = {s.structure: {s.b_skeleton: 1}}
    again = plan_batch(SIDES, seed=9)
    domains.assign(again, table, seed=9)
    solvefirst.assign(again, PACKS, seed=9, table=table)
    solvefirst.lock_skeletons(again, PACKS, prior=prior)
    twin = next(x for x in again if x.id == s.id)
    assert twin.b_skeleton != s.b_skeleton and twin.b_reseeds >= 1


def test_the_skeleton_floor_refuses_a_thin_structure(domain_table: Path, monkeypatch):
    table = domains.load(domain_table)
    slots = plan_batch(SIDES, seed=9)
    domains.assign(slots, table, seed=9)
    solvefirst.assign(slots, PACKS, seed=9, table=table)
    monkeypatch.setattr(solvefirst, "skeleton_of", lambda locked: "same")
    monkeypatch.setattr(solvefirst, "RESEED_MAX", 2)
    with pytest.raises(solvefirst.SolveFirstError, match="distinct skeletons"):
        solvefirst.lock_skeletons(slots, PACKS)


def test_no_dag_fact_names_both_claim_entities():
    """No single fact decides the claim (R&D, 2026-10-17): no fact of any form, order or neighbour, names both of the
    claim's entities."""

    for i, pid in enumerate(sorted(PACKS)):
        locked = solvefirst.generate("tree_with_pruning", PACKS[pid], 900 + i)
        pair = set(locked.model["claim"][1:3])
        assert len(locked.model["facts"]) >= solvefirst.MIN_FACTS["tree_with_pruning"]
        for f in locked.model["facts"]:
            assert not pair <= {x for x in f["rule"][1:3] if isinstance(x, str)}, (pid, f)
        assert {f["rule"][0] for f in locked.model["facts"]} <= set(solvefirst.DAG_FORMS)


def test_the_floor_holds_across_the_whole_regeneration(domain_table: Path):
    """The real draw (R&D on PR (a)): about 21 uses per structure across 21 lessons, counted cumulatively through B4's
    state, batch after batch. Every structure keeps 75% distinct skeletons and none used more than 3 times."""

    table = domains.load(domain_table)
    prior: dict = {}
    for batch in range(11):
        slots = plan_batch(SIDES, seed=100 + batch)
        domains.assign(slots, table, seed=100 + batch)
        solvefirst.assign(slots, PACKS, seed=100 + batch, table=table)
        prior = solvefirst.lock_skeletons(slots, PACKS, prior=prior)
    for structure in solvefirst.SOLVE_FIRST:
        c = prior[structure]
        assert sum(c.values()) >= 21, (structure, sum(c.values()))
        assert len(c) >= solvefirst.SKELETON_DISTINCT_SHARE * sum(c.values()) and max(c.values()) <= solvefirst.SKELETON_USES_MAX


def test_answer_early_reads_clauses_and_lets_candidate_lists_through():
    """Argon §9.4 #12: a negation covers only its own clause, and a clause still naming a rival answer is weighing
    candidates, not stating one."""

    locked = solvefirst.generate("scheduling_with_exclusions", PACKS["clinic-rota"], 11)
    attr = locked.model["asks"]["attribute"]
    rival = next(v for v in locked.model["attributes"][attr]["values"] if v != locked.key)
    facts = [f["line"] for f in locked.model["facts"]]
    cites = " ".join(f"Then line {n} applies." for n in facts)
    pad = " Working through it step by step." * 6
    tail = f" So {locked.subject} has {locked.key}."
    # "Not A, so B" up front states B: refused.
    sneaky = f"{locked.subject} does not have {rival}, so {locked.subject} has {locked.key}. {cites}{pad}{tail}"
    assert "before its final third" in solvefirst.answer_early(locked, sneaky)
    # Open candidates up front are fine.
    listing = f"From the setting, {locked.subject} has {locked.key} or {rival}. {cites}{pad}{tail}"
    assert solvefirst.answer_early(locked, listing) is None
    # A plain early pairing is still refused.
    assert "before its final third" in solvefirst.answer_early(locked, f"{locked.subject} has {locked.key}. {cites}{pad}{tail}")


def test_answer_early_lets_an_elimination_question_restatement_through():
    locked = solvefirst.generate("elimination_from_a_set", PACKS["clinic-rota"], 5)
    rival = next(e for e in locked.model["entities"] if e not in locked.keyed)
    cites = " ".join(f"Line {f['line']} trims it." for f in locked.model["facts"])
    text = (f"I check whether {locked.keyed[0]} or {rival} has {locked.subject}. {cites}"
            + " Working through it step by step." * 6 + f" So {locked.keyed[0]} has {locked.subject}.")
    assert solvefirst.answer_early(locked, text) is None


NEW_SOLVE_FIRST = ("stopping_problem", "falsifier_first", "when_to_stop")
PACK_CHUNKS = [tuple(sorted(PACKS)[i:i + 6]) for i in range(0, len(PACKS), 6)]


@pytest.mark.sweep
@pytest.mark.parametrize("structure", NEW_SOLVE_FIRST)
@pytest.mark.parametrize("pack_ids", PACK_CHUNKS)
def test_new_generators_prove_and_prune_seeds_1_to_200_in_every_pack(structure: str, pack_ids: tuple[str, ...]):
    for pid in pack_ids:
        pack = PACKS[pid]
        saw_trailing = False
        saw_exact = False
        for seed in range(1, 201):
            locked = solvefirst.generate(structure, pack, seed)
            fields = {**solvefirst.render_for_mock(locked), "model": locked.model, "answer": locked.key}
            got = check_puzzle(fields, structure, strict=True)
            assert got.answer == locked.key and not got.lookup
            assert solvefirst.answer_early(locked, fields["thinking"]) is None
            if structure == "stopping_problem":
                last = locked.model["facts"][-1]["line"]
                assert locked.key == str(got.stop_at) and last - 2 <= got.stop_at <= last
                saw_trailing = saw_trailing or (got.stop_at < last)
                saw_exact = saw_exact or (got.stop_at == last)
            elif structure == "falsifier_first":
                assert locked.key == "unsupported" and "claim" in locked.model and "asks" in locked.model
            elif structure == "when_to_stop":
                assert locked.key in ("supported", "unsupported") and "claim" in locked.model and got.stop_at is None
        if structure == "stopping_problem":
            assert saw_trailing and saw_exact, pid


@pytest.mark.parametrize("structure", NEW_SOLVE_FIRST)
def test_new_generators_prove_and_prune_seeds_1_to_10_in_every_pack(structure: str):
    saw_trailing = False
    saw_exact = False
    for pid in sorted(PACKS):
        pack = PACKS[pid]
        for seed in range(1, 11):
            locked = solvefirst.generate(structure, pack, seed)
            fields = {**solvefirst.render_for_mock(locked), "model": locked.model, "answer": locked.key}
            got = check_puzzle(fields, structure, strict=True)
            assert got.answer == locked.key and not got.lookup
            assert solvefirst.answer_early(locked, fields["thinking"]) is None
            if structure == "stopping_problem":
                last = locked.model["facts"][-1]["line"]
                assert locked.key == str(got.stop_at) and last - 2 <= got.stop_at <= last
                saw_trailing = saw_trailing or (got.stop_at < last)
                saw_exact = saw_exact or (got.stop_at == last)
            elif structure == "falsifier_first":
                assert locked.key == "unsupported" and "claim" in locked.model and "asks" in locked.model
            elif structure == "when_to_stop":
                assert locked.key in ("supported", "unsupported") and "claim" in locked.model and got.stop_at is None
    if structure == "stopping_problem":
        assert saw_trailing and saw_exact


def test_cited_lines_handles_oxford_commas_and_does_not_expand_ranges():
    cases = [
        ("lines 2, 3, and 4", {2, 3, 4}),
        ("lines 2, 3 and 4", {2, 3, 4}),
        ("lines 2 and 3", {2, 3}),
        ("lines 2-4", {2}),
        ("line 2, line 3", {2, 3}),
        ("(lines 2, 3, 4)", {2, 3, 4}),
        ("In line 2, 9 am is ruled out", {2}),
        ("In lines 1 and 2, 10 am is out", {1, 2}),
        ("lines 2 & 3", {2, 3}),
    ]
    for text, want in cases:
        assert solvefirst._cited_lines(text) == want, text


def test_answer_early_stopping_problem_rules_and_brief():
    locked = next(
        solvefirst.generate("stopping_problem", PACKS["clinic-rota"], s)
        for s in range(1, 50)
        if int(solvefirst.generate("stopping_problem", PACKS["clinic-rota"], s).key)
        < solvefirst.generate("stopping_problem", PACKS["clinic-rota"], s).model["facts"][-1]["line"]
    )
    subject = locked.subject
    attr = locked.model["asks"]["attribute"]
    stop_at = int(locked.key)
    all_lines = [f["line"] for f in locked.model["facts"]]
    needed = [n for n in all_lines if n < stop_at]
    trailing = [n for n in all_lines if n > stop_at]
    assert len(needed) >= 2 and len(trailing) >= 1

    want_msg = f"the thinking pairs {subject} with line {stop_at} before citing every earlier needed line"

    # 1. Upfront spoiler refused.
    cites_all = " ".join(f"Line {n} narrows it." for n in all_lines)
    spoiler = f"{subject}'s {attr} is settled at line {stop_at}, as I'll show. {cites_all}"
    assert solvefirst.answer_early(locked, spoiler) == want_msg

    # 2. Skipped needed line refused.
    skipped = (
        f"Line {needed[0]} narrows it. So {subject}'s {attr} is settled at line {stop_at}. "
        + " ".join(f"Line {n} applies." for n in all_lines[1:])
    )
    assert solvefirst.answer_early(locked, skipped) == want_msg

    # 3. Honest early settle accepted (even before 2/3 of the thinking), followed by trailing lines.
    prefix = " ".join(f"Line {n} trims the options." for n in needed)
    settle_step = f"Line {stop_at} rules out the other option, so {subject} is settled at line {stop_at}."
    tail_steps = " ".join(
        f"Line {n} changes nothing after line {stop_at} and leaves the rest of the rota consistent." for n in trailing
    )
    closing = "Every remaining line has been checked and nothing changes the stopping point."
    honest = f"{prefix} {settle_step} {tail_steps} {closing}"
    assert honest.index(settle_step) < len(honest) * solvefirst.LATE_SHARE
    assert solvefirst.answer_early(locked, honest) is None

    # 4. A clock-time digit isn't a citation of line {stop_at}.
    clock = (
        f"{subject} is at {stop_at}.30 pm; line {needed[0]} narrows it. "
        + " ".join(f"Line {n} trims it." for n in needed[1:])
        + f" {settle_step} {tail_steps} {closing}"
    )
    assert solvefirst.answer_early(locked, clock) is None

    # 5. A candidate list naming both line {stop_at} and a rival fact line is allowed before stop_at.
    cand = (
        f"Whether {subject} is settled at line {needed[0]} or line {stop_at} starts at line {needed[0]}. "
        + " ".join(f"Line {n} trims it." for n in needed[1:])
        + f" {settle_step} {tail_steps} {closing}"
    )
    assert solvefirst.answer_early(locked, cand) is None

    # 6. Brief wording for stopping_problem vs the other seven structures.
    b_stop = solvefirst.brief(locked)
    assert "it may say the answer is settled at the settling line once it has cited every earlier line, then works through any remaining lines" in b_stop
    assert "it reaches the answer only at its end" not in b_stop
    for other in solvefirst.SOLVE_FIRST:
        if other != "stopping_problem":
            b_other = solvefirst.brief(solvefirst.generate(other, PACKS["clinic-rota"], 1))
            assert "it reaches the answer only at its end" in b_other


ORIGINAL_FIVE_PINS = {
    ("constraint_grid", "bakery-line", 1): (
        "c586fe119ee5fff354174a228f78b1d12741669df60663ee69b9797ca2997f8d",
        "the tunnel oven",
        "Which oven does Agnieszka have?",
        1,
    ),
    ("constraint_grid", "brewery-cellar", 1): (
        "0babb45f5e40b2d53158e8ae02ae54aac89c8703e4df30b377ec6d3054123fd6",
        "the mash tun",
        "Which vessel does Hanne have?",
        2,
    ),
    ("elimination_from_a_set", "bakery-line", 1): (
        "857028b786fce54e6a2b54aeb1fbf5bcbdcef1c45d04bfccfa956f94e91e6088",
        "Elif",
        "Who has oven the rack oven?",
        1,
    ),
    ("elimination_from_a_set", "brewery-cellar", 1): (
        "427c109ad5243a9058297891d0d9e0687e825b1cecf72d76572ee2e06f0f4bbf",
        "Jolene",
        "Who has vessel the bright tank?",
        1,
    ),
    ("scheduling_with_exclusions", "bakery-line", 1): (
        "d8e63bc3761fb0ef2f24bf1582f72bb0d575f4609735afae0a355a16ab2eb3d2",
        "4.40 am",
        "Which bake does Coralie have?",
        1,
    ),
    ("scheduling_with_exclusions", "brewery-cellar", 1): (
        "35106d157fd97351b07891f716e981023a4759d03cf8e07e9fa1787523823e59",
        "Wednesday",
        "Which brew does Ivo have?",
        1,
    ),
    ("tree_with_pruning", "bakery-line", 1): (
        "7fb6a1203e4c8e08a02237d0bd5861ae8a52d91c1d16b83b8cff260087cf9b44",
        "supported",
        "Is Fintan's bake earlier than Babajide's? Name the lines that settle it, or say it can't be told.",
        1,
    ),
    ("tree_with_pruning", "brewery-cellar", 1): (
        "2de6ede168e790f6b7dde3353e0d628a0096f8e24b3fa6581971caef78d6cda9",
        "supported",
        "Is Kemal's brew later than Gideon's? Name the lines that settle it, or say it can't be told.",
        2,
    ),
    ("underdetermined_with_explicit_set", "bakery-line", 1): (
        "59daa14be7bd565186a8e4ff8421684da904575edeaeb05ff99a5bc67158fd93",
        "4.40 am, 5.20 am",
        "Which bake could Fintan have?",
        1,
    ),
    ("underdetermined_with_explicit_set", "brewery-cellar", 7): (
        "8dbf6f8d520ed706ca01e4146a3ae121e930a973fc2d4d642a2ed47bf55f1f70",
        "Friday, Thursday, Wednesday",
        "Which brew could Ivo have?",
        1,
    ),
}


def test_original_five_structures_are_byte_unchanged_from_main():
    import random

    assert solvefirst._grow_and_prune(random.Random(0), [], lambda f: {}, lambda p: True) is None
    assert solvefirst._grow_and_prune(random.Random(0), [["bad"]], lambda f: {"entities": []}, lambda p: True) is None
    for (structure, pid, seed), want in ORIGINAL_FIVE_PINS.items():
        locked = solvefirst.generate(structure, PACKS[pid], seed)
        assert (locked.sha256, locked.key, locked.question, locked.tries) == want


def test_t10_replan_changes_only_b05_b07_b08_and_journal_replays_unchanged_slots(tmp_path: Path, monkeypatch, capsys):
    import hashlib

    from role_set import callcache
    from role_set.cli import main as cli_main
    from role_set.pack import PackManifest, pack_hash
    from role_set.write import WriteConfig, run

    root = Path(__file__).resolve().parents[1]
    naics = root / "roles" / "domains" / "naics-2022.json"

    old_plan = tmp_path / "t10-old-plan.json"
    new_plan = tmp_path / "t10-new-plan.json"
    with monkeypatch.context() as mp:
        mp.setattr(solvefirst, "SOLVE_FIRST", solvefirst.SOLVE_FIRST[:5])
        assert cli_main(["plan", "--lessons", "T10", "--buffers", "--seed", "0", "--domains", str(naics), "--out", str(old_plan)]) == 0
    assert cli_main(["plan", "--lessons", "T10", "--buffers", "--seed", "0", "--domains", str(naics), "--out", str(new_plan)]) == 0
    capsys.readouterr()

    old_sha = hashlib.sha256(old_plan.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    new_sha = hashlib.sha256(new_plan.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert old_sha == "78ec357230b1ae4fdd1671862d4e378957df08215216d6b9b3fb2f99f2dde784"
    assert new_sha == "086b0a1649a0d84c8a1c71daf4ad03d1081a90af5abd71f3af2a110e38b1f5f6"

    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(new_plan)]) == 0
    out_text = capsys.readouterr().out
    assert "T10-B-05 (stopping_problem):" in out_text and "'3edb2792cd00290d'" in out_text
    assert "T10-B-07 (falsifier_first):" in out_text and "'9e3af05fbe55def0'" in out_text
    assert "T10-B-08 (when_to_stop):" in out_text and "'c43dbf8eea6e2b34'" in out_text
    assert "unchanged: 57/60 slots byte-identical" in out_text

    # Negative checks for t10_replan_check.py: touching top-level, another slot, a forbidden field, b_skeleton, or slot count fails.
    bad_top = tmp_path / "bad-top.json"
    dt = json.loads(new_plan.read_text(encoding="utf-8"))
    dt["seed"] = 99
    bad_top.write_text(json.dumps(dt), encoding="utf-8")
    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(bad_top)]) == 1

    bad_slot = tmp_path / "bad-slot.json"
    d = json.loads(new_plan.read_text(encoding="utf-8"))
    d["slots"][0]["framing_seed"] = 999999
    bad_slot.write_text(json.dumps(d), encoding="utf-8")
    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(bad_slot)]) == 1

    bad_field = tmp_path / "bad-field.json"
    d2 = json.loads(new_plan.read_text(encoding="utf-8"))
    for s in d2["slots"]:
        if s["id"] == "T10-B-05":
            s["pack"] = "other-pack"
    bad_field.write_text(json.dumps(d2), encoding="utf-8")
    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(bad_field)]) == 1

    bad_skel = tmp_path / "bad-skel.json"
    ds = json.loads(new_plan.read_text(encoding="utf-8"))
    for s in ds["slots"]:
        if s["id"] == "T10-B-05":
            s["b_skeleton"] = ""
    bad_skel.write_text(json.dumps(ds), encoding="utf-8")
    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(bad_skel)]) == 1

    bad_order = tmp_path / "bad-order.json"
    d3 = json.loads(new_plan.read_text(encoding="utf-8"))
    d3["slots"] = d3["slots"][1:]
    bad_order.write_text(json.dumps(d3), encoding="utf-8")
    assert t10_replan_check.main(["--old", str(old_plan), "--new", str(bad_order)]) == 1

    # Journal replay test: mock write on old_plan, then replay its journal on new_plan.
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

    res_old = run(WriteConfig(plan=old_plan, pack=pack_dir, out=tmp_path / "run_old", mock=True, wave="inline", training=(train_file,)))
    assert res_old["accepted"] == 60 and res_old["batch_check"] == "clean"
    old_journal = tmp_path / "run_old" / "mock" / callcache.JOURNAL_FILE
    res_new = run(
        WriteConfig(
            plan=new_plan,
            pack=pack_dir,
            out=tmp_path / "run_new",
            mock=True,
            wave="inline",
            training=(train_file,),
            reuse_calls=old_journal,
        )
    )
    assert res_new["accepted"] == 60 and res_new["batch_check"] == "clean"
    # Every call of the 57 unchanged slots (+ warm-up) is reused; only the 3 changed B slots (5 calls each) are called.
    assert res_new["calls"]["reused"] == res_old["calls"]["called"] - 15
    assert res_new["calls"]["called"] == 15
    assert res_new["calls"]["unused_from_reuse"] == 15

    # Wave write mode: 82 writer calls reused, only the 3 changed B slots' writer calls are made.
    w_old = run(WriteConfig(plan=old_plan, pack=pack_dir, out=tmp_path / "wave_old", mock=True, wave="write", training=(train_file,)))
    w_journal = tmp_path / "wave_old" / "mock" / callcache.JOURNAL_FILE
    w_new = run(
        WriteConfig(
            plan=new_plan,
            pack=pack_dir,
            out=tmp_path / "wave_new",
            mock=True,
            wave="write",
            training=(train_file,),
            reuse_calls=w_journal,
        )
    )
    assert w_new["calls"]["reused"] == w_old["calls"]["called"] - 3
    assert w_new["calls"]["called"] == 3
    assert w_new["calls"]["unused_from_reuse"] == 3
