"""BUILD-SPEC B3a: solve-first kind B. The model and the key are made in code; the writer renders them."""

import json
from pathlib import Path

import pytest

from role_set import domains, solvefirst
from role_set.plan import Slot, plan_batch
from role_set.puzzle import PuzzleError, check_puzzle, parse_model, solve

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
        assert sum(s.b_path == "solve_first" for s in bs) == 5 and sum(s.b_path == "freehand" for s in bs) == 3
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
    free = next(s for s in slots if s.kind == "B" and s.b_path == "freehand")
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
    assert len(bs) == 8 and sum(b["b_path"] == "solve_first" for b in bs) == 5
    first = next(b for b in bs if b["b_path"] == "solve_first")
    assert {"family", "pack", "seed", "tries", "model_sha256"} <= set(first) and len(first["model_sha256"]) == 64
    assert all(b["pack"] for b in bs if b["b_path"] == "freehand")


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
