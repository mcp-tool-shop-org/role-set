"""BUILD-SPEC B4: the cumulative exclusion state across batches."""

import json
from pathlib import Path

import pytest

from role_set import exclusion, waves
from role_set.cli import main
from role_set.pack import PackManifest, pack_hash
from role_set.write import WriteConfig, WriteRefused, run

FILES = {"role.md": "# L\n\nr\n", "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
         "traits/T03.md": "## 3. As a verifier, you are precise\n\nExact.\n", "kinds.md": "## 2. What to deliver\n\nk\n",
         "amendments/1.md": "## Amendment 1\n\nq\n"}


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pack"
    for name, text in FILES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(FILES)
    (d / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    return d


def _plan(tmp_path: Path, table: Path, lessons: str, seed: int, *extra) -> Path:
    out = tmp_path / f"plan-{lessons}-{seed}.json"
    assert main(["plan", "--lessons", lessons, "--seed", str(seed), "--out", str(out), "--domains", str(table), *extra]) == 0
    return out


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_the_state_round_trips_and_refuses_an_edit(tmp_path: Path):
    s = exclusion.State(batches=["b1"], rows=[{"id": "T02-C-01", "kind": "C", "turns": [{"user": "Check it.\nMaterial:\nx"}]}],
                        models={}, skeletons={"constraint_grid": {"abc": 1}}, domains={"621": 1}, frames={"ask": 1})
    p = tmp_path / exclusion.FILE
    sha = exclusion.save(s, p)
    back = exclusion.load(p)
    assert back.sha256 == sha == s.sha256 and back.framings() == ["Check it."]
    data = json.loads(p.read_text(encoding="utf-8"))
    data["rows"][0]["id"] = "T02-C-99"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(exclusion.ExclusionError, match="match its contents"):
        exclusion.load(p)
    p.write_text(json.dumps({**data, "format": "other"}), encoding="utf-8")
    with pytest.raises(exclusion.ExclusionError, match="not an exclusion state"):
        exclusion.load(p)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(exclusion.ExclusionError, match="cannot read"):
        exclusion.load(p)
    assert exclusion.load(None).sha256 == exclusion.State().sha256
    with pytest.raises(exclusion.ExclusionError, match="already in the exclusion state"):
        exclusion.with_batch(back, "b1", [], {}, [])


def test_a_batch_leaves_its_state_and_the_next_batch_checks_against_it(domain_table: Path, pack_dir: Path, tmp_path: Path):
    plan1 = _plan(tmp_path, domain_table, "T02", 3)
    first = run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "o1", mock=True))
    state_path = tmp_path / "o1" / "mock" / exclusion.FILE
    state = exclusion.load(state_path)
    assert first["accepted"] == 40 and first["exclusion_state"] == state.sha256
    assert len(state.rows) == 40 and len(state.batches) == 1 and sum(state.frames.values()) == 18
    assert sum(sum(v.values()) for v in state.skeletons.values()) == 5  # the five solve-first B slots
    # The same rows under another batch id: the rerun's identical drafts are refused against them.
    copy = exclusion.State(batches=["earlier"], rows=state.rows, models=state.models)
    prior = tmp_path / "prior.json"
    exclusion.save(copy, prior)
    # A plan is made against the state it runs with (Argon §9.4 #15): the old plan is refused against the copy...
    with pytest.raises(WriteRefused, match="plan was made against exclusion state"):
        run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "o2", mock=True, exclusion_state=prior))
    replan = tmp_path / "replan"
    replan.mkdir()
    plan1b = _plan(replan, domain_table, "T02", 3, "--exclusion-state", str(prior))
    second = run(WriteConfig(plan=plan1b, pack=pack_dir, out=tmp_path / "o2", mock=True, exclusion_state=prior))
    refusals = _jsonl(tmp_path / "o2" / "mock" / "refusals.jsonl")
    assert second["accepted"] < 40 and any(r["code"] == "AGAINST" for r in refusals)
    receipts = _jsonl(tmp_path / "o2" / "mock" / "receipts.jsonl") if (tmp_path / "o2" / "mock" / "receipts.jsonl").exists() else []
    assert all(r["exclusion_state_sha256"] == copy.sha256 and r["exclusion_batches"] == 1 for r in receipts)
    # A plan already in the state is refused before any call.
    with pytest.raises(WriteRefused, match="already in the exclusion state"):
        run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "o3", mock=True, exclusion_state=state_path))
    # The next batch (another plan) starts from the state and adds itself. (The mock's warm-up only knows T02.)
    plan2 = _plan(tmp_path, domain_table, "T02", 4, "--exclusion-state", str(state_path))
    third = run(WriteConfig(plan=plan2, pack=pack_dir, out=tmp_path / "o4", mock=True, exclusion_state=state_path))
    # The mock's words come from the slot's place in the plan, so a second T02 plan repeats the first's drafts and the
    # state refuses every one: no row an earlier batch accepted can come back. Nothing accepted, so no state is left.
    assert third["accepted"] == 0 and third["exclusion_state"] is None
    assert {r["code"] for r in _jsonl(tmp_path / "o4" / "mock" / "refusals.jsonl")} == {"AGAINST"}


def test_a_batch_adds_its_rows_and_counts_to_the_state():
    from role_set.plan import Slot

    s1 = Slot(id="T02-C-01", lesson="T02", kind="C", side="x", tier=1, domain={"code": "621", "title": "t"}, frame="ask")
    s2 = Slot(id="T02-B-01", lesson="T02", kind="B", side="x", tier=1, structure="constraint_grid", b_skeleton="abc", b_path="solve_first")
    s3 = Slot(id="T02-C-02", lesson="T02", kind="C", side="x", tier=1, domain={"code": "622", "title": "t"}, frame="ask")
    rows = [{"id": "T02-C-01", "kind": "C"}, {"id": "T02-B-01", "kind": "B"}]
    new = exclusion.with_batch(exclusion.State(batches=["b0"]), "b1", rows, {"T02-B-01": {"m": 1}, "T02-X": {}}, [s1, s2, s3])
    # Only the accepted slots count; the dropped T02-C-02 adds nothing.
    assert new.batches == ["b0", "b1"] and len(new.rows) == 2 and new.models == {"T02-B-01": {"m": 1}}
    assert new.skeletons == {"constraint_grid": {"abc": 1}} and new.domains == {"621": 1} and new.frames == {"ask": 1}


def test_the_plan_counts_earlier_skeletons(domain_table: Path, pack_dir: Path, tmp_path: Path):
    plan1 = _plan(tmp_path, domain_table, "T02", 3)
    run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "o1", mock=True))
    state_path = tmp_path / "o1" / "mock" / exclusion.FILE
    # The same lesson and seed again, now against the state: no solve-first skeleton repeats an earlier batch's (the
    # least-used packs come first now, so most differ by pack; a collision would re-seed).
    (tmp_path / "x").mkdir()
    again = _plan(tmp_path / "x", domain_table, "T02", 3, "--exclusion-state", str(state_path))
    plan = json.loads(again.read_text(encoding="utf-8"))
    earlier = exclusion.load(state_path)
    locked = [s for s in plan["slots"] if s.get("b_path") == "solve_first"]
    assert locked and all(s["b_skeleton"] not in earlier.skeletons.get(s["structure"], {}) for s in locked)
    # And the packs the first batch used are dealt last.
    first_packs = set(earlier.packs)
    assert first_packs and not first_packs & {s["pack"] for s in plan["slots"] if s.get("pack")}
    assert plan["exclusion_state_sha256"] == exclusion.load(state_path).sha256


def test_every_wave_of_a_batch_uses_the_same_state(domain_table: Path, pack_dir: Path, tmp_path: Path):
    plan1 = _plan(tmp_path, domain_table, "T02", 3)
    run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "w0", mock=True, wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    other = tmp_path / "other.json"
    exclusion.save(exclusion.State(batches=["x"]), other)
    with pytest.raises(WriteRefused, match="plan was made against exclusion state"):
        run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "w1", mock=True, wave="critique", drafts=drafts, exclusion_state=other))
    done = run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "w1", mock=True, wave="critique", drafts=drafts))
    # The wave that leaves nothing pending writes the batch's state.
    assert done["exclusion_state"] and len(exclusion.load(tmp_path / "w1" / "mock" / exclusion.FILE).rows) == 40


def test_only_a_clean_batch_enters_the_state(domain_table: Path, pack_dir: Path, tmp_path: Path, monkeypatch):
    from functools import partial

    import role_set.write as w
    from role_set.mock import MockCritic

    # The critic refuses every verdict: C, D and verdict E cells halt, so the batch check never runs clean.
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_all=True))
    plan1 = _plan(tmp_path, domain_table, "T02", 3)
    result = run(WriteConfig(plan=plan1, pack=pack_dir, out=tmp_path / "o", mock=True))
    assert result["accepted"] > 0 and result["batch_check"] != "clean"
    assert result["exclusion_state"] is None and not (tmp_path / "o" / "mock" / exclusion.FILE).exists()



def test_pack_uses_count_across_batches(domain_table: Path):
    """Argon §9.4 #10: a pack is used at most PACK_USES_MAX times across the regeneration, counted through B4's state;
    the least-used packs are dealt first."""

    from role_set import domains, solvefirst
    from role_set.plan import Slot, plan_batch

    packs = solvefirst.load_packs()
    table = domains.load(domain_table)
    sides = {"T02": ("quotes the source", "stops at the source")}
    # Every pack but a few is already at the cap: the batch draws only from those left.
    left = sorted(packs)[:16]
    prior = {p: solvefirst.PACK_USES_MAX for p in packs if p not in left}
    slots = plan_batch(sides, seed=5)
    domains.assign(slots, table, seed=5)
    solvefirst.assign(slots, packs, seed=5, table=table, prior=prior)
    assert {s.pack for s in slots if s.kind == "B"} <= set(left)
    # A pack over the cap with the earlier batches counted is refused.
    b = next(s for s in slots if s.kind == "B")
    with pytest.raises(solvefirst.SolveFirstError, match="across the regeneration"):
        solvefirst.check_pack_uses(slots, {b.pack: solvefirst.PACK_USES_MAX})
    # The state counts the packs of accepted B slots.
    s1 = Slot(id="T02-B-01", lesson="T02", kind="B", side="x", tier=1, structure="constraint_grid", pack="clinic-rota")
    new = exclusion.with_batch(exclusion.State(), "b1", [{"id": "T02-B-01", "kind": "B"}], {}, [s1])
    assert new.packs == {"clinic-rota": 1}
    assert exclusion.with_batch(new, "b2", [{"id": "T02-B-01", "kind": "B"}], {}, [s1]).packs == {"clinic-rota": 2}


def test_a_skeleton_an_earlier_batch_used_is_reseeded_away_from(domain_table: Path, tmp_path: Path):
    """R&D on #81: once packs repeat, a plan can meet an earlier batch's skeleton; the plan re-seeds, records it, and
    nothing repeats the prior. Forced here with a state that holds the first plan's skeletons but no pack counts, so the
    second plan is dealt the same packs."""

    plan1 = json.loads(_plan(tmp_path, domain_table, "T02", 3).read_text(encoding="utf-8"))
    locked1 = [s for s in plan1["slots"] if s.get("b_path") == "solve_first"]
    skeletons: dict = {}
    for s in locked1:
        skeletons.setdefault(s["structure"], {})[s["b_skeleton"]] = 1
    state = tmp_path / "skeletons-only.json"
    exclusion.save(exclusion.State(batches=["earlier"], skeletons=skeletons), state)
    (tmp_path / "x").mkdir()
    plan2 = json.loads(_plan(tmp_path / "x", domain_table, "T02", 3, "--exclusion-state", str(state)).read_text(encoding="utf-8"))
    locked2 = {s["id"]: s for s in plan2["slots"] if s.get("b_path") == "solve_first"}
    # Same packs as the first plan, so every slot met its own earlier skeleton and re-seeded.
    assert all(locked2[s["id"]]["pack"] == s["pack"] for s in locked1)
    assert all(s["b_reseeds"] >= 1 for s in locked2.values())
    assert all(s["b_skeleton"] not in skeletons[s["structure"]] for s in locked2.values())
