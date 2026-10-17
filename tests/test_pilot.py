"""The wave pilot (BUILD-SPEC §9.2, PLAN.md §7; composition ruled by R&D, 2026-10-17) and Argon §9.4 #7."""

import json
from collections import Counter
from pathlib import Path

import pytest

from role_set import solvefirst, waves
from role_set.cli import main
from role_set.pack import PackManifest, pack_hash
from role_set.plan import Slot
from role_set.runner import LOOKUP_CAP, lookup_cap
from role_set.write import WriteConfig, WriteRefused, build

FILES = {"role.md": "# L\n\nr\n", "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
         "traits/T04.md": "## 4. As a verifier, you say what is missing\n\nSilence is not support.\n",
         "kinds.md": "## 2. What to deliver\n\nk\n", "amendments/1.md": "## Amendment 1\n\nq\n"}


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pack"
    for name, text in FILES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(FILES)
    (d / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    return d


@pytest.fixture
def pilot_plan(tmp_path: Path, domain_table: Path) -> Path:
    out = tmp_path / "pilot.json"
    assert main(["plan", "--pilot", "--seed", "7", "--out", str(out), "--domains", str(domain_table)]) == 0
    return out


def test_the_pilot_plan_has_its_composition(pilot_plan: Path):
    plan = json.loads(pilot_plan.read_text(encoding="utf-8"))
    assert plan["pilot"] is True and plan["exclusion_state_sha256"]
    main_slots = plan["slots"]
    assert {s["lesson"] for s in main_slots} == {"T02"} and len(main_slots) == 20
    assert Counter(s["kind"] for s in main_slots) == {"A": 4, "B": 5, "C": 4, "D": 4, "E": 3}
    assert {s["structure"] for s in main_slots if s["kind"] == "B"} == set(solvefirst.SOLVE_FIRST)
    assert all(s["b_path"] == "solve_first" and s["b_skeleton"] for s in main_slots if s["kind"] == "B")
    assert sum(1 for s in main_slots if s["two_turn"]) == 1 and not any(s["pair"] for s in main_slots)
    assert Counter(s["e_type"] for s in main_slots if s["kind"] == "E") == {"written": 2, "verdict": 1}
    # Every slot with material has a domain; the floors are reported, not enforced.
    assert all(s["domain"] for s in plan["slots"] if s["kind"] in ("A", "C", "D") or s.get("e_type") == "verdict")
    assert plan["figures"][0]["pilot"] is True


def test_the_twins_are_their_own_batch(tmp_path: Path, domain_table: Path):
    """R&D, 2026-10-17: the T04 apply twins are planned apart and written after the pilot, its accepted rows their
    --training."""

    out = tmp_path / "twins.json"
    assert main(["plan", "--pilot-twins", "--seed", "7", "--out", str(out), "--domains", str(domain_table)]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["pilot"] is True and Counter(s["kind"] for s in plan["slots"]) == {"C": 2, "D": 2}
    assert all(s["lesson"] == "T04" and s["split"] == "apply" and s["verdict"] != "cannot_tell" and s["domain"]
               for s in plan["slots"])


def test_the_pilot_plan_refuses_lessons_apply_or_no_domains(tmp_path: Path, domain_table: Path, capsys):
    out = str(tmp_path / "p.json")
    assert main(["plan", "--pilot", "--lessons", "T02", "--out", out, "--domains", str(domain_table)]) == 2
    assert main(["plan", "--pilot", "--apply", "--out", out, "--domains", str(domain_table)]) == 2
    assert main(["plan", "--pilot", "--out", out]) == 2
    assert main(["plan", "--pilot", "--pilot-twins", "--out", out, "--domains", str(domain_table)]) == 2
    assert "REFUSED: --pilot" in capsys.readouterr().err


def test_a_pilot_write_refuses_exemplars_and_a_production_partition(pilot_plan: Path, pack_dir: Path, tmp_path: Path):
    with pytest.raises(WriteRefused, match="without exemplars"):
        build(WriteConfig(plan=pilot_plan, pack=pack_dir, out=tmp_path / "o", mock=True, exemplar_files=("x.jsonl",)))
    with pytest.raises(WriteRefused, match="trial:<name> partition"):
        build(WriteConfig(plan=pilot_plan, pack=pack_dir, out=tmp_path / "o", partition="production"))
    pins = build(WriteConfig(plan=pilot_plan, pack=pack_dir, out=tmp_path / "o", mock=True, partition="trial:j1-wave1"))[7]
    assert pins["pilot"] is True


def test_a_solve_first_slot_takes_no_lookup():
    """Argon §9.4 #7: a lookup on a locked puzzle is a defect, so its cap is 0; freehand keeps LOOKUP_CAP."""

    locked = Slot(id="T02-B-01", lesson="T02", kind="B", side="x", tier=1, b_path="solve_first")
    free = Slot(id="T02-B-06", lesson="T02", kind="B", side="x", tier=1, b_path="freehand")
    assert lookup_cap(locked) == 0 and lookup_cap(free) == LOOKUP_CAP


def test_a_rewrite_sees_the_spent_list_as_it_stands():
    """Argon §9.4 #7: the rewrite wave's first prompt carries what the batch has spent since the write wave."""

    head = "## The slot\nWrite it.\n\n"
    old = head + waves.AVOID_HEADER + "- first framing\n\n## The lesson's sentences\n- s" + waves.PROMPT_TAIL
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": old}]
    new = waves.refresh_avoid(msgs, ["first framing", "accepted in the critique wave"])
    assert "- accepted in the critique wave" in new[1]["content"] and "## The lesson's sentences" in new[1]["content"]
    assert new[1]["content"].endswith(waves.PROMPT_TAIL) and msgs[1]["content"] == old  # the stored draft is untouched
    # A prompt written with nothing spent gains the list before its last line.
    bare = waves.refresh_avoid([{"role": "user", "content": head + "x" + waves.PROMPT_TAIL}], ["one"])
    assert waves.AVOID_HEADER + "- one" + waves.PROMPT_TAIL in bare[0]["content"]


def test_a_mock_pilot_runs_in_waves_then_its_twins_and_leaves_no_state(pilot_plan: Path, pack_dir: Path, tmp_path: Path,
                                                                        domain_table: Path):
    """The pilot's three-wave cycle end to end on the mock (BUILD-SPEC §9.1): every slot reaches a final state and a
    pilot never enters the exclusion state. Then the T04 twins, with the pilot's accepted rows as --training."""

    from role_set.write import run

    def cycle(plan: Path, base: str, **kw) -> tuple[dict, Path]:
        def cfg(out, **more):
            return WriteConfig(plan=plan, pack=pack_dir, out=tmp_path / out, mock=True, partition="trial:j1-wave1", **kw, **more)

        result = run(cfg(f"{base}0", wave="write"))
        last = tmp_path / f"{base}0" / "mock"
        for i in range(1, 7):
            result = run(cfg(f"{base}{i}", wave="critique" if i % 2 else "rewrite", drafts=last / waves.DRAFTS_FILE))
            last = tmp_path / f"{base}{i}" / "mock"
            if not result["states"]["awaiting_critique"] and not result["states"]["needs_rewrite"]:
                break
        assert result["states"]["awaiting_critique"] == result["states"]["needs_rewrite"] == 0
        assert result["exclusion_state"] is None
        return result, last

    result, last = cycle(pilot_plan, "p")
    assert result["slots"] == 20 and result["accepted"] > 0
    twins_plan = tmp_path / "twins.json"
    assert main(["plan", "--pilot-twins", "--seed", "7", "--out", str(twins_plan), "--domains", str(domain_table)]) == 0
    twins, _ = cycle(twins_plan, "t", training=(last / "accepted.jsonl",))
    assert twins["slots"] == 4


def test_a_mock_pilot_runs_in_waves_and_leaves_no_state(pilot_plan: Path, pack_dir: Path, tmp_path: Path):
    """The 20-slot pilot alone, through the three-wave cycle on the mock: every slot reaches a final state, and a
    pilot never enters the exclusion state."""

    from role_set.write import run

    def cfg(out, **kw):
        return WriteConfig(plan=pilot_plan, pack=pack_dir, out=tmp_path / out, mock=True, partition="trial:j1-wave1", **kw)

    result = run(cfg("w0", wave="write"))
    last = tmp_path / "w0" / "mock"
    for i in range(1, 7):
        result = run(cfg(f"w{i}", wave="critique" if i % 2 else "rewrite", drafts=last / waves.DRAFTS_FILE))
        last = tmp_path / f"w{i}" / "mock"
        if not result["states"]["awaiting_critique"] and not result["states"]["needs_rewrite"]:
            break
    assert result["states"]["awaiting_critique"] == result["states"]["needs_rewrite"] == 0
    assert result["slots"] == 20 and result["accepted"] > 0 and result["exclusion_state"] is None


def test_an_apply_brief_asks_for_its_twin_flag():
    """The pilot's mock run found that no prompt asked for the twin flags assembly requires (aspire-si#137)."""

    from role_set.prompts import slot_brief

    t04 = Slot(id="T04-C-AP-01", lesson="T04", kind="C", side="x", tier=1, verdict="supported", split="apply", content_kind="notice")
    t09 = Slot(id="T09-D-AP-01", lesson="T09", kind="D", side="x", tier=1, verdict="unsupported", split="apply", content_kind="notice")
    silent = Slot(id="T04-C-AP-02", lesson="T04", kind="C", side="x", tier=1, verdict="cannot_tell", split="apply", content_kind="notice")
    assert "silent_without_deciding" in slot_brief(t04, [])
    assert "order_free" in slot_brief(t09, [])
    assert "silent_without_deciding" not in slot_brief(silent, [])  # a cannot_tell item has no deciding lines to drop
