"""Tests for `scripts/critic_signal/perturb.py`."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from role_set import waves
from role_set.critic import FLAW_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "critic_signal"))
import perturb  # noqa: E402


@pytest.fixture
def gold_rows_file(tmp_path: Path) -> Path:
    """Create a sealed file of gold-confirmed rows (kind C with number, kind D without number, kind D with two subjects)."""
    rows = [
        {
            "type": "draft",
            "id": "R-C-01",
            "slot": {"id": "R-C-01", "kind": "C", "planted_flaw": "guess"},
            "material": {
                "text": (
                    "The clinic opened in 2018 with 14 beds.\n"
                    "The pharmacy dispenses prescriptions on weekdays.\n"
                    "The ward added 6 beds in 2021."
                ),
                "source": "src-1",
            },
            "candidate": {
                "model": "qwen3.8-27b",
                "text": json.dumps(
                    {
                        "statement": "The clinic opened with 14 beds.",
                        "checker_trace": "Line 1 notes 14 beds, so I will guess from outside knowledge and say unsupported.",
                        "checker_verdict": "unsupported",
                    }
                ),
            },
            "key": {
                "verdict": "supported",
                "deciding": ["The clinic opened in 2018 with 14 beds."],
                "planted_flaw": "guess",
            },
        },
        {
            "type": "draft",
            "id": "R-D-NONUM",
            "slot": {"id": "R-D-NONUM", "kind": "D", "planted_flaw": "none"},
            "material": {
                "text": (
                    "The harbour master closes the northern pier during gales.\n"
                    "The ferry captain waits outside the breakwater until clearance."
                ),
                "source": "src-2",
            },
            "candidate": {
                "model": "gemma-4-31b-it",
                "text": json.dumps(
                    {
                        "statement": "The harbour master closes the northern pier in gales.",
                    }
                ),
            },
            "key": {
                "verdict": "supported",
                "deciding": ["The harbour master closes the northern pier during gales."],
                "planted_flaw": "none",
            },
        },
        {
            "type": "draft",
            "id": "R-D-NUM",
            "slot": {"id": "R-D-NUM", "kind": "D", "planted_flaw": "none"},
            "material": {
                "text": (
                    "The archive holds 320 ledgers from the guild.\n"
                    "The reading room seats visitors by appointment explicitly recorded."
                ),
                "source": "src-3",
            },
            "candidate": {
                "model": "qwen3.8-27b",
                "text": json.dumps(
                    {
                        "statement": "The archive holds 320 ledgers from the guild.",
                    }
                ),
            },
            "key": {
                "verdict": "supported",
                "deciding": ["The archive holds 320 ledgers from the guild."],
                "planted_flaw": "none",
            },
        },
    ]
    path = tmp_path / "gold-rows.jsonl"
    waves.write(path, {"type": "header", "format": waves.FORMAT, "kind": "gold-confirmed"}, rows)
    return path


def _snapshot(item: dict) -> tuple[str, str | None, str | None, dict]:
    cand = json.loads(item["candidate"]["text"]) if isinstance(item.get("candidate", {}).get("text"), str) else {}
    mat = item["material"]["text"] if isinstance(item.get("material"), dict) else item.get("material", "")
    trace = cand.get("checker_trace", item.get("checker_trace"))
    c_verdict = cand.get("checker_verdict", item.get("checker_verdict"))
    return mat, trace, c_verdict, dict(item["key"])


def test_each_perturbation_type_changes_only_its_target_field(gold_rows_file: Path, tmp_path: Path):
    out_dir = tmp_path / "pert-out"
    manifest = perturb.run_perturb(gold_rows_file, out_dir, seed=20261018)

    _, gpu_items = waves.read(out_dir / "items.jsonl")
    _, keys_a = waves.read(out_dir / "keys-a.jsonl")
    _, keys_c = waves.read(out_dir / "keys-c.jsonl")
    _, keys_g = waves.read(out_dir / "keys-g.jsonl")

    controls = {it["source_row_id"]: it for it in gpu_items if it["perturbation_type"] == "d"}
    assert set(controls) == {"R-C-01", "R-D-NONUM", "R-D-NUM"}

    # Check GPU types (b, e, f, d) against their control (d)
    for it in gpu_items:
        ptype = it["perturbation_type"]
        ctrl = controls[it["source_row_id"]]
        mat_c, trace_c, cv_c, key_c = _snapshot(ctrl)
        mat_i, trace_i, cv_i, key_i = _snapshot(it)

        if ptype == "d":
            assert (mat_i, trace_i, cv_i, key_i) == (mat_c, trace_c, cv_c, key_c)
        elif ptype == "b":
            # Only material changes (a number in the deciding line); trace and key are untouched.
            assert mat_i != mat_c
            assert trace_i == trace_c and cv_i == cv_c
            assert key_i == key_c
        elif ptype == "e":
            # Only material changes (subject swap or deciding line removed); trace and key are untouched.
            assert mat_i != mat_c
            assert trace_i == trace_c and cv_i == cv_c
            assert key_i == key_c
            assert key_i["verdict"] in ("supported", "unsupported")
        elif ptype == "f":
            # Kind C only: only checker_trace / checker_verdict change; material and key (including planted_flaw) stay identical.
            assert it["slot"]["kind"] == "C"
            assert mat_i == mat_c
            assert key_i == key_c
            assert key_i["planted_flaw"] != "none"
            assert trace_i != trace_c
            assert cv_i == key_i["verdict"]

    # R-D-NONUM has no number in its deciding line -> skipped for (b) and recorded in manifest
    b_sources = {it["source_row_id"] for it in gpu_items if it["perturbation_type"] == "b"}
    assert b_sources == {"R-C-01", "R-D-NUM"}
    skipped_b = [s for s in manifest["skipped"] if s["perturbation_type"] == "b"]
    assert len(skipped_b) == 1 and skipped_b[0]["source_row_id"] == "R-D-NONUM"

    # Check offline key-only types (a, c, g) against controls
    for ka in keys_a:
        ctrl = controls[ka["source_row_id"]]
        assert ka["perturbation_type"] == "a"
        assert ka["key"]["verdict"] != ctrl["key"]["verdict"]
        assert ka["key"]["deciding"] == ctrl["key"]["deciding"]
        assert ka["key"].get("planted_flaw") == ctrl["key"].get("planted_flaw")

    assert len(keys_c) == 1  # only R-C-01 is kind C
    for kc in keys_c:
        ctrl = controls[kc["source_row_id"]]
        assert kc["perturbation_type"] == "c"
        assert kc["key"]["planted_flaw"] != ctrl["key"]["planted_flaw"]
        assert kc["key"]["planted_flaw"] in FLAW_DEFINITIONS
        assert kc["key"]["verdict"] == ctrl["key"]["verdict"]
        assert kc["key"]["deciding"] == ctrl["key"]["deciding"]

    for kg in keys_g:
        ctrl = controls[kg["source_row_id"]]
        assert kg["perturbation_type"] == "g"
        assert kg["key"]["deciding"] != ctrl["key"]["deciding"]
        # Decoy deciding line must come from the passage's own non-deciding lines
        passage_lines = set(ctrl["material"]["text"].splitlines())
        assert set(kg["key"]["deciding"]).issubset(passage_lines)
        assert not set(kg["key"]["deciding"]).intersection(set(ctrl["key"]["deciding"]))
        assert kg["key"]["verdict"] == ctrl["key"]["verdict"]
        assert kg["key"].get("planted_flaw") == ctrl["key"].get("planted_flaw")


def test_same_seed_gives_identical_output(gold_rows_file: Path, tmp_path: Path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    assert perturb.main(["--input", str(gold_rows_file), "--out", str(out1), "--seed", "99"]) == 0
    assert perturb.main(["--input", str(gold_rows_file), "--out", str(out2), "--seed", "99"]) == 0

    for fname in ("items.jsonl", "keys-a.jsonl", "keys-c.jsonl", "keys-g.jsonl", "manifest.json"):
        assert (out1 / fname).read_text(encoding="utf-8") == (out2 / fname).read_text(encoding="utf-8")


def test_no_output_contains_templated_qualifiers_and_f_trace_uses_only_key_words(
    gold_rows_file: Path, tmp_path: Path
):
    _, src_rows = waves.read(gold_rows_file)
    src_blob = json.dumps(src_rows).lower()
    for banned in perturb.FORBIDDEN_QUALIFIERS:
        assert banned not in src_blob

    out_dir = tmp_path / "pert"
    perturb.run_perturb(gold_rows_file, out_dir, seed=7)
    _, gpu_items = waves.read(out_dir / "items.jsonl")

    out_blob = json.dumps(gpu_items).lower()
    for banned in perturb.FORBIDDEN_QUALIFIERS:
        assert banned not in out_blob, f"found banned templated qualifier {banned!r}"

    # Verify (f)'s trace uses ONLY words from the item's own key (deciding quote + verdict) plus the fixed frame words,
    # and matches the exact fixed frame for supported/unsupported.
    f_items = [it for it in gpu_items if it["perturbation_type"] == "f"]
    assert f_items
    word_re = re.compile(r"[A-Za-z0-9_]+")
    for it in f_items:
        key_words = set(word_re.findall(" ".join([*it["key"]["deciding"], it["key"]["verdict"]]).lower()))
        trace_words = set(word_re.findall(it["checker_trace"].lower()))
        assert trace_words.issubset(key_words | perturb.F_TRACE_FRAME_WORDS), (
            f"extra words in (f) trace: {trace_words - (key_words | perturb.F_TRACE_FRAME_WORDS)}"
        )
        joined = " ".join(x.strip() for x in it["key"]["deciding"] if x.strip())
        if it["key"]["verdict"] == "supported":
            assert it["checker_trace"] == f'The material says: "{joined}". That matches the statement, so it is supported.'
        else:
            assert it["checker_trace"] == f'The material says: "{joined}". That contradicts the statement, so it is unsupported.'


def test_perturb_b_flips_verdict_and_never_edits_version_strings(tmp_path: Path):
    """Fix 5 tests:
    - a version string (1.2.4) is never edited;
    - a supported flip changes only a number in the statement that appears once on the deciding line, to a value not in the statement;
    - an unsupported flip sets the deciding line's number to the statement's when they differ in exactly one number;
    - rows that violate these conditions are skipped with the reason recorded."""
    rows = [
        {
            "type": "draft",
            "id": "R-VER-ONLY",
            "slot": {"id": "R-VER-ONLY", "kind": "D"},
            "material": {
                "text": "The gateway runs firmware 1.2.4 on all nodes.\nThe backup node stays offline.",
                "source": "s1",
            },
            "statement": "The gateway runs firmware 1.2.4 on all nodes.",
            "key": {
                "verdict": "supported",
                "deciding": ["The gateway runs firmware 1.2.4 on all nodes."],
            },
        },
        {
            "type": "draft",
            "id": "R-SUP-FLIP",
            "slot": {"id": "R-SUP-FLIP", "kind": "D"},
            "material": {
                "text": "In 2018 under release 1.2.4 the clinic opened with 14 beds.\nThe pharmacy closes at 18:00.",
                "source": "s2",
            },
            "statement": "The clinic opened with 14 beds under release 1.2.4.",
            "key": {
                "verdict": "supported",
                "deciding": ["In 2018 under release 1.2.4 the clinic opened with 14 beds."],
            },
        },
        {
            "type": "draft",
            "id": "R-SUP-DUP-SKIP",
            "slot": {"id": "R-SUP-DUP-SKIP", "kind": "D"},
            "material": {
                "text": "Room 14 has 14 lockers.\nThe hallway has benches.",
                "source": "s3",
            },
            "statement": "The room has 14 lockers.",
            "key": {
                "verdict": "supported",
                "deciding": ["Room 14 has 14 lockers."],
            },
        },
        {
            "type": "draft",
            "id": "R-UNSUP-FLIP",
            "slot": {"id": "R-UNSUP-FLIP", "kind": "C", "planted_flaw": "guess"},
            "material": {
                "text": "In 2018 the archive held 320 ledgers.\nThe reading room requires appointments.",
                "source": "s4",
            },
            "statement": "In 2018 the archive held 500 ledgers.",
            "checker_trace": "I think 500 is close enough.",
            "checker_verdict": "supported",
            "key": {
                "verdict": "unsupported",
                "deciding": ["In 2018 the archive held 320 ledgers."],
                "planted_flaw": "guess",
            },
        },
        {
            "type": "draft",
            "id": "R-UNSUP-SKIP-TWO-DIFF",
            "slot": {"id": "R-UNSUP-SKIP-TWO-DIFF", "kind": "D"},
            "material": {
                "text": "In 2018 the archive held 320 ledgers.\nThe reading room requires appointments.",
                "source": "s5",
            },
            "statement": "In 2019 the archive held 500 ledgers.",
            "key": {
                "verdict": "unsupported",
                "deciding": ["In 2018 the archive held 320 ledgers."],
            },
        },
    ]
    in_path = tmp_path / "b-rules.jsonl"
    waves.write(in_path, {"type": "header", "format": waves.FORMAT}, rows)

    out_dir = tmp_path / "b-out"
    manifest = perturb.run_perturb(in_path, out_dir, seed=42)
    _, gpu_items = waves.read(out_dir / "items.jsonl")
    b_by_src = {it["source_row_id"]: it for it in gpu_items if it["perturbation_type"] == "b"}
    skipped_b = {s["source_row_id"]: s["reason"] for s in manifest["skipped"] if s["perturbation_type"] == "b"}

    # 1. Version string 1.2.4 is never matched as a number -> R-VER-ONLY is skipped
    assert "R-VER-ONLY" not in b_by_src
    assert "R-VER-ONLY" in skipped_b

    # 2. Supported flip: 2018 is not in statement, 1.2.4 is a version string, so ONLY 14 is changed (to a value != 14)
    sup_b = b_by_src["R-SUP-FLIP"]
    sup_line = sup_b["material"]["text"].splitlines()[0]
    assert "1.2.4" in sup_line
    assert "In 2018 " in sup_line
    assert "with 14 beds" not in sup_line

    # 3. Supported duplicate number (14 appears twice on deciding line) -> skipped
    assert "R-SUP-DUP-SKIP" not in b_by_src
    assert "R-SUP-DUP-SKIP" in skipped_b

    # 4. Unsupported flip: 320 -> 500 (matching the statement's number so the true verdict flips to supported)
    unsup_b = b_by_src["R-UNSUP-FLIP"]
    unsup_line = unsup_b["material"]["text"].splitlines()[0]
    assert unsup_line == "In 2018 the archive held 500 ledgers."

    # Also verify unsupported (f) trace frame on R-UNSUP-FLIP
    f_item = next(it for it in gpu_items if it["source_row_id"] == "R-UNSUP-FLIP" and it["perturbation_type"] == "f")
    assert f_item["checker_trace"] == (
        'The material says: "In 2018 the archive held 320 ledgers.". '
        "That contradicts the statement, so it is unsupported."
    )

    # 5. Unsupported differing in 2 numbers (2018 vs 2019 AND 320 vs 500) -> skipped
    assert "R-UNSUP-SKIP-TWO-DIFF" not in b_by_src
    assert "R-UNSUP-SKIP-TWO-DIFF" in skipped_b


def test_perturb_e_skips_single_line_material_without_truncating(tmp_path: Path):
    """Fix 6 test: single-line material with no other subject to swap is skipped for (e), never truncated."""
    rows = [
        {
            "type": "draft",
            "id": "R-SINGLE-LINE",
            "slot": {"id": "R-SINGLE-LINE", "kind": "D"},
            "material": {"text": "The boiler pressure stays below 40 psi.", "source": "s1"},
            "statement": "The boiler pressure stays below 40 psi.",
            "key": {"verdict": "supported", "deciding": ["The boiler pressure stays below 40 psi."]},
        }
    ]
    in_path = tmp_path / "single.jsonl"
    waves.write(in_path, {"type": "header", "format": waves.FORMAT}, rows)
    out_dir = tmp_path / "single-out"
    manifest = perturb.run_perturb(in_path, out_dir, seed=0)
    _, gpu_items = waves.read(out_dir / "items.jsonl")
    assert not [it for it in gpu_items if it["perturbation_type"] == "e"]
    skipped_e = [s for s in manifest["skipped"] if s["perturbation_type"] == "e"]
    assert len(skipped_e) == 1
    assert skipped_e[0]["source_row_id"] == "R-SINGLE-LINE"
    assert "single-line material" in skipped_e[0]["reason"]


def test_perturb_two_turn_edits_new_material_and_preserves_turn1_key(tmp_path: Path):
    """Fix 2 perturb test: on a two-turn D item whose turn-2 deciding line lives in turns[-1].new_material:
    - (b) edits the line inside new_material, leaves base material and turn_keys['1'] untouched, and records turn=2;
    - (e) edits new_material and records turn=2;
    - (g) swaps turn 2's deciding line for a decoy while keeping turn_keys['1'] intact;
    - (a) flips turn 2's verdict while keeping turn_keys['1'] intact."""
    two_turn_row = {
        "type": "draft",
        "id": "R-D-2T",
        "slot": {"id": "R-D-2T", "kind": "D", "two_turn": "changes"},
        "material": {
            "text": "The desk is open on Sundays from 10:00 to 14:00.\nThe side door is locked after 18:00.",
            "source": "s-2t",
        },
        "row": {
            "id": "R-D-2T",
            "kind": "D",
            "material": "The desk is open on Sundays from 10:00 to 14:00.\nThe side door is locked after 18:00.",
            "statement": "The courier drop box accepts 12 parcels on Mondays.",
            "turns": [
                {"user": "Check:\nMaterial:\nThe desk is open on Sundays from 10:00 to 14:00.\nThe side door is locked after 18:00.\nStatement: The courier drop box accepts 12 parcels on Mondays.", "thinking": "t1", "reply": "r1"},
                {"user": "New line pinned: The courier drop box accepts 12 parcels on Mondays.", "thinking": "t2", "reply": "r2", "new_material": ["The courier drop box accepts 12 parcels on Mondays."]},
            ],
            "turn_keys": {"1": {"verdict": "cannot_tell", "deciding": [], "escalate": "no"}},
            "verdict": "supported",
            "deciding": ["The courier drop box accepts 12 parcels on Mondays."],
            "escalate": "no",
        },
    }
    in_path = tmp_path / "two-turn.jsonl"
    waves.write(in_path, {"type": "header", "format": waves.FORMAT}, [two_turn_row])

    out_dir = tmp_path / "two-turn-out"
    manifest = perturb.run_perturb(in_path, out_dir, seed=11)
    _, gpu_items = waves.read(out_dir / "items.jsonl")
    _, keys_a = waves.read(out_dir / "keys-a.jsonl")
    _, keys_g = waves.read(out_dir / "keys-g.jsonl")

    item_b = next(it for it in gpu_items if it["perturbation_type"] == "b")
    # Base material is untouched; the edit is inside new_material!
    assert item_b["material"]["text"] == two_turn_row["material"]["text"]
    assert item_b["new_material"] != ["The courier drop box accepts 12 parcels on Mondays."]
    assert item_b["row"]["turns"][-1]["new_material"] == item_b["new_material"]
    assert "12" not in item_b["new_material"][0]
    assert item_b["turn_keys"]["1"] == {"verdict": "cannot_tell", "deciding": [], "escalate": "no"}
    b_manifest = next(m for m in manifest["items"] if m["perturbation_type"] == "b")
    assert b_manifest["turn"] == 2
    assert b_manifest["changes"]["material"]["location"] == "new_material"

    # (a) flips the final turn's verdict while turn_keys["1"] stays cannot_tell
    assert keys_a[0]["key"]["verdict"] == "unsupported"
    assert keys_a[0]["turn_keys"]["1"]["verdict"] == "cannot_tell"
    a_manifest = next(m for m in manifest["items"] if m["perturbation_type"] == "a")
    assert a_manifest["turn"] == 2

    # (g) swaps turn 2's deciding line for a decoy while turn_keys["1"] stays empty
    assert keys_g[0]["key"]["deciding"] != ["The courier drop box accepts 12 parcels on Mondays."]
    assert keys_g[0]["turn_keys"]["1"]["deciding"] == []
    g_manifest = next(m for m in manifest["items"] if m["perturbation_type"] == "g")
    assert g_manifest["turn"] == 2


def test_unsealed_or_keyless_input_refuses(tmp_path: Path):
    unsealed = tmp_path / "unsealed.jsonl"
    unsealed.write_text('{"type": "header", "format": "role-set waves 1"}\n{"type": "draft", "id": "1"}\n', encoding="utf-8")
    assert perturb.main(["--input", str(unsealed), "--out", str(tmp_path / "out1"), "--seed", "0"]) == 2

    # Sealed file, but row lacks a key block
    keyless = tmp_path / "keyless.jsonl"
    waves.write(
        keyless,
        {"type": "header", "format": waves.FORMAT},
        [
            {
                "type": "draft",
                "id": "R1",
                "slot": {"id": "R1", "kind": "D"},
                "material": {"text": "Line 1.\nLine 2."},
                "statement": "Line 1.",
            }
        ],
    )
    with pytest.raises(perturb.PerturbError, match="REFUSED: input row .* lacks a valid key verdict"):
        perturb.run_perturb(keyless, tmp_path / "out2", seed=0)


def test_planted_none_draft_written_with_blank_checker_verdict_perturbs(tmp_path: Path):
    from role_set.assemble import Material, assemble
    from role_set.mock import mock_gates
    from role_set.plan import Slot
    from role_set.runner import Runner
    from role_set.transport import Completion
    from role_set.waves import write_wave

    slot = Slot(
        id="T18-C-02",
        lesson="T18",
        kind="C",
        side="doesn't rewrite the statement",
        tier=2,
        content_kind="label",
        verdict="unsupported",
        escalate="no",
        real_source=True,
    )
    mat = Material(
        "Pure Cotton Blend\nMachine wash cold with like colors.\nTumble dry low.\nDo not bleach.\n",
        {"repo": "x/y", "commit": "abc", "path": "label.txt", "lines": "L1-L4", "licence": "MIT"},
    )
    raw_fields = {
        "framing": "Looking at the care label for this apron, which line settles the claim?",
        "statement": "The apron is made from a synthetic blend and should be tumble dried on high.",
        "checker_trace": (
            "The label says 'Pure Cotton Blend', contradicting 'synthetic blend', and 'Tumble dry low.', "
            "contradicting 'tumble dried on high', so the claim is unsupported."
        ),
        "checker_verdict": "",
        "deciding": ["Pure Cotton Blend", "Tumble dry low."],
        "thinking": (
            "Line 1 states 'Pure Cotton Blend' and line 3 states 'Tumble dry low.', which contradict the claim without "
            "needing to rewrite the statement."
        ),
        "reply": "The other checker identified both contradictions and reached unsupported without rewriting the claim.",
    }

    # Direct assemble fills fields["checker_verdict"] with the same value row["checker_verdict"] gets.
    fields_copy = dict(raw_fields)
    built_row = assemble(slot, "doesn't fix things", fields_copy, mat)
    assert built_row["checker_verdict"] == "unsupported"
    assert fields_copy["checker_verdict"] == "unsupported"

    # Write wave with a blank checker_verdict on a planted-none slot writes the filled checker_verdict into candidate.text
    pack = {
        "traits/T18.md": "## 18. As a verifier, you don't fix things\n\nJudge the statement as written.\n",
        "kinds.md": "kinds\n",
        "amendments/1.md": "## Amendment 1\n\nq\n",
    }

    def writer(messages, seed=0):
        text = messages[-1]["content"]
        if "In your own words: the trait's main point" in text:
            return Completion(
                text=json.dumps({
                    "main_point": "doesn't fix things: judge as written",
                    "counterweight": "still quote the deciding line",
                    "deciding_case": "a care label contradiction",
                }),
                prompt_tokens=10,
                completion_tokens=10,
                model="mock",
                seconds=0.01,
            )
        return Completion(
            text=json.dumps(raw_fields),
            prompt_tokens=20,
            completion_tokens=20,
            model="mock",
            seconds=0.01,
        )

    runner = Runner(
        gates=mock_gates(),
        pack=pack,
        trait_names={"T18": "doesn't fix things"},
        writer=writer,
        critic=None,
        out_dir=tmp_path / "w0",
        excerpts={"label": [mat]},
    )
    drafts = write_wave(runner, [slot])
    assert len(drafts) == 1
    d = drafts[0]
    assert d["state"] == "awaiting_critique"
    assert d["row"] is None
    assert json.loads(d["candidate"]["text"])["checker_verdict"] == "unsupported"

    # Attach gold key (as in Track A) and run perturb: perturbs cleanly without loosening perturb's check
    d_keyed = {
        **d,
        "key": {
            "verdict": "unsupported",
            "deciding": ["Pure Cotton Blend", "Tumble dry low."],
            "planted_flaw": "none",
        },
    }
    in_path = tmp_path / "t18-c-02.jsonl"
    waves.write(in_path, {"type": "header", "format": waves.FORMAT}, [d_keyed])
    out_dir = tmp_path / "t18-perturb"
    manifest = perturb.run_perturb(in_path, out_dir, seed=0)
    assert manifest["counts"]["d"] == 1
    _, gpu_items = waves.read(out_dir / "items.jsonl")
    ctrl = next(it for it in gpu_items if it["perturbation_type"] == "d")
    assert ctrl["checker_verdict"] == "unsupported"


