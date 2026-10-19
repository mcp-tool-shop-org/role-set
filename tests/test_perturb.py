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

    # Verify (f)'s trace uses ONLY words from the item's own key (deciding quote + verdict).
    f_items = [it for it in gpu_items if it["perturbation_type"] == "f"]
    assert f_items
    word_re = re.compile(r"[A-Za-z0-9_]+")
    for it in f_items:
        key_words = set(word_re.findall(" ".join([*it["key"]["deciding"], it["key"]["verdict"]]).lower()))
        trace_words = set(word_re.findall(it["checker_trace"].lower()))
        assert trace_words.issubset(key_words), f"extra words in (f) trace: {trace_words - key_words}"


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
