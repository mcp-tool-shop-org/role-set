import types

import pytest

from role_set.gates import GatesUnavailable, ModelsMissing, load_gates


DEFAULTS = {"carry": "paraphrase", "share": 0.5, "distractor_max": 0.8, "distractor_margin": 0.15, "span": 8,
            "exempt_keyed_key_points": False}


def _settings(cfg=None):
    out = {**DEFAULTS, **(cfg or {})}
    if out["carry"] not in ("verbatim", "paraphrase", "either"):
        raise ValueError(f"closed-form settings out of range: {out}")
    return out


def _fake_module():
    m = types.SimpleNamespace()
    m.calls = []
    m.check_item = lambda row, closed_form=None: m.calls.append(("item", row["id"], (closed_form or {}).get("carry")))
    m.check_against = lambda row, accepted, models=None: m.calls.append(("against", row["id"], len(accepted), models)) or []
    m.check_batch = lambda rows, closed_form=None, models=None: m.calls.append(("batch", len(rows), (closed_form or {}).get("carry")))
    m.role_words_in_prompt = lambda row: ("verifier",) if "verifier" in row.get("user", "") else ()
    m.DataError = ValueError
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {"verdict": reply.split()[-1]})
    m.gates_identity = lambda: {"version": "1.4.0", "sha256": "ab" * 32}
    m.norm = lambda x: " ".join(x.split())
    m.critic_mismatch = lambda key, critic, material: False
    m.closed_form_settings = _settings
    m.key_point_share = lambda key_point, option: 0.0
    m.check_trace_leaks_label = lambda row: m.calls.append(("leak", row["id"]))
    return m


def test_load_gates_wraps_the_package():
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake)
    assert (g.version, g.sha256[:4]) == ("1.4.0", "abab")
    g.check_item({"id": "T02-A-01"})
    assert g.check_against({"id": "T02-A-02"}, [{}, {}], models={}) == []
    g.check_batch([{}, {}, {}], models={})
    # The package's default closed-form settings reach every check.
    assert fake.calls == [("item", "T02-A-01", "paraphrase"), ("against", "T02-A-02", 2, {}), ("batch", 3, "paraphrase")]
    assert g.parse_answer("VERDICT: supported")["verdict"] == "supported"
    assert g.role_words_in_prompt({"user": "as a verifier"}) == ["verifier"]
    assert g.error_type() is ValueError


def test_missing_package_is_a_refusal():
    def importer(name):
        raise ImportError(name)

    with pytest.raises(GatesUnavailable, match="pin it"):
        load_gates(importer=importer)


def test_gates_without_the_scorer_or_closed_form_rules_are_refused_at_load():
    from role_set.gates import RULE_VERSIONS

    for missing, ver in RULE_VERSIONS.items():
        fake = _fake_module()
        delattr(fake, missing)
        with pytest.raises(GatesUnavailable, match=rf"has no {missing} \(added in {missing} \({ver}\)\); pin {ver} or later"):
            load_gates(importer=lambda name, f=fake: f)


def test_closed_form_settings_are_resolved_through_the_gates_and_reach_every_check():
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake, closed_form={"carry": "either"})
    assert g.closed_form == {**DEFAULTS, "carry": "either"} and g.closed_form_carry == "either"
    g.check_item({"id": "T02-E-05"})
    g.check_batch([{}], models={})
    assert fake.calls == [("item", "T02-E-05", "either"), ("batch", 1, "either")]
    with pytest.raises(GatesUnavailable, match="closed-form settings the gates reject"):
        load_gates(importer=lambda name: _fake_module(), closed_form={"carry": "loose"})


def test_a_stand_in_without_settings_calls_the_checks_as_before():
    from role_set.gates import Gates

    fake = _fake_module()
    g = Gates(module=fake, version="t", sha256="t")
    g.check_item({"id": "T02-A-01"})
    g.check_batch([{}], models={})
    assert g.closed_form_carry is None and fake.calls == [("item", "T02-A-01", None), ("batch", 1, None)]


def test_trait_words_come_from_the_gates_constants():
    fake = _fake_module()
    fake.constants = types.SimpleNamespace(TRAIT_NAMES={"T02": ["to the source", "goes to the source"]})
    g = load_gates(importer=lambda name: fake)
    assert g.trait_words("T02") == ["to the source", "goes to the source"] and g.trait_words("T99") == []
    bare = _fake_module()
    assert load_gates(importer=lambda name: bare).trait_words("T02") == []


def test_puzzles_reach_the_gates_with_their_models():
    # R&D (2026-10-15): check_against signs puzzles by the sidecar model; without one it falls back to the regex
    # signature. The wrapper passes the map through, and a kind B row missing from it stops the check.
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake)
    puzzle, prior = {"id": "T11-B-01", "kind": "B"}, {"id": "T11-B-02", "kind": "B"}
    models = {"T11-B-01": {"entities": ["x"]}, "T11-B-02": {"entities": ["y"]}}
    assert g.check_against(puzzle, [prior], models=models) == []
    assert fake.calls[-1] == ("against", "T11-B-01", 1, models)
    with pytest.raises(ModelsMissing, match="T11-B-02"):
        g.check_against(puzzle, [prior], models={"T11-B-01": {}})
    with pytest.raises(ModelsMissing, match="T11-B-01"):
        g.check_batch([puzzle, {"id": "T11-A-01", "kind": "A"}], models={})
    with pytest.raises(TypeError):
        g.check_against(puzzle, [prior])  # models has no default: a caller can't forget it


def test_verification_b_rows_without_models_pass_while_puzzles_require_models():
    verif_b = {
        "id": "T18-B-01",
        "lesson": "T18",
        "kind": "B",
        "turns": [{"user": "u", "thinking": "t", "reply": "The line settles it.\nVERDICT: supported\nDECIDING: line 1\nESCALATE: no"}],
    }
    puzzle_b = {
        "id": "T11-B-01",
        "lesson": "T11",
        "kind": "B",
        "turns": [{"user": "u", "thinking": "t", "reply": "1. A is out.\nAnswer: B"}],
    }
    tagged_puzzle_b = {
        "id": "T18-B-02",
        "lesson": "T18",
        "kind": "B",
        "tags": ["non-verification"],
        "turns": [{"user": "u", "thinking": "t", "reply": "The line settles it.\nVERDICT: supported\nDECIDING: NONE\nESCALATE: no"}],
    }
    fake = _fake_module()
    fake._is_puzzle = lambda r: r.get("kind") == "B" and (
        "non-verification" in (r.get("tags") or [])
        or "VERDICT:" not in ((r.get("turns") or [{}])[-1].get("reply") or "")
    )
    g = load_gates(importer=lambda name: fake)
    assert g.check_against({"id": "T10-D-AP-01", "kind": "D"}, [verif_b], models={}) == []
    g.check_batch([verif_b, {"id": "T10-A-01", "kind": "A"}], models={})
    with pytest.raises(ModelsMissing, match="T11-B-01"):
        g.check_against({"id": "T10-D-AP-01", "kind": "D"}, [verif_b, puzzle_b], models={})
    with pytest.raises(ModelsMissing, match="T11-B-01"):
        g.check_batch([verif_b, puzzle_b], models={})
    with pytest.raises(ModelsMissing, match="T18-B-02"):
        g.check_against(tagged_puzzle_b, [verif_b], models={})

    real = load_gates()
    assert callable(getattr(real.module, "_is_puzzle", None))
    assert not real.module._is_puzzle(verif_b)
    assert real.module._is_puzzle(puzzle_b)
    assert real.module._is_puzzle(tagged_puzzle_b)
    with pytest.raises(ModelsMissing, match="T11-B-01"):
        real.check_against({"id": "T10-D-AP-01", "kind": "D"}, [verif_b, puzzle_b], models={})


def test_apply_slot_critique_with_verification_b_training_row_without_model_is_accepted(tmp_path):
    from role_set.mock import MockCritic, MockWriter, mock_excerpts
    from role_set.plan import plan_apply
    from role_set.runner import Runner

    pack = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you read literally\n\nRead what is written.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    traits = {"T10": "reads literally"}
    verif_b = {
        "id": "T18-B-01",
        "lesson": "T18",
        "trait": "checks consistency",
        "kind": "B",
        "side": "s",
        "content_kind": "notice",
        "tier": 2,
        "sources": [{"invented": "notice, written for this item"}],
        "invented": True,
        "key_basis": "construction",
        "tags": [],
        "material": "Line one of the notice.\nLine two of the notice.\n",
        "verdict": "supported",
        "deciding": ["Line one of the notice."],
        "escalate": "no",
        "turns": [{
            "user": "Framing.\nMaterial:\nLine one of the notice.\nLine two of the notice.\nStatement: Line one holds.",
            "thinking": "Line one settles it.",
            "reply": "Line one settles it.\nVERDICT: supported\nDECIDING: Line one of the notice.\nESCALATE: no",
        }],
    }
    answers: dict = {}
    writer = MockWriter(answers=answers)
    critic = MockCritic(answers=answers, trait="T10")
    runner = Runner(
        gates=load_gates(),
        pack=pack,
        trait_names=traits,
        writer=writer,
        critic=critic,
        out_dir=tmp_path / "out",
        excerpts=mock_excerpts(),
        training=[verif_b],
        training_models={},
    )
    slot = next(s for s in plan_apply("T10", ("reads literally", "neither strengthens nor weakens"), seed=0) if s.kind == "D")
    out = runner.run_slot(slot)
    assert out.accepted, out.refusals


def test_mock_write_then_critique_over_plan_with_apply_slot_accepts_apply_row(tmp_path):
    import json
    from role_set import waves
    from role_set.cli import main
    from role_set.mock import MockCritic, answers_from_drafts
    from role_set.pack import PackManifest, pack_hash
    from role_set.write import WriteConfig, run

    root = tmp_path / "pack"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you read literally\n\nRead what is written.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for name, text in trait_files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(trait_files)
    (root / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    plan = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T10", "--seed", "0", "--out", str(plan), "--apply"]) == 0
    plan_data = json.loads(plan.read_text(encoding="utf-8"))
    plan_data["pilot"] = True
    # Keep a small plan of two apply slots (one supported C, one cannot_tell D) so both paths are exercised.
    c_slot = next(s for s in plan_data["slots"] if s["kind"] == "C" and s["verdict"] == "supported")
    d_slot = next(s for s in plan_data["slots"] if s["kind"] == "D" and s["verdict"] == "cannot_tell")
    plan_data["slots"] = [c_slot, d_slot]
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    verif_b = {
        "id": "T18-B-01",
        "lesson": "T18",
        "trait": "checks consistency",
        "kind": "B",
        "side": "s",
        "content_kind": "notice",
        "tier": 2,
        "sources": [{"invented": "notice, written for this item"}],
        "invented": True,
        "key_basis": "construction",
        "tags": [],
        "material": "Line one of the notice.\nLine two of the notice.\n",
        "verdict": "supported",
        "deciding": ["Line one of the notice."],
        "escalate": "no",
        "turns": [{
            "user": "Framing.\nMaterial:\nLine one of the notice.\nLine two of the notice.\nStatement: Line one holds.",
            "thinking": "Line one settles it.",
            "reply": "Line one settles it.\nVERDICT: supported\nDECIDING: Line one of the notice.\nESCALATE: no",
        }],
    }
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps(verif_b) + "\n", encoding="utf-8", newline="\n")
    (tmp_path / "train.models.jsonl").write_text("", encoding="utf-8", newline="\n")

    w0 = run(WriteConfig(plan=plan, pack=root, out=tmp_path / "w0", mock=True, partition="trial:apply", wave="write", training=[train_file]))
    assert w0["states"]["awaiting_critique"] == 2
    d0_path = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    h0, d0 = waves.read(d0_path)
    # Strip mock_answers from the header to prove a critique wave seeds from drafts by default even without it.
    h0.pop("mock_answers", None)
    waves.write(d0_path, h0, d0)
    assert answers_from_drafts(d0)

    w1 = run(WriteConfig(plan=plan, pack=root, out=tmp_path / "w1", mock=True, partition="trial:apply", wave="critique", drafts=d0_path, training=[train_file]))
    assert w1["accepted"] >= 1 and w1["states"]["accepted"] == 2
    accepted_rows = [json.loads(x) for x in (tmp_path / "w1" / "mock" / "accepted.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(r.get("split") == "apply" for r in accepted_rows)

    # Also verify MockCritic's unseeded HTTP/StubLlama fallback on a cannot_tell apply item with r10_twin.
    unseeded = MockCritic(answers={})
    r1 = unseeded([{"role": "system", "content": "verdict"}, {"role": "user", "content": "Material:\nFirst line.\n\nStatement: Main claim."}])
    r2 = unseeded([{"role": "system", "content": "verdict"}, {"role": "user", "content": "Material:\nFirst line.\n\nStatement: Twin claim."}])
    assert "VERDICT: cannot_tell" in r1.text and "VERDICT: supported" in r2.text


