import types

import pytest

from role_set.gates import GatesUnavailable, load_gates


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
    m.check_against = lambda row, accepted: m.calls.append(("against", row["id"], len(accepted))) or []
    m.check_batch = lambda rows, closed_form=None: m.calls.append(("batch", len(rows), (closed_form or {}).get("carry")))
    m.role_words_in_prompt = lambda row: ("verifier",) if "verifier" in row.get("user", "") else ()
    m.DataError = ValueError
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {"verdict": reply.split()[-1]})
    m.gates_identity = lambda: {"version": "1.4.0", "sha256": "ab" * 32}
    m.norm = lambda x: " ".join(x.split())
    m.critic_mismatch = lambda key, critic, material: False
    m.closed_form_settings = _settings
    m.key_point_share = lambda key_point, option: 0.0
    return m


def test_load_gates_wraps_the_package():
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake)
    assert (g.version, g.sha256[:4]) == ("1.4.0", "abab")
    g.check_item({"id": "T02-A-01"})
    assert g.check_against({"id": "T02-A-02"}, [{}, {}]) == []
    g.check_batch([{}, {}, {}])
    # The package's default closed-form settings reach every check.
    assert fake.calls == [("item", "T02-A-01", "paraphrase"), ("against", "T02-A-02", 2), ("batch", 3, "paraphrase")]
    assert g.parse_answer("VERDICT: supported")["verdict"] == "supported"
    assert g.role_words_in_prompt({"user": "as a verifier"}) == ["verifier"]
    assert g.error_type() is ValueError


def test_missing_package_is_a_refusal():
    def importer(name):
        raise ImportError(name)

    with pytest.raises(GatesUnavailable, match="pin it"):
        load_gates(importer=importer)


def test_gates_without_the_scorer_or_closed_form_rules_are_refused_at_load():
    for missing in ("norm", "critic_mismatch", "closed_form_settings", "key_point_share"):
        fake = _fake_module()
        delattr(fake, missing)
        with pytest.raises(GatesUnavailable, match=f"has no {missing}; pin 1.4.0 or later"):
            load_gates(importer=lambda name, f=fake: f)


def test_closed_form_settings_are_resolved_through_the_gates_and_reach_every_check():
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake, closed_form={"carry": "either"})
    assert g.closed_form == {**DEFAULTS, "carry": "either"} and g.closed_form_carry == "either"
    g.check_item({"id": "T02-E-05"})
    g.check_batch([{}])
    assert fake.calls == [("item", "T02-E-05", "either"), ("batch", 1, "either")]
    with pytest.raises(GatesUnavailable, match="closed-form settings the gates reject"):
        load_gates(importer=lambda name: _fake_module(), closed_form={"carry": "loose"})


def test_a_stand_in_without_settings_calls_the_checks_as_before():
    from role_set.gates import Gates

    fake = _fake_module()
    g = Gates(module=fake, version="t", sha256="t")
    g.check_item({"id": "T02-A-01"})
    g.check_batch([{}])
    assert g.closed_form_carry is None and fake.calls == [("item", "T02-A-01", None), ("batch", 1, None)]


def test_trait_words_come_from_the_gates_constants():
    fake = _fake_module()
    fake.constants = types.SimpleNamespace(TRAIT_NAMES={"T02": ["to the source", "goes to the source"]})
    g = load_gates(importer=lambda name: fake)
    assert g.trait_words("T02") == ["to the source", "goes to the source"] and g.trait_words("T99") == []
    bare = _fake_module()
    assert load_gates(importer=lambda name: bare).trait_words("T02") == []
