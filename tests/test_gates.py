import types

import pytest

from role_set.gates import GatesUnavailable, load_gates


def _fake_module():
    m = types.SimpleNamespace()
    m.calls = []
    m.check_item = lambda row: m.calls.append(("item", row["id"]))
    m.check_against = lambda row, accepted: m.calls.append(("against", row["id"], len(accepted))) or []
    m.check_batch = lambda rows: m.calls.append(("batch", len(rows)))
    m.role_words_in_prompt = lambda row: ("verifier",) if "verifier" in row.get("user", "") else ()
    m.DataError = ValueError
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {"verdict": reply.split()[-1]})
    m.gates_identity = lambda: {"version": "1.0.0", "sha256": "ab" * 32}
    m.norm = lambda x: " ".join(x.split())
    m.critic_mismatch = lambda key, critic, material: False
    return m


def test_load_gates_wraps_the_package():
    fake = _fake_module()
    g = load_gates(importer=lambda name: fake)
    assert (g.version, g.sha256[:4]) == ("1.0.0", "abab")
    g.check_item({"id": "T02-A-01"})
    assert g.check_against({"id": "T02-A-02"}, [{}, {}]) == []
    g.check_batch([{}, {}, {}])
    assert fake.calls == [("item", "T02-A-01"), ("against", "T02-A-02", 2), ("batch", 3)]
    assert g.parse_answer("VERDICT: supported")["verdict"] == "supported"
    assert g.role_words_in_prompt({"user": "as a verifier"}) == ["verifier"]
    assert g.error_type() is ValueError


def test_missing_package_is_a_refusal():
    def importer(name):
        raise ImportError(name)

    with pytest.raises(GatesUnavailable, match="pin it"):
        load_gates(importer=importer)


def test_gates_without_the_scorer_rules_are_refused_at_load():
    for missing in ("norm", "critic_mismatch"):
        fake = _fake_module()
        delattr(fake, missing)
        with pytest.raises(GatesUnavailable, match=f"has no {missing}; pin 1.1.0 or later"):
            load_gates(importer=lambda name, f=fake: f)
