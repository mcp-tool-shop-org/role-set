import json
import types
from pathlib import Path

from role_set.critic_control import Case, cases_from_rows, load_rows, plant_errors, run_control
from role_set.gates import Gates

MATERIAL = "Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n"


def _gates():
    m = types.SimpleNamespace()
    m.critic_mismatch = lambda key, critic, material: not ({" ".join(x.split()) for x in key} & {" ".join(x.split()) for x in critic})
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    return Gates(module=m, version="t", sha256="t")


def _rows():
    user_c = f"Check it.\nMaterial:\n{MATERIAL}Statement: The desk is closed on Sundays.\nChecker verdict: supported\nThe other checker skimmed."
    return [
        {"id": "T02-C-01", "kind": "C", "material": MATERIAL, "verdict": "unsupported", "deciding": ["The desk is open on Sundays from 10:00 to 14:00."],
         "planted_flaw": "guess", "checker_verdict": "supported", "turns": [{"user": user_c, "thinking": "", "reply": ""}]},
        {"id": "T02-D-01", "kind": "D", "material": MATERIAL, "verdict": "supported", "deciding": ["Desk hours"],
         "turn_keys": {"1": {"verdict": "cannot_tell", "deciding": [], "escalate": "no"}},
         "turns": [{"user": f"Material:\n{MATERIAL}Statement: The desk sells stamps.", "thinking": "", "reply": ""}, {"user": "new", "thinking": "", "reply": ""}]},
        {"id": "T02-E-07", "kind": "E", "material": MATERIAL, "quiz": {"type": "verdict", "verdict": "supported", "deciding": ["The desk is open on Sundays from 10:00 to 14:00."], "escalate": "no"},
         "turns": [{"user": f"Material:\n{MATERIAL}Statement: The desk opens on Sundays at 10:00.", "thinking": "", "reply": ""}]},
        {"id": "T02-A-01", "kind": "A", "turns": [{"user": "passage", "thinking": "", "reply": ""}]},
        {"id": "T02-E-01", "kind": "E", "quiz": {"type": "written"}, "turns": [{"user": "q?", "thinking": "", "reply": ""}]},
    ]


class _Critic:
    """Answers correctly for known statements; a 'noisy' statement gets a wrong verdict."""

    def __init__(self, noisy=()):
        self.noisy = set(noisy)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        text = messages[-1]["content"]
        system = messages[0]["content"]
        if "Name the single flaw" in system:
            return types.SimpleNamespace(text="guess")
        stmt = text.split("Statement: ", 1)[1].split("\n", 1)[0].strip()
        if stmt in self.noisy:
            return types.SimpleNamespace(text="VERDICT: unsupported\nDECIDING: Desk hours\nESCALATE: no")
        if "closed on Sundays" in stmt:
            return types.SimpleNamespace(text="VERDICT: unsupported\nDECIDING: The desk is open on Sundays from 10:00 to 14:00.\nESCALATE: no")
        if "sells stamps" in stmt:
            return types.SimpleNamespace(text="VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no")
        return types.SimpleNamespace(text="VERDICT: supported\nDECIDING: The desk is open on Sundays from 10:00 to 14:00.\nESCALATE: no")


def test_cases_come_from_verdict_bearing_rows_only():
    cases = cases_from_rows(_rows())
    assert [c.item_id for c in cases] == ["T02-C-01", "T02-D-01", "T02-E-07"]
    assert cases[0].flaw == "guess" and "skimmed" in cases[0].checker_trace and "Checker verdict" not in cases[0].checker_trace
    assert cases[1].verdict == "cannot_tell" and cases[1].deciding == []  # the first turn's key, not the final one


def test_control_reports_agreement_false_refusals_and_catches(tmp_path: Path):
    cases = cases_from_rows(_rows())
    planted = plant_errors(cases, 2, seed=1)
    assert len(planted) == 2 and all(p.planted and p.item_id.endswith("+planted") for p in planted)
    critic = _Critic()
    rep = run_control(_gates(), critic, cases, planted=planted)
    d = rep.as_dict()
    assert d["items"] == 3 and d["agree"] == 3 and d["false_refusals"] == 0 and d["usable_as_critic"] is True
    assert d["planted"] == 2 and d["catches"] == 2 and d["flaw_asks"] == 1 and d["flaw_agree"] == 1

    noisy = _Critic(noisy={"The desk opens on Sundays at 10:00."})
    logged = []
    rep2 = run_control(_gates(), noisy, cases, planted=planted, log=logged.append)
    d2 = rep2.as_dict()
    assert d2["false_refusals"] == 1 and d2["false_refusal_detail"][0]["item"] == "T02-E-07" and d2["usable_as_critic"] is False

    f = tmp_path / "a.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _rows()) + "\n\n", encoding="utf-8")
    assert len(load_rows([f])) == 5


def test_void_answers_are_asked_again_and_counted():
    class Void:
        def __init__(self):
            self.n = 0

        def __call__(self, messages):
            self.n += 1
            return types.NamespaceType(text="I cannot say.") if False else types.SimpleNamespace(text="I cannot say.")

    case = Case("x", "D", MATERIAL, "The desk is open on Sundays.", "supported", ["Desk hours"])
    rep = run_control(_gates(), Void(), [case])
    assert rep.voids == 2 and len(rep.false_refusals) == 1 and rep.false_refusals[0]["code"] == "VOID"
