import json
import types
from pathlib import Path

from role_set.critic_control import Case, cases_from_rows, load_rows, plant_errors, run_agreement, run_control
from role_set.gates import Gates

MATERIAL = "Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n"


def _gates():
    m = types.SimpleNamespace()
    m.norm = lambda x: " ".join(x.replace("**", "").replace("__", "").split())
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

    def __call__(self, messages, **_):
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

        def __call__(self, messages, **_):
            self.n += 1
            return types.NamespaceType(text="I cannot say.") if False else types.SimpleNamespace(text="I cannot say.")

    case = Case("x", "D", MATERIAL, "The desk is open on Sundays.", "supported", ["Desk hours"])
    rep = run_control(_gates(), Void(), [case])
    assert rep.voids == 2 and len(rep.false_refusals) == 1 and rep.false_refusals[0]["code"] == "VOID"


def test_runs_record_each_item_at_their_seed_and_agreement_compares_them():
    """R&D, 2026-10-17: a sampling critic runs twice at different seeds; the per-item verdicts give run-to-run
    agreement. Concurrency doesn't change the report's order or counts."""

    cases = cases_from_rows(_rows())
    planted = plant_errors(cases, 2, seed=1)
    seen = []

    class Seeded(_Critic):
        def __call__(self, messages, seed=0, **kw):
            seen.append(seed)
            return super().__call__(messages)

    a = run_control(_gates(), Seeded(), cases, planted=planted, seed=0).as_dict()
    b = run_control(_gates(), Seeded(noisy={"The desk opens on Sundays at 10:00."}), cases, planted=planted, seed=1, concurrency=3).as_dict()
    assert a["seed"] == 0 and b["seed"] == 1
    assert [r["item"] for r in a["per_item"]] == [r["item"] for r in b["per_item"]] == [c.item_id for c in cases + planted]
    assert a["per_item"][0]["flaw_ok"] is True and a["per_item"][-1]["planted"] is True
    agree = run_agreement(a, b)
    # The noisy statement is T02-E-07's, so its planted copy (same statement) differs too: reported apart.
    assert agree["unplanted"] == {"items": 3, "differ": 1, "rate": 0.333, "differing": ["T02-E-07"]}
    planted_differ = [r["item"] for r in a["per_item"] if r["planted"] and r["item"].startswith("T02-E-07")]
    assert agree["planted"]["items"] == 2 and agree["planted"]["differing"] == planted_differ and agree["seeds"] == [0, 1]
    # Spaced seeds: run 0 asks at 0..1 (and 500 for a flaw), run 1 at 1000..1001 (and 1500).
    assert set(seen) <= {0, 1, 500, 1000, 1001, 1500} and {0, 1000} <= set(seen)


def test_control_forwards_content_kind_and_preserves_dunders_in_code_cases():
    code_mat = (
        "class Loader:\n"
        "    def __init__(self, config_path: str):\n"
        "        self.ready = True\n"
        "    def init(self, config_path: str):\n"
        "        self.ready = False\n"
    )
    stmt = "Loader defines __init__(self, config_path: str)."
    user_c = (
        f"Check it.\nMaterial:\n{code_mat}Statement: {stmt}\n"
        "Checker verdict: unsupported\nThe other checker misread the method name."
    )
    rows = [
        {
            "id": "T02-C-12",
            "kind": "C",
            "content_kind": "code",
            "material": code_mat,
            "verdict": "supported",
            "deciding": ["    def __init__(self, config_path: str):"],
            "planted_flaw": "guess",
            "checker_verdict": "unsupported",
            "turns": [{"user": user_c, "thinking": "", "reply": ""}],
        },
        {
            "id": "T02-D-12",
            "kind": "D",
            "content_kind": "code",
            "material": code_mat,
            "verdict": "supported",
            "deciding": ["    def __init__(self, config_path: str):"],
            "turns": [{"user": f"Material:\n{code_mat}Statement: {stmt}", "thinking": "", "reply": ""}],
        },
        {
            "id": "T02-E-12",
            "kind": "E",
            "content_kind": "code",
            "material": code_mat,
            "quiz": {
                "type": "verdict",
                "verdict": "cannot_tell",
                "deciding": [],
                "escalate": "no",
            },
            "turns": [{"user": f"Material:\n{code_mat}Statement: Loader caches files on disk.", "thinking": "", "reply": ""}],
        },
    ]
    cases = cases_from_rows(rows)
    assert [c.content_kind for c in cases] == ["code", "code", "code"]
    planted = plant_errors(cases, 3, seed=7)
    assert [p.content_kind for p in planted] == ["code", "code", "code"]

    def _code_critic(messages, **_):
        system = messages[0]["content"]
        text = messages[-1]["content"]
        if "Name the single flaw" in system:
            return types.SimpleNamespace(text="guess")
        if "caches files on disk" in text:
            return types.SimpleNamespace(text="VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no")
        # Backtick-wrapped `__init__` line: only resolves uniquely when `content_kind="code"` is passed
        # (stripping `__` makes both `def __init__` and `def init` collide and voids the reply).
        return types.SimpleNamespace(
            text="VERDICT: supported\nDECIDING: `def __init__(self, config_path: str):`\nESCALATE: no"
        )

    rep = run_control(_gates(), _code_critic, cases, planted=planted, seed=0).as_dict()
    assert rep["items"] == 3
    assert rep["agree"] == 3
    assert rep["false_refusals"] == 0
    assert rep["voids"] == 0
    assert rep["planted"] == 3
    assert rep["catches"] == 3

