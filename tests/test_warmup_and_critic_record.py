"""PR D: the warm-up retry marks the missed point and accepts the lesson's own terms, without becoming a rubber stamp;
every critic check is recorded per draft; the pilot launch script carries no machine paths."""

import json
import types
from pathlib import Path

from role_set.plan import plan_lesson
from role_set.runner import lesson_terms
from test_runner import SIDES, _runner

T07 = ("## 7. As a verifier, you keep things in proportion\n\n"
       "As a verifier, you're quick on the obvious and careful on the hard. If one line settles it, you say so and\n"
       "stop. If it needs several steps, you take them. This matters to a verifier because overthinking a simple\n"
       "case wastes effort and invites doubt where there's none, while rushing a hard case misses the part that's\n"
       "wrong.\n\n**A short trace beside a long one.**\n")

# The rehearsal's qwen3:8b restatement of T07 (2026-10-11): accurate, but without the word "proportion".
ACCURATE = json.dumps({"main_point": "As a verifier, you should balance speed and thoroughness, recognizing when a case is simple enough to stop quickly and when it requires careful, step-by-step analysis.",
                       "counterweight": "Avoid overthinking simple cases, which wastes effort and introduces unnecessary doubt, and avoid underthinking complex cases.",
                       "deciding_case": "A notice that says 'Closed on Mondays' settles 'It's closed on Mondays' in one line, so I stop."})
VAGUE = json.dumps({"main_point": "Be accurate and thorough in every check.", "counterweight": "Don't be sloppy.", "deciding_case": "Any check at all."})
OFF_TRAIT = json.dumps({"main_point": "Never invent evidence or quote words that aren't there.", "counterweight": "Don't assume the missing piece.",
                        "deciding_case": "A parcel sent on Monday says nothing about its arrival, so cannot_tell."})


def _say(text):
    return types.SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, model="m", seconds=0)


def _t07_runner(tmp_path, replies):
    r, _, _ = _runner(tmp_path)
    r.pack = {**r.pack, "traits/T07.md": T07}
    r.trait_names = {**r.trait_names, "T07": "keeps things in proportion"}
    asked = []

    def writer(messages, seed=0):
        asked.append(messages)
        return _say(replies[min(len(asked), len(replies)) - 1])

    r.writer = writer
    return r, asked


def test_an_accurate_restatement_in_the_lessons_terms_passes_without_the_trait_name(tmp_path: Path):
    assert {"quick", "careful", "obvious", "overthinking", "simple", "rushing"} <= lesson_terms(T07)
    r, asked = _t07_runner(tmp_path, [ACCURATE])
    assert r.warmup("T07") and len(asked) == 1
    kept = [json.loads(x) for x in (tmp_path / "out" / "warmups.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept[0]["passed"] and kept[0]["terms"] == [] and {"careful", "simple", "overthinking"} <= set(kept[0]["lesson_terms"])


def test_a_vague_or_off_trait_restatement_still_fails(tmp_path: Path):
    for reply in (VAGUE, OFF_TRAIT):
        r, asked = _t07_runner(tmp_path / reply[:12].strip("{\" :"), [reply])
        assert not r.warmup("T07") and len(asked) == 2


def test_the_retry_marks_the_missed_point(tmp_path: Path):
    r, asked = _t07_runner(tmp_path, [VAGUE, ACCURATE])
    assert r.warmup("T07") and len(asked) == 2
    retry = asked[1]
    assert retry[-2] == {"role": "assistant", "content": VAGUE}
    assert "name this trait as the lesson does: 'keeps things in proportion'" in retry[-1]["content"]


def test_every_critic_check_is_recorded_on_the_draft_agreeing_or_not(tmp_path: Path):
    r, _, _ = _runner(tmp_path)
    out = r.run_slot(next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "C" and s.verdict != "cannot_tell"))
    assert out.accepted
    drafts = [json.loads(x) for x in (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    checks = drafts[-1]["critic"]
    assert [c["check"] for c in checks][:1] == ["verdict"] and checks[-1]["check"] == "trait" and all(c["code"] == "AGREE" for c in checks if c["check"] != "flaw")
    assert "votes T02, T02, T02" in checks[-1]["detail"] and json.loads(checks[-1]["raw"])


def test_a_draft_that_never_reached_the_critic_has_no_critic_field(tmp_path: Path):
    r, _, _ = _runner(tmp_path)
    real, calls = r.writer, []

    def first_unparseable(messages, seed=0):
        calls.append(seed)
        return _say("not json") if len(calls) == 1 else real(messages, seed)

    r.writer = first_unparseable
    r.run_slot(next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A"))
    drafts = [json.loads(x) for x in (tmp_path / "out" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert drafts[0]["outcome"] == "PARSE" and "critic" not in drafts[0] and "critic" in drafts[-1]


def test_the_pilot_launch_script_carries_no_machine_paths():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "pilot-write.sh").read_text(encoding="utf-8")
    for needle in ("C:/", "C:\\", "E:/", "/e/AI", "Users/", "mikey"):
        assert needle not in script, needle
    assert "RS_HEARTBEAT:?" in script and "RS_EXEMPLAR_REPO:?" in script and "keep_alive" in script and "timeout -k 15" in script
