"""BUILD-SPEC B2: a NAICS domain for every slot with material, and an opening frame for every framing.

The fixture table holds no real NAICS titles (R&D, 2026-10-16): the real table is built from the pinned census.gov
file once it is downloaded.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from role_set import domains, frames
from role_set.cli import main
from role_set.plan import plan_apply_batch, plan_batch
from conftest import fixture_xlsx as _xlsx, naics_script as _script

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def table_file(domain_table: Path) -> Path:
    return domain_table


def test_the_table_is_built_verbatim_from_the_pinned_source(table_file: Path, tmp_path: Path):
    data = json.loads(table_file.read_text(encoding="utf-8"))
    src = data["source"]
    assert src["registry_id"] == "census-naics-2022" and src["url"].startswith("https://www.census.gov/") and len(src["sha256"]) == 64
    subs = {e["code"]: e["title"] for e in data["subsectors"]}
    assert len(subs) >= 60 + 6 and subs["621"] == "Fixture title 621" and "5411" not in subs and "23" not in subs
    assert [e["code"] for e in data["escalate"]["medical"]] == ["621", "622", "623"]
    assert [e["code"] for e in data["escalate"]["safety"]] == ["23", "2211", "325", "481", "482", "5413", "9221"]
    assert data["escalate"]["legal"] == [{"code": "5411", "title": "Fixture title 5411"}]
    table = domains.load(table_file)
    assert table.pin()["domain_registry_id"] == "census-naics-2022" and len(table.sha256) == 64
    # A mapped code the source doesn't hold stops the build.
    short = _xlsx(tmp_path / "short.xlsx", [str(c) for c in range(100, 140)])
    with pytest.raises(SystemExit, match="mapped codes not in the source"):
        _script().build(short)


def test_a_table_without_its_pins_is_refused(table_file: Path, tmp_path: Path):
    good = json.loads(table_file.read_text(encoding="utf-8"))

    def refused(match, change):
        bad = json.loads(json.dumps(good))
        change(bad)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(domains.DomainTableError, match=match):
            domains.load(p)

    refused("registry_id", lambda d: d["source"].update(registry_id="apache-2.0"))
    refused("census.gov", lambda d: d["source"].update(url="https://example.org/naics.xlsx"))
    refused("sha256", lambda d: d["source"].update(sha256=None))
    refused("bad NAICS code", lambda d: d["subsectors"].append({"code": "1234", "title": "x"}))
    refused("no title", lambda d: d["subsectors"].append({"code": "199", "title": ""}))
    refused("repeats a code", lambda d: d["subsectors"].append(dict(d["subsectors"][0])))
    refused("no subsectors", lambda d: d.update(subsectors=[]))
    refused("cover exactly", lambda d: d["escalate"].pop("money"))
    refused("at least one code", lambda d: d["escalate"].update(money=[]))


def test_every_slot_with_material_gets_a_distinct_domain_and_every_framing_a_balanced_frame(table_file: Path):
    table = domains.load(table_file)
    lessons = {"T02": ("quotes the source", "stops at the source"), "T03": ("exact", "not fussy"), "T04": ("a", "b"), "T05": ("c", "d")}
    slots = plan_batch({k: lessons[k] for k in ("T02", "T03")}, seed=3)
    domains.assign(slots, table, seed=3)
    for lesson in ("T02", "T03"):
        mine = [s for s in slots if s.lesson == lesson]
        bearing = [s for s in mine if domains.bears_domain(s)]
        assert len(bearing) == 26 and all(s.domain for s in bearing)
        assert not any(s.domain for s in mine if s.kind == "B" or (s.kind == "E" and s.e_type == "written"))
        assert len({s.domain["code"] for s in bearing}) == 26
        esc = [s for s in mine if s.escalate.startswith("yes (")]
        dom = esc[0].escalate[5:].split(")")[0]
        assert esc[0].domain["escalate"] == dom and esc[0].domain["code"] in {e["code"] for e in table.escalate[dom]}
        counts = Counter(s.frame for s in mine if s.frame)
        assert sum(counts.values()) == 18 and set(counts) == set(frames.FRAMES) and max(counts.values()) - min(counts.values()) <= 1
        assert not any(s.frame for s in mine if s.kind == "A")
    # The same seed, the same plan.
    again = plan_batch({k: lessons[k] for k in ("T02", "T03")}, seed=3)
    domains.assign(again, table, seed=3)
    assert [s.as_dict() for s in again] == [s.as_dict() for s in slots]
    # Apply slots: C and D only, each with a domain and a frame.
    apply = plan_apply_batch({"T02": lessons["T02"]}, seed=3)
    domains.assign(apply, table, seed=3)
    assert all(s.domain and s.frame for s in apply)


def test_the_plan_checks_refuse_repeats_gaps_and_imbalance(table_file: Path):
    table = domains.load(table_file)
    slots = plan_batch({"T02": ("quotes the source", "stops at the source")}, seed=1)
    domains.assign(slots, table, seed=1)
    bearing = [s for s in slots if domains.bears_domain(s)]
    first = bearing[0].domain
    bearing[1].domain = dict(first)
    with pytest.raises(domains.DomainTableError, match="repeats a domain"):
        domains.check(slots)
    bearing[1].domain = None
    with pytest.raises(domains.DomainTableError, match="no domain"):
        domains.check(slots)
    bearing[1].domain = {"code": "999", "title": "x"}
    framed = [s for s in slots if s.frame]
    for s in framed[:6]:
        s.frame = "ask"
    with pytest.raises(domains.DomainTableError, match="out of balance"):
        domains.check(slots)
    framed[0].frame = None
    with pytest.raises(domains.DomainTableError, match="no frame"):
        domains.check(slots)


def test_the_deck_spreads_codes_across_a_batch_and_refuses_a_lesson_it_cannot_fill(table_file: Path):
    table = domains.load(table_file)
    deck = domains.Deck(table, random.Random(0))
    seen = Counter()
    for _ in range(4):
        used: set[str] = set()
        for _ in range(26):
            seen[deck.deal(used)["code"]] += 1
            used.add(list(seen)[-1])
    assert max(seen.values()) - min(seen.values()) <= 2
    small = domains.DomainTable(subsectors=table.subsectors[:3], escalate=table.escalate, source=table.source, sha256="x")
    with pytest.raises(domains.DomainTableError, match="more distinct domains"):
        d = domains.Deck(small, random.Random(0))
        used = set()
        for _ in range(4):
            used.add(d.deal(used)["code"])
    # An ESCALATE domain whose codes are all used in the lesson still deals one of its own.
    assert domains.Deck(table, random.Random(0)).escalate("legal", {"5411"})["code"] == "5411"


@pytest.mark.parametrize("frame,framing,ok", [
    ("ask", "Please check the statement against the notice below.", True),
    ("ask", "Is this right?", False),
    ("ask", "A tenant sent this. Check it.", False),
    ("question", "Does the statement hold against the notice below?", True),
    ("question", "Check the statement.", False),
    ("situation_then_ask", "A tenant forwarded the notice from the landlord. Check the statement against it.", True),
    ("situation_then_ask", "Check the statement against it.", False),
    ("situation_then_ask", "Did the landlord send this? Check it.", False),
    ("line_question", "Which line of the notice settles the statement?", True),
    ("line_question", "Which part settles the statement?", False),
    ("line_question", "Does line 3 settle the statement?", False),
    ("line_question", "Which line says the desk is open on Sundays?", False),
    # Argon §9.4 #13: no ordinal or position, no three-word quote, no paraphrase in the material's words.
    ("line_question", "Does the first line of the notice settle the statement?", False),
    ("line_question", "Which line, the opening line or the last line, settles it?", False),
    ("line_question", "Which line says the front office desk?", False),
    ("line_question", "Which line shows when the desk opens?", False),
    ("line_question", "Which line of the landlord's notice decides it?", True),
])
def test_the_frame_gate(frame: str, framing: str, ok: bool):
    material = "Desk hours at the front office\nThe desk is open on Sundays from 10:00 to 14:00.\n"
    user = framing + "\n\nMaterial:\n" + material + "\nStatement: the desk opens on Sundays."
    assert (frames.check_frame(frame, user, material) is None) is ok


def test_the_frame_gate_needs_a_framing_and_a_known_frame():
    assert frames.check_frame("ask", "Material:\nx\n", "x") == "the framing is empty"
    assert "unknown frame" in frames.check_frame("poem", "Hi.\nMaterial:\nx", "x")


def test_plan_and_mock_write_through_the_cli(table_file: Path, tmp_path: Path, capsys):
    from role_set.pack import PackManifest, pack_hash

    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(out), "--domains", str(table_file)]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["domain_table"]["domain_registry_id"] == "census-naics-2022"
    assert sum(1 for s in plan["slots"] if s["domain"]) == 26 + 8 and sum(1 for s in plan["slots"] if s["frame"]) == 18
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["plan", "--lessons", "T02", "--out", str(tmp_path / "p2.json"), "--domains", str(bad)]) == 2
    assert "REFUSED" in capsys.readouterr().err

    files = {"role.md": "# Lesson 1\n\nthe role\n", "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
             "kinds.md": "## 2. What to deliver\n\nfive kinds\n", "amendments/1.md": "## Amendment 1\n\nquote exactly\n"}
    pack = tmp_path / "pack"
    for name, text in files.items():
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    assert main(["write", "--plan", str(out), "--pack", str(pack), "--out", str(tmp_path / "o"), "--mock"]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["accepted"] == 40 and result["batch_check"] == "clean"
    receipts = [json.loads(x) for x in (tmp_path / "o" / "mock" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    framed = [r for r in receipts if r["slot"]["frame"]]
    assert len(framed) == 18 and all(r["slot"]["domain"]["title"].startswith("Fixture title") for r in framed)
    prompts = (tmp_path / "o" / "mock" / "drafts.jsonl").read_text(encoding="utf-8")
    assert prompts  # the call log exists; the brief itself is checked in the next test


def test_the_writer_brief_names_the_domain_and_frame(table_file: Path):
    from role_set.prompts import slot_brief

    slots = plan_batch({"T02": ("quotes the source", "stops at the source")}, seed=3)
    domains.assign(slots, domains.load(table_file), seed=3)
    c = next(s for s in slots if s.kind == "C")
    a = next(s for s in slots if s.kind == "A")
    brief = slot_brief(c, [])
    assert f"Domain: {c.domain['title']} (NAICS {c.domain['code']})" in brief and f"Frame: {c.frame}." in brief
    assert frames.FRAMES[c.frame] in brief
    a_brief = slot_brief(a, [])
    assert "Set the passage in this field of work" in a_brief and "Frame:" not in a_brief


def test_cd_domain_line_applies_to_framing_only_and_grounding_sentence_on_decided_cd(table_file: Path, tmp_path: Path):
    from role_set import callcache
    from role_set.cli import VERIFIER_SIDES, main as cli_main
    from role_set.guards import GuardError
    from role_set.pack import PackManifest, pack_hash
    from role_set.prompts import slot_brief, writer_messages
    from role_set.transport import Completion
    from role_set.write import WriteConfig, WriteRefused, run as run_write

    slots = plan_batch({"T10": VERIFIER_SIDES["T10"]}, seed=0, buffers=True)
    domains.assign(slots, domains.load(table_file), seed=0)

    grounding = "The statement names only people, products, places and terms that the material names."

    # 1. default is byte-identical to 4033ebd's base brief for kinds A, C and D (and B and E).
    a = next(s for s in slots if s.kind == "A")
    c_sup = next(s for s in slots if s.kind == "C" and s.verdict == "supported")
    c_unsup = next(s for s in slots if s.kind == "C" and s.verdict == "unsupported")
    c_ct = next(s for s in slots if s.kind == "C" and s.verdict == "cannot_tell")
    d_sup = next(s for s in slots if s.kind == "D" and s.verdict == "supported")
    e_verdict = next(s for s in slots if s.kind == "E" and s.e_type == "verdict")

    for s in (a, c_sup, c_unsup, c_ct, d_sup, e_verdict):
        assert slot_brief(s, []) == slot_brief(s, [], domain_scope="default")
        assert grounding not in slot_brief(s, [], domain_scope="default")

    assert (
        f"Domain: {a.domain['title']} (NAICS {a.domain['code']}). Set the passage in this field of work, "
        "with the people, documents and terms it really has; never a famous or implausible subject."
    ) in slot_brief(a, [], domain_scope="default")
    for s in (c_sup, c_unsup, c_ct, d_sup, e_verdict):
        assert (
            f"Domain: {s.domain['title']} (NAICS {s.domain['code']}). Set the material, the statement and the framing "
            "in this field of work, with the people, documents and terms it really has; never a famous or implausible subject."
        ) in slot_brief(s, [], domain_scope="default")

    # 2. framing_only changes only C/D briefs; the grounding sentence appears only on supported and unsupported C/D slots.
    for s in slots:
        b_def = slot_brief(s, [], domain_scope="default")
        b_frm = slot_brief(s, [], domain_scope="framing_only")
        if s.kind not in ("C", "D"):
            assert b_frm == b_def, s.id
            assert grounding not in b_frm, s.id
        else:
            assert b_frm != b_def, s.id
            assert "Set the framing in this field of work" in b_frm, s.id
            assert "Set the material, the statement and the framing in this field of work" not in b_frm, s.id
            if s.verdict in ("supported", "unsupported"):
                assert grounding in b_frm, s.id
                b_frm_without_grounding = b_frm.replace(f"{grounding}\n", "")
                expected_without_grounding = b_def.replace(
                    "Set the material, the statement and the framing in this field of work",
                    "Set the framing in this field of work",
                )
                assert b_frm_without_grounding == expected_without_grounding, s.id
            else:
                assert grounding not in b_frm, s.id
                expected_from_base = b_def.replace(
                    "Set the material, the statement and the framing in this field of work",
                    "Set the framing in this field of work",
                )
                assert b_frm == expected_from_base, s.id

    with pytest.raises( ValueError, match="domain_scope must be one of"):
        slot_brief(c_sup, [], domain_scope="unknown")

    # 3. Recorded per draft and part of the call key, like flaw_prompt.
    msgs_def = writer_messages(c_sup, {"traits/T10.md": "# T10\n\nx\n"}, material_text="m", exemplars=[], avoid=[], domain_scope="default")
    msgs_frm = writer_messages(c_sup, {"traits/T10.md": "# T10\n\nx\n"}, material_text="m", exemplars=[], avoid=[], domain_scope="framing_only")
    assert grounding not in msgs_def[1]["content"] and grounding in msgs_frm[1]["content"]

    k_none = callcache.request_key("writer", (msgs_def,), {"seed": 0})
    k_def = callcache.request_key("writer", (msgs_def,), {"seed": 0}, domain_scope="default")
    k_frm = callcache.request_key("writer", (msgs_def,), {"seed": 0}, domain_scope="framing_only")
    assert k_def == k_none and k_frm != k_def

    j_path = tmp_path / "calls.jsonl"
    j = callcache.CallJournal(j_path, None, domain_scope="framing_only")
    w_wrapped = j.wrap(
        "writer",
        lambda m, seed=0, **kw: Completion(text="ok", prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1),
    )
    w_wrapped(msgs_frm, seed=0)
    assert len(callcache.load(j_path, expected_domain_scope="framing_only")) == 1
    with pytest.raises(GuardError, match="domain_scope"):
        callcache.load(j_path, expected_domain_scope="default")

    plan_file = tmp_path / "plan.json"
    assert cli_main(["plan", "--lessons", "T10", "--seed", "0", "--domains", str(table_file), "--out", str(plan_file)]) == 0
    files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you judge the exact size\n\nCheck the exact size.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    pack_dir = tmp_path / "pack"
    for name, text in files.items():
        (pack_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (pack_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    out_frm = tmp_path / "out_frm"
    assert (
        cli_main(
            [
                "write",
                "--plan",
                str(plan_file),
                "--pack",
                str(pack_dir),
                "--out",
                str(out_frm),
                "--mock",
                "--domain-scope",
                "framing_only",
            ]
        )
        == 0
    )
    draft_rows = [json.loads(x) for x in (out_frm / "mock" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert draft_rows and all(r.get("domain_scope") == "framing_only" for r in draft_rows)

    with pytest.raises(WriteRefused, match="--domain-scope"):
        run_write(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "bad_ds", mock=True, domain_scope="invalid"))



def test_a_real_write_refuses_a_plan_without_domains(tmp_path: Path):
    from role_set.write import WriteConfig, WriteRefused, build

    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(out)]) == 0
    with pytest.raises(WriteRefused, match="no domains"):
        build(WriteConfig(plan=out, pack=ROOT / "roles" / "verifier", out=tmp_path / "o", endpoint="http://127.0.0.1:8080/v1"))


def test_the_committed_table_is_the_pinned_census_file():
    table = domains.load(ROOT / "roles" / "domains" / "naics-2022.json")
    assert table.source["sha256"] == "be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316"
    assert table.source["url"] == "https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx"
    assert len(table.subsectors) == 89 and {e["code"] for e in table.subsectors if e["code"].startswith("92")} == {"927"} and table.subsectors[0] == {"code": "111", "title": "Crop Production"}
    assert [e["title"] for e in table.escalate["legal"]] == ["Legal Services"]
    assert "9221" in {e["code"] for e in table.escalate["safety"]}
    # Every lesson's 26 domain-bearing slots fit, with room.
    assert len(table.subsectors) >= 3 * 26


def test_line_question_may_name_its_subject_but_not_the_deciding_line():
    """R&D on #79: the content kind and the domain title name the material, not the answer; a word the framing shares
    only with the deciding line is the leak."""

    material = ("Bus timetable, Harbour Line\nThe early service leaves the depot at dawn on weekdays.\n"
                "The late service runs on Saturdays only.\n")
    deciding = ["The late service runs on Saturdays only."]
    subject = ["timetable", "Transit and Ground Passenger Transportation"]

    def user(framing):
        return framing + "\n\nMaterial:\n" + material + "\nStatement: the late service runs on Sundays."

    named = "Which line of the bus timetable settles the statement?"
    assert frames.check_frame("line_question", user(named), material, deciding=deciding, subject_words=subject) is None
    # Without the subject words, "bus" and "timetable" count as two shared words.
    assert "material's words" in frames.check_frame("line_question", user(named), material, deciding=deciding)
    leak = "Which line settles whether Saturday matters here?"
    assert "only on the deciding line" in frames.check_frame("line_question", user(leak), material, deciding=deciding,
                                                             subject_words=subject)
