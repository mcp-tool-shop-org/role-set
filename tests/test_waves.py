"""Wave mode (BUILD-SPEC B1): write, critique and rewrite as separate runs, with the drafts file between them.

The equivalence tests are the spec's CI gate: a mock batch split into waves accepts the same rows, with the same
receipts apart from the wave fields, as one inline run of the same plan and seed.
"""

import json
from functools import partial
from pathlib import Path

import pytest
from test_cards import cards_file

from role_set import waves
from role_set.cli import main
from role_set.mock import MockCritic
from role_set.pack import PackManifest, pack_hash
from role_set.write import WriteConfig, WriteRefused, run

TRAIT_FILES = {
    "role.md": "# Lesson 1\n\nthe role\n",
    "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
    "traits/T03.md": "## 3. As a verifier, you are precise\n\nExact.\n",
    "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
    "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
}
# Fields that differ between an inline run and a wave run by design.
WAVE_FIELDS = ("wave", "drafts_sha256", "writer_host", "critic_model")


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pack"
    for name, text in TRAIT_FILES.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(TRAIT_FILES)
    (d / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    return d


@pytest.fixture
def plan_file(tmp_path: Path, domain_table: Path) -> Path:
    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(out), "--domains", str(domain_table)]) == 0
    return out


def _sealed(path: Path, text: str) -> None:
    """A hand-written drafts file with a good seal, so the test reaches the check after the seal."""

    import hashlib

    path.write_bytes(text.encode("utf-8"))
    waves.seal_path(path).write_text(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {path.name}\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _cfg(plan: Path, pack: Path, out: Path, **kw) -> WriteConfig:
    return WriteConfig(plan=plan, pack=pack, out=out, mock=True, partition="trial:waves", **kw)


def _batch(dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    """Accepted rows and receipts across waves, by row id, with the wave fields taken out."""

    rows = sorted((r for d in dirs for r in _jsonl(d / "mock" / "accepted.jsonl")), key=lambda r: r["id"])
    receipts = sorted(({k: v for k, v in r.items() if k not in WAVE_FIELDS} for d in dirs for r in _jsonl(d / "mock" / "receipts.jsonl")),
                      key=lambda r: r["slot"]["id"])
    return rows, receipts


def _waves_until_done(plan: Path, pack: Path, base: Path, *, limit: int = 8) -> tuple[list[Path], dict]:
    """write, then critique and rewrite in turn until nothing is pending. Returns the critique dirs and the last result."""

    result = run(_cfg(plan, pack, base / "w0", wave="write"))
    drafts = base / "w0" / "mock" / waves.DRAFTS_FILE
    critiques: list[Path] = []
    for i in range(1, limit):
        wave = "critique" if i % 2 else "rewrite"
        out = base / f"w{i}"
        result = run(_cfg(plan, pack, out, wave=wave, drafts=drafts))
        drafts = out / "mock" / waves.DRAFTS_FILE
        if wave == "critique":
            critiques.append(out)
        if result["states"]["awaiting_critique"] == 0 and result["states"]["needs_rewrite"] == 0:
            return critiques, result
    raise AssertionError("the waves did not finish")


def test_waves_without_a_rewrite_match_one_inline_run(pack_dir: Path, plan_file: Path, tmp_path: Path):
    inline = run(_cfg(plan_file, pack_dir, tmp_path / "inline"))
    critiques, last = _waves_until_done(plan_file, pack_dir, tmp_path / "waves")
    assert inline["accepted"] == 40 and len(critiques) == 1
    assert _batch([tmp_path / "inline"]) == _batch(critiques)
    assert last["batch_check"] == inline["batch_check"] == "clean"
    assert last["states"] == {"awaiting_critique": 0, "needs_rewrite": 0, "accepted": 40, "dropped": 0, "skipped": 0}


def test_waves_with_critic_refusals_match_one_inline_run(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    # The critic disagrees with the first draft of three D slots, so each needs a rewrite wave.
    run(_cfg(plan_file, pack_dir, tmp_path / "probe", wave="write"))
    _, drafts = waves.read(tmp_path / "probe" / "mock" / waves.DRAFTS_FILE)
    first = [json.loads(d["candidate"]["text"])["statement"] for d in drafts if d["slot"]["kind"] == "D"][:3]
    assert len(first) == 3
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_once=set(first)))
    inline = run(_cfg(plan_file, pack_dir, tmp_path / "inline"))
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_once=set(first)))
    critiques, last = _waves_until_done(plan_file, pack_dir, tmp_path / "waves")
    assert inline["accepted"] == 40 and len(critiques) == 2
    # The wave that leaves nothing pending holds the whole batch (Argon §9.4 #8).
    rows, receipts = _batch(critiques[-1:])
    assert (rows, receipts) == _batch([tmp_path / "inline"])
    refused = len(set(first))  # two D slots may share a statement; the critic refuses each statement once
    assert refused >= 2 and sorted(r["attempt"] for r in receipts if r["attempt"] > 1) == [2] * refused
    assert last["batch_check"] == inline["batch_check"] == "clean"
    # The second critique wave saw the first one's accepted rows: its receipts chain to the rewrite's drafts file.
    # The folder holds the whole batch now; the rows this wave accepted are the rewritten ones (attempt 2).
    second = [r for r in _jsonl(critiques[1] / "mock" / "receipts.jsonl") if r["attempt"] > 1]
    assert len(second) == refused and {r["wave"] for r in second} == {"critique"}
    rewrite_out = tmp_path / "waves" / "w2" / "mock" / waves.DRAFTS_FILE
    assert {r["drafts_sha256"] for r in second} == {waves.file_sha256(rewrite_out)}


def test_one_attempt_budget_across_waves(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_all=True))
    critiques, last = _waves_until_done(plan_file, pack_dir, tmp_path / "waves")
    header, drafts = waves.read(tmp_path / "waves" / "w5" / "mock" / waves.DRAFTS_FILE)
    assert len(critiques) == 3 and max(d["attempt"] for d in drafts) == 3
    # Every verdict-bearing slot spent its three attempts and was dropped; the andon counted the drops.
    verdict_bearing = [d for d in drafts if d["slot"]["kind"] in "CD" or (d["slot"]["kind"] == "E" and d["slot"]["e_type"] == "verdict")]
    assert verdict_bearing and all(d["state"] == "dropped" and d["attempt"] == 3 for d in verdict_bearing)
    assert header["andon"]["cell_halts"] and last["batch_check"] is None
    # Each wave is on record with the hash of its input.
    assert [x["wave"] for x in header["waves"]] == ["write", "critique", "rewrite", "critique", "rewrite", "critique"]
    assert header["waves"][0]["input_sha256"] is None and all(len(x["input_sha256"]) == 64 for x in header["waves"][1:])


def test_a_wave_refuses_a_drafts_file_it_cannot_continue(pack_dir: Path, plan_file: Path, tmp_path: Path, domain_table: Path):
    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    with pytest.raises(WriteRefused, match="needs --drafts"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="critique"))
    with pytest.raises(WriteRefused, match="for a critique or rewrite wave"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="write", drafts=drafts))
    with pytest.raises(WriteRefused, match="--wave must be"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="polish"))
    with pytest.raises(WriteRefused, match="nothing for a rewrite wave"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="rewrite", drafts=drafts))
    # Another plan is another batch.
    other = tmp_path / "plan2.json"
    assert main(["plan", "--lessons", "T02", "--seed", "4", "--out", str(other), "--domains", str(domain_table)]) == 0
    with pytest.raises(WriteRefused, match="other batch pins.*plan_sha256"):
        run(_cfg(other, pack_dir, tmp_path / "x", wave="critique", drafts=drafts))
    with pytest.raises(WriteRefused, match="other batch pins.*partition"):
        run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "x", mock=True, partition="production", wave="critique", drafts=drafts))
    bad = tmp_path / "bad.jsonl"
    with pytest.raises(WriteRefused, match="cannot read the drafts file and its seal"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="critique", drafts=bad))
    _sealed(bad, '{"type": "draft"}\n')
    with pytest.raises(WriteRefused, match="not a drafts file"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="critique", drafts=bad))
    _sealed(bad, "not json\n")
    with pytest.raises(WriteRefused, match="cannot read the drafts file \\("):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="critique", drafts=bad))


def test_the_drafts_file_carries_the_handoff(pack_dir: Path, plan_file: Path, tmp_path: Path):
    result = run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    header, drafts = waves.read(tmp_path / "w0" / "mock" / waves.DRAFTS_FILE)
    assert result["states"]["awaiting_critique"] == 40 and result["accepted"] == 0 and result["batch_check"] is None
    assert header["batch_pins"]["partition"] == "trial:waves" and header["writer_pins"] == {"writer_model": "mock"}
    assert header["gates"]["version"] and header["mock_answers"]
    d = drafts[0]
    assert d["state"] == "awaiting_critique" and d["attempt"] == 1 and d["messages"][0]["role"] == "system"
    assert d["candidate"]["text"] and d["row"] is None and d["receipt"] is None
    # No accepted file in a write wave: nothing is accepted before the critic has read it.
    assert not (tmp_path / "w0" / "mock" / "accepted.jsonl").exists()
    report = json.loads((tmp_path / "w0" / "mock" / "batch-report.json").read_text(encoding="utf-8"))
    assert report["wave"] == "write" and report["pending"] == 40


def test_a_changed_draft_is_checked_again_before_the_critic(pack_dir: Path, plan_file: Path, tmp_path: Path):
    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    path = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    header, drafts = waves.read(path)
    drafts[0]["candidate"]["text"] = "no longer JSON"
    waves.write(path, header, drafts)
    result = run(_cfg(plan_file, pack_dir, tmp_path / "w1", wave="critique", drafts=path))
    _, after = waves.read(tmp_path / "w1" / "mock" / waves.DRAFTS_FILE)
    assert after[0]["state"] == "needs_rewrite" and after[0]["refusals"][-1]["code"] == "PARSE"
    assert result["states"]["accepted"] == 39


def test_the_mock_answers_round_trip():
    answers = {("a statement", "first line"): ("supported", ["first line"]), ("flaw", "trace"): "quote_drift", ("answer", "tail"): "gold"}
    header = {"mock_answers": waves.dump_answers(answers)}
    assert waves.load_answers(json.loads(json.dumps(header))) == answers


def test_waves_through_the_cli(pack_dir: Path, plan_file: Path, tmp_path: Path, capsys):
    base = ["write", "--plan", str(plan_file), "--pack", str(pack_dir), "--mock", "--partition", "trial:waves"]
    assert main([*base, "--out", str(tmp_path / "a"), "--wave", "write"]) == 0
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert first["wave"] == "write" and first["states"]["awaiting_critique"] == 40
    drafts = tmp_path / "a" / "mock" / waves.DRAFTS_FILE
    assert main([*base, "--out", str(tmp_path / "b"), "--wave", "critique", "--drafts", str(drafts)]) == 0
    second = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert second["states"]["accepted"] == 40 and second["batch_check"] == "clean"
    assert main([*base, "--out", str(tmp_path / "c"), "--wave", "rewrite", "--drafts", str(drafts)]) == 2
    assert "nothing for a rewrite wave" in capsys.readouterr().err


def test_waves_on_the_pool_and_under_a_time_budget(pack_dir: Path, plan_file: Path, tmp_path: Path):
    result = run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write", concurrency=2))
    assert result["states"]["awaiting_critique"] == 40
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    result = run(_cfg(plan_file, pack_dir, tmp_path / "w1", wave="critique", drafts=drafts, concurrency=2))
    assert result["states"]["accepted"] == 40
    # No new slot starts once the budget is spent; a batch with skipped slots gets no batch check.
    short = run(_cfg(plan_file, pack_dir, tmp_path / "w2", wave="write", stop_after_minutes=1e-9))
    assert short["states"]["skipped"] == 40 and short["skipped"] == {"time budget": 40} and short["batch_check"] is None


def test_a_halt_keeps_the_drafts_in_hand(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    from role_set.runner import RunHalted

    real = waves._write_one
    calls = []

    def halting(runner, st):
        calls.append(st.slot.id)
        if len(calls) == 5:
            runner.status.halted = "systemic: test"
            raise RunHalted("systemic: test")
        return real(runner, st)

    monkeypatch.setattr(waves, "_write_one", halting)
    result = run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    assert result["halted"] == "systemic: test"
    assert result["states"]["awaiting_critique"] == 4 and result["states"]["skipped"] == 36
    header, drafts = waves.read(tmp_path / "w0" / "mock" / waves.DRAFTS_FILE)
    assert header["andon"]["halted"] == "systemic: test"
    assert {d["skip_reason"] for d in drafts if d["state"] == "skipped"} == {"run halted during the slot"}


def test_other_gates_and_another_writer_are_refused(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    header, rows = waves.read(drafts)
    other = tmp_path / "other-gates.jsonl"
    waves.write(other, {**header, "gates": {"version": "0.0.1", "sha256": "x"}}, rows)
    with pytest.raises(WriteRefused, match="checked by gates 0.0.1"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="critique", drafts=other))
    # A rewrite continues the drafts' own writer.
    monkeypatch.setattr(w, "MockCritic", partial(MockCritic, disagree_all=True))
    run(_cfg(plan_file, pack_dir, tmp_path / "w1", wave="critique", drafts=drafts))
    refused = tmp_path / "w1" / "mock" / waves.DRAFTS_FILE
    header, rows = waves.read(refused)
    swapped = tmp_path / "swapped.jsonl"
    waves.write(swapped, {**header, "writer_pins": {"writer_model": "another"}}, rows)
    with pytest.raises(WriteRefused, match="own writer.*writer_model"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="rewrite", drafts=swapped))
    mixed = tmp_path / "mixed.jsonl"
    _sealed(mixed, json.dumps(header) + "\n" + json.dumps({"type": "note"}) + "\n")
    with pytest.raises(WriteRefused, match="not a draft"):
        run(_cfg(plan_file, pack_dir, tmp_path / "x", wave="rewrite", drafts=mixed))


def test_run_wave_refuses_an_unknown_wave_and_a_started_batch(tmp_path: Path):
    from role_set.mock import MockWriter, mock_gates
    from role_set.runner import Runner

    r = Runner(gates=mock_gates(), pack={}, trait_names={}, writer=MockWriter(), critic=None, out_dir=tmp_path / "o")
    with pytest.raises(waves.WaveError, match="unknown wave"):
        waves.run_wave(r, "polish", slots=[], header_in=None, drafts_in=None)
    with pytest.raises(waves.WaveError, match="takes no drafts file"):
        waves.run_wave(r, "write", slots=[], header_in={}, drafts_in=[{"type": "draft"}])


def test_a_real_critique_wave_loads_no_writer_and_refuses_the_writers_family(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.mock import mock_gates
    from test_proven import SHA_C, SHA_S, proven_file
    from role_set.write import build

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    header, rows = waves.read(drafts)
    real = tmp_path / "real.jsonl"
    from role_set import proven

    listed = proven.load(proven_file(tmp_path))
    entry = listed.writers["big-writer"]
    full = {"writer_model": "big-writer", **{f"writer_{k}": v for k, v in proven.require_pinned(entry, entry["gguf_sha256"]).items()}}
    waves.write(real, {**header, "writer_pins": full}, rows)
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"driver": "580", "compat": True, "runtime": "13.4", "card": "NVIDIA A100-SXM4-40GB",
                                   "colab_release_tag": "release-colab_20261015"}), encoding="utf-8")
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    cfg = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=proven_file(tmp_path), cards=cards_file(tmp_path), critic_endpoint="http://127.0.0.1:8081/v1",
               critic_gguf_sha256=SHA_C, server_sha256=SHA_S, machine=machine, partition="trial:waves", wave="critique", drafts=real)
    with pytest.raises(WriteRefused, match="own family"):
        build(WriteConfig(critic_model="same-critic", **cfg))
    parts = build(WriteConfig(critic_model="other-critic", **cfg))
    writer, critic, pins = parts[5], parts[6], parts[7]
    assert writer is None and callable(critic) and "writer_gguf_sha256" not in pins
    assert pins["wave"] == "critique" and pins["drafts_sha256"] == waves.file_sha256(real) and pins["critic_family"] == "muse"


def test_the_hand_off_is_sealed_against_a_one_byte_change(pack_dir: Path, plan_file: Path, tmp_path: Path):
    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    path = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    # The seal is in sha256sum's format, so any host can check it before a wave starts.
    seal = waves.seal_path(path).read_text(encoding="utf-8").split()
    assert seal == [waves.file_sha256(path), waves.DRAFTS_FILE]
    data = bytearray(path.read_bytes())
    data[data.index(b"awaiting_critique")] = ord("A")  # one byte, inside a draft's state
    path.write_bytes(bytes(data))
    with pytest.raises(WriteRefused, match="doesn't match its seal"):
        run(_cfg(plan_file, pack_dir, tmp_path / "w1", wave="critique", drafts=path))


def test_a_critique_wave_refuses_an_unlisted_or_repinned_writer(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    """Argon §9.4 #2: the drafts' writer is looked up in writers.json; a blank family or a re-pinned GGUF is refused."""

    import role_set.write as w
    from role_set import proven
    from role_set.mock import mock_gates
    from role_set.write import build
    from test_proven import SHA_C, SHA_S, proven_file

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    header, rows = waves.read(tmp_path / "w0" / "mock" / waves.DRAFTS_FILE)
    listed = proven.load(proven_file(tmp_path))
    entry = listed.writers["big-writer"]
    full = {"writer_model": "big-writer", **{f"writer_{k}": v for k, v in proven.require_pinned(entry, entry["gguf_sha256"]).items()}}
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"driver": "580", "compat": True, "runtime": "13.4", "card": "NVIDIA A100-SXM4-40GB",
                                   "colab_release_tag": "release-colab_20261015"}), encoding="utf-8")
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    for bad, match in (({**full, "writer_model": "ghost"}, "not on the proven writer list"),
                       ({**full, "writer_family": ""}, "differ from writers.json"),
                       ({**full, "writer_gguf_sha256": "f" * 64}, "differs from the pinned")):
        real = tmp_path / f"real-{match[:5]}.jsonl"
        waves.write(real, {**header, "writer_pins": bad}, rows)
        cfg = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=proven_file(tmp_path), cards=cards_file(tmp_path),
                   critic_endpoint="http://127.0.0.1:8081/v1", critic_gguf_sha256=SHA_C, server_sha256=SHA_S, machine=machine,
                   partition="trial:waves", wave="critique", drafts=real, critic_model="other-critic")
        with pytest.raises(WriteRefused, match=match):
            build(WriteConfig(**cfg))


def test_a_wave_refuses_drafts_that_dont_match_the_plan(pack_dir: Path, plan_file: Path, tmp_path: Path):
    """Argon §9.4 #6: a truncated or reordered drafts file is refused, even resealed."""

    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    drafts = tmp_path / "w0" / "mock" / waves.DRAFTS_FILE
    header, rows = waves.read(drafts)
    for name, bad in (("short", rows[:5]), ("swapped", [rows[1], rows[0], *rows[2:]])):
        cut = tmp_path / name / waves.DRAFTS_FILE
        cut.parent.mkdir()
        waves.write(cut, header, bad)
        with pytest.raises(WriteRefused, match="don't match this plan"):
            run(_cfg(plan_file, pack_dir, tmp_path / f"o-{name}", wave="critique", drafts=cut))


def test_the_last_wave_holds_the_whole_batch_and_its_models(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    """Argon §9.4 #8: after write -> critique (-> rewrite -> critique), the wave that leaves nothing pending holds every
    accepted row of the batch, its receipts and the B models sidecar, and reports the batch's counts."""

    import role_set.write as w
    from functools import partial

    from role_set.mock import MockCritic

    # The critic refuses one slot's first draft, so the batch needs a rewrite wave and a second critique.
    calls = {"n": 0}
    real = MockCritic.__call__

    def flaky(self, messages, *a, **k):
        calls["n"] += 1
        return real(self, messages, *a, **k)

    monkeypatch.setattr(MockCritic, "__call__", flaky)
    run(_cfg(plan_file, pack_dir, tmp_path / "w0", wave="write"))
    r1 = run(_cfg(plan_file, pack_dir, tmp_path / "w1", wave="critique", drafts=tmp_path / "w0" / "mock" / waves.DRAFTS_FILE))
    last = tmp_path / "w1" / "mock"
    if r1["states"]["needs_rewrite"]:
        run(_cfg(plan_file, pack_dir, tmp_path / "w2", wave="rewrite", drafts=last / waves.DRAFTS_FILE))
        r1 = run(_cfg(plan_file, pack_dir, tmp_path / "w3", wave="critique", drafts=tmp_path / "w2" / "mock" / waves.DRAFTS_FILE))
        last = tmp_path / "w3" / "mock"
    header, drafts = waves.read(last / waves.DRAFTS_FILE)
    accepted = [d for d in drafts if d["state"] == "accepted"]
    rows = [json.loads(x) for x in (last / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(accepted) == r1["accepted"] and len(rows) > 0
    assert len((last / "receipts.jsonl").read_text(encoding="utf-8").splitlines()) == len(rows)
    b_rows = {d["row"]["id"] for d in accepted if d["slot"]["kind"] == "B"}
    side = [json.loads(x) for x in (last / "accepted.models.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {m["id"] for m in side} == b_rows
    # The gates' own loader reads the sidecar, each row_sha256 matching its line as written.
    gates = pytest.importorskip("aspire_si_gates")
    assert set(gates.load_models(last / "accepted.jsonl")) == b_rows
