"""The wave chain on role-set's real path (mistakes #33, R&D 2026-10-17): write, critique, rewrite and critique again
through build() with writers.json's pins, rnd's card values and a stub llama-server on the loopback. The mock waves
skip the card path, which is where window 12b's critique was refused; this runs it."""

import json
from pathlib import Path

import pytest
from stub_llama import StubLlama
from test_waves import pack_dir  # noqa: F401 - fixture

from role_set import waves
from role_set.cli import main
from role_set.gates import GatesUnavailable, load_gates
from role_set.mock import MockCritic, mock_gates
from role_set.write import WriteConfig, run

ROOT = Path(__file__).resolve().parents[1]
CARDS = Path(__file__).parent / "fixtures" / "card-values.json"
WRITER, CRITIC = "granite-4.2-30b", "muse-glimmer-30b"


def _pins() -> tuple[str, str, str]:
    listed = json.loads((ROOT / "writers.json").read_text(encoding="utf-8"))
    w = next(e for e in listed["writers"] if e["served_as"] == WRITER)
    c = next(e for e in listed["critics"] if e["served_as"] == CRITIC)
    return w["gguf_sha256"], c["gguf_sha256"], listed["llama_server"]["sha256"]


@pytest.fixture
def real_path(tmp_path: Path, pack_dir: Path, monkeypatch):  # noqa: F811
    try:
        load_gates()
    except GatesUnavailable:
        import role_set.write as w

        monkeypatch.setattr(w, "load_gates", lambda **kw: mock_gates())
    plan = tmp_path / "plan.json"
    assert main(["plan", "--pilot", "--seed", "0", "--domains", str(ROOT / "roles/domains/naics-2022.json"), "--out", str(plan)]) == 0
    data = json.loads(plan.read_text(encoding="utf-8"))
    # A short screen of verdict-bearing kinds, so the mock critic's refusals reach every slot and a rewrite follows.
    data["slots"] = [s for s in data["slots"] if s["kind"] in ("C", "D")][:4]
    plan.write_text(json.dumps(data), encoding="utf-8")
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"card": "NVIDIA A100-SXM4-40GB", "memory": "40960 MiB", "driver": "580.82.07", "compat": False,
                                   "driver_cuda": "13.0", "colab_release_tag": "test", "runtime": "13.4"}), encoding="utf-8")
    stub = StubLlama(WRITER, CRITIC)
    yield {"plan": plan, "pack": pack_dir, "machine": machine, "stub": stub, "tmp": tmp_path}
    stub.close()


def _cfg(rp: dict, out: str, wave: str, *, drafts: Path | None = None) -> WriteConfig:
    w_sha, c_sha, server = _pins()
    common = dict(plan=rp["plan"], pack=rp["pack"], out=rp["tmp"] / out, writers_file=ROOT / "writers.json", cards=CARDS,
                  partition="trial:chain", machine=rp["machine"], server_sha256=server, wave=wave, drafts=drafts)
    if wave == "critique":
        return WriteConfig(**common, critic_endpoint=rp["stub"].url, critic_model=CRITIC, critic_gguf_sha256=c_sha)
    return WriteConfig(**common, endpoint=rp["stub"].url, model=WRITER, writer_gguf_sha256=w_sha)


def test_write_critique_rewrite_critique_on_the_real_path(real_path):
    rp = real_path
    stub = rp["stub"]
    w0 = run(_cfg(rp, "w0", "write"))
    assert w0["states"]["awaiting_critique"] > 0, w0
    d0 = rp["tmp"] / "w0" / waves.DRAFTS_FILE
    header, _ = waves.read(d0)
    assert any(k.startswith("card_") for k in header["writer_pins"])  # the card is the writer's now
    # The writer really went through the card: its sampling reached the stub.
    granite = json.loads(CARDS.read_text(encoding="utf-8"))
    card = next(c for c in granite["candidates"] if c["candidate"] == WRITER)
    sent = next(r for r in stub.requests if r["model"] == WRITER and "## This slot" in json.dumps(r["messages"]))
    assert all(sent.get(k) == v for k, v in card["sampling"].items())

    stub.critic = MockCritic(answers=stub.writer.answers, disagree_all=True)  # every draft goes back for a rewrite
    w1 = run(_cfg(rp, "w1", "critique", drafts=d0))  # window 12b was refused here
    assert w1["states"]["needs_rewrite"] > 0, w1
    w2 = run(_cfg(rp, "w2", "rewrite", drafts=rp["tmp"] / "w1" / waves.DRAFTS_FILE))
    assert w2["states"]["awaiting_critique"] > 0, w2
    stub.critic = MockCritic(answers=stub.writer.answers)
    w3 = run(_cfg(rp, "w3", "critique", drafts=rp["tmp"] / "w2" / waves.DRAFTS_FILE))
    assert w3["accepted"] > 0 and w3["states"]["awaiting_critique"] == 0, w3
    receipts = [json.loads(x) for x in (rp["tmp"] / "w3" / "receipts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert receipts and all(r.get("card_candidate") == WRITER and r.get("critic_model") == CRITIC for r in receipts)


def test_a_rewrite_under_another_writer_card_is_refused(real_path, tmp_path):
    rp = real_path
    run(_cfg(rp, "w0", "write"))
    rp["stub"].critic = MockCritic(answers=rp["stub"].writer.answers, disagree_all=True)
    run(_cfg(rp, "w1", "critique", drafts=rp["tmp"] / "w0" / waves.DRAFTS_FILE))
    other = json.loads(CARDS.read_text(encoding="utf-8"))
    for c in other["candidates"]:
        if c["candidate"] == WRITER:
            c["sampling"] = {**c["sampling"], "temperature": 1.37}
    changed = tmp_path / "cards-changed.json"
    changed.write_text(json.dumps(other), encoding="utf-8")
    cfg = _cfg(rp, "w2", "rewrite", drafts=rp["tmp"] / "w1" / waves.DRAFTS_FILE)
    from dataclasses import replace

    with pytest.raises(Exception, match="card"):
        run(replace(cfg, cards=changed))
