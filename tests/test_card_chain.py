"""The card-pin chain on the real files (stub rung 12s5a; R&D and Argon, 2026-10-18). Granite's and Bolt's w1 (12g) and
Muse's w1 (12h) came from old headers through a critique that dropped the writer's card pins; their w0, the write
waves' sealed drafts that critique read, still hold them. waves.chain_card_pins takes them back, one hop, or refuses."""

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from role_set import waves

FIX = Path(__file__).parent / "fixtures" / "window12g12h-chain.tar.gz"
FIX12F = Path(__file__).parent / "fixtures" / "window12f-w1.tar.gz"
GRANITE, BOLT, MUSE = "granite-4.2-30b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"


@pytest.fixture
def real(tmp_path: Path) -> Path:
    with tarfile.open(FIX) as tf:
        tf.extractall(tmp_path, filter="data")
    return tmp_path


def _reseal(path: Path, header: dict, drafts: list[dict]) -> None:
    waves.write(path, header, drafts)  # a forger can always write a matching seal; (iii) is what must catch it


def test_the_real_old_format_w1_files_lost_their_card_pins_and_their_w0_kept_them(real):
    for arm in (GRANITE, BOLT, MUSE):
        h1, _ = waves.read(real / "w1" / f"{arm}.jsonl")
        h0, _ = waves.read(real / "w0" / f"{arm}.jsonl")
        assert not waves.CARD_KEYS & set(waves.header_writer_pins(h1)), arm
        assert waves.CARD_KEYS <= set(h0["batch_pins"]), arm  # the old header kept them among its batch pins


@pytest.mark.parametrize("arm", [GRANITE, BOLT, MUSE])
def test_the_chain_restores_all_seven_card_pins_from_the_w0_the_critique_read(real, arm):
    w1, w0 = real / "w1" / f"{arm}.jsonl", real / "w0" / f"{arm}.jsonl"
    h1, _ = waves.read(w1)
    pins, sha = waves.chain_card_pins(h1, w0)
    assert set(pins) == waves.CARD_KEYS and sha == waves.file_sha256(w0) == h1["waves"][-1]["input_sha256"]
    assert pins == {k: waves.read(w0)[0]["batch_pins"][k] for k in waves.CARD_KEYS}
    # With them, a rewrite at the same card passes the writer-pin check that refused it in 12s5a and 12s5b.
    chained = {**h1, "writer_pins": {**waves.header_writer_pins(h1), **pins}}
    rewrite = {**h1["batch_pins"], **waves.header_writer_pins(h1), **pins, "wave": "rewrite"}
    gates = SimpleNamespace(version=h1["gates"]["version"], sha256=h1["gates"]["sha256"])
    waves.require_same_pins(chained, rewrite, gates, "rewrite")
    with pytest.raises(waves.WaveError, match="card_"):
        waves.require_same_pins(h1, rewrite, gates, "rewrite")


def test_a_w0_with_one_byte_changed_and_resealed_is_refused_on_the_critiques_input(real):
    w0 = real / "w0" / f"{GRANITE}.jsonl"
    h0, d0 = waves.read(w0)
    d0[0]["slot"]["id"] = d0[0]["slot"]["id"] + " "
    _reseal(w0, h0, d0)
    waves.read(w0)  # its own seal now verifies
    with pytest.raises(waves.WaveError, match="is not the input"):
        waves.chain_card_pins(waves.read(real / "w1" / f"{GRANITE}.jsonl")[0], w0)


def test_a_w0_changed_without_a_new_seal_is_refused_on_its_seal(real):
    w0 = real / "w0" / f"{GRANITE}.jsonl"
    w0.write_bytes(w0.read_bytes().replace(b"T02-A-01", b"T02-A-02", 1))
    with pytest.raises(waves.WaveError, match="doesn't match its seal"):
        waves.chain_card_pins(waves.read(real / "w1" / f"{GRANITE}.jsonl")[0], w0)


def test_another_arms_w0_is_refused(real):
    with pytest.raises(waves.WaveError, match="is not the input"):
        waves.chain_card_pins(waves.read(real / "w1" / f"{GRANITE}.jsonl")[0], real / "w0" / f"{MUSE}.jsonl")


def test_drafts_that_record_card_pins_never_take_a_prior(real, tmp_path):
    # Qwen3.8's real w1 (12f, new format) keeps its card pins through the critique; a prior beside it is refused (i).
    with tarfile.open(FIX12F) as tf:
        tf.extractall(tmp_path / "w12f", filter="data")
    h1, _ = waves.read(tmp_path / "w12f" / "qwen3.8-27b" / "wave-drafts.jsonl")
    assert waves.CARD_KEYS <= set(h1["writer_pins"])
    with pytest.raises(waves.WaveError, match="record card pins of their own"):
        waves.chain_card_pins(h1, real / "w0" / f"{GRANITE}.jsonl")


def test_the_chain_takes_one_hop_only(real):
    w1 = real / "w1" / f"{GRANITE}.jsonl"
    h1, d1 = waves.read(w1)
    longer = {**h1, "waves": [*h1["waves"], {"wave": "rewrite", "input_sha256": "0" * 64}]}
    with pytest.raises(waves.WaveError, match="one write wave and one critique"):
        waves.chain_card_pins(longer, real / "w0" / f"{GRANITE}.jsonl")
    with pytest.raises(waves.WaveError, match="not one write wave's drafts"):
        waves.chain_card_pins(h1, w1)  # a w1 is no prior: its waves are write and critique


def test_a_prior_with_other_batch_or_writer_pins_is_refused(real):
    w0 = real / "w0" / f"{GRANITE}.jsonl"
    h0, d0 = waves.read(w0)
    h1, _ = waves.read(real / "w1" / f"{GRANITE}.jsonl")
    for key, where in (("plan_sha256", "batch_pins"), ("writer_gguf_sha256", "writer_pins")):
        changed = json.loads(json.dumps(h0))
        changed[where][key] = "f" * 64
        _reseal(w0, changed, d0)
        forged = {**h1, "waves": [h1["waves"][0], {**h1["waves"][1], "input_sha256": waves.file_sha256(w0)}]}
        with pytest.raises(waves.WaveError, match=key):
            waves.chain_card_pins(forged, w0)


@pytest.mark.parametrize("key, value", [("version", "9.9.9"), ("sha256", "f" * 64), ("closed_form", {"carry": "verbatim"})])
def test_a_prior_checked_by_other_gates_is_refused(real, key, value):
    w0 = real / "w0" / f"{GRANITE}.jsonl"
    h0, d0 = waves.read(w0)
    h1, _ = waves.read(real / "w1" / f"{GRANITE}.jsonl")
    _reseal(w0, {**h0, "gates": {**h0["gates"], key: value}}, d0)
    forged = {**h1, "waves": [h1["waves"][0], {**h1["waves"][1], "input_sha256": waves.file_sha256(w0)}]}
    with pytest.raises(waves.WaveError, match=f"gates {key}"):
        waves.chain_card_pins(forged, w0)


def test_a_prior_with_no_gates_sha256_is_refused_even_beside_drafts_without_one(real):
    w0 = real / "w0" / f"{GRANITE}.jsonl"
    h0, d0 = waves.read(w0)
    h1, _ = waves.read(real / "w1" / f"{GRANITE}.jsonl")
    _reseal(w0, {**h0, "gates": {k: v for k, v in h0["gates"].items() if k != "sha256"}}, d0)
    bare = {**h1, "gates": {k: v for k, v in h1["gates"].items() if k != "sha256"},
            "waves": [h1["waves"][0], {**h1["waves"][1], "input_sha256": waves.file_sha256(w0)}]}
    with pytest.raises(waves.WaveError, match="records no gates sha256"):
        waves.chain_card_pins(bare, w0)
