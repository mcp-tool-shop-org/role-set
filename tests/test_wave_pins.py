"""Wave-to-wave pins with real card values (window 12b, 2026-10-17): the writer's card pins are the writer's, so a
critique wave, which loads no writer, continues the drafts; a rewrite still has to match the writer's card."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from role_set import waves

HEADER = json.loads((Path(__file__).parent / "fixtures" / "window12b-granite-w0-header.json").read_text(encoding="utf-8"))
GATES = SimpleNamespace(version=HEADER["gates"]["version"], sha256=HEADER["gates"]["sha256"])
CARD_KEYS = [k for k in HEADER["batch_pins"] if k.startswith("card_") or k == "cards_sha256"]


def _critique_pins() -> dict:
    """What a critique wave's pins hold: the batch's own pins, the critic's, and no writer card (it loads no writer)."""

    batch = {k: v for k, v in HEADER["batch_pins"].items() if k not in CARD_KEYS}
    return {**batch, "critic_model": "muse-glimmer-30b", "critic_card_candidate": "muse-glimmer-30b", "wave": "critique",
            "machine": {"card": "A100"}, "drafts_sha256": "x" * 64}


def test_the_real_header_kept_the_writer_card_among_its_batch_pins():
    assert CARD_KEYS and "card_sampling" in CARD_KEYS  # the shape that refused window 12b's critique


def test_a_critique_wave_continues_drafts_whose_header_kept_card_pins_as_batch_pins():
    waves.require_same_pins(HEADER, _critique_pins(), GATES, "critique")  # refused before the fix


def test_a_critique_wave_still_refuses_another_batch():
    pins = {**_critique_pins(), "plan_sha256": "f" * 64}
    with pytest.raises(waves.WaveError, match="plan_sha256"):
        waves.require_same_pins(HEADER, pins, GATES, "critique")


def test_a_rewrite_continues_the_same_writer_card_and_refuses_another():
    write_pins = {**HEADER["batch_pins"], **HEADER["writer_pins"]}
    # The critique wave's header carries the old writer pins forward, without the card; the card is still found.
    waves.require_same_pins(HEADER, {**write_pins, "wave": "rewrite"}, GATES, "rewrite")
    other = {**write_pins, "card_sampling": {**HEADER["batch_pins"]["card_sampling"], "temperature": 1.5}}
    with pytest.raises(waves.WaveError, match="card_sampling"):
        waves.require_same_pins(HEADER, other, GATES, "rewrite")


def test_a_new_header_keeps_the_card_with_the_writer():
    write_pins = {**HEADER["batch_pins"], **HEADER["writer_pins"]}
    assert set(CARD_KEYS) <= set(waves.writer_pins(write_pins))
    assert not set(CARD_KEYS) & set(waves.batch_pins(write_pins))
    new_header = {**HEADER, "batch_pins": waves.batch_pins(write_pins), "writer_pins": waves.writer_pins(write_pins)}
    waves.require_same_pins(new_header, _critique_pins(), GATES, "critique")
    waves.require_same_pins(new_header, {**write_pins, "wave": "rewrite"}, GATES, "rewrite")
