"""Offline reconstruction of the 60 non-trait P1r critic requests for the batching-drift study.

Reconstructs every critic request body from P1r's `wave-drafts.jsonl` (sha256 0b2fc676...) and `pack.tar`
(pack_sha256 de2c7ed0...) using role-set's own prompt-building code, verifies 100% of `messages_sha256` and `key`
values against P1r's `calls-w1-muse-glimmer-30b.jsonl` (178 total rows, 70 non-trait rows), selects the 60
non-trait calls specified in DESIGN-v1.md §3 (40 verdict, 12 twin_verdict, 8 flaw from Kind C stratified 5 pilot /
3 apply), orders them by a fixed seeded permutation, assigns `request_index` 0..59, and verifies 60/60 hashes
and explicit integer `seed: 0` on every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from role_set import assemble, callcache, cards, critic, gates, pack, prompts, runner, waves, write  # noqa: E402

P1R_RECEIPTS_DEFAULT = Path("E:/AI/Publisher/receipts/colab/window17n-driftv3-P1r-20261023")
P1R_WAVE_DRAFTS_SHA256 = "0b2fc67663a541e5415b1eddeef1d5f77deeb8891ef7b36adc151f4f0aff340d"
P1R_W0_DRAFTS_SHA256 = "2394dc0ed1025531e63dafc61c31e402c5663db3976a29782862176726e0d8fe"
P1R_PACK_SHA256 = "de2c7ed0bdfa945aefec9745b5ac0a3514a15ca98f63211929b09561fe38dcae"
P1R_JOURNAL_SHA256 = "61d1fc6d055b01ffb291fee02fe6c9297b1bb5d310c547c66396e0819a09ba27"
P1R_GENERATIONS_SHA256 = "bd62780448a5b5862c72676f1570933999071053802f7bb50b3598bc76f5cf87"
P1R_CARDS_SHA256 = "2927865f3b6d257b5d11e458faaef538fef20d81ec221d9bc70ff3d03d80c36e"
P1R_CRITIC_MODEL = "muse-glimmer-30b"
DEFAULT_SEED = 0
PILOT_FLAW_SAMPLE = 5
APPLY_FLAW_SAMPLE = 3
EXPECTED_TOTAL_SELECTED = 60


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_payload_sha256(messages: list[dict], seed: int) -> str:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    blob = json.dumps({"messages": messages, "seed": seed}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _candidate_text_from_draft(draft: dict) -> str:
    cand = draft.get("candidate")
    if isinstance(cand, dict) and isinstance(cand.get("text"), str):
        return cand["text"]
    for msg in reversed(draft.get("messages") or []):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            return msg["content"]
    raise RuntimeError(f"draft {draft.get('slot', {}).get('id')!r} has neither candidate.text nor an assistant message")


def _completion_text(journal_row: dict) -> str:
    comp = journal_row.get("completion")
    if isinstance(comp, dict):
        return str(comp.get("text") or "")
    return str(comp or "")


def reconstruct_p1r_non_trait_calls(
    *,
    drafts_path: Path,
    pack_dir: Path,
    journal_path: Path,
    cards_path: Path,
    critic_model: str = P1R_CRITIC_MODEL,
    verify_pins: bool = True,
) -> list[dict]:
    """Reconstruct all non-trait critic calls from wave-drafts and pack, verifying 100% against the P1r journal."""

    if verify_pins:
        drafts_sha = file_sha256(drafts_path)
        if drafts_sha not in (P1R_WAVE_DRAFTS_SHA256, P1R_W0_DRAFTS_SHA256):
            raise RuntimeError(
                f"REFUSED: wave-drafts.jsonl sha256 {drafts_sha} is neither P1r w1 ({P1R_WAVE_DRAFTS_SHA256}) "
                f"nor P1r w0 ({P1R_W0_DRAFTS_SHA256})"
            )
        journal_sha = file_sha256(journal_path)
        if journal_sha != P1R_JOURNAL_SHA256:
            raise RuntimeError(
                f"REFUSED: journal sha256 {journal_sha} != P1r journal ({P1R_JOURNAL_SHA256})"
            )
        cards_sha = file_sha256(cards_path)
        if cards_sha != P1R_CARDS_SHA256:
            raise RuntimeError(
                f"REFUSED: card-values.json sha256 {cards_sha} != P1r card-values ({P1R_CARDS_SHA256})"
            )

    pack_info = pack.verify(pack_dir)
    if verify_pins and pack_info.pack_sha256 != P1R_PACK_SHA256:
        raise RuntimeError(
            f"REFUSED: pack_sha256 {pack_info.pack_sha256} != P1r pack ({P1R_PACK_SHA256})"
        )
    pack_data = prompts.load_pack(pack_dir)
    card_set, _ = cards.load(cards_path)
    if critic_model not in card_set:
        raise RuntimeError(f"REFUSED: critic {critic_model!r} not found in {cards_path}")
    critic_card = card_set[critic_model]
    g = gates.load_gates()

    lines = [json.loads(ln) for ln in drafts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines or lines[0].get("type") != "header":
        raise RuntimeError(f"REFUSED: {drafts_path} is missing a wave header")
    header = lines[0]
    drafts = [d for d in lines[1:] if d.get("type") == "draft"]
    verdict_prompt = (header.get("verdict_prompt") or "old").strip()
    flaw_prompt = (header.get("flaw_prompt") or "old").strip()

    # waves.run_wave processes train slots before apply slots.
    train_drafts = [d for d in drafts if d["slot"].get("split") != "apply"]
    apply_drafts = [d for d in drafts if d["slot"].get("split") == "apply"]
    ordered_drafts = train_drafts + apply_drafts

    journal_rows = [json.loads(ln) for ln in journal_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    non_trait_journal = [
        (idx, r) for idx, r in enumerate(journal_rows) if r.get("check") in ("verdict", "twin_verdict", "flaw")
    ]

    reconstructed: list[dict] = []
    j_cursor = 0

    def _emit_and_verify(
        *,
        slot: dict,
        check_name: str,
        msgs: list[dict],
        call_occurrence: int,
        key_verdict: str | None = None,
        key_deciding: list[str] | None = None,
        material: str | None = None,
        key_flaw: str | None = None,
    ) -> tuple[dict, str]:
        nonlocal j_cursor
        if j_cursor >= len(non_trait_journal):
            raise RuntimeError(
                f"reconstructed call {slot['id']} ({check_name}) exceeded P1r non-trait journal length ({len(non_trait_journal)})"
            )
        j_idx, j_row = non_trait_journal[j_cursor]
        j_cursor += 1

        sent = cards.apply(critic_card, msgs)
        req_key = callcache.request_key("critic", (msgs,), {})
        msg_sha = callcache.messages_sha256(sent)
        attempt = int(j_row.get("attempt", 0))
        if attempt != 0:
            raise RuntimeError(f"expected attempt 0 on P1r call {j_idx} ({slot['id']} {check_name}), got {attempt}")
        seed = 0  # Every attempt-0 call in P1r sent explicit integer seed: 0 (write.py:208-212, transport.py:115-116).

        if j_row.get("slot") != slot["id"] or j_row.get("check") != check_name:
            raise RuntimeError(
                f"journal order mismatch at non-trait call {j_cursor - 1} (journal row {j_idx}): "
                f"expected ({slot['id']}, {check_name}), got ({j_row.get('slot')}, {j_row.get('check')})"
            )
        if j_row.get("key") != req_key:
            raise RuntimeError(
                f"key mismatch on {slot['id']} {check_name} (journal row {j_idx}): "
                f"reconstructed {req_key} != journal {j_row.get('key')}"
            )
        if j_row.get("messages_sha256") != msg_sha:
            raise RuntimeError(
                f"messages_sha256 mismatch on {slot['id']} {check_name} (journal row {j_idx}): "
                f"reconstructed {msg_sha} != journal {j_row.get('messages_sha256')}"
            )

        comp_text = _completion_text(j_row)
        comp_sha = hashlib.sha256(comp_text.encode("utf-8")).hexdigest()
        rec = {
            "key": req_key,
            "slot": slot["id"],
            "split": slot.get("split", "train"),
            "kind": slot["kind"],
            "check": check_name,
            "call_occurrence": call_occurrence,
            "attempt": attempt,
            "seed": seed,
            "messages": sent,
            "messages_sha256": msg_sha,
            "payload_sha256": canonical_payload_sha256(sent, seed),
            "key_verdict": key_verdict,
            "key_deciding": key_deciding,
            "material": material,
            "content_kind": slot.get("content_kind"),
            "key_flaw": key_flaw,
            "p1r_journal_index": j_idx,
            "p1r_completion_sha256": comp_sha,
            "p1r_completion_text": comp_text,
        }
        reconstructed.append(rec)
        return rec, comp_text

    trait_names = write.trait_names_from_pack(pack_data)
    for d in ordered_drafts:
        slot = d["slot"]
        if j_cursor >= len(non_trait_journal) or non_trait_journal[j_cursor][1].get("slot") != slot["id"]:
            continue
        raw_text = _candidate_text_from_draft(d)
        st = waves.state_of(d)
        raw_fields = prompts.parse_json_object(raw_text)
        fields, _ = runner.normalize_glyphs(raw_fields)
        if st.slot.kind == "C" and "planted_flaw" not in fields:
            fields["planted_flaw"] = assemble.slot_flaw(st.slot)
        trait = trait_names.get(st.slot.lesson, st.slot.lesson)
        row = assemble.assemble(st.slot, trait, fields, st.material)
        material = st.material.text if st.material else ""
        kind = slot["kind"]

        if kind in ("C", "D") or (kind == "E" and slot.get("e_type") == "verdict"):
            statement = fields["statement"]
            turns: list[tuple[str, str, list[str]]] = []
            if kind == "E":
                turns.append((material, row["quiz"]["verdict"], row["quiz"]["deciding"]))
            elif row.get("turn_keys"):
                first = row["turn_keys"]["1"]
                turns.append((material, first["verdict"], first["deciding"]))
                new_lines = row["turns"][-1].get("new_material") or []
                turns.append((material + "\n" + "\n".join(new_lines), row["verdict"], row["deciding"]))
            else:
                turns.append((material, row["verdict"], row["deciding"]))

            for turn_idx, (seen_material, key_verdict, key_deciding) in enumerate(turns):
                v_msgs = critic.verdict_messages(seen_material, statement, verdict_prompt=verdict_prompt)
                _, comp_text = _emit_and_verify(
                    slot=slot,
                    check_name="verdict",
                    msgs=v_msgs,
                    call_occurrence=turn_idx,
                    key_verdict=key_verdict,
                    key_deciding=key_deciding,
                    material=seen_material,
                )
                first_cv = critic.compare_verdict(
                    g,
                    comp_text,
                    key_verdict,
                    key_deciding,
                    seen_material,
                    content_kind=slot.get("content_kind"),
                )
                if first_cv.code == "VOID":
                    _emit_and_verify(
                        slot=slot,
                        check_name="verdict",
                        msgs=v_msgs,
                        call_occurrence=turn_idx,
                        key_verdict=key_verdict,
                        key_deciding=key_deciding,
                        material=seen_material,
                    )
                if not first_cv.ok:
                    break

            twin = row.get("r10_twin")
            if isinstance(twin, dict) and twin.get("statement") and twin.get("verdict"):
                tv_msgs = critic.verdict_messages(material, twin["statement"], verdict_prompt=verdict_prompt)
                _, tv_comp = _emit_and_verify(
                    slot=slot,
                    check_name="twin_verdict",
                    msgs=tv_msgs,
                    call_occurrence=0,
                    key_verdict=twin["verdict"],
                    key_deciding=[],
                    material=material,
                )
                first_tv = critic.compare_twin_verdict(g, tv_comp, twin["verdict"])
                if first_tv.code == "VOID":
                    _emit_and_verify(
                        slot=slot,
                        check_name="twin_verdict",
                        msgs=tv_msgs,
                        call_occurrence=1,
                        key_verdict=twin["verdict"],
                        key_deciding=[],
                        material=material,
                    )

        if kind == "C":
            f_msgs = critic.flaw_messages(
                material,
                fields["statement"],
                fields["checker_trace"],
                row["checker_verdict"],
                flaw_prompt=flaw_prompt,
            )
            _emit_and_verify(
                slot=slot,
                check_name="flaw",
                msgs=f_msgs,
                call_occurrence=0,
                material=material,
                key_flaw=row["planted_flaw"],
            )

    if j_cursor != len(non_trait_journal):
        raise RuntimeError(
            f"reconstructed {j_cursor} non-trait calls, but P1r journal has {len(non_trait_journal)}"
        )
    return reconstructed


def select_and_permute_60(
    calls: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    pilot_flaw_k: int = PILOT_FLAW_SAMPLE,
    apply_flaw_k: int = APPLY_FLAW_SAMPLE,
) -> list[dict]:
    """Select all 40 verdict, all 12 twin_verdict, and 8 stratified Kind C flaw calls (5 pilot, 3 apply),
    then order by a fixed seeded permutation and assign request_index 0..59."""

    verdict_calls = [c for c in calls if c["check"] == "verdict"]
    twin_calls = [c for c in calls if c["check"] == "twin_verdict"]
    flaw_calls = [c for c in calls if c["check"] == "flaw"]

    if any(c["kind"] != "C" for c in flaw_calls):
        raise RuntimeError("expected all flaw calls to come from Kind C slots")

    pilot_flaw = [c for c in flaw_calls if c["split"] != "apply"]
    apply_flaw = [c for c in flaw_calls if c["split"] == "apply"]
    if len(pilot_flaw) < pilot_flaw_k or len(apply_flaw) < apply_flaw_k:
        raise RuntimeError(
            f"insufficient flaw calls for stratified sample: have {len(pilot_flaw)} pilot (need {pilot_flaw_k}) "
            f"and {len(apply_flaw)} apply (need {apply_flaw_k})"
        )

    rng = random.Random(seed)
    chosen_pilot_flaw = rng.sample(pilot_flaw, pilot_flaw_k)
    chosen_apply_flaw = rng.sample(apply_flaw, apply_flaw_k)

    selected = list(verdict_calls) + list(twin_calls) + chosen_pilot_flaw + chosen_apply_flaw
    if len(selected) != EXPECTED_TOTAL_SELECTED:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL_SELECTED} selected calls (40 verdict + 12 twin_verdict + 8 flaw), "
            f"got {len(selected)} ({len(verdict_calls)} verdict + {len(twin_calls)} twin_verdict + "
            f"{len(chosen_pilot_flaw) + len(chosen_apply_flaw)} flaw)"
        )

    rng.shuffle(selected)
    out: list[dict] = []
    for idx, item in enumerate(selected):
        row_seed = item.get("seed")
        if not isinstance(row_seed, int) or isinstance(row_seed, bool) or row_seed < 0:
            raise RuntimeError(f"row at request_index={idx} is missing a valid non-negative integer seed: {row_seed!r}")
        msg_sha = callcache.messages_sha256(item["messages"])
        if msg_sha != item["messages_sha256"]:
            raise RuntimeError(
                f"row at request_index={idx} failed messages_sha256 self-check: {msg_sha} != {item['messages_sha256']}"
            )
        out.append({"request_index": idx, **item})
    return out


def verify_against_journal(selected_rows: list[dict], journal_path: Path) -> dict:
    """Verify every selected row's messages_sha256 and key against P1r's journal (60/60 required)."""

    journal_rows = [json.loads(ln) for ln in journal_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_key: dict[str, list[dict]] = {}
    for r in journal_rows:
        by_key.setdefault(r["key"], []).append(r)

    matched = 0
    total_p1r_s = 0.0
    for row in selected_rows:
        if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool) or row["seed"] < 0:
            raise RuntimeError(f"request_index {row.get('request_index')}: invalid seed {row.get('seed')!r}")
        j_idx = row.get("p1r_journal_index")
        if j_idx is None or not (0 <= j_idx < len(journal_rows)):
            raise RuntimeError(f"request_index {row.get('request_index')}: invalid p1r_journal_index {j_idx!r}")
        j_row = journal_rows[j_idx]
        recomputed_sha = callcache.messages_sha256(row["messages"])
        if recomputed_sha != row["messages_sha256"]:
            raise RuntimeError(
                f"request_index {row['request_index']}: recomputed messages_sha256 {recomputed_sha} != {row['messages_sha256']}"
            )
        if j_row.get("messages_sha256") != row["messages_sha256"]:
            raise RuntimeError(
                f"request_index {row['request_index']}: journal messages_sha256 {j_row.get('messages_sha256')} "
                f"!= {row['messages_sha256']}"
            )
        if j_row.get("key") != row["key"]:
            raise RuntimeError(
                f"request_index {row['request_index']}: journal key {j_row.get('key')} != {row['key']}"
            )
        if row["key"] not in by_key:
            raise RuntimeError(f"request_index {row['request_index']}: key {row['key']} not found in journal")
        comp = j_row.get("completion")
        if isinstance(comp, dict) and isinstance(comp.get("seconds"), (int, float)):
            total_p1r_s += float(comp["seconds"])
        matched += 1

    if matched != len(selected_rows):
        raise RuntimeError(f"verified {matched}/{len(selected_rows)} rows against journal")
    return {
        "verified": matched,
        "total": len(selected_rows),
        "verdict": sum(1 for r in selected_rows if r["check"] == "verdict"),
        "twin_verdict": sum(1 for r in selected_rows if r["check"] == "twin_verdict"),
        "flaw": sum(1 for r in selected_rows if r["check"] == "flaw"),
        "flaw_pilot": sum(1 for r in selected_rows if r["check"] == "flaw" and r["split"] != "apply"),
        "flaw_apply": sum(1 for r in selected_rows if r["check"] == "flaw" and r["split"] == "apply"),
        "p1r_c1_seconds": round(total_p1r_s, 2),
    }


def build_from_paths(
    *,
    drafts_path: Path,
    pack_dir: Path,
    journal_path: Path,
    cards_path: Path,
    out_path: Path,
    seed: int = DEFAULT_SEED,
    verify_pins: bool = True,
) -> dict:
    all_non_trait = reconstruct_p1r_non_trait_calls(
        drafts_path=drafts_path,
        pack_dir=pack_dir,
        journal_path=journal_path,
        cards_path=cards_path,
        verify_pins=verify_pins,
    )
    selected = select_and_permute_60(all_non_trait, seed=seed)
    verification = verify_against_journal(selected, journal_path)
    if verification["verified"] != EXPECTED_TOTAL_SELECTED:
        raise RuntimeError(f"expected {EXPECTED_TOTAL_SELECTED}/{EXPECTED_TOTAL_SELECTED} verified, got {verification}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    out_sha = file_sha256(out_path)
    return {
        "out": str(out_path),
        "sha256": out_sha,
        "seed": seed,
        **verification,
    }


def build_from_receipts(
    receipts_dir: Path,
    out_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    verify_pins: bool = True,
) -> dict:
    """Extract P1r's tars into a temporary scratch directory (leaving receipts untouched) and build requests-60.jsonl."""

    pilot_tar = receipts_dir / "pilot-out.tar"
    pack_tar = receipts_dir / "pack.tar"
    cards_path = receipts_dir / "card-values.json"
    for p in (pilot_tar, pack_tar, cards_path):
        if not p.is_file():
            raise RuntimeError(f"missing P1r receipt file: {p}")

    with tempfile.TemporaryDirectory(prefix="batchdrift-p1r-") as tmp:
        tmp_root = Path(tmp)
        pilot_extract = tmp_root / "pilot-out"
        pack_extract = tmp_root / "pack"
        pilot_extract.mkdir(parents=True)
        pack_extract.mkdir(parents=True)
        with tarfile.open(pilot_tar) as tf:
            tf.extractall(pilot_extract, filter="data")
        with tarfile.open(pack_tar) as tf:
            tf.extractall(pack_extract, filter="data")

        drafts_path = pilot_extract / "pilot" / "pilot" / "gemma-4-31b-it" / "w1" / "wave-drafts.jsonl"
        journal_path = pilot_extract / "pilot" / "pilot" / "gemma-4-31b-it" / "calls-w1-muse-glimmer-30b.jsonl"
        return build_from_paths(
            drafts_path=drafts_path,
            pack_dir=pack_extract,
            journal_path=journal_path,
            cards_path=cards_path,
            out_path=out_path,
            seed=seed,
            verify_pins=verify_pins,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build and verify requests-60.jsonl from P1r receipts or unpacked paths.")
    ap.add_argument("--receipts-dir", type=Path, default=P1R_RECEIPTS_DEFAULT)
    ap.add_argument("--drafts", type=Path, default=None, help="Path to wave-drafts.jsonl (overrides --receipts-dir)")
    ap.add_argument("--pack", type=Path, default=None, help="Path to unpacked pack directory (overrides --receipts-dir)")
    ap.add_argument("--journal", type=Path, default=None, help="Path to P1r calls journal (overrides --receipts-dir)")
    ap.add_argument("--cards", type=Path, default=None, help="Path to card-values.json (overrides --receipts-dir)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-pin-check", action="store_true", help="Skip P1r sha256 pin assertions (for synthetic test fixtures)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "requests-60.jsonl",
    )
    args = ap.parse_args(argv)

    if args.drafts or args.pack or args.journal or args.cards:
        if not (args.drafts and args.pack and args.journal and args.cards):
            raise RuntimeError("when overriding paths, provide all of --drafts, --pack, --journal, and --cards")
        summary = build_from_paths(
            drafts_path=args.drafts,
            pack_dir=args.pack,
            journal_path=args.journal,
            cards_path=args.cards,
            out_path=args.out,
            seed=args.seed,
            verify_pins=not args.no_pin_check,
        )
    else:
        summary = build_from_receipts(
            args.receipts_dir,
            args.out,
            seed=args.seed,
            verify_pins=not args.no_pin_check,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
