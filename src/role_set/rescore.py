"""`role-set rescore`: blind second-critic rescore of a sample of each writer arm's w0 drafts (R&D, 2026-10-17).

Before any writer arm goes on to its second lesson, a second critic re-scores a blind sample of 4 w0 drafts per arm:
2 that passed both the mechanical gates and Muse at the w0 critique, and 2 that Muse's critic check refused (a draft
refused by a mechanical gate never reached Muse, so it is not in the refused group). An arm whose writer is the second
critic's own family is skipped and reported as unsampled.

The draw is seeded random within each group, and all arms' drawn items are shuffled into one blind queue with the same
seed. `<out>/sample-manifest.json` is written before anything is drawn and updated with the sample before anything is
scored. Each item is checked with `Runner.mechanical` (asserted equal to its w0 result) and `Runner.critique_step` at
the second critic's card values. The batch-mix check (`check_batch`) is never asserted on the 4-item sample. Scores map
back to arms only after scoring, from the manifest.
"""

from __future__ import annotations

import json
import random
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from role_set import cards, proven, waves
from role_set.assemble import FLAWS
from role_set.critic import flaw_messages
from role_set.gates import Gates, GatesUnavailable, load_gates
from role_set.guards import GuardError
from role_set.mock import MockCritic, mock_gates
from role_set.pack import PackError, verify
from role_set.prompts import load_pack
from role_set.runner import Runner
from role_set.write import critic_caller, resolve_critic, trait_names_from_pack

SAMPLE_PER_GROUP = 2
FLAG_DISAGREEMENTS = 2
VERBATIM_NOTE = "4 items per arm finds gross bias only, not small leans."
CRITIC_CODES = frozenset({
    "VERDICT_MISMATCH",
    "EVIDENCE_MISMATCH",
    "VOID",
    "FLAW_MISMATCH",
    "ANSWER_MISMATCH",
    "TRAIT_MISMATCH",
})
WINDOW_RE = re.compile(r"window[0-9a-z_-]*", re.I)


class RescoreError(RuntimeError):
    """A rescore input, seal, pin or mechanical-gate check failed."""


@dataclass(frozen=True)
class RescoreConfig:
    pack: Path = Path("roles/verifier")
    writers_file: Path = Path("writers.json")
    cards: Path | None = None
    critic_endpoint: str | None = None
    critic_model: str | None = None
    critic_gguf_sha256: str | None = None
    seed: int = 0
    out: Path = Path("rescore")
    runs: tuple[Path, ...] = ()
    drafts: tuple[str, ...] = ()
    mock: bool = False


@dataclass
class _ArmInput:
    arm: str
    w0_path: Path
    w1_path: Path
    header0: dict
    drafts0: list[dict]
    header1: dict
    drafts1: list[dict]
    w1_call_notes: dict[tuple[str, int], list[dict]]
    w1_journal_flaws: dict[str, str]
    window_tokens: set[str]


def _run_dir(wave_dir: Path) -> Path:
    for d in (wave_dir, wave_dir / "mock", wave_dir / "rehearsal"):
        if (d / waves.DRAFTS_FILE).exists():
            return d
    return wave_dir


def _flaw_word(reply: str | None) -> str | None:
    if not reply or not isinstance(reply, str):
        return None
    word = reply.strip().split()[0].strip(".,:;'\"").lower() if reply.strip() else ""
    return word if word in FLAWS + ("none",) else None


def _read_w1_call_notes(w1_file: Path) -> dict[tuple[str, int], list[dict]]:
    """Critic check notes per (slot_id, attempt) from w1's drafts.jsonl, if present beside wave-drafts.jsonl."""

    notes: dict[tuple[str, int], list[dict]] = {}
    drafts_log = w1_file.parent / "drafts.jsonl"
    if not drafts_log.exists():
        return notes
    for line in drafts_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        slot_id = rec.get("slot")
        attempt = int(rec.get("attempt") or 0)
        if slot_id and isinstance(rec.get("critic"), list):
            notes[(slot_id, attempt)] = rec["critic"]
            notes[(slot_id, 0)] = rec["critic"]
    return notes


def _read_w1_journal_flaws(w1_file: Path) -> dict[str, str]:
    """Flaw replies from a call journal (e.g. checkpoints.py extract's calls/<arm>.jsonl), keyed by request key."""

    from role_set import callcache  # noqa: PLC0415

    candidates = [
        w1_file.parent / "calls" / w1_file.name,
        w1_file.parent / callcache.JOURNAL_FILE,
        *sorted(w1_file.parent.parent.glob(f"calls-*{w1_file.stem}*.jsonl")),
    ]
    out: dict[str, str] = {}
    for p in candidates:
        if p.is_file():
            for key, q in callcache.load(p).items():
                if q:
                    out[key] = q[0].text
    return out


def _window_tokens_from_paths(*paths: Path) -> set[str]:
    tokens: set[str] = set()
    for p in paths:
        for part in p.parts:
            for m in WINDOW_RE.findall(part):
                if len(m) >= 7:
                    tokens.add(m)
    return tokens


def _parse_drafts_spec(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec:
        raise RescoreError(f"REFUSED: --drafts must be <arm>=<w0 wave-drafts.jsonl>,<w1 wave-drafts.jsonl>, got {spec!r}")
    arm, rest = spec.split("=", 1)
    parts = [x.strip() for x in rest.split(",") if x.strip()]
    if not arm.strip() or len(parts) != 2:
        raise RescoreError(f"REFUSED: --drafts must be <arm>=<w0 wave-drafts.jsonl>,<w1 wave-drafts.jsonl>, got {spec!r}")
    return arm.strip(), Path(parts[0]), Path(parts[1])


def _load_one_arm(arm: str, w0_path: Path, w1_path: Path, extra_tokens: set[str]) -> _ArmInput:
    try:
        header0, drafts0 = waves.read(w0_path)
        header1, drafts1 = waves.read(w1_path)
    except waves.WaveError as e:
        raise RescoreError(str(e)) from e
    slots0 = [d.get("slot", {}).get("id") for d in drafts0]
    slots1 = [d.get("slot", {}).get("id") for d in drafts1]
    if slots0 != slots1:
        raise RescoreError(f"REFUSED: {arm}'s w0 and w1 slots do not match ({len(slots0)} vs {len(slots1)})")
    tokens = set(extra_tokens) | _window_tokens_from_paths(w0_path, w1_path)
    for h in (header0, header1):
        tag = ((h.get("writer_host") or {}).get("machine") or {}).get("colab_release_tag")
        if isinstance(tag, str) and tag.strip():
            tokens.add(tag.strip())
        for w in h.get("waves") or []:
            wtag = ((w.get("pins") or {}).get("machine") or {}).get("colab_release_tag")
            if isinstance(wtag, str) and wtag.strip():
                tokens.add(wtag.strip())
    return _ArmInput(
        arm=arm,
        w0_path=w0_path,
        w1_path=w1_path,
        header0=header0,
        drafts0=drafts0,
        header1=header1,
        drafts1=drafts1,
        w1_call_notes=_read_w1_call_notes(w1_path),
        w1_journal_flaws=_read_w1_journal_flaws(w1_path),
        window_tokens=tokens,
    )


def _discover_tar_arms(tar_path: Path, extract_root: Path) -> list[_ArmInput]:
    if not tar_path.is_file():
        raise RescoreError(f"REFUSED: cannot read --runs tar {tar_path}")
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_root, filter="data")
    except (OSError, tarfile.TarError) as e:
        raise RescoreError(f"REFUSED: cannot unpack {tar_path.name} ({type(e).__name__})") from e
    if (extract_root / "pilot" / "pilot").is_dir():
        base = extract_root / "pilot" / "pilot"
    elif (extract_root / "pilot").is_dir():
        base = extract_root / "pilot"
    else:
        base = extract_root
    tokens = _window_tokens_from_paths(tar_path)
    found: list[_ArmInput] = []
    for arm_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name not in ("twins", "pilot")):
        w0_dir = _run_dir(arm_dir / "w0")
        w1_dir = _run_dir(arm_dir / "w1")
        w0_file = w0_dir / waves.DRAFTS_FILE
        w1_file = w1_dir / waves.DRAFTS_FILE
        if not w0_file.exists() and not w1_file.exists():
            continue
        if not w0_file.exists() or not w1_file.exists():
            raise RescoreError(f"REFUSED: arm {arm_dir.name} in {tar_path.name} lacks {'w0' if not w0_file.exists() else 'w1'} {waves.DRAFTS_FILE}")
        found.append(_load_one_arm(arm_dir.name, w0_file, w1_file, tokens | {arm_dir.name}))
    if not found:
        raise RescoreError(f"REFUSED: no arm with w0 and w1 drafts found in {tar_path.name}")
    return found


def _muse_flaw_for_c(arm_in: _ArmInput, d0: dict, d1: dict) -> str | None:
    """Muse's flaw label on a kind-C draft in w1, from w1's drafts.jsonl, refusal, call journal, or mock answers."""

    slot_id = d0["slot"]["id"]
    attempt = int(d0.get("attempt") or 1)
    notes = arm_in.w1_call_notes.get((slot_id, attempt)) or arm_in.w1_call_notes.get((slot_id, 0)) or d1.get("critic") or []
    for entry in notes:
        if entry.get("check") == "flaw":
            word = _flaw_word(entry.get("raw"))
            if word:
                return word
    refusals = d1.get("refusals") or []
    if refusals and refusals[-1].get("code") == "FLAW_MISMATCH":
        word = _flaw_word(refusals[-1].get("critic_raw"))
        if word:
            return word
    try:
        fields = json.loads((d0.get("candidate") or {}).get("text") or "{}")
    except ValueError:
        fields = {}
    trace = str(fields.get("checker_trace") or "").strip()
    if trace and arm_in.w1_journal_flaws:
        from role_set import callcache  # noqa: PLC0415

        material = (d0.get("material") or {}).get("text") or ""
        statement = str(fields.get("statement") or "")
        checker_verdict = str(fields.get("checker_verdict") or "")
        msgs = flaw_messages(material, statement, trace, checker_verdict)
        for seed in (0, None):
            kw = {"seed": seed} if seed is not None else {}
            k = callcache.request_key("critic", (msgs,), kw)
            if k in arm_in.w1_journal_flaws:
                word = _flaw_word(arm_in.w1_journal_flaws[k])
                if word:
                    return word
    if trace:
        mock_ans = waves.load_answers(arm_in.header1) or waves.load_answers(arm_in.header0)
        if ("flaw", trace) in mock_ans:
            return str(mock_ans[("flaw", trace)])
    return None


def _classify_slot(arm_in: _ArmInput, d0: dict, d1: dict) -> dict:
    """Classify one slot at the w0 critique: accepted, critic-refused, or gate-refused."""

    slot = d0["slot"]
    slot_id = slot["id"]
    w0_state = d1["state"]
    if d0.get("state") != "awaiting_critique" or d0.get("candidate") is None:
        refs0 = d0.get("refusals") or []
        r = refs0[-1] if refs0 else None
        return {
            "arm": arm_in.arm,
            "slot": slot_id,
            "kind": slot["kind"],
            "w0_state": w0_state,
            "refusal_code": r["code"] if r else None,
            "refused_by": "gate" if r else None,
            "check": "gate" if r else None,
            "group": None,
        }
    if w0_state == "accepted":
        return {
            "arm": arm_in.arm,
            "slot": slot_id,
            "kind": slot["kind"],
            "w0_state": "accepted",
            "refusal_code": None,
            "refused_by": None,
            "check": None,
            "group": "accepted",
        }
    refs0 = d0.get("refusals") or []
    refs1 = d1.get("refusals") or []
    added = refs1[len(refs0):]
    r = added[-1] if added else (refs1[-1] if refs1 else None)
    code = r.get("code") if r else None
    is_critic = bool(r and (code in CRITIC_CODES or "critic_raw" in r))
    check = "critic" if is_critic else ("gate" if r else None)
    return {
        "arm": arm_in.arm,
        "slot": slot_id,
        "kind": slot["kind"],
        "w0_state": w0_state,
        "refusal_code": code,
        "refused_by": check,
        "check": check,
        "group": "refused" if is_critic else None,
    }


def _writer_entry_for(arm_in: _ArmInput, listed: proven.ProvenList | None, *, mock: bool) -> tuple[dict, dict]:
    """Read the arm's writer pins one way only (waves.header_writer_pins) and match writers.json."""

    w_pins = {**waves.header_writer_pins(arm_in.header1), **waves.header_writer_pins(arm_in.header0)}
    model = (w_pins.get("writer_model") or "").strip()
    writers = listed.writers if listed is not None else {}
    if model in writers:
        entry = dict(writers[model])
    elif w_pins.get("writer_family"):
        entry = {"served_as": model or arm_in.arm, "family": w_pins["writer_family"], "card": w_pins.get("card_candidate")}
    elif arm_in.arm in writers:
        entry = dict(writers[arm_in.arm])
    elif mock or model == "mock":
        entry = {"served_as": model or arm_in.arm, "family": ""}
    else:
        raise RescoreError(f"REFUSED: {arm_in.arm}'s drafts name writer {model!r}, which is not on the proven writer list")
    return w_pins, entry


def _forbidden_strings_for(arm_in: _ArmInput, w_pins: dict, writer_entry: dict) -> set[str]:
    """Strings that must never reach the blinded item or the second critic's messages."""

    out: set[str] = set(arm_in.window_tokens)
    for val in (
        arm_in.arm,
        w_pins.get("writer_model"),
        w_pins.get("card_candidate"),
        w_pins.get("writer_repo"),
        w_pins.get("writer_gguf_file"),
        writer_entry.get("served_as"),
        writer_entry.get("card"),
        writer_entry.get("repo"),
        writer_entry.get("gguf_file"),
    ):
        if isinstance(val, str) and val.strip() and val.strip() != "mock":
            out.add(val.strip())
    return out


def _strip_tokens(text: str, tokens: set[str]) -> str:
    out = text
    for tok in sorted(tokens, key=len, reverse=True):
        if tok and tok in out:
            out = out.replace(tok, "")
    return out


def _blind_draft(d0: dict, tokens: set[str]) -> dict:
    """A copy of the w0 draft with writer name, served_as, arm directory and window stripped."""

    cand = dict(d0["candidate"])
    cand["model"] = ""
    mat = None
    if d0.get("material"):
        src = d0["material"].get("source")
        if isinstance(src, dict):
            src = {k: (_strip_tokens(v, tokens) if isinstance(v, str) else v) for k, v in src.items()}
        mat = {"text": d0["material"]["text"], "source": src}
    return {
        "type": "draft",
        "slot": json.loads(_strip_tokens(json.dumps(d0["slot"], ensure_ascii=False), tokens)),
        "state": "awaiting_critique",
        "attempt": d0["attempt"],
        "ended": False,
        "messages": [],
        "material": mat,
        "facts": d0.get("facts"),
        "spent": [],
        "refusals": [],
        "candidate": cand,
        "row": None,
        "model": None,
        "lookup": bool(d0.get("lookup")),
        "receipt": None,
        "accepted_seq": None,
    }


def _blind_critic(critic, forbidden: set[str], queue_index: int):
    """Wrap the critic so any forbidden string in its messages refuses the run (naming the queue index, never the string)."""

    def wrapped(messages: list[dict], **kwargs):
        for m in messages:
            text = str(m.get("content") or "")
            for tok in forbidden:
                if tok and tok in text:
                    raise RescoreError(
                        f"REFUSED: unblinded identifier in critic messages at queue index {queue_index}"
                    )
        return critic(messages, **kwargs)

    return wrapped


def _w1_critics(header1: dict) -> set[str]:
    out: set[str] = set()
    for w in header1.get("waves") or []:
        if w.get("wave") == "critique":
            cm = ((w.get("pins") or {}).get("critic_model") or "").strip()
            if cm:
                out.add(cm)
    for key in ("critic_pins", "batch_pins"):
        cm = ((header1.get(key) or {}).get("critic_model") or "").strip()
        if cm:
            out.add(cm)
    return out


def render_table(report: dict) -> str:
    lines = [
        "| arm | n | verdict | groups | flaw (C both refused) | flag |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, r in report["arms"].items():
        if r.get("unsampled"):
            fam = r.get("writer_family") or "same family"
            lines.append(f"| {arm} | 0 | unsampled ({fam}) | unsampled | - | unsampled |")
            continue
        verdict_col = f"{r['verdict_agreements']}/{r['n']}"
        flaw_col = f"{r['flaw_agreements']}/{r['c_both_refused']}"
        flag_col = "FLAG" if r["flagged"] else "ok"
        lines.append(f"| {arm} | {r['n']} | {verdict_col} | {r['group_split']} | {flaw_col} | {flag_col} |")
    lines.append("")
    lines.append(VERBATIM_NOTE)
    return "\n".join(lines)


def run_rescore(cfg: RescoreConfig, *, critic=None) -> dict:
    """Run the blind second-critic rescore and write sample-manifest.json, rescore.json and rescore-report.json."""

    if not cfg.runs and not cfg.drafts:
        raise RescoreError("REFUSED: rescore needs at least one --runs <pilot-out.tar> or --drafts <arm>=<w0>,<w1>")
    try:
        verify(cfg.pack)
    except PackError as e:
        raise RescoreError(str(e)) from e
    pack = load_pack(cfg.pack)
    trait_names = trait_names_from_pack(pack)

    listed: proven.ProvenList | None = None
    if cfg.writers_file and cfg.writers_file.exists():
        try:
            listed = proven.load(cfg.writers_file)
        except GuardError as e:
            raise RescoreError(str(e)) from e
    elif not cfg.mock:
        raise RescoreError(f"REFUSED: cannot read writers file {cfg.writers_file}")

    c_model = (cfg.critic_model or ("mock" if cfg.mock else "")).strip()
    c_pins: dict | None = None
    c_card_pins: dict | None = None
    critic_entry: dict = {"served_as": c_model, "family": ""}
    if listed is not None and c_model in listed.critics:
        critic_entry = dict(listed.critics[c_model])
    elif listed is not None and c_model in listed.writers:
        critic_entry = dict(listed.writers[c_model])

    if not cfg.mock:
        if listed is None:
            raise RescoreError(f"REFUSED: cannot read writers file {cfg.writers_file}")
        if cfg.cards is None:
            raise RescoreError("REFUSED: the critic runs at its card values; pass --cards <card-values.json>")
        try:
            c_endpoint, c_model, c_pins = resolve_critic(listed, cfg.critic_model, cfg.critic_endpoint, cfg.critic_gguf_sha256)
            critic_entry = dict(listed.critics[c_model])
            card_set, card_sha = cards.load(cfg.cards)
            c_card = cards.select(card_set, critic_entry, c_model)
            c_card_pins = cards.pins(c_card, card_sha)
        except GuardError as e:
            raise RescoreError(str(e)) from e
        if critic is None:
            critic = critic_caller(c_endpoint, c_model, c_card)

    c_family = (critic_entry.get("family") or "").strip() or None
    if not cfg.mock and not c_family:
        raise RescoreError(f"REFUSED: critic {c_model!r} has no family on record")

    with tempfile.TemporaryDirectory(prefix="rescore-") as tmp_str:
        tmp = Path(tmp_str)
        arms_in: list[_ArmInput] = []
        seen_arms: set[str] = set()
        for idx, tar_path in enumerate(cfg.runs):
            for a in _discover_tar_arms(Path(tar_path), tmp / f"tar-{idx}"):
                if a.arm in seen_arms:
                    raise RescoreError(f"REFUSED: arm {a.arm!r} is given more than once")
                seen_arms.add(a.arm)
                arms_in.append(a)
        for spec in cfg.drafts:
            arm, w0_p, w1_p = _parse_drafts_spec(spec)
            if arm in seen_arms:
                raise RescoreError(f"REFUSED: arm {arm!r} is given more than once")
            seen_arms.add(arm)
            arms_in.append(_load_one_arm(arm, w0_p, w1_p, {arm}))

        arms_in.sort(key=lambda a: a.arm)

        closed_form = None
        for a in arms_in:
            cf = (a.header0.get("gates") or {}).get("closed_form")
            if cf is not None:
                closed_form = cf
                break
        try:
            gates: Gates = load_gates(closed_form=closed_form)
        except GatesUnavailable as e:
            if not cfg.mock:
                raise RescoreError(str(e)) from e
            gates = mock_gates(closed_form=closed_form)

        for a in arms_in:
            g = a.header0.get("gates") or {}
            if (g.get("version"), g.get("sha256")) != (gates.version, gates.sha256):
                raise RescoreError(
                    f"REFUSED: {a.arm}'s drafts were checked by gates {g.get('version')}; "
                    f"rescore has {gates.version}"
                )
            if c_model in _w1_critics(a.header1):
                raise RescoreError(
                    f"REFUSED: second critic {c_model!r} is the same critic that judged {a.arm}'s w1"
                )

        if cfg.mock and critic is None:
            merged_answers: dict = {}
            for a in arms_in:
                merged_answers.update(waves.load_answers(a.header0))
                merged_answers.update(waves.load_answers(a.header1))
            critic = MockCritic(answers=merged_answers)

        # Write the seed into sample-manifest.json before anything is drawn.
        cfg.out.mkdir(parents=True, exist_ok=True)
        manifest_path = cfg.out / "sample-manifest.json"
        manifest: dict = {
            "seed": cfg.seed,
            "critic_model": c_model,
            "critic_family": c_family,
            "unsampled": [],
            "arms": {},
            "items": [],
            "queue": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        rng = random.Random(cfg.seed)
        drawn_pool: list[tuple[dict, dict, str | None]] = []  # (manifest_item_meta, blinded_draft, muse_flaw)
        forbidden_all: set[str] = set()

        for a in arms_in:
            w_pins, w_entry = _writer_entry_for(a, listed, mock=cfg.mock)
            forbidden_all |= _forbidden_strings_for(a, w_pins, w_entry)
            w_model = w_pins.get("writer_model") or w_entry.get("served_as") or a.arm
            w_family = (w_entry.get("family") or w_pins.get("writer_family") or "").strip() or None
            if not cfg.mock and not w_family:
                raise RescoreError(f"REFUSED: {a.arm}'s writer {w_model!r} has no family on record")

            if w_family and c_family:
                try:
                    proven.require_other_family({**w_entry, "family": w_family}, {**critic_entry, "family": c_family})
                except GuardError as e:
                    reason = str(e)
                    manifest["unsampled"].append({
                        "arm": a.arm,
                        "writer_model": w_model,
                        "writer_family": w_family,
                        "critic_model": c_model,
                        "critic_family": c_family,
                        "reason": reason,
                    })
                    manifest["arms"][a.arm] = {
                        "arm": a.arm,
                        "writer_model": w_model,
                        "writer_family": w_family,
                        "sampled": False,
                        "unsampled": True,
                        "skip_reason": reason,
                        "available": {"accepted": 0, "refused": 0},
                        "drawn": {"accepted": 0, "refused": 0},
                        "short": False,
                        "short_groups": {},
                        "note": reason,
                        "slots": [],
                    }
                    continue

            by_slot0 = {d["slot"]["id"]: d for d in a.drafts0}
            slot_records: list[dict] = []
            acc_candidates: list[tuple[dict, dict, dict]] = []
            ref_candidates: list[tuple[dict, dict, dict]] = []
            for d0, d1 in zip(a.drafts0, a.drafts1):
                info = _classify_slot(a, d0, d1)
                slot_records.append(info)
                if info["group"] == "accepted":
                    acc_candidates.append((info, d0, d1))
                elif info["group"] == "refused":
                    ref_candidates.append((info, d0, d1))

            acc_pick = rng.sample(acc_candidates, min(SAMPLE_PER_GROUP, len(acc_candidates)))
            ref_pick = rng.sample(ref_candidates, min(SAMPLE_PER_GROUP, len(ref_candidates)))
            short_groups: dict[str, str] = {}
            if len(acc_candidates) < SAMPLE_PER_GROUP:
                short_groups["accepted"] = f"only {len(acc_candidates)} of {SAMPLE_PER_GROUP} available"
            if len(ref_candidates) < SAMPLE_PER_GROUP:
                short_groups["refused"] = f"only {len(ref_candidates)} of {SAMPLE_PER_GROUP} available"
            note = "; ".join(f"{g}: {msg}" for g, msg in short_groups.items()) if short_groups else None

            manifest["arms"][a.arm] = {
                "arm": a.arm,
                "writer_model": w_model,
                "writer_family": w_family,
                "sampled": True,
                "unsampled": False,
                "skip_reason": None,
                "available": {"accepted": len(acc_candidates), "refused": len(ref_candidates)},
                "drawn": {"accepted": len(acc_pick), "refused": len(ref_pick)},
                "short": bool(short_groups),
                "short_groups": short_groups,
                "note": note,
                "slots": slot_records,
            }

            arm_tokens = _forbidden_strings_for(a, w_pins, w_entry)
            for info, d0, d1 in acc_pick + ref_pick:
                muse_flaw = _muse_flaw_for_c(a, by_slot0[info["slot"]], d1) if info["kind"] == "C" and info["group"] == "refused" else None
                drawn_pool.append((dict(info), _blind_draft(d0, arm_tokens), muse_flaw))

        rng.shuffle(drawn_pool)
        blind_queue: list[dict] = []
        muse_flaws_by_q: dict[int, str | None] = {}
        for q_idx, (meta, blinded_d0, muse_flaw) in enumerate(drawn_pool):
            item_entry = {"queue_index": q_idx, **meta}
            manifest["items"].append(item_entry)
            manifest["queue"].append({
                "queue_index": q_idx,
                "arm": meta["arm"],
                "slot": meta["slot"],
                "group": meta["group"],
            })
            muse_flaws_by_q[q_idx] = muse_flaw
            blind_queue.append({"queue_index": q_idx, "draft": blinded_d0})

        # Write the complete sample-manifest.json before any scoring starts.
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        # Score the blinded queue with Runner.mechanical and Runner.critique_step.
        raw_scores: list[dict] = []
        scratch_root = tmp / "runner-scratch"
        for q_item in blind_queue:
            q_idx = q_item["queue_index"]
            d_blind = q_item["draft"]
            runner = Runner(
                gates=gates,
                pack=pack,
                trait_names=trait_names,
                writer=None,
                critic=_blind_critic(critic, forbidden_all, q_idx),
                out_dir=scratch_root / f"q-{q_idx}",
            )
            st = waves.state_of(d_blind)
            c = waves.completion_of(d_blind)
            m_refusal, cand, _ = runner.mechanical(st, c)
            if m_refusal is not None or cand is None or cand.lookup != bool(d_blind.get("lookup")):
                detail = f"{m_refusal['code']}: {m_refusal['detail']}" if m_refusal else "lookup mismatch"
                raise RescoreError(
                    f"REFUSED: mechanical check on {st.slot.id} (queue {q_idx}) differed from w0 ({detail})"
                )
            c_refusal = runner.critique_step(st, cand)
            notes = list(runner._critic_notes.get(st.slot.id) or [])
            if not notes and (scratch_root / f"q-{q_idx}" / "drafts.jsonl").exists():
                for ln in (scratch_root / f"q-{q_idx}" / "drafts.jsonl").read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        notes = json.loads(ln).get("critic") or []
            second_flaw = None
            for n_entry in notes:
                if n_entry.get("check") == "flaw":
                    second_flaw = _flaw_word(n_entry.get("raw"))
            raw_scores.append({
                "queue_index": q_idx,
                "mechanical": "pass",
                "second_verdict": "accepted" if c_refusal is None else "refused",
                "second_refusal_code": c_refusal["code"] if c_refusal is not None else None,
                "second_refusal_detail": c_refusal["detail"] if c_refusal is not None else None,
                "second_flaw": second_flaw,
                "critic_checks": notes,
            })

    # Map scores back to arms only after scoring, from the manifest.
    manifest_by_q = {item["queue_index"]: item for item in manifest["items"]}
    scored_items: list[dict] = []
    items_by_arm: dict[str, list[dict]] = {arm: [] for arm in manifest["arms"]}

    for score in raw_scores:
        q_idx = score["queue_index"]
        meta = manifest_by_q[q_idx]
        muse_verdict = "accepted" if meta["group"] == "accepted" else "refused"
        second_verdict = score["second_verdict"]
        verdict_agree = muse_verdict == second_verdict
        both_refused_c = meta["kind"] == "C" and muse_verdict == "refused" and second_verdict == "refused"
        muse_flaw = muse_flaws_by_q.get(q_idx) if both_refused_c else None
        second_flaw = score["second_flaw"] if both_refused_c else None
        flaw_agree = bool(muse_flaw and second_flaw and muse_flaw == second_flaw) if both_refused_c else None
        item_out = {
            "queue_index": q_idx,
            "arm": meta["arm"],
            "slot": meta["slot"],
            "kind": meta["kind"],
            "group": meta["group"],
            "w0_state": meta["w0_state"],
            "refusal_code": meta["refusal_code"],
            "w0_refusal_code": meta["refusal_code"],
            "refused_by": meta["refused_by"],
            "check": meta["check"],
            "mechanical": score["mechanical"],
            "muse_verdict": muse_verdict,
            "second_verdict": second_verdict,
            "second_refusal_code": score["second_refusal_code"],
            "second_refusal_detail": score["second_refusal_detail"],
            "verdict_agree": verdict_agree,
            "agree": verdict_agree,
            "muse_flaw": muse_flaw,
            "second_flaw": second_flaw,
            "flaw_agree": flaw_agree,
            "critic_checks": score["critic_checks"],
        }
        scored_items.append(item_out)
        items_by_arm[meta["arm"]].append(item_out)

    rescore_data = {
        "seed": cfg.seed,
        "critic_model": c_model,
        "items": scored_items,
    }
    (cfg.out / "rescore.json").write_text(json.dumps(rescore_data, indent=1), encoding="utf-8")

    arms_report: dict[str, dict] = {}
    for arm, arm_meta in manifest["arms"].items():
        if arm_meta.get("unsampled"):
            arms_report[arm] = {
                "arm": arm,
                "writer_model": arm_meta["writer_model"],
                "writer_family": arm_meta["writer_family"],
                "sampled": False,
                "unsampled": True,
                "skip_reason": arm_meta["skip_reason"],
                "n": 0,
                "verdict_agreements": None,
                "agree": None,
                "disagreements": 0,
                "by_group": {
                    "accepted": {"n": 0, "agree": 0, "disagree": 0},
                    "refused": {"n": 0, "agree": 0, "disagree": 0},
                },
                "group_split": "unsampled",
                "c_both_refused": 0,
                "flaw_agreements": 0,
                "flaw_agree": 0,
                "flagged": False,
                "short": False,
            }
            continue
        arm_items = items_by_arm[arm]
        n = len(arm_items)
        v_agree = sum(1 for x in arm_items if x["verdict_agree"])
        disagree = n - v_agree
        acc_items = [x for x in arm_items if x["group"] == "accepted"]
        ref_items = [x for x in arm_items if x["group"] == "refused"]
        acc_n, acc_agree = len(acc_items), sum(1 for x in acc_items if x["verdict_agree"])
        ref_n, ref_agree = len(ref_items), sum(1 for x in ref_items if x["verdict_agree"])
        c_both = [x for x in arm_items if x["flaw_agree"] is not None]
        c_both_n = len(c_both)
        f_agree = sum(1 for x in c_both if x["flaw_agree"])
        flagged = disagree >= FLAG_DISAGREEMENTS
        arms_report[arm] = {
            "arm": arm,
            "writer_model": arm_meta["writer_model"],
            "writer_family": arm_meta["writer_family"],
            "sampled": True,
            "unsampled": False,
            "n": n,
            "verdict_agreements": v_agree,
            "agree": v_agree,
            "disagreements": disagree,
            "by_group": {
                "accepted": {"n": acc_n, "agree": acc_agree, "disagree": acc_n - acc_agree},
                "refused": {"n": ref_n, "agree": ref_agree, "disagree": ref_n - ref_agree},
            },
            "group_split": f"accepted {acc_agree}/{acc_n}, refused {ref_agree}/{ref_n}",
            "c_both_refused": c_both_n,
            "flaw_agreements": f_agree,
            "flaw_agree": f_agree,
            "flagged": flagged,
            "short": arm_meta["short"],
        }

    report = {
        "seed": cfg.seed,
        "critic_model": c_model,
        "critic_family": critic_entry.get("family") or None,
        "critic_pins": c_pins,
        "critic_card": c_card_pins,
        "gates": {"version": gates.version, "sha256": gates.sha256, "closed_form": gates.closed_form},
        "unsampled": [arm for arm, r in arms_report.items() if r.get("unsampled")],
        "flagged": [arm for arm, r in arms_report.items() if r.get("flagged")],
        "arms": arms_report,
        "note": VERBATIM_NOTE,
    }
    report["table"] = render_table(report)
    (cfg.out / "rescore-report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
