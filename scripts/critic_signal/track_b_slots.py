"""Track B Rung 0 bank V slot census, frozen 48-slot list builder, and post-write material check (TRACK-B-DESIGN.md §5).

Selects 48 fresh kind C base slots across non-T10 verifier lessons at plan seed 2:
- 24 odd decided slots (`slot_flaw` in `FLAWS`, verdict in `supported`/`unsupported`) -> `f_eligible`
- 24 even decided slots (`slot_flaw == "none"`, verdict in `supported`/`unsupported`) -> `controls`
- Zero `T10` slots and zero pilot-used C slots (`T02-C-01`, `T02-C-08`).

Priority order within each 24-slot half (sorted in lesson order, then slot ID order):
1. `new`: slots in neither Phase 4 nor Track A.
2. `phase4_only`: slots used only by Phase 4 (re-planned at fresh seed 2).
3. `track_a`: slots Track A used (re-planned at fresh seed 2).

Real-source rule:
- `Runner._material_for` picks a real-source excerpt from a hash of `slot.id` alone (`runner.py` 687-699).
- Therefore, a slot that is `real_source` in V's seed-2 plan is taken only if its ID was never a `real_source` slot in
  Phase 4 or Track A. Any slot skipped under this rule is recorded in `real_source_skipped` with its reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from track_a_slots import (  # noqa: E402
    DECIDED_VERDICTS,
    DEFAULT_DOMAINS,
    NON_T10_LESSONS,
    PHASE4_CONTROLS,
    PHASE4_F_ELIGIBLE,
    PILOT_USED_C_SLOTS,
    ROOT,
    USED_BEFORE_IDS,
    _slot_order_key,
)

PHASE4_NON_T10_IDS: frozenset[str] = frozenset(
    sid for sid in USED_BEFORE_IDS if not sid.startswith("T10-")
)
DEFAULT_POOL: Path = ROOT / "roles" / "verifier" / "pool.jsonl"
BUCKET_ORDER: tuple[str, ...] = ("new", "phase4_only", "track_a")


def sha256_lf_bytes(raw: bytes) -> str:
    """SHA-256 over LF-normalised bytes (matching Linux line endings)."""
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _git_head_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown"


def load_track_a_slots(
    track_a_path: Path,
    expected_sha256: str | None = None,
) -> tuple[dict, str, frozenset[str], frozenset[str]]:
    """Load the frozen Track A slots JSON, verify its sha256 against `expected_sha256`, and return
    `(data, sha256_lf, track_a_slot_ids, track_a_real_source_ids)`."""
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import plan_batch

    if not track_a_path.is_file():
        raise RuntimeError(f"REFUSED: --track-a-slots file does not exist: {track_a_path}")
    raw = track_a_path.read_bytes()
    sha_raw = hashlib.sha256(raw).hexdigest()
    sha_lf = sha256_lf_bytes(raw)
    if expected_sha256 is not None:
        want = expected_sha256.strip().lower()
        if not want or want not in (sha_raw, sha_lf):
            raise RuntimeError(
                f"REFUSED: --track-a-slots {track_a_path.name} sha256 mismatch: "
                f"got {sha_lf} (raw {sha_raw}), expected {want}"
            )
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise RuntimeError(f"REFUSED: cannot parse --track-a-slots {track_a_path}: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"REFUSED: --track-a-slots {track_a_path} is not a JSON object")

    raw_ids = data.get("slot_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise RuntimeError(f"REFUSED: --track-a-slots {track_a_path} has no non-empty 'slot_ids' list")
    track_a_ids = frozenset(str(x) for x in raw_ids)

    rs_ids: set[str] = set()
    slots_list = data.get("slots")
    if (
        isinstance(slots_list, list)
        and slots_list
        and all(isinstance(entry, dict) and "real_source" in entry for entry in slots_list)
    ):
        for entry in slots_list:
            if entry.get("real_source") is True and entry.get("id"):
                rs_ids.add(str(entry["id"]))
    else:
        ta_seed = int(data.get("plan_seed", 1))
        for lesson in NON_T10_LESSONS:
            for s in plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=ta_seed):
                if s.id in track_a_ids and s.real_source:
                    rs_ids.add(s.id)

    return data, sha_lf, track_a_ids, frozenset(rs_ids)


def phase4_real_source_slot_ids() -> frozenset[str]:
    """Return the non-T10 Phase 4 slot IDs that were `real_source` in Phase 4's seed-0 plan."""
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import plan_batch

    rs: set[str] = set()
    for lesson in NON_T10_LESSONS:
        for s in plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=0):
            if s.id in PHASE4_NON_T10_IDS and s.real_source:
                rs.add(s.id)
    return frozenset(rs)


def _plan_lesson_full(
    lesson: str,
    *,
    seed: int,
    table,
    packs: list,
    prior,
) -> tuple[list, dict, str]:
    """Run the full `role-set plan --lessons <lesson> --seed <seed> --domains <domains>` pipeline for one lesson
    and return `(slots, plan_json_dict, plan_lf_sha256)`."""
    from role_set import domains as dom_mod
    from role_set import solvefirst
    from role_set.cli import VERIFIER_SIDES, _check_slot_list_fields
    from role_set.plan import check_plan, plan_batch

    if lesson == "T10":
        raise ValueError("REFUSED: T10 is excluded from Track B bank V")

    slots = plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=seed)
    dom_mod.assign(slots, table, seed=seed)
    solvefirst.assign(slots, packs, seed=seed, table=table, prior=prior.packs)
    solvefirst.lock_skeletons(slots, packs, prior=prior.skeletons)
    dom_mod.check(slots)
    _check_slot_list_fields(slots)
    figures = [check_plan(slots)]
    plan_obj = {
        "role": "verifier",
        "seed": seed,
        "side_pairing": "default",
        "figures": figures,
        "slots": [s.as_dict() for s in slots],
        "domain_table": table.pin(),
        "exclusion_state_sha256": prior.sha256,
    }
    lf_bytes = json.dumps(plan_obj, indent=1).replace("\r\n", "\n").encode("utf-8")
    plan_sha = hashlib.sha256(lf_bytes).hexdigest()
    return slots, plan_obj, plan_sha


def _source_bucket(slot_id: str, phase4_ids: frozenset[str], track_a_ids: frozenset[str]) -> str:
    if slot_id not in phase4_ids and slot_id not in track_a_ids:
        return "new"
    if slot_id in phase4_ids and slot_id not in track_a_ids:
        return "phase4_only"
    return "track_a"


def build_track_b_slots(
    *,
    track_a_slots_path: Path,
    track_a_slots_sha256: str | None = None,
    against_paths: list[Path] | tuple[Path, ...] = (),
    pool_path: Path = DEFAULT_POOL,
    plan_seed: int = 2,
    domains_path: Path = DEFAULT_DOMAINS,
    target_per_half: int = 24,
    out_path: Path | None = None,
    role_set_commit: str | None = None,
) -> dict:
    """Compute the Track B bank V census at `plan_seed` (default 2) and build the 48-slot JSON payload."""
    from role_set import domains as dom_mod
    from role_set import exclusion, solvefirst
    from role_set.assemble import FLAWS, slot_flaw
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import plan_batch
    from role_set.write import load_pool

    if not against_paths:
        raise RuntimeError("REFUSED: --against <D .jsonl>... is required when building the slot list")
    if not pool_path.is_file():
        raise RuntimeError(f"REFUSED: --pool file does not exist: {pool_path}")

    _, ta_sha256, track_a_ids, track_a_rs_ids = load_track_a_slots(
        track_a_slots_path,
        expected_sha256=track_a_slots_sha256,
    )
    p4_rs_ids = phase4_real_source_slot_ids()

    pool_sha256 = sha256_lf_bytes(pool_path.read_bytes())
    excerpts = load_pool(pool_path)

    against_sha256: dict[str, str] = {}
    d_by_hash: dict[str, list[tuple[str, str]]] = {}
    for ap in against_paths:
        if not ap.is_file():
            raise RuntimeError(f"REFUSED: --against file does not exist: {ap}")
        against_sha256[str(ap)] = sha256_lf_bytes(ap.read_bytes())
        for did, norm in _extract_materials_from_file(ap, require_all_rows=False):
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            d_by_hash.setdefault(h, []).append((did, str(ap)))

    table = dom_mod.load(domains_path)
    packs = solvefirst.load_packs()
    exclusion_sha256 = exclusion.load(None).sha256

    seed0_by_id: dict[str, object] = {}
    seed1_by_id: dict[str, object] = {}
    for lesson in NON_T10_LESSONS:
        s0_list = plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=0)
        dom_mod.assign(s0_list, table, seed=0)
        for s in s0_list:
            if s.kind == "C":
                seed0_by_id[s.id] = s

        s1_list = plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=1)
        dom_mod.assign(s1_list, table, seed=1)
        for s in s1_list:
            if s.kind == "C":
                seed1_by_id[s.id] = s

    per_lesson_plan_sha256: dict[str, str] = {}
    all_c_records: list[dict] = []
    raw_slot_by_id: dict[str, object] = {}

    for lesson in NON_T10_LESSONS:
        prior = exclusion.load(None)
        slots, _, plan_sha = _plan_lesson_full(
            lesson,
            seed=plan_seed,
            table=table,
            packs=packs,
            prior=prior,
        )
        per_lesson_plan_sha256[lesson] = plan_sha
        for s in slots:
            if s.kind != "C" or s.id in PILOT_USED_C_SLOTS:
                continue
            raw_slot_by_id[s.id] = s
            flaw = slot_flaw(s)
            num = int(s.id.rsplit("-", 1)[1])
            parity = "even" if num % 2 == 0 else "odd"
            if parity == "even" and flaw != "none":
                raise RuntimeError(f"REFUSED: even slot {s.id} has slot_flaw={flaw!r}, expected 'none'")
            if parity == "odd" and flaw not in FLAWS:
                raise RuntimeError(f"REFUSED: odd slot {s.id} has slot_flaw={flaw!r}, expected one of {FLAWS}")

            bucket = _source_bucket(s.id, PHASE4_NON_T10_IDS, track_a_ids)
            s0 = seed0_by_id[s.id]
            s1 = seed1_by_id[s.id]
            s_dict = s.as_dict()
            rec = {
                **s_dict,
                "number": num,
                "parity": parity,
                "flaw": flaw,
                "source_bucket": bucket,
                "needs_novelty_check": bucket != "new",
                "domain_fresh": s.domain,
                "frame_fresh": s.frame,
                "domain_seed0": s0.domain,
                "frame_seed0": s0.frame,
                "domain_seed1": s1.domain,
                "frame_seed1": s1.frame,
                "domain_changed": (
                    (s.id not in PHASE4_NON_T10_IDS or s.domain != s0.domain)
                    and (s.id not in track_a_ids or s.domain != s1.domain)
                ),
            }
            all_c_records.append(rec)

    decided_odd = [r for r in all_c_records if r["parity"] == "odd" and r["verdict"] in DECIDED_VERDICTS]
    decided_even = [r for r in all_c_records if r["parity"] == "even" and r["verdict"] in DECIDED_VERDICTS]

    real_source_skipped: list[dict] = []
    usable_odd: list[dict] = []
    usable_even: list[dict] = []

    for r in sorted(decided_odd + decided_even, key=lambda x: _slot_order_key(x["id"])):
        sid = r["id"]
        used_rs_in: list[str] = []
        if sid in p4_rs_ids:
            used_rs_in.append("phase4")
        if sid in track_a_rs_ids:
            used_rs_in.append("track_a")
        if r["real_source"] and used_rs_in:
            real_source_skipped.append({
                "id": sid,
                "lesson": r["lesson"],
                "parity": r["parity"],
                "verdict": r["verdict"],
                "flaw": r["flaw"],
                "content_kind": r["content_kind"],
                "source_bucket": r["source_bucket"],
                "real_source": True,
                "used_as_real_source_in": used_rs_in,
                "reason": (
                    f"{sid} is real_source at seed {plan_seed} and was already a real_source slot in "
                    f"{' and '.join(used_rs_in)} (Runner._material_for hashes slot.id alone)"
                ),
            })
        else:
            if r["parity"] == "odd":
                usable_odd.append(r)
            else:
                usable_even.append(r)

    def _by_bucket(items: list[dict], bucket: str) -> list[dict]:
        return sorted((r for r in items if r["source_bucket"] == bucket), key=lambda x: _slot_order_key(x["id"]))

    odd_decided_by_bucket = {b: _by_bucket(decided_odd, b) for b in BUCKET_ORDER}
    even_decided_by_bucket = {b: _by_bucket(decided_even, b) for b in BUCKET_ORDER}

    while True:
        odd_usable_by_bucket = {b: _by_bucket(usable_odd, b) for b in BUCKET_ORDER}
        even_usable_by_bucket = {b: _by_bucket(usable_even, b) for b in BUCKET_ORDER}

        ordered_usable_odd = (
            odd_usable_by_bucket["new"]
            + odd_usable_by_bucket["phase4_only"]
            + odd_usable_by_bucket["track_a"]
        )
        ordered_usable_even = (
            even_usable_by_bucket["new"]
            + even_usable_by_bucket["phase4_only"]
            + even_usable_by_bucket["track_a"]
        )

        selected_odd = ordered_usable_odd[:target_per_half]
        selected_even = ordered_usable_even[:target_per_half]

        odd_skipped = [s for s in real_source_skipped if s["parity"] == "odd"]
        even_skipped = [s for s in real_source_skipped if s["parity"] == "even"]

        counts = {
            "total_non_t10_c_slots": len(all_c_records),
            "odd_decided_available": len(decided_odd),
            "odd_real_source_skipped": len(odd_skipped),
            "odd_usable": len(usable_odd),
            "even_decided_available": len(decided_even),
            "even_real_source_skipped": len(even_skipped),
            "even_usable": len(usable_even),
            "odd_by_bucket": {
                "new_available": len(odd_decided_by_bucket["new"]),
                "new_usable": len(odd_usable_by_bucket["new"]),
                "new_taken": sum(1 for r in selected_odd if r["source_bucket"] == "new"),
                "phase4_only_available": len(odd_decided_by_bucket["phase4_only"]),
                "phase4_only_usable": len(odd_usable_by_bucket["phase4_only"]),
                "phase4_only_taken": sum(1 for r in selected_odd if r["source_bucket"] == "phase4_only"),
                "track_a_available": len(odd_decided_by_bucket["track_a"]),
                "track_a_usable": len(odd_usable_by_bucket["track_a"]),
                "track_a_taken": sum(1 for r in selected_odd if r["source_bucket"] == "track_a"),
            },
            "even_by_bucket": {
                "new_available": len(even_decided_by_bucket["new"]),
                "new_usable": len(even_usable_by_bucket["new"]),
                "new_taken": sum(1 for r in selected_even if r["source_bucket"] == "new"),
                "phase4_only_available": len(even_decided_by_bucket["phase4_only"]),
                "phase4_only_usable": len(even_usable_by_bucket["phase4_only"]),
                "phase4_only_taken": sum(1 for r in selected_even if r["source_bucket"] == "phase4_only"),
                "track_a_available": len(even_decided_by_bucket["track_a"]),
                "track_a_usable": len(even_usable_by_bucket["track_a"]),
                "track_a_taken": sum(1 for r in selected_even if r["source_bucket"] == "track_a"),
            },
        }
        buckets = {
            "odd": {
                "new": [r["id"] for r in odd_usable_by_bucket["new"]],
                "phase4_only": [r["id"] for r in odd_usable_by_bucket["phase4_only"]],
                "track_a_available": [r["id"] for r in odd_decided_by_bucket["track_a"]],
                "track_a_usable": [r["id"] for r in odd_usable_by_bucket["track_a"]],
                "track_a_taken": [r["id"] for r in selected_odd if r["source_bucket"] == "track_a"],
                "real_source_skipped": [s["id"] for s in odd_skipped],
            },
            "even": {
                "new": [r["id"] for r in even_usable_by_bucket["new"]],
                "phase4_only": [r["id"] for r in even_usable_by_bucket["phase4_only"]],
                "track_a_available": [r["id"] for r in even_decided_by_bucket["track_a"]],
                "track_a_usable": [r["id"] for r in even_usable_by_bucket["track_a"]],
                "track_a_taken": [r["id"] for r in selected_even if r["source_bucket"] == "track_a"],
                "real_source_skipped": [s["id"] for s in even_skipped],
            },
        }
        census = {
            "counts": counts,
            "buckets": buckets,
            "real_source_skipped_count": len(real_source_skipped),
            "real_source_skipped": real_source_skipped,
        }

        if len(selected_odd) < target_per_half or len(selected_even) < target_per_half:
            raise RuntimeError(
                f"REFUSED: cannot reach {target_per_half} + {target_per_half} at seed {plan_seed} "
                f"(usable odd={len(usable_odd)}, usable even={len(usable_even)}); "
                f"census={json.dumps(census, ensure_ascii=False)}"
            )

        candidate_48 = sorted(
            selected_odd + selected_even,
            key=lambda x: _slot_order_key(x["id"]),
        )
        used_excerpts: dict[str, set[int]] = {}
        collided_ids: set[str] = set()
        for r in candidate_48:
            if not r["real_source"]:
                continue
            sid = r["id"]
            kind = r["content_kind"] or "notice"
            kind_pool = excerpts.get(kind) or []
            if not kind_pool:
                raise RuntimeError(f"REFUSED: real_source slot {sid} ({kind}) has no excerpt in the pool")
            used = used_excerpts.setdefault(kind, set())
            start = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16) % len(kind_pool)
            idx = next(
                ((start + k) % len(kind_pool) for k in range(len(kind_pool)) if (start + k) % len(kind_pool) not in used),
                None,
            )
            if idx is None:
                raise RuntimeError(f"REFUSED: real_source pool for {kind!r} exhausted ({len(kind_pool)} excerpts) at {sid}")
            used.add(idx)
            mat = kind_pool[idx]
            norm = _normalize_material(mat.text)
            mat_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if mat_sha in d_by_hash:
                d_id, d_file = d_by_hash[mat_sha][0]
                real_source_skipped.append({
                    "id": sid,
                    "lesson": r["lesson"],
                    "parity": r["parity"],
                    "verdict": r["verdict"],
                    "flaw": r["flaw"],
                    "content_kind": kind,
                    "source_bucket": r["source_bucket"],
                    "real_source": True,
                    "reason": "excerpt_collision",
                    "d_slot_id": d_id,
                    "d_id": d_id,
                    "d_file": d_file,
                    "pool_index": idx,
                    "material_sha256": mat_sha,
                })
                collided_ids.add(sid)

        if not collided_ids:
            break
        usable_odd = [r for r in usable_odd if r["id"] not in collided_ids]
        usable_even = [r for r in usable_even if r["id"] not in collided_ids]

    unchanged_domain_ids = [
        r["id"]
        for r in sorted(selected_odd + selected_even, key=lambda x: _slot_order_key(x["id"]))
        if r["source_bucket"] != "new" and not r["domain_changed"]
    ]
    if unchanged_domain_ids:
        raise RuntimeError(
            f"REFUSED: selected reused slot(s) have unchanged domain at seed {plan_seed}: "
            f"{', '.join(unchanged_domain_ids)}"
        )

    # Re-verify every selected slot against role-set's raw Slot and slot_flaw
    for r in selected_odd:
        sid = r["id"]
        if r["lesson"] == "T10" or sid.startswith("T10-"):
            raise RuntimeError(f"REFUSED: T10 slot {sid} in selected_odd")
        if sid in PILOT_USED_C_SLOTS:
            raise RuntimeError(f"REFUSED: pilot-used slot {sid} in selected_odd")
        raw_slot = raw_slot_by_id[sid]
        flaw = slot_flaw(raw_slot)
        if flaw == "none" or flaw not in FLAWS:
            raise RuntimeError(f"REFUSED: selected odd slot {sid} has slot_flaw={flaw!r}")
        if raw_slot.verdict not in DECIDED_VERDICTS:
            raise RuntimeError(f"REFUSED: selected odd slot {sid} has undecided verdict {raw_slot.verdict!r}")
        if raw_slot.real_source and (sid in p4_rs_ids or sid in track_a_rs_ids):
            raise RuntimeError(f"REFUSED: selected odd real_source slot {sid} was already real_source in D")

    for r in selected_even:
        sid = r["id"]
        if r["lesson"] == "T10" or sid.startswith("T10-"):
            raise RuntimeError(f"REFUSED: T10 slot {sid} in selected_even")
        if sid in PILOT_USED_C_SLOTS:
            raise RuntimeError(f"REFUSED: pilot-used slot {sid} in selected_even")
        raw_slot = raw_slot_by_id[sid]
        flaw = slot_flaw(raw_slot)
        if flaw != "none":
            raise RuntimeError(f"REFUSED: selected even slot {sid} has slot_flaw={flaw!r}, expected 'none'")
        if raw_slot.verdict not in DECIDED_VERDICTS:
            raise RuntimeError(f"REFUSED: selected even slot {sid} has undecided verdict {raw_slot.verdict!r}")
        if raw_slot.real_source and (sid in p4_rs_ids or sid in track_a_rs_ids):
            raise RuntimeError(f"REFUSED: selected even real_source slot {sid} was already real_source in D")

    selected_odd_sorted = sorted(selected_odd, key=lambda x: _slot_order_key(x["id"]))
    selected_even_sorted = sorted(selected_even, key=lambda x: _slot_order_key(x["id"]))
    all_selected_sorted = sorted(
        selected_odd_sorted + selected_even_sorted,
        key=lambda x: _slot_order_key(x["id"]),
    )
    lessons_used = [l for l in NON_T10_LESSONS if any(r["lesson"] == l for r in all_selected_sorted)]
    needs_novelty_check_count = sum(1 for r in all_selected_sorted if r["needs_novelty_check"])

    payload = {
        "what": "Track B bank V 48-slot list (TRACK-B-DESIGN.md §5, rung 0)",
        "source": (
            "TRACK-B-DESIGN.md §5 (Track B Rung 0); computed with role-set's plan_batch, "
            "domains.assign, solvefirst.assign, and assemble.slot_flaw at seed 2 "
            "(T10, pilot C slots, and D's real_source slot IDs and excerpt collisions excluded)"
        ),
        "role_set_commit": role_set_commit or _git_head_commit(),
        "plan_seed": plan_seed,
        "fresh_domain_seed": plan_seed,
        "how": (
            f"role-set plan --lessons <L> --seed {plan_seed} --domains roles/domains/naics-2022.json per lesson "
            "(the pin_window plan_pilot_lessons path), filtered to the census slot_ids from "
            "scripts/critic_signal/track_b_slots.py"
        ),
        "track_a_slots_sha256": ta_sha256,
        "pool_sha256": pool_sha256,
        "against_sha256": against_sha256,
        "per_lesson_plan_sha256": {l: per_lesson_plan_sha256[l] for l in lessons_used},
        "exclusion_state_sha256": exclusion_sha256,
        "lessons": lessons_used,
        "excluded_lessons": ["T10"],
        "excluded_pilot_slots": list(PILOT_USED_C_SLOTS),
        "slots_per_lesson": 8,
        "f_eligible": [r["id"] for r in selected_odd_sorted],
        "controls": [r["id"] for r in selected_even_sorted],
        "slot_ids": [r["id"] for r in all_selected_sorted],
        "draft_slot_count": len(all_selected_sorted),
        "needs_novelty_check_count": needs_novelty_check_count,
        "real_source_count": sum(1 for r in all_selected_sorted if r["real_source"]),
        "two_turn_count": sum(1 for r in all_selected_sorted if r.get("two_turn") is not None),
        "real_source_skipped": real_source_skipped,
        "census": census,
        "counts": counts,
        "buckets": buckets,
        "slots": all_selected_sorted,
        "slot_details": {r["id"]: r for r in all_selected_sorted},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


def _normalize_material(text: str) -> str:
    return " ".join(text.split())


def _extract_materials_from_file(path: Path, *, require_all_rows: bool = False) -> list[tuple[str, str]]:
    """Read `(row_id, normalized_material)` pairs from a drafts or items `.jsonl` file.

    Refuses on any non-header row with an empty material, naming the row id and line.
    When `require_all_rows` is False (for D `--against` files), allows an empty material only on a row whose
    kind is A or B, or E with `e_type == "written"`. When `require_all_rows` is True (for the `--check-material`
    V file), refuses on every row with an empty material.
    """
    if not path.is_file():
        raise RuntimeError(f"REFUSED: material check file does not exist: {path}")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[tuple[str, str]] = []
    for idx, line in enumerate(lines, start=1):
        try:
            d = json.loads(line)
        except ValueError as e:
            raise RuntimeError(f"REFUSED: cannot parse JSON at {path}:{idx}: {e}") from e
        if not isinstance(d, dict):
            raise RuntimeError(f"REFUSED: non-object JSON row at {path}:{idx}")
        if d.get("type") == "header":
            continue
        slot = d.get("slot") if isinstance(d.get("slot"), dict) else {}
        row_id = str(
            d.get("item_id")
            or d.get("id")
            or slot.get("id")
            or f"{path.name}:{idx}"
        )
        mat_field = d.get("material")
        if isinstance(mat_field, dict) and isinstance(mat_field.get("text"), str):
            mat_text = mat_field["text"]
        elif isinstance(mat_field, str):
            mat_text = mat_field
        else:
            mat_text = ""
        norm = _normalize_material(mat_text)
        if not norm:
            if not require_all_rows:
                kind = d.get("kind") or slot.get("kind")
                quiz = d.get("quiz") if isinstance(d.get("quiz"), dict) else {}
                e_type = d.get("e_type") or slot.get("e_type") or quiz.get("type")
                if kind in ("A", "B") or (kind == "E" and e_type == "written"):
                    continue
            raise RuntimeError(f"REFUSED: empty material on row {row_id} at {path}:{idx}")
        out.append((row_id, norm))
    if not out:
        raise RuntimeError(f"REFUSED: no rows with material found in {path}")
    return out


def check_material_overlap(v_path: Path, against_paths: list[Path]) -> dict:
    """Compare whitespace-normalised material hashes in `v_path` against `against_paths`.
    Returns a report dict with `ok: bool`, counts, and `shared_pairs`."""
    if not against_paths:
        raise RuntimeError("REFUSED: --check-material requires at least one --against file")

    v_entries = _extract_materials_from_file(v_path, require_all_rows=True)
    v_hashed = [
        (vid, hashlib.sha256(norm.encode("utf-8")).hexdigest())
        for vid, norm in v_entries
    ]

    d_by_hash: dict[str, list[tuple[str, str]]] = {}
    d_total_rows = 0
    for ap in against_paths:
        d_entries = _extract_materials_from_file(ap, require_all_rows=False)
        d_total_rows += len(d_entries)
        for did, norm in d_entries:
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            d_by_hash.setdefault(h, []).append((did, str(ap)))

    shared_pairs: list[dict] = []
    for vid, vh in v_hashed:
        if vh in d_by_hash:
            for did, dfile in d_by_hash[vh]:
                shared_pairs.append({
                    "v_id": vid,
                    "d_id": did,
                    "d_file": dfile,
                    "material_sha256": vh,
                })

    return {
        "ok": len(shared_pairs) == 0,
        "v_file": str(v_path),
        "v_rows": len(v_hashed),
        "v_unique_materials": len({vh for _, vh in v_hashed}),
        "d_files": [str(p) for p in against_paths],
        "d_rows": d_total_rows,
        "d_unique_materials": len(d_by_hash),
        "shared_count": len(shared_pairs),
        "shared_pairs": shared_pairs,
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2, help="plan_batch seed (default: 2)")
    p.add_argument("--domains", type=Path, default=DEFAULT_DOMAINS, help="NAICS domain table path")
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="real-source excerpt pool path")
    p.add_argument("--track-a-slots", type=Path, default=None, help="frozen Track A slots JSON (track-a-slots-10c1ebf.json)")
    p.add_argument("--track-a-slots-sha256", type=str, default=None, help="expected sha256 of --track-a-slots")
    p.add_argument("--target-per-half", type=int, default=24, help="target decided slots per half (default: 24)")
    p.add_argument("--role-set-commit", type=str, default=None, help="override role_set_commit field")
    p.add_argument("--out", type=Path, default=None, help="write frozen JSON to this file")
    p.add_argument("--check-material", type=Path, default=None, help="V drafts.jsonl to check for shared material")
    p.add_argument(
        "--against",
        type=Path,
        nargs="+",
        action="extend",
        default=[],
        help="D drafts or items .jsonl file(s) to check V materials against",
    )
    args = p.parse_args(argv)

    if args.check_material is not None:
        try:
            report = check_material_overlap(args.check_material, list(args.against))
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 2
        print(json.dumps(report, indent=1, ensure_ascii=False))
        if not report["ok"]:
            pairs_str = ", ".join(f"{x['v_id']}=={x['d_id']}" for x in report["shared_pairs"])
            print(f"REFUSED: V shares material with D ({len(report['shared_pairs'])} pair(s)): {pairs_str}", file=sys.stderr)
            return 1
        return 0

    if args.track_a_slots is None:
        print("REFUSED: --track-a-slots <json> is required", file=sys.stderr)
        return 2
    if args.track_a_slots_sha256 is None:
        print("REFUSED: --track-a-slots-sha256 <sha256> is required", file=sys.stderr)
        return 2
    if not args.against:
        print("REFUSED: --against <D .jsonl>... is required", file=sys.stderr)
        return 2

    try:
        payload = build_track_b_slots(
            track_a_slots_path=args.track_a_slots,
            track_a_slots_sha256=args.track_a_slots_sha256,
            against_paths=list(args.against),
            pool_path=args.pool,
            plan_seed=args.seed,
            domains_path=args.domains,
            target_per_half=args.target_per_half,
            out_path=args.out,
            role_set_commit=args.role_set_commit,
        )
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
