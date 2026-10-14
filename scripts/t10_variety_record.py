"""Measure T10 builds on seeds 1-20 and locked seeds 61-80 across every pack and question.

Writes src/role_set/generate/t10_record.json with the sha256 over the generator source, t10_cells.json,
t10_capacity.json and pack data, plus per-build records (built, cell, tries, canonical_skeleton).

    uv run python scripts/t10_variety_record.py
    uv run python scripts/t10_variety_record.py --check
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from role_set.generate.build import MAX_TRIES, check_model
from role_set.generate.recipes import PACKS, T10, build_t10
from role_set.generate.render import render_puzzle
from role_set.generate.shortcuts import REGISTRY, prove
from role_set.generate.t10_variety import (
    RECORD_HASH_FILES,
    RECORD_SEEDS,
    REGENERATE_RECORD_CMD,
    canonical_skeleton,
    cell_for,
    check_record_hash,
    load_record,
    record_sha256,
)
from role_set.puzzle import check_puzzle

OUT = Path(__file__).resolve().parents[1] / "src" / "role_set" / "generate" / "t10_record.json"
LOCKED = set(range(61, 81))


def _kind(pack: str) -> str:
    return "two" if len(PACKS[pack]["values"]) == 2 else "three"


def _prove_built(pack: str, question: str, seed: int, cell: dict, built) -> bool:
    structure = "falsifier_first" if question == "claim" else "underdetermined_with_explicit_set"
    if check_model(built.model, structure) != built.answer:
        return False
    text = render_puzzle(built.model, PACKS[pack]["plural"], random.Random(seed))
    reply = ("x\n" + built.answer) if question == "claim" else ("x\nAnswer: " + built.answer)
    if check_puzzle({"puzzle": text, "model": built.model, "answer": built.answer, "reply": reply}, structure).answer != built.answer:
        return False
    proofs = prove(built.model, structure, list(T10.shortcuts))
    if not any(p["differs"] for p in proofs.values()):
        return False
    op = built.trait_clue[3]
    own = [name for name in T10.shortcuts if REGISTRY[name].get("from") == op]
    if not any(proofs.get(name, {}).get("differs") for name in own):
        return False
    counts = [fact["rule"] for fact in built.model["facts"] if fact["rule"][0] == "count"]
    if counts != [built.trait_clue]:
        return False
    if question == "claim" and op != "=":
        return False
    ifs = sum(1 for fact in built.model["facts"] if fact["rule"][0] == "if")
    return (
        ifs == cell["ifs"]
        and built.trait_clue[3:] == [cell["op"], cell["k"]]
        and len(built.model["entities"]) == cell["n"]
    )


def _measure_one(job: tuple[str, str, int]) -> dict:
    pack, question, seed = job
    cell = cell_for(pack, question, seed)
    try:
        built = build_t10(pack, seed, question)
        ok = _prove_built(pack, question, seed, cell, built)
    except Exception:
        return {
            "pack": pack,
            "question": question,
            "seed": seed,
            "built": False,
            "cell": cell,
            "tries": 0,
            "canonical_skeleton": "",
        }
    return {
        "pack": pack,
        "question": question,
        "seed": seed,
        "built": ok,
        "cell": cell,
        "tries": built.tries,
        "canonical_skeleton": canonical_skeleton(built.model) if ok else "",
    }


def _print_locked(builds: list[dict]) -> bool:
    ok = True
    for question in ("set", "claim"):
        for kind in ("two", "three"):
            rows = [
                r for r in builds
                if r["question"] == question and _kind(r["pack"]) == kind and r["seed"] in LOCKED
            ]
            skins = [r["canonical_skeleton"] for r in rows]
            distinct = len(set(skins))
            top = Counter(skins).most_common(1)[0][1]
            by_pack: dict[str, list[str]] = {}
            for r in rows:
                by_pack.setdefault(r["pack"], []).append(r["canonical_skeleton"])
            within = {p: sum(n - 1 for n in Counter(s).values()) for p, s in by_pack.items()}
            print(f"{question} {kind}: {distinct}/{len(skins)} top {top} within {within}", flush=True)
            if distinct / len(skins) < 0.75 or top > 3 or any(within.values()):
                ok = False
    return ok


def check_committed() -> None:
    record = check_record_hash(load_record())
    builds = record["builds"]
    expected = [(p, q, s) for p in PACKS for q in ("set", "claim") for s in RECORD_SEEDS]
    got = [(r["pack"], r["question"], r["seed"]) for r in builds]
    if got != expected:
        raise SystemExit("record build keys do not match expected (pack, question, seed) list")
    for r in builds:
        if not r["built"] or not (1 <= r["tries"] <= MAX_TRIES) or not r["canonical_skeleton"]:
            raise SystemExit(f"unproven or failed build in record: {r}")
    if not _print_locked(builds):
        raise SystemExit(2)


def main() -> None:
    jobs = [(pack, question, seed) for pack in PACKS for question in ("set", "claim") for seed in RECORD_SEEDS]
    workers = min(16, os.cpu_count() or 4)
    print(f"0/{len(jobs)} workers {workers}", flush=True)
    builds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, row in enumerate(pool.map(_measure_one, jobs), 1):
            builds.append(row)
            if done % 80 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)
    payload = {
        "regenerate": REGENERATE_RECORD_CMD,
        "files": list(RECORD_HASH_FILES),
        "sha256": record_sha256(),
        "seeds": list(RECORD_SEEDS),
        "packs": list(PACKS),
        "questions": ["set", "claim"],
        "builds": builds,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not all(r["built"] for r in builds) or not _print_locked(builds):
        raise SystemExit(2)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_committed()
    else:
        main()
