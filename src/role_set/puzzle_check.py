"""`role-set puzzle-check`: the kind B solver over a reviewed data file and its models sidecar.

A reviewed batch keeps the trainer's schema untouched: each data file `<name>.jsonl` gets a sidecar
`<name>.models.jsonl` with one line per B item, `{"id", "structure", "model", "row_sha256"}`, where row_sha256 is the
sha256 of the item's line as committed (UTF-8, no newline), so a model can't drift from its puzzle. The trainer never
reads it. This
command reads both, runs `check_puzzle` on every modelled item (the puzzle is the item's first user turn, the key its
`answer`, the reply its last reply), and reports each refusal with its counterexample. An item whose structure is
`verdict` is a verdict-keyed B item over free-text material: it stays checked as verdict items are (verbatim
DECIDING, the key-check, the critic), and is counted here as outside the solver's scope, never as passed.

Exit 0 only when every B item has a sidecar line whose row_sha256 matches, every modelled item passes, no lesson has
more than one lookup, and the sidecar names no item the data file doesn't hold as kind B (R&D's conditions).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from role_set.assemble import VERDICTS
from role_set.plan import PUZZLE_STRUCTURES
from role_set.puzzle import PuzzleError, check_puzzle

VERDICT_STRUCTURE = "verdict"
STRUCTURES = tuple(name for name, _ in PUZZLE_STRUCTURES) + (VERDICT_STRUCTURE,)
LOOKUPS_PER_LESSON = 1


def models_path(data: Path) -> Path:
    return data.with_name(data.name[: -len(".jsonl")] + ".models.jsonl") if data.name.endswith(".jsonl") else data.with_suffix(".models.jsonl")


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def row_sha256(line: str) -> str:
    return hashlib.sha256(line.rstrip("\r\n").encode("utf-8")).hexdigest()


def check_file(data: Path, models: Path | None = None) -> dict:
    models = models or models_path(data)
    raw = [x for x in data.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = {r["id"]: r for r in map(json.loads, raw) if r.get("kind") == "B"}
    shas = {json.loads(x)["id"]: row_sha256(x) for x in raw}
    side = {m["id"]: m for m in _lines(models)} if models.exists() else {}
    items: list[dict] = []
    lookups: dict[str, list[str]] = {}
    for rid, row in rows.items():
        entry = side.get(rid)
        if entry is None:
            items.append({"id": rid, "result": "missing", "detail": "no line in the models sidecar"})
            continue
        structure = entry.get("structure")
        if entry.get("row_sha256") != shas[rid]:
            items.append({"id": rid, "result": "refused", "structure": structure,
                          "detail": "row_sha256 doesn't match the row as committed; the row changed after its model was written"})
            continue
        if structure not in STRUCTURES:
            # A typo would otherwise run as a plain one-answer puzzle and skip its structure's own rules (R&D on #32).
            items.append({"id": rid, "result": "refused", "structure": structure,
                          "detail": f"structure {structure!r} is not one of {', '.join(STRUCTURES)}"})
            continue
        if structure == VERDICT_STRUCTURE:
            # Only a verdict-label key may skip the solver: a value-keyed puzzle that ends on a VERDICT line still
            # has an answer the solver can check, so `verdict` is never an escape from it.
            keyed = (row.get("answer") or "").strip().lower() in VERDICTS
            verdict_reply = "VERDICT:" in (row["turns"][-1].get("reply") or "")
            detail = ("verdict-keyed, outside the solver's scope" if keyed and verdict_reply
                      else f"keyed by a value ({row.get('answer')!r}), so it needs a model" if not keyed
                      else "marked verdict, but its reply carries no VERDICT line")
            items.append({"id": rid, "result": "verdict" if keyed and verdict_reply else "refused", "structure": structure, "detail": detail})
            continue
        fields = {"puzzle": row["turns"][0].get("user") or "", "model": entry.get("model"), "answer": row.get("answer") or "",
                  "reply": row["turns"][-1].get("reply") or "", "thinking": row["turns"][-1].get("thinking") or ""}
        try:
            got = check_puzzle(fields, structure)
        except PuzzleError as e:
            items.append({"id": rid, "result": "refused", "structure": structure, "detail": str(e)})
            continue
        items.append({"id": rid, "result": "pass", "structure": structure, "detail": f"key {got.answer!r}" + (", a lookup" if got.lookup else "")})
        if got.lookup:
            lookups.setdefault(row.get("lesson") or rid[:3], []).append(rid)
    for lesson, ids in lookups.items():
        for rid in ids[LOOKUPS_PER_LESSON:]:
            item = next(i for i in items if i["id"] == rid)
            item.update(result="refused", detail=f"a second lookup in {lesson} (the first is {ids[0]}); make the answer need elimination")
    stray = sorted(set(side) - set(rows))
    counts = {k: sum(1 for i in items if i["result"] == k) for k in ("pass", "verdict", "refused", "missing")}
    ok = counts["refused"] == 0 and counts["missing"] == 0 and not stray
    by_structure = dict(sorted(Counter(f"{i.get('structure') or '-'}: {i['result']}" for i in items).items()))
    return {"data": data.name, "models": models.name, "b_items": len(rows), **counts, "by_structure": by_structure, "stray": stray,
            "ok": ok, "items": items}


def render(report: dict) -> str:
    out = [f"{report['data']} + {report['models']}: {report['b_items']} B items; pass {report['pass']}, "
           f"verdict, outside solver scope: {report['verdict']}, refused {report['refused']}, missing {report['missing']}"
           + (f"; sidecar ids not B items here: {', '.join(report['stray'])}" if report["stray"] else "")]
    out.append("  by structure: " + ", ".join(f"{k} {v}" for k, v in report["by_structure"].items()))
    for i in report["items"]:
        if i["result"] != "pass":
            out.append(f"  {i['id']}: {i['result'].upper()}: {i['detail']}")
    return "\n".join(out)


def stamp(data: Path, models: Path | None = None) -> int:
    """Write each sidecar line's row_sha256 from the data file as it stands. Run once, when a row and its model are
    final; a reviewer re-runs puzzle-check without it, so any later edit to the row shows up as a stale hash."""

    models = models or models_path(data)
    shas = {json.loads(x)["id"]: row_sha256(x) for x in data.read_text(encoding="utf-8").splitlines() if x.strip()}
    side = _lines(models)
    n = 0
    for m in side:
        if m["id"] in shas:
            m["row_sha256"] = shas[m["id"]]
            n += 1
    models.write_bytes("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in side).encode("utf-8"))
    return n
