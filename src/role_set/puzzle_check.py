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
more than one lookup, the sidecar names no item the data file doesn't hold as kind B (R&D's conditions), and no two
passing puzzles across the files checked together share a fact skeleton (`refuse_repeats`).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from role_set.assemble import VERDICTS
from role_set.plan import PUZZLE_STRUCTURES
from role_set.puzzle import PuzzleError, check_puzzle

VERDICT_STRUCTURE = "verdict"
STRUCTURES = tuple(name for name, _ in PUZZLE_STRUCTURES) + (VERDICT_STRUCTURE,)
LOOKUPS_PER_LESSON = 1
# Readings tried when canonicalising a skeleton (fact orders times argument orders of symmetric clues); past this,
# a puzzle is compared as written.
SKELETON_READINGS_MAX = 50_000
SYMMETRIC = ("adjacent", "not_adjacent", "same", "different")


def _canon(rule) -> list:
    """`after a b` is `before b a`, and `offset a b -k` is `offset b a k`; every other form as written."""

    if rule[0] == "after":
        return ["before", rule[2], rule[1], *rule[3:]]
    if rule[0] == "offset" and rule[4] < 0:
        return ["offset", rule[2], rule[1], rule[3], -rule[4]]
    if rule[0] in ("not", "and", "or", "if"):
        return [rule[0], *(_canon(r) for r in rule[1:])]
    return list(rule)


def _variants(rule) -> list:
    """Each way of writing a clue that says the same thing: a symmetric relation in either argument order (R&D on
    role-set#37: `adjacent Z X` and `adjacent X Z` are one clue), nested ones included."""

    if rule[0] in SYMMETRIC:
        return [rule, [rule[0], rule[2], rule[1], *rule[3:]]]
    if rule[0] in ("not", "and", "or", "if"):
        return [[rule[0], *parts] for parts in itertools.product(*(_variants(r) for r in rule[1:]))]
    return [rule]


def _rename(model: dict, facts: list) -> str:
    """The facts with entities, attributes and values renamed in order of first appearance. An attribute keeps its
    shape (value count, ordered, distinct), so a two-value and a three-value elimination are different skeletons."""

    attrs = model["attributes"]
    ents, anames, vnames = {}, {}, {}

    def attr(a):
        if a not in anames:
            spec = attrs[a]
            anames[a] = f"A{len(anames) + 1}({len(spec['values'])},{bool(spec.get('ordered'))},{spec.get('distinct', True)})"
        return anames[a]

    def value(a, v):
        attr(a)
        return vnames.setdefault((a, str(v)), f"V{len(vnames) + 1}")

    def ref(x):
        if isinstance(x, dict):
            (a, v), = x.items()
            return {attr(a): value(a, v)}
        return ents.setdefault(x, f"E{len(ents) + 1}")

    def rule(r):
        f = r[0]
        if f in ("is", "is_not"):
            return [f, ref(r[1]), attr(r[2]), value(r[2], r[3])]
        if f in ("in", "not_in"):
            return [f, ref(r[1]), attr(r[2]), sorted(value(r[2], v) for v in r[3])]
        if f in ("same", "different"):
            return [f, ref(r[1]), ref(r[2])]
        if f in ("before", "adjacent", "not_adjacent"):
            return [f, ref(r[1]), ref(r[2]), attr(r[3])]
        if f == "offset":
            return [f, ref(r[1]), ref(r[2]), attr(r[3]), r[4]]
        if f == "count":
            return [f, attr(r[1]), value(r[1], r[2]), r[3], r[4]]
        return [f, *(rule(x) for x in r[1:])]

    return json.dumps([rule(r) for r in facts], sort_keys=True)


def skeleton(model: dict) -> str:
    """A puzzle's fact skeleton: its facts renamed (see `_rename`), with flavour, the question and any claim dropped,
    taken in the reading (fact order, and argument order of each symmetric clue) whose serialisation sorts first, so line
    order, naming and a swapped symmetric clue don't hide a copy (R&D found six copies by hand in aspire-si#95)."""

    facts = [_canon(f["rule"]) for f in model["facts"]]
    variants = [_variants(r) for r in facts]
    readings = math.factorial(len(facts)) * math.prod(len(v) for v in variants)
    if readings > SKELETON_READINGS_MAX:
        return _rename(model, facts)
    return min(_rename(model, list(choice)) for order in itertools.permutations(variants) for choice in itertools.product(*order))


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
        items.append({"id": rid, "result": "pass", "structure": structure, "detail": f"key {got.answer!r}" + (", a lookup" if got.lookup else ""),
                      "skeleton": skeleton(entry["model"])})
        if got.lookup:
            lookups.setdefault(row.get("lesson") or rid[:3], []).append(rid)
    for lesson, ids in lookups.items():
        for rid in ids[LOOKUPS_PER_LESSON:]:
            item = next(i for i in items if i["id"] == rid)
            item.update(result="refused", detail=f"a second lookup in {lesson} (the first is {ids[0]}); make the answer need elimination")
    report = {"data": data.name, "models": models.name, "b_items": len(rows), "stray": sorted(set(side) - set(rows)), "items": items}
    return _tally(report)


def _tally(report: dict) -> dict:
    items = report["items"]
    report.update({k: sum(1 for i in items if i["result"] == k) for k in ("pass", "verdict", "refused", "missing")})
    report["ok"] = report["refused"] == 0 and report["missing"] == 0 and not report["stray"]
    report["by_structure"] = dict(sorted(Counter(f"{i.get('structure') or '-'}: {i['result']}" for i in items).items()))
    return report


def refuse_repeats(reports: list[dict]) -> list[dict]:
    """Across every file checked together, a passing puzzle whose fact skeleton an earlier one already has is
    refused, naming the first. Earlier means earlier in the argument order, then in the file."""

    first: dict[str, tuple[str, str]] = {}
    for report in reports:
        hit = False
        for item in report["items"]:
            sk = item.get("skeleton")
            if item["result"] != "pass" or not sk:
                continue
            if sk in first:
                rid, where = first[sk]
                item.update(result="refused", detail=f"repeats the fact skeleton of {rid} ({where}) once entities and values "
                                                     "are renamed; build it on other fact forms")
                hit = True
            else:
                first[sk] = (item["id"], report["data"])
        if hit:
            _tally(report)
    return reports


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
