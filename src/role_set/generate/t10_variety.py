"""T10's structural cells, shared across packs of one value-count.

A set cell is (how many entities, which count word, k, how many if clues). A claim cell adds if_shape: none for 0 ifs,
consequent polarity (pos or neg) for 1 if, and entity graph shape (chain, shared_consequent, shared_antecedent, or
disjoint) for 2 ifs. The usable cells are the ones that build, recorded in t10_cells.json by scripts/t10_usable_cells.py.
Set cells keep all three count words. Claim cells are exactly only.

Packs with the same number of values draw on one skeleton space, so one assignment covers the whole stratum for a file
of 20 seeds. A pack's 20 seeds take 20 different cells. A cell with no if clue is used only when a pack has fewer than
20 if-cells, and at most twice in a file in every stratum. The other visits spread over if-cells, more of them on cells
that produced more distinct skeletons on the tuning seeds (t10_capacity.json). The next file rotates that assignment.

The capacity check reads the table. It does not run inside a build, and a finished puzzle is never discarded because its
skeleton has been seen. The builder still retries inside one seed when the reduced puzzle loses its if count or if shape.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import deque
from pathlib import Path

_DATA = Path(__file__).with_name("t10_cells.json")
_CAPACITY = Path(__file__).with_name("t10_capacity.json")
_RECORD = Path(__file__).with_name("t10_record.json")
_CELLS: dict | None = None
_CAP: dict | None = None
_REC: dict | None = None
_PLANS: dict = {}
# One file is 20 seeds. The next file of the same packs is the next 20.
FILE = 20
RECORD_SEEDS = (
    tuple(range(1, 21))
    + tuple(range(61, 81))
    + tuple(range(81, 101))
    + tuple(range(101, 121))
)
REGENERATE_RECORD_CMD = "uv run python scripts/t10_variety_record.py"
RECORD_HASH_FILES = (
    "src/role_set/assemble.py",
    "src/role_set/plan.py",
    "src/role_set/puzzle.py",
    "src/role_set/puzzle_check.py",
    "src/role_set/span.py",
    "src/role_set/generate/__init__.py",
    "src/role_set/generate/build.py",
    "src/role_set/generate/recipes.py",
    "src/role_set/generate/render.py",
    "src/role_set/generate/shortcuts.py",
    "src/role_set/generate/t10_variety.py",
    "src/role_set/generate/t10_cells.json",
    "src/role_set/generate/t10_capacity.json",
)
IF_SHAPES: dict[int, tuple[str, ...]] = {
    0: ("none",),
    1: ("pos", "neg"),
    2: ("chain", "shared_consequent", "shared_antecedent", "disjoint"),
}
_INF = 10**18


def clues_if_shape(clues: list[list]) -> str:
    """The if-shape of a clue list (raw rules or canonical skeleton clues).

    For 1 if clue, 'touches' and 'not' (whether the if mentions the claimed entity) are merged into consequent polarity
    ('pos' vs 'neg') because canonical_skeleton records facts only, not the claim, so touches vs not never differs in
    skeleton and cannot be decided from the skeleton alone."""

    ifs = [c for c in clues if c[0] == "if"]
    if not ifs:
        return "none"
    if len(ifs) == 1:
        return "pos" if ifs[0][2][0] == "is" else "neg"
    if len(ifs) == 2:
        p1, q1 = ifs[0][1][1], ifs[0][2][1]
        p2, q2 = ifs[1][1][1], ifs[1][2][1]
        ents = {p1, q1, p2, q2}
        if len(ents) == 4:
            return "disjoint"
        if len(ents) == 3:
            if q1 == p2 or q2 == p1:
                return "chain"
            if q1 == q2:
                return "shared_consequent"
            if p1 == p2:
                return "shared_antecedent"
    return "other"


def load_cells() -> dict:
    """The committed list: pack, question, then the cells that built within MAX_TRIES."""

    global _CELLS
    if _CELLS is None:
        _CELLS = json.loads(_DATA.read_text(encoding="utf-8"))
    return _CELLS


def record_sha256(root: Path | None = None, files: list[str] | tuple[str, ...] = RECORD_HASH_FILES) -> str:
    """SHA-256 over the generator source, cell tables and pack data that decide T10 builds."""

    base = Path(__file__).resolve().parents[3] if root is None else root
    digest = hashlib.sha256()
    for rel in files:
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update((base / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def load_record() -> dict:
    """The committed build record for seeds 1-20 and locked seeds 61-80 and 81-100 across every pack and question."""

    global _REC
    if _REC is None:
        _REC = json.loads(_RECORD.read_text(encoding="utf-8"))
    return _REC


def check_record_hash(record: dict | None = None, root: Path | None = None) -> dict:
    """Verify the committed record lists the hashed files and matches their current sha256."""

    data = load_record() if record is None else record
    if data.get("regenerate") != REGENERATE_RECORD_CMD or data.get("files") != list(RECORD_HASH_FILES):
        raise ValueError(
            f"T10 variety record metadata mismatch; regenerate with: {REGENERATE_RECORD_CMD}"
        )
    actual = record_sha256(root=root, files=data["files"])
    if data.get("sha256") != actual:
        raise ValueError(
            f"T10 variety record sha256 mismatch ({data.get('sha256')} != {actual}); "
            f"regenerate with: {REGENERATE_RECORD_CMD}"
        )
    return data


def record_index(record: dict | None = None) -> dict[tuple[str, str, int], dict]:
    """Map (pack, question, seed) to its build entry in the committed record."""

    data = load_record() if record is None else record
    return {(row["pack"], row["question"], row["seed"]): row for row in data["builds"]}


def load_capacity() -> dict:
    """Distinct skeletons per cell on the tuning seeds. Missing means the weights are even."""

    global _CAP
    if _CAP is None:
        if _CAPACITY.exists():
            _CAP = json.loads(_CAPACITY.read_text(encoding="utf-8"))
        else:
            _CAP = {"cells": {}}
    return _CAP


def cell_id(cell: dict) -> tuple:
    if "if_shape" in cell:
        return (cell["n"], cell["op"], cell["k"], cell["ifs"], cell["if_shape"])
    return (cell["n"], cell["op"], cell["k"], cell["ifs"])


def cell_key(cell: dict) -> str:
    return "|".join(str(part) for part in cell_id(cell))


def zero_if_cap(kind: str, question: str) -> int:
    """Visits one zero-if cell may take in one file."""

    _ = (kind, question)
    return 2


def _value_counts() -> dict[str, int]:
    from role_set.generate.recipes import PACKS

    return {name: len(pack["values"]) for name, pack in PACKS.items()}


def _unique(cells: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for cell in cells:
        cid = cell_id(cell)
        if cid not in seen:
            seen.add(cid)
            out.append(cell)
    return out


def _by_size_and_word(cells: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["op"], cell["n"]), []).append(cell)
    keys = sorted(groups)
    out: list[dict] = []
    while any(groups[key] for key in keys):
        for key in keys:
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def _spread(cells: list[dict]) -> list[dict]:
    """Deal if-counts in turn, and inside each if-count deal the count word and the size in turn."""

    buckets = {ifs: [] for ifs in (2, 1, 0)}
    for cell in cells:
        buckets[cell["ifs"]].append(cell)
    dealt = {ifs: _by_size_and_word(bucket) for ifs, bucket in buckets.items()}
    cursors = {ifs: 0 for ifs in dealt}
    out: list[dict] = []
    moving = True
    while moving:
        moving = False
        for ifs in (2, 1, 0):
            cursor = cursors[ifs]
            if cursor < len(dealt[ifs]):
                out.append(dealt[ifs][cursor])
                cursors[ifs] = cursor + 1
                moving = True
    return out


def _stride(length: int) -> int:
    """A step coprime to the list, near a third of its length."""

    target = max(1, length // 3)
    for step in range(target, length):
        if math.gcd(step, length) == 1:
            return step
    return 1


def _max_placed(packs: list[str], cells_of: dict[str, list[dict]], zero_cap: int, per_pack: int = FILE) -> int:
    """How many of the per-pack cells can be placed without passing the zero-if cap."""

    lists: dict[str, dict[tuple, dict]] = {}
    union: list[dict] = []
    seen: set[tuple] = set()
    for pack in packs:
        lists[pack] = {}
        for cell in cells_of[pack]:
            cid = cell_id(cell)
            lists[pack][cid] = cell
            if cid not in seen:
                seen.add(cid)
                union.append(cell)
    if not packs or not union:
        return 0
    n_packs = len(packs)
    n_cells = len(union)
    source = 0
    pack0 = 1
    cell0 = pack0 + n_packs
    sink = cell0 + n_cells
    graph: list[dict[int, int]] = [dict() for _ in range(sink + 1)]

    def add(u: int, v: int, cap: int) -> None:
        graph[u][v] = graph[u].get(v, 0) + cap
        graph[v].setdefault(u, 0)

    c_index = {cell_id(cell): i for i, cell in enumerate(union)}
    for i, pack in enumerate(packs):
        add(source, pack0 + i, per_pack)
        for cid in lists[pack]:
            add(pack0 + i, cell0 + c_index[cid], 1)
    for cell in union:
        cap = zero_cap if cell["ifs"] == 0 else n_packs
        add(cell0 + c_index[cell_id(cell)], sink, cap)
    flow = 0
    needed = per_pack * n_packs
    while flow < needed:
        parent = [-1] * len(graph)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v, cap in graph[u].items():
                if cap > 0 and parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] < 0:
            break
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= 1
            graph[v][u] = graph[v].get(u, 0) + 1
            v = u
        flow += 1
    return flow


def _marginal(weight: int, k: int, zero: bool, rot: int) -> int:
    """Cost of the k-th visit. A cell with more skeletons stays cheaper for longer. A zero-if visit costs more than any
    if visit, so those cells are used only when a pack has no if cell left."""

    load = (1000 * k + weight - 1) // weight
    if zero:
        load += 1_000_000
    return load * 100 + rot


def assign_cells(packs: list[str], cells_of: dict[str, list[dict]], zero_cap: int, weights: dict[tuple, int],
                 file_index: int, per_pack: int = FILE) -> dict[str, list[dict]]:
    """One file's cells. `weights` maps a cell id to how many distinct skeletons it yielded."""

    lists: dict[str, dict[tuple, dict]] = {}
    union: list[dict] = []
    seen: set[tuple] = set()
    for pack in packs:
        lists[pack] = {}
        for cell in cells_of[pack]:
            cid = cell_id(cell)
            lists[pack][cid] = cell
            if cid not in seen:
                seen.add(cid)
                union.append(cell)
    needed = per_pack * len(packs)
    if _max_placed(packs, cells_of, zero_cap, per_pack) < needed:
        raise ValueError(f"no cell assignment places {needed} builds under a zero-if cap of {zero_cap}")
    order = _spread(union)
    index = {cell_id(cell): i for i, cell in enumerate(order)}
    stride = _stride(len(order))
    n_packs = len(packs)
    n_cells = len(order)
    caps = []
    for cell in order:
        caps.append(zero_cap if cell["ifs"] == 0 else n_packs)
    source = 0
    pack0 = 1
    cell0 = pack0 + n_packs
    slot0 = cell0 + n_cells
    n_slots = sum(caps)
    sink = slot0 + n_slots
    graph: list[list[list]] = [[] for _ in range(sink + 1)]

    def add(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, cap, cost, len(graph[v]), cap])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1, 0])

    for i, pack in enumerate(packs):
        add(source, pack0 + i, per_pack, 0)
        pack_cost = (i + file_index) % 17
        for cid in lists[pack]:
            add(pack0 + i, cell0 + index[cid], 1, pack_cost)
    slot = slot0
    for i, cell in enumerate(order):
        weight = max(1, weights.get(cell_id(cell), 1))
        rot = (i * stride + file_index * stride) % 97
        for k in range(1, caps[i] + 1):
            add(cell0 + i, slot, 1, _marginal(weight, k, cell["ifs"] == 0, rot))
            add(slot, sink, 1, 0)
            slot += 1
    potential = [0] * len(graph)
    flow = 0
    while flow < needed:
        dist = [_INF] * len(graph)
        dist[source] = 0
        parent_v = [-1] * len(graph)
        parent_e = [-1] * len(graph)
        heap = [(0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for ei, edge in enumerate(graph[u]):
                v, cap, cost = edge[0], edge[1], edge[2]
                if cap <= 0:
                    continue
                nd = d + cost + potential[u] - potential[v]
                if nd < dist[v]:
                    dist[v] = nd
                    parent_v[v] = u
                    parent_e[v] = ei
                    heapq.heappush(heap, (nd, v))
        if dist[sink] >= _INF:
            break
        for v in range(len(graph)):
            if dist[v] < _INF:
                potential[v] += dist[v]
        v = sink
        while v != source:
            u = parent_v[v]
            edge = graph[u][parent_e[v]]
            edge[1] -= 1
            graph[v][edge[3]][1] += 1
            v = u
        flow += 1
    if flow < needed:
        raise ValueError(f"no cell assignment places {needed} builds under a zero-if cap of {zero_cap}")
    chosen: dict[str, list[dict]] = {pack: [] for pack in packs}
    for i, pack in enumerate(packs):
        for edge in graph[pack0 + i]:
            v, residual, _cost, _rev, original = edge
            if cell0 <= v < slot0 and original > residual:
                chosen[pack].append(lists[pack][cell_id(order[v - cell0])])
        # Spread order, then a coprime walk. Adjacent seeds take cells far apart in that
        # order. The pack index and the file shift where the walk starts.
        chosen[pack].sort(key=lambda cell: (index[cell_id(cell)], cell_id(cell)))
        n = len(chosen[pack])
        step = _stride(n)
        start = (i + file_index * step) % n
        chosen[pack] = [chosen[pack][(start + slot * step) % n] for slot in range(n)]
    return chosen


def _packs_of(table: dict, nvalues: int, value_counts: dict[str, int]) -> list[str]:
    return [pack for pack in table["packs"] if value_counts[pack] == nvalues]


def assignment(question: str, nvalues: int, file_index: int) -> dict[str, list[dict]]:
    """The file's cells for every pack with this many values. Computed once, then reused."""

    key = (question, nvalues, file_index)
    if key not in _PLANS:
        table = load_cells()
        counts = _value_counts()
        packs = _packs_of(table, nvalues, counts)
        cells_of = {pack: table["cells"][pack][question] for pack in packs}
        kind = "three" if nvalues == 3 else "two"
        q_data = load_capacity().get("cells", {}).get(question, {})
        data = q_data.get(kind, q_data)
        weights = {}
        for pack in packs:
            for cell in cells_of[pack]:
                found = data.get(cell_key(cell))
                if found and found.get("builds"):
                    weights[cell_id(cell)] = max(1, int(found["distinct"]))
        _PLANS[key] = assign_cells(packs, cells_of, zero_if_cap(kind, question), weights, file_index)
    return _PLANS[key]


def cell_for(pack: str, question: str, seed: int) -> dict:
    """The cell for this pack, question and seed. One file's seeds land on different cells."""

    if seed < 1:
        raise ValueError(f"T10 seed is 1 or more, not {seed}")
    nvalues = _value_counts()[pack]
    plan = assignment(question, nvalues, (seed - 1) // FILE)
    return plan[pack][(seed - 1) % FILE]


def placed_builds(question: str, nvalues: int, zero_cap: int) -> tuple[int, int]:
    """How many builds the stratum can place at this zero-if cap, and how many a file needs."""

    table = load_cells()
    packs = _packs_of(table, nvalues, _value_counts())
    cells_of = {pack: table["cells"][pack][question] for pack in packs}
    return _max_placed(packs, cells_of, zero_cap), FILE * len(packs)


def _headroom(packs: list[str], cells_of: dict[str, list[dict]], zero_cap: int, candidates: list[dict]) -> int | None:
    """How many more packs fit when each new one lists `candidates`.

    None means those cells include a whole file of if-cells, so the zero-if cap never runs out: an if cell's visits grow
    with the number of packs."""

    cells = _unique(candidates)
    if len(cells) < FILE or not packs:
        return 0
    if sum(1 for cell in cells if cell["ifs"] != 0) >= FILE:
        return None

    def fits(extras: int) -> bool:
        names = list(packs) + [f"extra-{i}" for i in range(extras)]
        lists = dict(cells_of)
        for i in range(extras):
            lists[f"extra-{i}"] = cells
        return _max_placed(names, lists, zero_cap) >= FILE * len(names)

    if not fits(1):
        return 0
    zero_cells = 0
    seen: set[tuple] = set()
    for pack in packs:
        for cell in cells_of[pack]:
            cid = cell_id(cell)
            if cid not in seen and cell["ifs"] == 0:
                seen.add(cid)
                zero_cells += 1
    limit = max(1, zero_cells * zero_cap)
    lo, hi = 1, 1
    while hi < limit and fits(hi):
        lo = hi
        hi = min(limit, hi * 2)
    if not fits(hi):
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid - 1
    return lo


def capacity_report(table: dict | None = None, value_counts: dict[str, int] | None = None) -> list[dict]:
    """Supply, demand and headroom for each stratum. `ok` is false when the file cannot be placed.

    The next step named on a failure is polarity_mode or in/not_in for claim cells. This checks the table. It is not
    called from a build."""

    table = load_cells() if table is None else table
    counts = _value_counts() if value_counts is None else value_counts
    rows = []
    for question in ("set", "claim"):
        for kind, nvalues in (("two", 2), ("three", 3)):
            packs = _packs_of(table, nvalues, counts)
            cells_of = {pack: table["cells"][pack][question] for pack in packs}
            cap = zero_if_cap(kind, question)
            union: list[dict] = []
            seen: set[tuple] = set()
            detail = []
            for pack in packs:
                cells = _unique(cells_of[pack])
                ifs = sum(1 for cell in cells if cell["ifs"] != 0)
                demand = max(0, FILE - ifs)
                detail.append({"pack": pack, "if_cells": ifs, "zero_if_demand": demand})
                for cell in cells:
                    cid = cell_id(cell)
                    if cid not in seen:
                        seen.add(cid)
                        union.append(cell)
            zero_cells = sum(1 for cell in union if cell["ifs"] == 0)
            supply = zero_cells * cap
            demand = sum(row["zero_if_demand"] for row in detail)
            needed = FILE * len(packs)
            placed = _max_placed(packs, cells_of, cap) if packs else 0
            ok = placed >= needed
            shared: list[dict] = []
            if packs:
                common: set[tuple] | None = None
                sample = {cell_id(cell): cell for cell in union}
                for pack in packs:
                    ids = {cell_id(cell) for cell in cells_of[pack]}
                    common = ids if common is None else common & ids
                shared = [sample[cid] for cid in sorted(common)]
            row = {
                "question": question,
                "kind": kind,
                "zero_if_cap": cap,
                "zero_if_cells": zero_cells,
                "supply": supply,
                "demand": demand,
                "placed": placed,
                "needed": needed,
                "headroom_shared": _headroom(packs, cells_of, cap, shared) if ok else 0,
                "headroom_any": _headroom(packs, cells_of, cap, union) if ok else 0,
                "packs": detail,
                "ok": ok,
            }
            if not ok:
                row["reason"] = (
                    f"{question} {kind}: placed {placed} of {needed}. "
                    f"Zero-if supply is {supply} ({zero_cells} cells × {cap}), demand is {demand}. "
                    f"Next step is polarity_mode or in/not_in for claim cells."
                )
            rows.append(row)
    return rows


def canonical_skeleton(model: dict) -> str:
    """Entities and values renamed by first appearance in the clue list. Attribute shapes and the clue list stay, so two
    puzzles from different packs share a skeleton only when the clues have the same shape."""

    ents: dict[str, str] = {}
    vals: dict[tuple[str, str], str] = {}

    def ent(name: str) -> str:
        if name not in ents:
            ents[name] = f"e{len(ents) + 1}"
        return ents[name]

    def val(attr: str, value: str) -> str:
        key = (attr, value)
        if key not in vals:
            n = 1 + sum(1 for a, _v in vals if a == attr)
            vals[key] = f"v{n}"
        return vals[key]

    def rename(rule: list) -> list:
        form = rule[0]
        if form in ("is", "is_not"):
            return [form, ent(rule[1]), "a", val(rule[2], rule[3])]
        if form in ("in", "not_in"):
            return [form, ent(rule[1]), "a", [val(rule[2], v) for v in rule[3]]]
        if form == "count":
            return ["count", "a", val(rule[1], rule[2]), rule[3], rule[4]]
        if form == "if":
            return ["if", rename(rule[1]), rename(rule[2])]
        return rule

    clues = [rename(fact["rule"]) for fact in model["facts"]]
    shapes = [{"values": len(spec["values"]), "ordered": spec.get("ordered") is True,
               "distinct": spec.get("distinct", True) is not False}
              for spec in model["attributes"].values()]
    return json.dumps({"n": len(model["entities"]), "attributes": shapes, "clues": clues}, separators=(",", ":"))
