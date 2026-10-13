"""The selector: a lesson's review file chosen under the per-file shortcut floor."""

import pytest

from role_set.generate.build import receipt
from role_set.generate.recipes import T12_PACKS, build_t12
from role_set.generate.select import Floor, SelectError, check_file, differing, loosens_eq, select

FLOOR_EQ, CEIL_EQ = "T10.exact_read_as_floor", "T10.exact_read_as_ceiling"
EQ_FLOOR, EQ_CEIL = "T10.floor_read_as_exact", "T10.ceiling_read_as_exact"
NEWEST, HIGHEST, BOUNDARY = "T12.newest", "T12.highest_number", "T12.boundary_next"


def rec(*names, same=()):
    shortcuts = {n: {"answer": "x", "differs": True} for n in names}
    shortcuts.update({n: {"answer": "k", "differs": False} for n in same})
    return {"shortcuts": shortcuts}


def _file(pool, picks):
    return [(pool[p.index], p.rests_on) for p in picks]


def test_a_real_t12_pool_where_newest_dominates_still_fills_a_file():
    pool = [receipt(build_t12("issues", s), "T12", T12_PACKS["issues"], "constraint_grid") for s in range(1, 21)]
    assert sum(NEWEST in differing(r) for r in pool) > len(pool) // 2
    picks = select(pool)
    assert len(picks) == 5 and check_file(_file(pool, picks)) == []
    rested = [p.rests_on for p in picks]
    assert rested.count(NEWEST) <= 2 and len(set(rested)) >= 2
    assert select(pool) == picks  # the same receipts give the same file


def test_each_item_rests_on_a_reading_that_differs_on_it():
    pool = [rec(NEWEST, same=[HIGHEST]), rec(NEWEST, HIGHEST), rec(BOUNDARY), rec(HIGHEST), rec(BOUNDARY, NEWEST),
            rec(NEWEST)]
    for p in select(pool):
        assert p.rests_on in differing(pool[p.index])


def test_a_pool_with_one_reading_is_refused_and_says_so():
    with pytest.raises(SelectError, match="generate more seeds"):
        select([rec(NEWEST) for _ in range(20)])


def test_loosening_eq_rests_on_at_most_half():
    assert loosens_eq(FLOOR_EQ) and loosens_eq(CEIL_EQ) and not loosens_eq(EQ_FLOOR) and not loosens_eq(NEWEST)
    pool = [rec(FLOOR_EQ)] * 3 + [rec(CEIL_EQ)] * 3 + [rec(EQ_FLOOR)] * 3
    with pytest.raises(SelectError, match="no file of 5"):  # two loosening items at most, and two per reading
        select(pool)
    pool += [rec(EQ_CEIL)] * 3
    picks = select(pool)
    assert sum(loosens_eq(p.rests_on) for p in picks) <= 2 and check_file(_file(pool, picks)) == []


def test_counting_every_differing_reading_is_stricter():
    pool = [rec(NEWEST, HIGHEST)] * 4 + [rec(BOUNDARY)] * 4
    picks = select(pool)
    assert check_file(_file(pool, picks)) == []
    strict = Floor(count="differs")
    assert any("counts on" in m for m in check_file(_file(pool, picks), strict))
    with pytest.raises(SelectError):
        select(pool, strict)
    pool += [rec(HIGHEST)] * 2 + [rec(NEWEST)] * 2
    assert check_file(_file(pool, select(pool, strict)), strict) == []


def test_check_file_names_what_a_file_breaks():
    a, b = rec(NEWEST), rec(HIGHEST, same=[NEWEST])
    got = check_file([(a, NEWEST)] * 3 + [(b, NEWEST)])
    assert any("has 4 items" in m for m in got)
    assert any("does not differ" in m for m in got)
    assert any("1 distinct" in m for m in got)
    assert any("counts on 4 items" in m for m in got)
    assert any("loosening" in m for m in check_file([(rec(FLOOR_EQ), FLOOR_EQ)] * 3 + [(rec(EQ_FLOOR), EQ_FLOOR)] * 2))


def test_a_file_is_chosen_per_lesson_and_the_count_is_known():
    with pytest.raises(SelectError, match="mixes"):
        select([rec(NEWEST), rec(FLOOR_EQ)] * 5)
    with pytest.raises(SelectError, match="count"):
        select([rec(NEWEST)] * 5, Floor(count="any"))
    with pytest.raises(SelectError, match="count"):
        check_file([], Floor(count="any"))


def test_the_floor_scales_with_the_file():
    assert Floor().cap(0.5) == 2 and Floor(size=4).cap(0.5) == 2 and Floor(size=6).cap(0.5) == 3
    pool = [rec(NEWEST)] * 3 + [rec(HIGHEST)] * 3
    assert len(select(pool, Floor(size=6))) == 6


def dated(position, *names):
    r = rec(*names)
    r["position"] = position
    return r


def test_a_dated_file_floors_answer_positions():
    # No answer is the newest edition: "never the newest" would work, so the pool is refused with the remedy.
    with pytest.raises(SelectError, match="position=n - 1"):
        select([dated("between", NEWEST), dated("earliest", HIGHEST), dated("between", BOUNDARY)] * 5)
    pool = [dated("between", NEWEST), dated("earliest", HIGHEST), dated("between", BOUNDARY), dated("earliest", NEWEST),
            dated("between", HIGHEST), dated("earliest", BOUNDARY), dated("newest", HIGHEST)]
    picks = select(pool)
    positions = [pool[p.index]["position"] for p in picks]
    assert positions.count("newest") == 1 and max(positions.count(x) for x in set(positions)) <= 2
    assert check_file(_file(pool, picks)) == []
    file = [(dated("between", NEWEST), NEWEST)] * 2 + [(dated("between", HIGHEST), HIGHEST)] + \
           [(dated("earliest", BOUNDARY), BOUNDARY)] * 2
    got = check_file(file)
    assert any("between edition on 3" in m for m in got) and any("newest edition on 0" in m for m in got)
    assert check_file(file, Floor(position_max_share=1, min_newest=0)) == []  # the floor is config


def test_the_newest_slot_is_kept_for_a_late_newest_answer():
    pool = [dated("earliest", NEWEST), dated("between", HIGHEST), dated("earliest", BOUNDARY), dated("between", NEWEST)] + \
           [dated("between", BOUNDARY)] * 30 + [dated("newest", HIGHEST)]
    assert pool[select(pool)[-1].index]["position"] == "newest"


def test_counting_every_differing_reading_cannot_hold_for_t12():
    # newest differs whenever the answer is not the newest edition, so the strict count wants three newest answers in
    # five, and the position floor allows two: the reason "rests_on" is the default.
    pool = [receipt(build_t12(p, s), "T12", T12_PACKS[p], "constraint_grid") for p in sorted(T12_PACKS) for s in (1, 2, 3)]
    pool += [receipt(build_t12(p, 1, position=2), "T12", T12_PACKS[p], "constraint_grid") for p in sorted(T12_PACKS)]
    assert check_file(_file(pool, select(pool))) == []
    with pytest.raises(SelectError):
        select(pool, Floor(count="differs"))
