---
title: Kinds and gates
description: What A through E check, and what the kind B solver will not claim.
sidebar:
  order: 3
---

The plan assigns every slot a kind. The runner does not ask the model to pick one. The five kinds below are the ones the program builds.

| Kind | What the item is | What refuses | What is recorded and does not refuse |
|---|---|---|---|
| A | A teaching passage | The validator, when the passage fails it | The trait check. There is no verdict. |
| B | A finite assignment puzzle | The solver, when the model of the puzzle fails it | The critic's answer, until that leg passes its bar |
| C | A checker with a planted flaw, or a checker who was right | Verdict and evidence | The flaw check, until a control shows the critic agrees on flaws |
| D | Two turns | Verdict and evidence, each turn against the material available then | The trait check |
| E | A written quiz, or a verdict quiz | Verdict and evidence, on a verdict-type E only | The trait check. A written E has no verdict check. |

Verdict and evidence checks refuse. Trait, answer, and flaw checks are recorded and do not refuse, until their controls pass. That split is the program, and the design says the same.

## Why the critic does not gate

`critic-control` measured the critic on reviewed items whose keys are already known.

The trait leg was 40.3% overall. The bar is 85% overall, 70% on every trait, and 90% unanimous. The worst trait was 12.5%. The critic was consistent and wrong often enough that turning the check into a refusal would throw out items known to be good.

The answer leg, on kind B's closed answers, was 62.7% combined. The bar is 90%.

The flaw control on 2026-10-11 had 9 of 24. Until a later control passes, a flaw mismatch is logged as advisory.

These figures are for the 2026-10-11 critic, `gemma4:31b`, which is off the proven list. The new critics, Muse-Glimmer and Nemotron 3.5 Lightning, get their own control before any of their checks gate.

```text
uv run role-set critic-control --items reviewed.jsonl --critic-endpoint http://127.0.0.1:8081/v1 --critic-model muse-glimmer-30b \
  --critic-gguf-sha256 <sha256> --llama-server-sha256 <sha256> --machine machine.json
uv run role-set critic-control --trait-leg --pack roles/verifier --items reviewed.jsonl <the same critic flags>
uv run role-set critic-control --answer-leg --items reviewed.jsonl <the same critic flags>
```

`--items` takes one or more reviewed data files. `--trait-leg` needs `--pack`, because the trait names come from the pack. The command returns 0 when every leg you ran passes its bar, and 2 otherwise. Both legs are still short of the bar. Running the command does not turn the gates on.

On the 5090, add `--grant` and `--heartbeat`. `write` refuses a critic listed with `role: critic-comparison`. Such a critic runs in a control only.

## The B solver

**Solve-first puzzles** (BUILD-SPEC B3a). For five of the eight structures the model and its answer are made in code before the writer sees the slot. A proving generator picks a hidden assignment, adds true facts until the solver leaves the intended answer, prunes every fact that changes nothing, and tries again on a lookup. The writer gets that locked model, a plain gloss of each line and the fixed key. It writes the puzzle's prose, the thinking and the reply, never the model or the answer. The same check below then runs on its text. Two more rules apply. The `ANSWER_EARLY` gate refuses thinking that pairs what is asked with the answer before its final third. And the thinking must cite every fact line, in any form ("(line 3)", "lines 2 and 4"). Each B slot's pack, one of 42 subject packs with its own NAICS domain, and its seed are fixed by the plan. The receipt records `b_path`, the family, the pack, the seed and the model's sha256. `stopping_problem`, `falsifier_first` and `when_to_stop` are still written freehand.

The solver checks a puzzle the writer has given twice: numbered lines a person can read, and a `model` the code can search. It refuses the draft unless the model is tied to those lines, the Question line names the entity the model asks about, every fact is needed, and the search produces the key. The writer is not asked whether the puzzle has one solution.

The closed rule forms are the ones the solver assigns an arity: `is`, `is_not`, `in`, `not_in`, `same`, `different`, `before`, `after`, `adjacent`, `not_adjacent`, `offset`, `count`, `not`, and `if`. A puzzle outside that set is not checked by being called kind B.

Limits, past which the solver refuses the puzzle as too large or too loose:

- at most 6 entities
- at most 3 attributes
- at most 2 flavour lines
- at most 200,000 assignments in the search

Two limits that are easy to miss:

- The thinking is not checked against the model. A trace can narrate a step the model does not take. R&D's key-check is what reads the trace.
- The direction check takes argument order from where the names first appear. A fronted clause can be right and still be refused.

`puzzle-check` is the same solver pointed at a reviewed data file and its `.models.jsonl` sidecar. It exits 0 only when every B item has a sidecar line whose `row_sha256` matches the row, every modelled item passes, no lesson has more than one lookup, the sidecar names nothing the data file does not hold as kind B, and no two passing puzzles across the files given share a fact skeleton. A skeleton is the facts with entities, attributes and values renamed in order of first appearance, each attribute keeping its value count and order, "after" read as "before" reversed, a symmetric clue (next to, not next to, same, different) read in either order, a negative clue on a two-value attribute read as the other value and a count on one read from either side ("at least 2 of 3 lit" is "at most 1 of 3 dark"), and line order set aside.

An underdetermined puzzle's Question asks what could be ("Which cups could hold the pen?"), because its key lists every possibility.

When an underdetermined puzzle asks which entities hold a value on an attribute that may repeat, its possible answers are groups. The key writes each group with "and" and separates the groups with semicolons ("red and blue; red and green", or "elm; oak; yew" for groups of one).

A puzzle that asks which entity holds a value, with a Question line that names neither that value nor a stem of it ("fastest" for 1st), passes with a line for the key-check. A paraphrase is fine as written. The pointer is there so a reader checks that the question is the one the model asks.

A dated puzzle declares its spans in `model.span`: the span attribute, the named days as typed records (label, day, month, ordinal, and a year on a wrap), each span's start and end day, and, where two spans share days, the line that states which one owns them. A day label carries no year; the record's year field only orders a year wrap. An owner rule is refused when no two spans share a day. The clues place entities on spans, and the solver works out which entity holds a named day (inclusive at both ends). The question asks `{"contains": day}` or a claim states it. A fact line never states it, and a date is never an entity.

An item whose structure is a verdict over free-text material is not run through this search. The command counts it as outside the solver. It does not count it as passed. Verdict and evidence for that item stay with the checks those kinds already have, plus the key-check.

Pilot 2 is why the solver exists. Three of its five B puzzles had more than one solution, and each accepted trace had a wrong forcing step. The runner's accept count was not the result that mattered. The key-check was.

It is also why the generator exists. For T10 and T12, code now builds the B puzzle, the solver proves it, and a named lazy reading must give a different answer before the puzzle is kept. That is on [The T10 generator](./generator/) page.
