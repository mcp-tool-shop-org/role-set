# role-set: a generator for role-shaping training data

*Design draft 17, from the Dataset Specialist, 2026-10-10, with the Publisher's and ASPIRE's reviews folded in.
Nothing here runs until the maintainer lifts the offrig hold and says which home it gets.*

## 1. What it is for

The Stage 1 dataset is 810 hand-written items, built one batch at a time by an outside model through the
maintainer, reviewed by R&D and the Dataset Specialist. Three batches in, the pattern is clear: a writer that
follows the spec by hand produces careful work slowly, and a writer that reaches for a template produces work
that passes the gates and teaches nothing. role-set is the tool that lets a large model on a rented pod write
batches that are **unique, in the schema, and faithful to the role-shaping method**, with every gate the
humans use run before a human sees the item.

It is a generator, not a judge. It never decides what a good verifier is; that lives in the role pack, which
humans write. It never scores items by model opinion; every refusal is structural (a check in code, or a
known-answer mismatch). It never launches, prices or shuts down a pod; the human does that through offrig.

## 2. Where it lives

A new repo, `mcp-tool-shop-org/role-set`, internal, created by the Publisher on the maintainer's word. Python
3.12. It imports aspire-si's gates through a pinned dependency, never a copy, so there is one source of truth.
ASPIRE makes the gates a small stdlib-only package inside aspire-si (no torch), installed by commit
(`pip install git+…aspire-si@<sha>#subdirectory=…`); the trainer imports the same package; ASPIRE owns it. The
sealed set, the held-out leakage leg, the tokenizer-side checks, `no_role.py` and the trainer stay in aspire-si
and never reach role-set. When a second role exists, the gates package can move to role-set and aspire-si pin
it instead; not before. (The package split is in §9.)

Why not inside aspire-si: the generator will outlive Stage 1 and be used for other roles, it has its own tests
and its own CLI, and a paid-model tool shouldn't sit inside the trainer's docs tree.

## 3. The philosophy, made mechanical

The maintainer's method, and where each part of it is enforced:

| Method | Where it lives in role-set |
|---|---|
| Teach the qualities, not the task | The role pack: lessons, the 18 traits with counterweights, the kind table. The writer gets the trait lesson, never a "write a verification exercise" instruction. |
| Teach the thinking process, and vary it | The plan assigns every B item a different structure from a menu, and the batch check refuses repeats. |
| Shape the role like a psychologist | Kind A and E prompts are built from the lesson's own framing (identity, purpose before rule, own words); exemplars come from approved items of a *different* trait, so form transfers but text can't. The psychologist's framing itself, and "the trait decides a step" in B and D, are not mechanical: only the human read covers them, and the table says so rather than pretending otherwise. |
| Fresh wording; never copy the lessons | No 8-word span shared with any role-pack file (role, traits, kinds, amendments), checked with the same span machinery; the span and phrase checks also run against the exemplars, so a skeleton can't cross traits through them. |
| Lesson, absorb, apply, quiz | The kind mix per trait (A/B/C/D/E, 8 each) and the E variants for the repeat loop. The runner applies the same loop to the builder: lesson served, quizzed in its own words, and only then allowed to write (§6b). |
| Well-timed critique and assistance | The runner's briefs before each slot and its plain-words suggestions after each refused draft, with the governing pack line quoted (§6b). |
| Stay inside the role | The critic names the trait an item teaches from the closed list of 18, key and slot hidden; a mismatch with the slot is a refusal (§6b). |
| Every answer known by construction or from the source | The hidden-key critic (§6): the key must be re-derivable from the material alone. |
| Instant agreement; a contested item is cut | The critic's mismatch is a refusal, not a debate. A slot that fails twice is dropped and reported, never argued for. |
| Not a test to aim at | role-set never reads the sealed set, the DEV sets or the interview; prompts are built only from the role pack, the slot and approved exemplars; the exclusion list is enforced by the leakage check on every item; the sealed leakage leg stays ASPIRE's, run after generation. |
| Fresh wording | The avoid-list (framing lines plus a fixed number of the most frequent phrases) goes to the writer as "do not use"; the full span, framing and phrase checks run in the batch checks, where they're free. Exemplars rotate per slot from a pool, recorded per item, and exemplar-derived spans are counted in the batch checks too. |

## 4. Inputs: the role pack

A versioned directory, hashed, recorded in every item:

```
roles/verifier/
  role.md            lessons 1–3 (the role, how it thinks, what is expected)
  traits/T01..T18.md one lesson each, with the counterweight named
  kinds.md           the per-kind required-parts table and content rules (from the handoff)
  amendments/        1 quoting, 2 silent≠contradicted / escalate / corrections, 3 role-neutral prompts
  rules.md           instant agreement, three lines, time words, length limits, provenance, no identity
  exclusions.txt     repos and sets the writer must never draw from
  structures.yaml    the B-puzzle structure menu (grid, elimination, scheduling with exclusions,
                     underdetermined with explicit set, stopping problems, …) with one worked example each
  pressures.yaml     T13 pressure kinds (seniority, urgency, flattery, repetition, hope, deadline)
  injections.yaml    T17 injection kinds and placements (comment, footer, quoted email, config value,
                     note to reviewers, mid-table row)
  exemplars/         approved items, by trait and kind, taken from merged batches
```

The pack is text the humans already wrote. role-set adds nothing to it; it only serves it in slices.

## 5. The plan: uniqueness before writing

`role-set plan --role verifier --lessons T02,T03,T06 --seed N` produces one **slot** per item, with no model
call. A slot fixes everything the writer must not be free to repeat:

- lesson, kind, side (which half of the counterweight), tier;
- target verdict and escalate value, chosen so the trait meets the floors (≥25% cannot_tell, ≥25% supported,
  ≥1 escalate-yes, ≥1 everyday cannot_tell with escalate-no, 2 contrast pairs with `pair` ids, 2 two-turn D
  items: one stands, one changes);
- content kind (notice, policy, label, timetable, changelog, code, record, design note) rotated so no kind
  dominates a trait;
- for B: a structure from the menu, with at least 3 of 8 non-verification, no structure twice in a trait, at
  most twice in a batch;
- for C: planted flaw or "the checker was right", from the flaw list, balanced;
- for T13: the pressure kind; for T17: the injection kind and placement;
- real-source or invented: at least 4 real-source C/D items per trait, each with its excerpt already fetched
  (§7);
- the multi-line quota: at least a quarter of verdict-bearing items need 2–4 DECIDING lines, assigned to slots
  whose material has that shape (a table row, a heading plus bullet).

The plan is a quota table that meets every floor by construction, so the end-of-batch floor check should never
fail; the model is never asked to keep shares right. It is a file: R&D can read it before any money is spent,
and the same seed reproduces it.

**Cost before spend.** `plan` prints an estimate: slots × expected attempts × tokens per call (writer plus
critic), divided by the profile's measured throughput, as pod hours and dollars at the profile's rate. The
maintainer sees the figure before anything is launched.

## 6. The writer and the critic

**One model call per slot, and the model never holds the pen on the material.** The generator chooses the
excerpt (real or from an invented-material pool it builds for the slot's content kind), and the model returns
only the parts a writer should write: the statement, the framing sentence, the thinking and the reply, as JSON
fields (plus the other checker's trace for C, the pressure or new-material turn for two-turn D, the question
and variant for E). The generator assembles the user turn itself, with `Material:` and the excerpt verbatim,
and fills `sources`, `content_kind`, `pair`, `checker_verdict` and the slot fields. Material drift, invented
citations and framing that doesn't fit the content then can't happen.

The prompt is:

1. the role lesson for the slot's trait and the kind's required parts;
2. the slot's fixed parameters and the material;
3. two or three exemplars of the same kind from a *different* trait;
4. a short avoid-list: every framing line already used, and the N most frequent spans and short phrases in
   the accepted set (N fixed in config, 60 to start; the batch checks catch the rest at accept time);
5. the generator's own computed facts where the slot needs them: counts, digit edits, the claim/material diff,
   orders of magnitude. The model is given these, never asked to compute them;
6. the output schema as JSON, with the three-line block spelled out.

Temperature is moderate (0.7) with a per-slot seed recorded. Requests run concurrently; the pod's throughput
needs many in flight.

**Mechanical truth checks on the thinking, before any critic:**
- every bracketed or quoted span in the thinking is verbatim in the material;
- every number in the thinking appears in the material or the statement, or is one the generator computed;
- every edit or difference the thinking claims is in the generator's claim/material diff.

**The critic is a second call with the key hidden, and it is structural.** It is a **different model family**
from the writer, so they don't share blind spots. The Publisher's choice: **gemma4:31b through the local
Ollama on the 5090** (Apache-2.0, confirmed at the pinned digest before first use; a different family from
every candidate writer; the lowest false-accept rate of the local models on our own verify rows, which are
indicative only since they sit on provisional gold; about 20 GB at 16k context, inside the 26 GB plan). It
runs inline, not as a pass afterwards, because the refusal loop needs its answer while the pod is up; a few
hundred short calls per batch is light work for the card. It only refuses; agreement is never proof, and the
agreement rate is logged per batch. If the card is busy or the control below fails, the fallback is a second,
smaller model on the pod from a third family, licence-checked the same way, with its cost added to the plan's
estimate. A same-model critic is never used; R&D's full read is mandatory either way. The no-model-judge rule
is about rewards and scores; a filter that only refuses is outside it, and the design says so here.

**Known-answer control before any paid batch.** The critic runs over the 160 merged items, whose keys are
known and reviewed, and over 20 of them with one key deliberately broken. Measured: false refusals (disagreeing
with a correct key) and catches on the planted errors. A false-refusal rate over 10% means it isn't usable as a
critic, and another local model or the pod fallback is chosen. One 5090 grant covers this control and the
40-slot pipeline rehearsal in §9.

**The grant pattern: one batch-window grant per batch,** issued by the Publisher after telling the maintainer
what, how hard and how long. The window opens when the pod is ready and closes 30 minutes after `write` ends.
The critic is held loaded (`keep_alive`) for the window and unloaded (`keep_alive: 0`) at its end; nothing else
uses the card in the window, and no switchyard 5090 leg is booked over a pod session. `write` refuses to start
without a `--grant` token and a fresh watchdog heartbeat, the same guard shape as switchyard's runner.

**Andon per lesson and kind; the run stops only on a systemic signal.** Three dropped slots end that lesson's kind
(its cell): the cell's remaining slots are skipped, and the lesson's other kinds and the other lessons go on. Each
cell halt is keyed by the specific rule its drops fell on (a gate rule, a check_against code, a critic code), never the
bare code. Two cells of different lessons in a row halting on the same kind and rule switch that kind off for the rest
of the run; the other kinds carry on. The whole run stops only when every kind is off, or when more than 20% of the
last 20 writer calls fail to parse or reach the writer. A draft cut off at the length limit is TRUNCATED: it counts
toward its cell's drops and never toward that rate. A failed warm-up ends its own lesson and is never part of the
systemic rule. Slots are dispatched interleaved: round by round, one from every lesson and kind, kind B last, so a
short budget still samples every lesson and kind. The 235B pilot (`docs/runs/2026-10-11-235b-pilot.md`) is why: with
the andon per lesson and B first in the plan, every lesson ended inside B, and two warm-up misses stopped the run. A time budget (`--stop-after-minutes`) starts no new slot after it; running slots finish. Every
halt is logged with its reason (`log.jsonl`). Every writer call, accepted or refused, is kept in `drafts.jsonl` with
its raw text, tokens, latency, outcome, glyphs normalised and the frozen stuck rule's measures; refused drafts also
go to `refusals.jsonl`. `role-set readout` reports per lesson and kind, so a failed run reads like a passing one.
The maintainer's rule for paid runs (2026-10-11): the hour returns diagnosable data either way. Every `write`
ends, halt included, with a reminder to shut the pod down; offrig's deadline watchdog is the backstop, not the plan.

- For verdict-bearing items (C, D, verdict-type E): the critic gets the material and the statement only, and
  must return VERDICT and DECIDING. For two-turn D items, each turn is checked with the material available
  at that turn. What counts as a mismatch: the verdict differs → refuse; the critic's DECIDING isn't verbatim
  in the material → that answer is void, ask once more; the key's lines and the critic's lines share no line,
  and neither set covers the other under the frozen scorer's coverage rule (`deciding_ok`) → refuse, because
  two readers found different evidence and that isn't instant agreement.
- For C with a planted flaw: the critic gets the other checker's trace and must name the flaw from the flaw
  list, each label with its one-line meaning. Mismatch = refuse once a control run shows the critic agrees on
  flaws (`flaw_gate`); until then it is logged as `flaw_advisory`. The 2026-10-11 control had 9 of 24.
- For B puzzles: the critic gets the puzzle's facts and question without the writer-facing pattern instruction,
  and ends with an ANSWER line in the key's closed form (a verdict label, a value, an ordered list or
  `name=value` assignments). The two closed answers are compared, never free text; a missing or open ANSWER line
  is void and asked once more. Candidates are compared without case or a leading article, never by word span,
  and named as the puzzle names them. (The 235B pilot's comparator matched free text against free text, and R&D
  found 14 of 20 refusals were the check's fault.)
- For B puzzles, before the critic, the solver (`puzzle.py`, PR H): the writer gives the puzzle twice, as numbered
  lines and as a `model` (entities, attributes, one rule per line in a closed set of forms, flavour lines, and what
  the question asks or a claim). The code refuses the draft unless the model is tied to the text (every line is a
  rule or flavour; a rule names only what its line names; a negative line has a negative rule; before and after keep
  the line's order; left, right, first or last come with the order stated; flavour names no partial domain), every
  fact is needed (re-solved without it, the answer changes), and exhaustive search gets the key: exactly one answer,
  or the computed set (underdetermined), the computed stopping line (stopping problem, when to stop), or the computed
  verdict on the claim (verification structures). An answer that follows from `is` facts alone is a lookup, one per
  lesson. A refusal names a counterexample, and the normal retry carries it; the writer is never asked whether its
  puzzle is unique. Pilot 2's key-check found three of five B puzzles with more than one solution.
- For every kind, the trait: the critic reads the whole item and names the trait whose step decides it (each trait
  listed with the step it decides, and the sibling traits it confused told apart), with the error the item
  contrasts against on its own line; pressure is structure except for T05 and T13. A majority of three calls at
  temperature 0.7 decides. `critic-control --trait-leg` and `--answer-leg` measure the two checks on reviewed
  items against R&D's bars (trait: 85% overall, every trait 70%, unanimous on 90%; answer: 90%), scored on items
  never used to tune the prompt. **Until a leg passes, its check is advisory** (`trait_gate`, `answer_gate` off):
  a mismatch is logged with the critic's votes and words, and never refuses. R&D reversed its first ruling once
  critic-control showed 40% trait agreement on known-good items. A critic that misses a bar gets a new prompt; the
  bar doesn't move.
- For A and written E: no verdict check is possible without opinion, so these get the validator, the trait
  check and the human read. The plan keeps them at 16 of 40 per trait, where they are now.

A mismatch is a refusal, never a vote. The critic can be the same pod model or a different one; the item
records which.

**Refusal loop.** Validator refusal or critic mismatch → one replacement request carrying the refusal reason,
then a second. After two, the slot is dropped and listed in the batch report. The writer never sees the
sealed set, the DEV sets or the interview. The runner (§6b) is what turns a refusal into a suggestion.

## 6b. The runner: it suggests, and it keeps the builder inside the role-set

`write` is not a loop that fires prompts and filters the answers. It is a runner that sits between the role
pack and the builder (the writer model) for the whole batch, and does three jobs: it suggests, it enforces the
confines of the role-set, and it keeps diagnostics. This is the maintainer's "well-timed critique and
assistance" applied to the builder, and the lesson-absorb-apply-quiz loop applied to the builder itself before
it writes a single item.

**Warm-up: the builder is quizzed before it builds.** Before the first slot of a trait, the runner serves the
trait lesson and asks the builder three things in its own words: the trait's main point, its counterweight,
and one case where the trait decides a step. The check is lexical, with no model opinion. All three parts must be
answered, and the trait must be named, either by a content word of its own name or by two content words of the
lesson's teaching paragraph (the prose the builder is shown). A vague or off-trait restatement hits neither. It never
asks for words the builder isn't shown: the pilot's check demanded the planner's side labels, so two traits could not
pass at any seed. A miss is re-asked with the missed point marked (the trait's name); the rehearsal's 8B gave the
identical reply to an unchanged prompt. Two misses stop the trait and report it. Every warm-up reply, passed or
missed, is kept in `warmups.jsonl`.

**Suggesting, before each slot.** The runner briefs the builder with what the batch still needs, read from its
coverage ledger, not from the builder's memory: which side of the counterweight is due, which content kinds
and puzzle structures are unused, which pressure or injection kinds haven't appeared, whether the next item
must carry a multi-line key or a pair id, and which framings and phrases are spent. It also points at the two
role-pack passages most relevant to the slot (the trait's own paragraph on that side, and the kind's required
parts). The builder gets a short, specific brief rather than the whole pack each time.

**Suggesting, after each draft.** A refusal code from the gates or the critic is not sent back bare. The
runner maps each code to a fix in plain words plus the role-pack line that governs it, with the offending
text quoted: "DECIDING quotes two cells; quote the whole row: `| Child weekday | £1 |` (amendment 1, rule 2)",
"the reply closes with a phrase already used in 61% of this trait's turns; end it differently", "the thinking
states a count the material doesn't give; the count is 3, computed from the material". The mapping is a table
in the repo, pinned by a test, so suggestions are consistent across batches and reviewable.

**The confines of the role-set.** The builder may draw on the role pack and nothing else, and every item has
to belong to the trait it was ordered for. Enforced:
- **Which trait the item teaches, named by the critic from a closed list.** With the key and the slot hidden,
  the critic is asked which of the 18 traits the item teaches, and must answer from the list. A mismatch with
  the slot's trait is a refusal, like the critic's other mismatches, and the named trait is logged as the
  drift destination. This is the check that catches a quoting lesson pinned onto "takes no orders". A
  term-overlap check against the lesson texts was probed first and rejected: on 280 approved items the own
  lesson was nearest for only 15%, because C items teach through another trait's failure and B/D thinking
  talks about the material, not the trait's vocabulary. That overlap ranking survives only as a figure in the
  batch report for kind A, where reviewers can read it; it refuses nothing.
- **The slot fixes the trait and its counterweight side,** and A and E items must use the lesson's own trait
  term (already enforced for A by the gates).
- **Pack-only sources.** No span shared with any text outside the pack and the chosen excerpt; the exclusion
  list; the role-neutral prompt rule; no copying from the lessons themselves (§3).
- **Scope of the batch.** The builder can't write a slot the plan didn't order, change a slot's parameters,
  or produce extra items; the runner assembles the row from the plan, so there's nowhere for that to go.
- **Voice and form.** The kind's required parts, the three-line block, the length limits and the identity
  rules, all from the gates package.

A confines refusal counts like any other refusal toward the drop andon, and the drift destination is logged.

**Live changes during a run.** The runner reports live and fixes what it may fix while the batch runs, so a
context problem or a transport error doesn't cost a pod hour. The boundary is fixed, from the Publisher:

| Allowed action | Trigger | Cap | Logged per item |
|---|---|---|---|
| Trim the avoid-list, then the exemplars (fewest first) | a prompt within 10% of the context limit | never below the framing list and one exemplar | tokens before and after, what was cut |
| Re-brief a slot, quoting the governing pack line | the same refusal code twice on one slot | once per slot, inside the two-retry limit | the code and the line quoted |
| Retry a transport error with backoff | a 5xx, a reset or a timeout | 3 tries, 2 s → 8 s → 32 s | the error and the wait |
| Resume from the last accepted item | a crash, or `write --resume` | n/a; the accepted file is append-only and the ledger is on disk | the resume point |
| Lower concurrency | timeouts on more than a quarter of in-flight calls | halve, floor of 1; never raised again in the run | old and new values |

Not allowed mid-run, under any trigger: changing the role pack, the gates, the floors, the critic, the plan's
slots or the model id. Those are human changes between batches. Anything outside the table stops the run as
an andon and reports; the runner never improvises a fix.

**Live data.** `write` keeps a status file (`status.json`, rewritten after every event, and the same content
on a local-only HTTP endpoint on 127.0.0.1 when `--status-port` is given) that the Publisher reads during the
run: items accepted, refused and dropped, by refusal code; the drop-rate andon and its distance from the
threshold; context headroom per call; critic agreement so far; tokens and elapsed time against the plan's
estimate; the current concurrency; and every runner action taken, in order. The batch report carries the full
action log, so the reviewers know exactly what the runner changed and why.

**Diagnostics, the metadata the maintainer asked for.** Per slot: attempts, every refusal code with the
suggestion that followed it, and whether the next draft cleared it. Per trait: the warm-up result, the drift
map (which other lessons items leaned toward, and how often), the spent-phrase list, critic agreement, and the
coverage ledger at the end. Per batch: which suggestion types cleared refusals on the first retry and which
didn't, so the prompt pack and the suggestion table are tuned for the next run from evidence rather than
guesses. All of it goes in `batch-report.json` beside the receipts and is read by R&D with the PR.

## 7. Real sources

`role-set harvest` reads an allow-list of repos pinned at a commit, checks the LICENSE first line against the
allowed set (Apache-2.0, BSD, MIT), refuses anything on the exclusion list, and extracts candidate excerpts
(changelog entries with their version heading, README paragraphs, tables, short functions) with
`repo@commit:path#lines`. The writer receives the excerpt verbatim and cites what it was given; it never
invents a citation. `_verify_sources`-style verification runs on every accepted item.

## 8. Gates, outputs and receipts

Every item passes, in order: the schema and per-item checks (`validate` on the row with the accepted set as
context), the critic, then the batch checks (floors, pairs, spans, framings, short phrases, puzzle
signatures, material reuse) against the accepted set plus this batch, then the leakage check and the
identity scan on the output files. Nothing is written to the output file until it has passed everything.

Outputs per batch:
- `batchN-<lesson>.jsonl` and the `.md` render, in aspire-si's shape;
- `batch-report.json`: the `--batch` and `--freeze` figures, dropped slots with reasons, attempts per slot;
- `receipts.jsonl`: per item, the writer model id and revision, the sampling settings and seed, the endpoint,
  the prompt-template sha, the role pack hash, the gates package sha and `GATES_VERSION`, the slot, the critic
  model and result, and the attempt count. The same pins go on the item as generator metadata;
- a PR body with the per-trait table the reviewers ask for.

Then the human path is unchanged: PR to aspire-si, R&D reads every item, the Dataset Specialist reads 15 per
trait, the Publisher merges. A model writer skips no gate. role-set's leakage check covers the DEV sets, the
interview and the gold corpora; the sealed leg is ASPIRE's, run on each generated PR. The receipts make every
item traceable, not byte-reproducible: SGLang at temperature 0.7 under batching doesn't replay a seed exactly.
The generator never tunes materials or lengths toward the NO-ROLE arm's token ratio; shaping data to the
control is out, by R&D's ruling.

## 9. Transport and the offrig rules

- The writer talks to an OpenAI-compatible `/v1` endpoint on `127.0.0.1` only: offrig's SSH tunnel port for
  the lane (11435 default, 11500+2i for project lanes). Any other host is refused. Cloud model tags are
  refused. Nothing but prompts leaves this machine; the validator, the accepted set and the role pack stay
  local.
- role-set never plans, launches or shuts down a pod. The human does that with offrig, which holds the
  budget cap. role-set takes an endpoint URL and a model id.
- The prompts go to rented hardware, so the role pack and the exemplars must be identity-clean: `write` runs
  the identity scan on the role pack directory, gated on its exit code, before it sends anything, and records
  the pack hash that was scanned.
- The standing rules apply in full: local models only and no Ollama Cloud; the 5090 only under a Publisher
  grant; any Claude subagent on Sonnet; repo visibility and releases are the maintainer's word.
- Writers (the Publisher owns the licence check at the pinned revision, for every model that touches an item,
  critic included): Qwen3-235B-A22B-Instruct-2507 for the pilot; GLM-4.5/4.6 (MIT) remains a later candidate;
  DeepSeek V3 doesn't fit a 4-card pod at 4-bit; coding models and Llama-family models are out.
- **Alternative transport, to decide:** offrig's own runner (`offrig_handoffs` with acceptance checks, then
  `offrig_run`) already drafts, revises twice against failed checks and sends the rest to review. role-set
  could emit one handoff per slot with the validator as the acceptance check. The gain is reuse; the loss is
  the critic step and the avoid-list, which need item-level state between calls. Draft 1 drives the endpoint
  directly and keeps the runner as a later mode.
- **Rehearsal before paying:** a `--mock` writer for tests and CI (canned rows, no model), then one local
  rehearsal on the 5090 through Ollama under the same grant as the critic's known-answer control (§6): a
  writer inside the ≤26 GB plan (qwen3:30b-a3b Q4 or qwen3:14b), one trait, 40 slots, told to the maintainer
  first (what, how hard, how long). Its output is a pipeline shakedown and is discarded, never merged; merged
  data comes only from the chosen writer.
- **A single-writer pilot before a full run** (the maintainer's call, 2026-10-10): no coding models, a
  creative-writing frontier, Qwen or DeepSeek. DeepSeek V3's 4-bit build is 352 GB and leaves the 4-card
  frontier no room for KV, so the pilot writer is **Qwen alone**: offrig profile `writer-qwen235` (2 × RTX
  PRO 6000, SGLang v0.5.20-cu130, context 32,768, parallel 32, about $4.20/hr), model
  `QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ` at revision 96548f65 (124 GB, about 8 minutes to download at
  launch), served as `qwen3-235b-instruct-2507`, base `Qwen/Qwen3-235B-A22B-Instruct-2507` at ac9c66cc,
  Apache-2.0 checked in the LICENSE file at both; both shas go in every item's provenance. One trait, 40
  slots, scored by acceptance rate, critic agreement and R&D's read. The human prices it with `offrig_plan`
  first; the pod shuts down when the quota table is filled.

**What moves in aspire-si (ASPIRE's plan, one mechanical PR, proven on every existing file).** The Publisher's
conditions: no behaviour change (identical pass or identical refusal text on the pilot, batch 2 and #84's
head, reviewed by R&D); the frozen eval files stay byte-identical, so the scorer's hash doesn't move; and it
merges between dataset batches, never while one is under review.
1. The gates become a stdlib-only package: schema, per-kind parts, amendments, diversity, pairs, floors,
   `freeze_report`, `ROLE_WORDS`, `TRAIT_NAMES`, `PRE_AMENDMENT_LESSONS`, with `GATES_VERSION` and a sha.
   `sequences()` (tokenizer), the sealed and held-out leakage leg, `no_role.py` and the trainer stay out.
2. `parse_answer` is copied into a light `answer_format.py`, with a test asserting the copy's source equals
   `harness.parse_answer`'s. The eval files are hash-frozen and don't change.
3. The `_evaldir` path hack leaves the package; only the trainer-side wrapper keeps it.
4. Three functions: `check_item(r)` (context-free), `check_against(r, accepted)` (the exclusivity checks,
   incremental, refusing with a reason the generator can feed back), `check_batch(rows)` (floors, shares,
   pair completeness, freeze figures). `validate(rows)` stays as their composition, so nothing downstream
   changes. For generated data, `check_against` runs the span check against the exemplars and the whole
   accepted set, not just the lesson.
5. The constants become data in one module, so another role's lessons, trait names and role words are files,
   not code edits.

## 9b. The learning loop: one store, observation, and the next batch

The maintainer's goal is a self-learning dataset generator: a runner that takes part in both the creation and
the training, with one database for both, that notices when the model is stuck on a question for longer than
a set time, observes it, and uses what it saw to improve the next batch. This section is the design for that,
agreed with the Publisher and ASPIRE and bound by R&D's ruling in the eval plan (a7b336b). Nothing in it
changes the gates, the critic, the human review chain or the boundary between rounds.

**One shared store, one write path.** A SQLite file with WAL at `E:/AI/role-store/role.db`, outside both repos
(it is data, not code), backed up to the archive drive at the end of every round. The schema, its numbered
migrations and the only code that writes to it live in role-set (`store/`). The trainer never imports
role-set and never opens the database: `run.py` writes JSONL receipts, as it does today, into an explicit
directory (`outputs/stage1-train/<round>/<arm>/<seed>/receipts/*.jsonl`, one `source` per row), and `role-set ingest <dir>` loads them, refusing any
path outside that directory and anything on the deny list below. role-set's runner writes its own rows through
the same `store/` code. So there is one write path into the database, and the firewall sits in it (and a
second time in `run.py`, which refuses to write a sealed, interview or DEV id or text, with the sealed shingle
check on aspire-si's side, the only side that holds the sealed set). Append-only tables, every row stamped
with its round, seed, arm and a `source` from a closed list (`training_item`, `generation`, `train_quiz`,
`rehearsal`, `diagnostics`, and `dev_between_stages`, which ingest accepts only with a dated amendment and a
recorded fired stop rule):
- `items`: every accepted item with its provenance and pins (the receipts of §8);
- `attempts`: every generation attempt, refusal codes, the suggestion that followed and whether it cleared;
- `quiz_results`: every training-time quiz answer: question id, lesson, kind, tier, round, seed, arm, adapter
  sha, attempt number, wall time, thinking tokens, answer tokens, correct or not under the frozen scorer, and
  the Leitner box and state of the repeat-until-correct **ledger**, which is a table from its first version so
  the trainer's loop and the generator read one record. (The quiz loop and the ledger are designed, by R&D's
  Leitner rule, and not yet coded; `train.py` is plain SFT today, so every field here goes in from the start.)
- `rehearsal_results`: the pipeline rehearsal's answers, same columns;
- `observations`: the stuck records below, with the trace only when the stuck rule fired;
- `runs` and `diagnostics`: each round's figures (the batch report, the training run's numbers, the plan that
  fed it);
- `plans`: every slot with the observation ids that drove it.

The deny list ingest refuses by name, all under aspire-si's `outputs/`: `stage1-sealed-FROZEN/`,
`stage1-sealed-draft/`, `stage1-baseline/sealed-130-s*/`, `stage1-baseline/interview/`,
`stage1-baseline/pilot-30-s*/` (DEV), and from round 1 on every `stage1-eval-*` sitting directory. The
explicit receipts directory is the primary guard; the deny list is the second.

**Stuck detection, passive, as R&D ruled.** For every question the model answers in training quizzes and
rehearsals, an observation is recorded when the model is stuck. Loop markers are the core (they measure
circling, which is what the maintainer means by stuck; length measures effort), and the no-progress markers
count only on long traces. **Stuck =**
- **(A)** the same 8-word span three or more times in the trace, at any length; **or**
- **(B)** the 16,384-token cap is reached; **or**
- **(C)** the trace is long **and** not progressing, where *long* = thinking tokens above max(the tier's
  frozen p95, 2 × the item's own target thinking tokens in its arm), and *not progressing* = the re-read rate
  is above its frozen baseline of **8.09 per 1,000 thinking tokens**, **or** a burst of three or more hedge
  phrases ("wait", "hmm", "actually", "let me re-check") within 200 thinking tokens (the maintainer's
  amendment, measured by ASPIRE and admitted by R&D's three conditions; the hedge rate itself is a
  measure, not a trigger).

Measured and dropped: "no new material line in the final 30%" (true of 98 of 102 untrained traces and every
ROLE target sampled, so it would have collapsed (C) to "long"). Recorded as measures, never triggers: the
hedge rate (p95 8.32 per 1,000), length above the untrained p95, length above 2 × the item's target.

Frozen before round 1 in aspire-si `docs/runs/stage1-stuck-rule.json`, **sha256 716dc251… at 4b857ff**, code
`stage1-train/stuck.py` sha256 8097b039… (supersedes 63928fea and 4dbcd12e; released by R&D and frozen for round 1). Definitions in the file:
- **re-read:** a quote of, or an explicit return to, a material line already quoted earlier in the same
  trace, matched by the scorer's lenient normalising, with a 5-word window shared by two material lines
  counting for neither; the rate is re-reads per 1,000 thinking tokens; baseline the p95 over the untrained
  pilot-30 traces, pooled over all tiers (102 turns): 8.09;
- **the tier p95s:** from the untrained pilot-30, tiers under 20 samples pooled (tiers 3 and 4 at 572
  thinking tokens, 5 and 6 at 453, the rest pooled at 575);
- **hedge fields in every receipt** (from stuck.py's counter; eval sittings carry the same fields in their own
  outputs, never in the store): per-phrase hedge counts; hedge rate; first and last hedge positions as
  fractions of thinking tokens; end clustering (null under 2 hedges); gaps in tokens; max in window at W =
  100, 200 and 400; bursts with start and end; burst_before_answer; `gap_unit: "tokens (no streaming
  timestamps)"`.

An observation records: question id, lesson, kind, tier, round, seed, arm, wall time, thinking and answer
tokens, the thinking trace, the final answer and whether it was right, and the trace analysis: where it loops,
where it stalls (a long stretch with no new line quoted from the material), and where it stops. Recording
only: nothing intervenes in an answer while it's being given, and the training run's timing isn't changed by
it; generation already records seconds and tokens per turn, and traces are dumped only on crossings.

**The reserve pool (R&D's rule; the size is the Dataset Specialist's call).** Steered slots, from round 2
on, draw only from a key-checked reserve built before round 1, never from mid-run generation, so every
trained item has been through the review lane. Grain: one cell per (lesson, kind) for kinds C, D and E, the
kinds where stuck and miss are measured, with tiers mixed inside the cell in the batch's own proportions, so
54 cells. Size: before round 1, at least one round's per-cell cap in every cell (at a round of 120 new items
the budget is 24 and the cap 6, so about 324 items), written, key-checked and leakage-checked like a normal
batch, NO-ROLE derived, reviewed by R&D the same way, and frozen by sha256; before round 2 begins, the
reserve is topped up to at least two rounds' worth per cell (648 at that round size; R&D's floor), by the
generator if it is running by then and by hand if not. Topping up the reserve between rounds is the generator's standing first
job. The initial reserve is a batch of its own in the dataset sequence, after the 18 traits and L1–L3.

**The loop.** At planning time, `role-set plan` reads a *weakness view* from the store. The steering cell is
**(lesson, kind), for kinds C, D and E only** (the kinds that produce quiz-ledger and rehearsal
observations), matching the reserve's grain; tier, content shape, verdict and thinking pattern are recorded
with each observation as descriptive fields, not steering keys. Coarser cells reach the n ≥ 8 minimum
sooner, so steering acts on signal rather than waiting on sparse tier cells. The view holds each cell's
stuck rate and miss rate from training quizzes and rehearsals only. The signal is the **pooled stuck rate of both arms' runners** (ROLE and NO-ROLE), treated as
one item stream with matched composition, never the ROLE model alone; and **each training seed runs its own
steering stream**. The plan keeps every amendment floor and share as fixed quotas in every round, and
allocates a **steering budget** above them toward the weakest cells, by the formula R&D approved: 20% of a
round's new items, at most 25% of that budget to any one cell and 40% to any one lesson; a cell's weight is
w_c = max(0, stuck_c − max(median, 0.10)), with Laplace smoothing on the pooled arms per seed, and only cells
with n ≥ 8 count; if no cell clears, the budget returns to the base plan; largest-remainder allocation with
ties broken by cell key, floors first under the `check_batch` gate; deterministic given a store snapshot.
Steering slots are **drawn from the reserve pool** (below), without replacement, taking tiers in the batch's
proportions by id order within each tier, so the draw is deterministic; an exhausted cell passes its slots on
by weight, then back to the base plan; the reserve's sha256 goes in the
plan and the drawn ids in each round receipt. Each slot records
`driven_by`, the observation ids behind it, so a reviewer can trace any item to the evidence that asked for it.
The steering rule (the formula, the floors kept in every round, the cap and the per-weakness maximum) is
**pre-registered before round 1** as a standing dated amendment; every plan's shift is recorded, so R&D can see
the curriculum's drift round by round. The validator, the critic and R&D's read are unchanged: the weakness
view changes what is ordered, never what is accepted. The NO-ROLE arm takes the same plan, so the arm
comparison stays matched.

**DEV is a measuring instrument, not a steering signal.** During a stage, the store holds training-time
observations only: no DEV ids, text, traces or aggregates (with pilot-30, a lesson × kind × tier cell has one
to three tasks, so an aggregate is an item result). The generator never proposes DEV tests; a DEV set is built
and key-checked by the DEV process to its own rules, and a generator that wrote DEV toward the model's
weaknesses would break the rule from the other side. Between stages, after the stop rule fires and the sealed
set is scored, DEV aggregates may steer the next stage's generator by a dated amendment; DEV is then spent,
and the next stage gets a fresh DEV.

**The firewall, mandatory, on the write side.** `ingest` and the store code refuse any row whose `source` is
outside the closed list, any sealed (`sx-`) id, any pre-interview id, and any DEV id, text or aggregate during a stage;
each refusal is logged. role-set's config never points at aspire-si's held-out directory, so the generator has
no read path to the sealed results or the interview transcript; the sealed shingle check runs on aspire-si's
side, so role-set never loads the sealed set. Sealed and interview aggregates reach the humans as reports, as
the baseline's did, and humans may write a dated amendment from them; the generator never reads them. Tests
plant a sealed-tagged row, an interview id and a DEV row and assert the refusal at `ingest` and at `run.py`.

**The boundary between rounds stays.** The runner observes and proposes; it doesn't change what is taught.
Inside a round, the only automatic effect of the store is the slot weighting above, within its pre-registered
rule. Changes to the role pack, the lessons, the floors, the thresholds, the cap or the steering formula are
human changes between rounds, through a dated amendment.

**What the maintainer sees each round.** A round report: where the model got stuck and on what, which cells
the next batch leans toward and why, the quiz ledger's state, the pooled and per-arm stuck rates, and the
sealed and interview aggregates as a separate section the generator never read.

## 10. CLI

Built (`role-set --help`; each command prints a refusal and exits 2 when something it needs is missing):

```
role-set plan     --lessons T02 --seed 7 --out plan.json                     # no model
role-set cost     --plan plan.json --profile two-card                        # no model
role-set pack     sync --repo <aspire-si clone> --commit <sha> --out pack/   # git only; verify --pack pack/
role-set harvest  --allow-list roles/verifier/sources.json --clones <dir> --out pool.jsonl  # git only
role-set write    --plan plan.json --pack pack/ --out <dir> --endpoint http://127.0.0.1:<lane port>/v1                   --model <served name> --grant <id> --heartbeat <file> [--pool pool.jsonl] [--exemplar-*]  # the runner; paid
role-set write    --plan plan.json --pack pack/ --out <dir> --mock           # no model; output is marked, never data
role-set critic-control --items <reviewed items> --grant <id> --heartbeat <file> --out control.json  # 5090, local critic
role-set ingest   --receipts outputs/stage1-train/<round>/<arm>/<seed>/receipts --store <db> --repo <aspire-si> --commit <sha>
role-set store    status --store <db>  |  void --store <db> --round N --reason <text>
```

The gates are aspire-si-gates 1.1.0, a pinned dependency (`pyproject.toml`, at aspire-si's merge commit). `write`
runs `check_item` and `check_against` per item and `check_batch` once on a finished, unhalted run; the batch
report and the result carry `batch_check` ("clean" or the refusal). The mock run goes through the same real gates:
its writer varies its wording per draft, so CI proves a whole mock lesson passes every gate.

Planned, not built: `warmup` as its own command (it runs inside `write` today), `check` (gates on files),
`report`, `weakness` and `round` (the learning loop, §9b), and `write --resume` / `--status-port`.

## 11. Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Role pack hash, model id and revision, prompt hash, seed and slot recorded per item; the plan is reproducible from its seed. Loop: the store schema version, the frozen threshold file's sha256, the steering rule and the cap are recorded per round, and every slot carries the observation ids that drove it. |
| ANDON_AUTHORITY | 2 | Validator refusal and critic mismatch stop the item; two failures drop the slot; nothing is written until every gate passes. Loop: `ingest` and `run.py` refuse sealed, interview and in-stage DEV rows, and any path outside the receipts directory; a plan over the steering budget or a per-cell or per-lesson maximum is refused; `TRACE_REUSED` refuses item text drawn from a trace. |
| NAMED_COMPENSATORS | 2 | Pods are the human's through offrig (shutdown is theirs); role-set's only writes are local files, undone by deleting the batch directory; the PR is closed and the branch deleted. Loop: store writes are append-only and carry a round id; a round is voided by id (`role-set void --round N`), owner the Dataset Specialist, and the next plan is regenerated. |
| DECOMPOSE_BY_SECRETS | 2 | Role pack (humans), plan (deterministic), writer (model), critic (model, key hidden), gates (aspire-si), transport (endpoint only) are separate modules. Loop: the store (role-set), the trainer's receipts (aspire-si), the stuck detector, the weakness view and the planner are separate; the database has one write path; role-set never reads the held-out directory. |
| UNCERTAINTY_GATED_HUMANS | 2 | The plan is reviewable before spend; dropped slots are reported, never filled by guess; the human review chain is unchanged. Loop: the runner observes and proposes; what is taught changes only between rounds by a dated amendment; the maintainer gets a round report. |
| EXTERNAL_VERIFIER | 2 | R&D (a different family from the writer) reads every item; the critic is a known-answer check, not a judge. Loop: R&D sees each plan's shift and the `driven_by` trail, and pre-registered the frozen thresholds, the steering rule, the cap and the DEV ruling (a7b336b). |

## 12. Decisions taken by the maintainer, 2026-10-10

1. The home: `mcp-tool-shop-org/role-set`, created by the Publisher, **private** until Codecov is at 90% or
   above; it goes internal on the maintainer's word.
2. The writer: no coding models; a creative-writing frontier, Qwen or DeepSeek. DeepSeek doesn't fit, so the
   pilot is Qwen alone (§9).
3. The offrig hold is lifted. Order: the scaffold PR; #84 then #85 merge; `plan` with its cost estimate; the
   Publisher's 5090 grant for the critic control and the rehearsal, told to the maintainer first; the paid
   pilot.
