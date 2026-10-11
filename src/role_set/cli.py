"""role-set: plan and cost (no model), write (the runner; paid unless --mock), pack, harvest, ingest, store, critic-control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from role_set import __version__
from role_set import autostart
from role_set import defects
from role_set import puzzle_check
from role_set.readout import readout
from role_set.cost import PROFILES, estimate
from role_set.gates import GatesUnavailable, load_gates
from role_set.guards import GuardError, require_local_endpoint
from role_set.harvest import harvest
from role_set.ingest import IngestError, ingest, interview_questions
from role_set.pack import PackError, sync, verify
from role_set.store import counts, open_store, void_round
from role_set.critic_control import FALSE_REFUSAL_CAP, answer_leg, cases_from_rows, load_rows, plant_errors, run_control, trait_leg
from role_set.guards import require_grant, require_heartbeat, require_ollama_digest
from role_set.prompts import load_pack
from role_set.trait_steps import TRAIT_STEPS
from role_set.write import allowed_critics, trait_names_from_pack
from role_set.transport import ollama_chat, unload
from role_set.write import WriteConfig, WriteRefused
from role_set.write import run as run_write
from role_set.plan import PlanError, check_plan, plan_batch

# The verifier role's lessons and their two sides. The full pack arrives with `pack sync`;
# until then the sides are the ones the hand-written batches used.
VERIFIER_SIDES = {
    "T02": ("quotes the source", "stops at the source"),
    "T03": ("exact", "equivalent is fine"),
    "T06": ("breaks it down", "knows when one step is enough"),
    # The paid pilot's trait: no injection items, where the critic is weakest (control 2026-10-11).
    "T18": ("doesn't rewrite the statement", "still quotes the deciding line"),
    # Each trait and its counterweight, as the lessons state them (pack at aspire-si 0801513).
    "T07": ("quick on the obvious", "careful on the hard"),
    "T08": ("flags advice for a person", "still gives the verdict"),
    "T09": ("same facts, same verdict", "a real difference changes the verdict"),
    "T10": ("judges the exact size", "an equivalent wording still matches"),
    "T11": ("makes acceptances earn their place", "doesn't reject by default"),
    "T12": ("checks the version and the date", "a matching version speaks to the statement"),
    "T15": ("gives the answer and stops", "takes the steps a hard case needs"),
}


def _plan(args: argparse.Namespace) -> int:
    lessons = {}
    for name in args.lessons.split(","):
        name = name.strip()
        if name not in VERIFIER_SIDES:
            print(f"REFUSED: no sides known for lesson {name!r}; known: {sorted(VERIFIER_SIDES)}", file=sys.stderr)
            return 2
        lessons[name] = VERIFIER_SIDES[name]
    try:
        slots = plan_batch(lessons, seed=args.seed)
    except PlanError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    figures = [check_plan([s for s in slots if s.lesson == name]) for name in lessons]
    out = {"role": args.role, "seed": args.seed, "figures": figures, "slots": [s.as_dict() for s in slots]}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for f in figures:
        print(json.dumps(f))
    print(f"wrote {len(slots)} slots to {args.out}")
    return 0


def _cost(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    try:
        est = estimate(len(plan["slots"]), args.profile, attempts=args.attempts, critic_on_pod=args.critic_on_pod)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(est.as_dict()))
    return 0


def _readout(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not (run_dir / "batch-report.json").exists():
        print(f"REFUSED: no batch-report.json in {run_dir.name}", file=sys.stderr)
        return 2
    path = Path(args.catalog) if args.catalog else defects.find()
    catalog = defects.load(path) if path else None
    data, md = readout(run_dir, [Path(p) for p in (args.compare or [])], catalog)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "readout.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out / "readout.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _closed_form_arg(text: str | None) -> dict | None:
    """--closed-form: a JSON object of the gates' closed-form settings (carry, share, distractor_max,
    distractor_margin, span, exempt_keyed_key_points). The gates check the values; this only parses the object."""

    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--closed-form is not JSON: {e}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--closed-form must be a JSON object, e.g. '{\"carry\": \"either\"}'")
    return value


def _write(args: argparse.Namespace) -> int:
    cfg = WriteConfig(
        plan=Path(args.plan), pack=Path(args.pack), out=Path(args.out), endpoint=args.endpoint, model=args.model,
        writers_file=Path(args.writers), critic_endpoint=args.critic_endpoint, critic_model=args.critic_model,
        grant=args.grant, heartbeat=Path(args.heartbeat) if args.heartbeat else None, concurrency=args.concurrency,
        mock=args.mock, no_critic=args.no_critic,
        pool=Path(args.pool) if args.pool else None,
        exemplar_repo=Path(args.exemplar_repo) if args.exemplar_repo else None,
        exemplar_commit=args.exemplar_commit, exemplar_files=tuple(args.exemplar_files or ()),
        exemplar_exclude=Path(args.exemplar_exclude) if args.exemplar_exclude else None,
        stuck_repo=Path(args.stuck_repo) if args.stuck_repo else None, stuck_commit=args.stuck_commit,
        stop_after_minutes=args.stop_after_minutes, closed_form=args.closed_form,
    )
    try:
        result = run_write(cfg)
    except WriteRefused as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 2 if result["halted"] else 0


def _pack_sync(args: argparse.Namespace) -> int:
    try:
        m = sync(Path(args.repo), args.commit, Path(args.out), role=args.role)
    except PackError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps({"role": m.role, "commit": m.commit, "files": len(m.files), "pack_sha256": m.pack_sha256}))
    return 0


def _pack_verify(args: argparse.Namespace) -> int:
    try:
        m = verify(Path(args.pack))
    except PackError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps({"role": m.role, "commit": m.commit, "pack_sha256": m.pack_sha256}))
    return 0


def _ingest(args: argparse.Namespace) -> int:
    try:
        questions = interview_questions(Path(args.repo), args.commit, expected_sha_prefix=args.questions_sha)
        result = ingest(Path(args.receipts), Path(args.store), questions=questions, allow_dev=args.dev_between_stages, amendment=args.amendment)
    except IngestError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=1))
    return 2 if result.refusals else 0


def _store_status(args: argparse.Namespace) -> int:
    conn = open_store(Path(args.store))
    print(json.dumps(counts(conn)))
    return 0


def _store_void(args: argparse.Namespace) -> int:
    conn = open_store(Path(args.store))
    void_round(conn, args.round, args.reason)
    print(json.dumps(counts(conn)))
    return 0


def _critic_control(args: argparse.Namespace) -> int:
    import time

    try:
        gates = load_gates()
        require_grant(args.grant, uses_card=True)
        require_heartbeat(Path(args.heartbeat), time.time())
        endpoint = require_local_endpoint(args.critic_endpoint)
        if "cloud" in args.critic_model.lower():
            raise GuardError("REFUSED: a local critic model is required, never a cloud tag")
        critics = allowed_critics(Path(args.writers))
        if args.critic_model not in critics:
            raise GuardError(f"REFUSED: {args.critic_model} is not on the licence-checked critic list")
        digest = require_ollama_digest(endpoint, args.critic_model, critics[args.critic_model]["digest"])
    except (GatesUnavailable, GuardError) as e:
        print(e, file=sys.stderr)
        return 2
    rows = load_rows([Path(p) for p in args.items])
    log: list = []

    def critic(messages, **sampling):
        return ollama_chat(endpoint, args.critic_model, messages, **sampling)

    if args.trait_leg or args.answer_leg:
        # The legs R&D set bars for (2026-10-11), in one grant: the trait each reviewed item teaches, three calls
        # each; and the kind-B answers, closed and compared closed.
        if args.trait_leg and not args.pack:
            print("REFUSED: --trait-leg needs --pack (the trait names come from the pack)", file=sys.stderr)
            return 2
        out = {"critic_model": args.critic_model, "critic_digest": digest, "grant": args.grant}
        try:
            if args.trait_leg:
                trait_names = trait_names_from_pack(load_pack(Path(args.pack)))
                out["trait_leg"] = trait_leg(critic, rows, trait_names, TRAIT_STEPS, concurrency=args.concurrency, log=log.append)
                out["trait_misses"] = log
            if args.answer_leg:
                out["answer_leg"] = answer_leg(critic, rows)
        finally:
            unload(endpoint, args.critic_model)
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        legs = [out[k] for k in ("trait_leg", "answer_leg") if k in out]
        if "trait_leg" in out:
            leg = out["trait_leg"]
            print(json.dumps({k: v for k, v in leg.items() if k not in ("confusion", "per_trait")}))
            for t, cell in leg["per_trait"].items():
                print(f"{t}: {cell['agree']}/{cell['items']} = {cell['rate']}" + (" (thin)" if cell["thin"] else ""))
        if "answer_leg" in out:
            print(json.dumps({k: v for k, v in out["answer_leg"].items() if k != "disagreements"}))
            if out["answer_leg"]["as_written_below_bar"]:
                print("NOTE: the keys closed as written fall below the bar on their own, though the combined figure is what gates")
        return 0 if all(leg["passes"] for leg in legs) else 2
    cases = cases_from_rows(rows)
    planted = plant_errors(cases, args.planted, seed=args.seed)

    try:
        report = run_control(gates, critic, cases, planted=planted, log=log.append)
    finally:
        unload(endpoint, args.critic_model)
    out = {"critic_model": args.critic_model, "critic_digest": digest, "gates": {"version": gates.version, "sha256": gates.sha256}, "grant": args.grant,
           "cap": FALSE_REFUSAL_CAP, "report": report.as_dict(), "missed_plants": log}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["report"].items() if k not in ("false_refusal_detail", "flaw_misses")}))
    return 0 if out["report"]["usable_as_critic"] else 2


def _harvest(args: argparse.Namespace) -> int:
    allow = json.loads(Path(args.allow_list).read_text(encoding="utf-8"))
    report = harvest(allow["repos"], Path(args.clones), Path(args.out))
    print(json.dumps(report.as_dict(), indent=1))
    return 0 if report.excerpts else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="role-set", description=__doc__)
    p.add_argument("--version", action="version", version=f"role-set {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("plan", help="fix every slot for a batch; no model call")
    a.add_argument("--role", default="verifier")
    a.add_argument("--lessons", required=True, help="comma-separated lesson ids, e.g. T02,T03,T06")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="plan.json")
    a.set_defaults(func=_plan)

    c = sub.add_parser("cost", help="pod hours and dollars for a plan; no model call")
    c.add_argument("--plan", required=True)
    c.add_argument("--profile", choices=sorted(PROFILES), required=True)
    c.add_argument("--attempts", type=float, default=1.5)
    c.add_argument("--critic-on-pod", action="store_true")
    c.set_defaults(func=_cost)

    k = sub.add_parser("pack", help="the role pack: the humans' text at a pinned commit")
    ksub = k.add_subparsers(dest="pack_command", required=True)
    ks = ksub.add_parser("sync", help="fetch and split the pack from a local aspire-si clone at a commit")
    ks.add_argument("--repo", required=True, help="path to the aspire-si clone (never recorded)")
    ks.add_argument("--commit", required=True, help="the commit sha to pin")
    ks.add_argument("--role", default="verifier")
    ks.add_argument("--out", default="roles/verifier")
    ks.set_defaults(func=_pack_sync)
    kv = ksub.add_parser("verify", help="the pack on disk still hashes to its manifest")
    kv.add_argument("--pack", default="roles/verifier")
    kv.set_defaults(func=_pack_verify)

    i = sub.add_parser("ingest", help="the trainer's receipts into the store; firewall at the door")
    i.add_argument("--receipts", required=True, help=".../stage1-train/<round>/<arm>/<seed>/receipts")
    i.add_argument("--store", required=True, help="the SQLite file (outside both repos)")
    i.add_argument("--repo", required=True, help="aspire-si clone, read only for the public interview questions")
    i.add_argument("--commit", required=True, help="the aspire-si commit the questions are read at")
    i.add_argument("--questions-sha", default="b59927e1", help="expected prefix of the questions' sha256")
    i.add_argument("--dev-between-stages", action="store_true")
    i.add_argument("--amendment", default=None, help="the dated amendment that admits DEV aggregates between stages")
    i.set_defaults(func=_ingest)

    st = sub.add_parser("store", help="the store: status and voids")
    stsub = st.add_subparsers(dest="store_command", required=True)
    ss = stsub.add_parser("status")
    ss.add_argument("--store", required=True)
    ss.set_defaults(func=_store_status)
    sv = stsub.add_parser("void", help="append a void for a round; readers skip it")
    sv.add_argument("--store", required=True)
    sv.add_argument("--round", type=int, required=True)
    sv.add_argument("--reason", required=True)
    sv.set_defaults(func=_store_void)

    cc = sub.add_parser("critic-control", help="the critic's known-answer control on reviewed items plus planted errors; needs a grant")
    cc.add_argument("--items", nargs="+", required=True, help="reviewed aspire-si data files (jsonl)")
    cc.add_argument("--planted", type=int, default=20)
    cc.add_argument("--seed", type=int, default=0)
    cc.add_argument("--critic-endpoint", default="http://127.0.0.1:11434")
    cc.add_argument("--critic-model", default="gemma4:31b")
    cc.add_argument("--writers", default="writers.json")
    cc.add_argument("--grant", required=True)
    cc.add_argument("--heartbeat", required=True)
    cc.add_argument("--out", default="critic-control.json")
    cc.add_argument("--trait-leg", action="store_true", help="run only the trait leg: three calls per reviewed item against R&D's bar")
    cc.add_argument("--answer-leg", action="store_true", help="run the kind-B answer leg: closed answers against the reviewed keys, bar 90%%")
    cc.add_argument("--pack", help="the synced role pack (for --trait-leg's trait names)")
    cc.add_argument("--concurrency", type=int, default=1, help="parallel critic calls for --trait-leg (match OLLAMA_NUM_PARALLEL)")
    cc.set_defaults(func=_critic_control)

    h = sub.add_parser("harvest", help="real excerpts from pinned, licence-checked repos into a local pool")
    h.add_argument("--allow-list", required=True, help='JSON: {"repos": [{"name", "commit"}]}')
    h.add_argument("--clones", required=True, help="directory holding the clones, by name (never recorded)")
    h.add_argument("--out", default="out/pool.jsonl")
    h.set_defaults(func=_harvest)

    w = sub.add_parser("write", help="the runner; paid unless --mock")
    w.add_argument("--plan", required=True)
    w.add_argument("--pack", default="roles/verifier")
    w.add_argument("--out", default="out")
    w.add_argument("--endpoint", default=None, help="http://127.0.0.1:<lane port>/v1 (offrig's tunnel); for a digest-pinned rehearsal writer, the local Ollama base http://127.0.0.1:11434")
    w.add_argument("--model", default=None, help="a served name from writers.json")
    w.add_argument("--writers", default="writers.json")
    w.add_argument("--critic-endpoint", default="http://127.0.0.1:11434")
    w.add_argument("--critic-model", default="gemma4:31b")
    w.add_argument("--grant", default=None, help="the Publisher's batch-window grant id for the 5090 critic")
    w.add_argument("--heartbeat", default=None, help="the watchdog heartbeat file (mtime only)")
    w.add_argument("--concurrency", type=int, default=1)
    w.add_argument("--mock", action="store_true", help="no model, no money; output is marked and never data")
    w.add_argument("--no-critic", action="store_true", help="dry runs only; items pass without the second reader")
    w.add_argument("--pool", default=None, help="the harvested excerpt pool (jsonl)")
    w.add_argument("--exemplar-repo", default=None, help="aspire-si clone, for approved items read at a pinned commit")
    w.add_argument("--exemplar-commit", default=None)
    w.add_argument("--exemplar-files", nargs="*", default=None, help="merged data files, e.g. pilot-t04.jsonl batch2-t01.jsonl")
    w.add_argument("--exemplar-exclude", default=None, help="roles/<role>/exemplar-exclude.json: reviewed ids never shown as exemplars")
    w.add_argument("--stuck-repo", default=None, help="aspire-si checkout for the frozen stuck rule (default: --exemplar-repo)")
    w.add_argument("--stuck-commit", default=None, help="commit for the frozen stuck rule (default: --exemplar-commit)")
    w.add_argument("--stop-after-minutes", type=float, default=None, help="no new slot starts after this many minutes")
    w.add_argument("--closed-form", type=_closed_form_arg, default=None,
                   help="the gates' closed-form settings as JSON, e.g. '{\"carry\": \"either\"}'; unset takes the pinned gates' "
                        "defaults. Every receipt records the settings used")
    w.set_defaults(func=_write)

    ro = sub.add_parser("readout", help="one run's results per lesson and kind, beside earlier runs; passing or failing")
    ro.add_argument("--run", required=True, help="the run's output directory (holding batch-report.json)")
    ro.add_argument("--compare", nargs="*", default=None, help="earlier run directories to compare with")
    ro.add_argument("--out", default=None, help="write readout.json and readout.md here (default: the run directory)")
    ro.add_argument("--catalog", default=None, help="the defect catalog whose open classes the readout lists "
                                                    "(default: roles/verifier/defects.jsonl)")
    ro.set_defaults(func=_readout)

    pa = sub.add_parser("pilot-auto", help="one paste for a paid pilot: wait for the lane, write, relaunch after a lost-call stop")
    pa.add_argument("port", type=int, help="the lane's port on the loopback")
    pa.add_argument("grant", help="the critic's 5090 grant")
    pa.add_argument("deadline", help="the pod's hard deadline, HH:MM UTC")
    pa.add_argument("work", help="work dir holding plan.json, pack/ and pool.jsonl; attempts go in attempt-N/")
    pa.add_argument("--serve-by", default=None, help="HH:MM UTC: give up if the lane hasn't served by then (the pod's no-serve shutdown)")
    pa.set_defaults(func=_pilot_auto)

    pc = sub.add_parser("puzzle-check", help="the kind B solver over reviewed data files and their .models.jsonl sidecars; "
                                             "no fact skeleton may repeat across the files given together")
    pc.add_argument("data", nargs="+", help="data files (<name>.jsonl); each sidecar is <name>.models.jsonl beside it. "
                                            "Give every file in the directory, so repeats across files are caught")
    pc.add_argument("--json", action="store_true", help="print the full report as JSON")
    pc.add_argument("--stamp", action="store_true", help="first write each sidecar line's row_sha256 from the rows as they stand")
    pc.set_defaults(func=_puzzle_check)
    return p


def _puzzle_check(args) -> int:
    if args.stamp:
        for d in args.data:
            print(f"stamped {puzzle_check.stamp(Path(d))} sidecar line(s) for {Path(d).name}")
    reports = puzzle_check.refuse_repeats([puzzle_check.check_file(Path(d)) for d in args.data])
    if args.json:
        print(json.dumps(reports, indent=1, ensure_ascii=False))
    else:
        print("\n".join(puzzle_check.render(r) for r in reports))
    return 0 if all(r["ok"] for r in reports) else 1


def _pilot_auto(args) -> int:
    state = autostart.run(args.port, args.grant, args.deadline, Path(args.work), serve_by=args.serve_by)
    print(json.dumps({k: v for k, v in state.items() if k != "attempts"}))
    for a in state["attempts"]:
        print(f"{a['dir']}: {a['ended_by']}; accepted {a['accepted']} of {a['attempted']}; {a['halted'] or ''}")
    return 0 if state["attempts"] else 3


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
