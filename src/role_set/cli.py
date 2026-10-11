"""role-set: plan and cost (no model), write (the runner; paid unless --mock), pack, harvest, ingest, store, critic-control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from role_set import __version__
from role_set.cost import PROFILES, estimate
from role_set.gates import GatesUnavailable, load_gates
from role_set.guards import GuardError, require_local_endpoint
from role_set.harvest import harvest
from role_set.ingest import IngestError, ingest, interview_questions
from role_set.pack import PackError, sync, verify
from role_set.store import counts, open_store, void_round
from role_set.critic_control import FALSE_REFUSAL_CAP, cases_from_rows, load_rows, plant_errors, run_control
from role_set.guards import require_grant, require_heartbeat, require_ollama_digest
from role_set.write import allowed_critics
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


def _write(args: argparse.Namespace) -> int:
    cfg = WriteConfig(
        plan=Path(args.plan), pack=Path(args.pack), out=Path(args.out), endpoint=args.endpoint, model=args.model,
        writers_file=Path(args.writers), critic_endpoint=args.critic_endpoint, critic_model=args.critic_model,
        grant=args.grant, heartbeat=Path(args.heartbeat) if args.heartbeat else None, concurrency=args.concurrency,
        mock=args.mock, no_critic=args.no_critic,
        pool=Path(args.pool) if args.pool else None,
        exemplar_repo=Path(args.exemplar_repo) if args.exemplar_repo else None,
        exemplar_commit=args.exemplar_commit, exemplar_files=tuple(args.exemplar_files or ()),
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
    cases = cases_from_rows(rows)
    planted = plant_errors(cases, args.planted, seed=args.seed)
    log: list = []

    def critic(messages):
        return ollama_chat(endpoint, args.critic_model, messages)

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
    w.add_argument("--endpoint", default=None, help="http://127.0.0.1:<lane port>/v1 (offrig's tunnel)")
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
    w.set_defaults(func=_write)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
