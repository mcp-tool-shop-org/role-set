"""role-set: plan (no model), cost (no model), write (refused until the gates are pinned)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from role_set import __version__
from role_set.cost import PROFILES, estimate
from role_set.guards import GuardError, require_local_endpoint
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
    try:
        require_local_endpoint(args.endpoint)
    except GuardError as e:
        print(e, file=sys.stderr)
        return 2
    print("REFUSED: write is not available until aspire-si's gates package is pinned (aspire-si #85)", file=sys.stderr)
    return 2


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

    w = sub.add_parser("write", help="the runner; paid; refused until the gates are pinned")
    w.add_argument("--plan", required=True)
    w.add_argument("--endpoint", required=True)
    w.add_argument("--model", required=True)
    w.set_defaults(func=_write)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
