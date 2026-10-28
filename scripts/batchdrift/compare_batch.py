"""Per-request, per-field comparison and statistical tests for the batching-drift study (DESIGN-v1.md §4, §5).

Joins `calls-{arm}.jsonl` and `generations-{arm}.jsonl` by `call_index == request_index + 1` (never by line
position or completion order), compares `A2` vs `A1`, each batched arm (`B2`, `B4`, `B8`, `B4r`) vs `A1`, and
`B4r` vs `B4` across the four pre-registered field groups:
  1. Pipeline decision outcome (`VERDICT` and `DECIDING` verbatim/coverage status via `compare_verdict` /
     `compare_twin_verdict`; bare first word via `compare_flaw`; `TRAIT` via `compare_trait`; `ANSWER` via
     `compare_answer`).
  2. Auxiliary structured lines (raw `DECIDING`, `ESCALATE`, `CONTRASTED`).
  3. Reasoning text (`reasoning_content` in `generations.jsonl`).
  4. Whole reply (raw `response` + `reasoning_content` + `tokens` + top-5 logprobs).

Computes exact Clopper-Pearson 95% confidence intervals on exact-match rate (k/60) and decision-drift rate (d/60),
and McNemar's exact test on paired key agreement. Detects known-bad inputs (cached prompt tokens > 0, payload or
messages sha256 mismatch, missing/dropped seed, reused rows, `/props` slot/ctx mismatch, out-of-order join mismatch).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "batchdrift",
):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_requests60 import canonical_payload_sha256, file_sha256  # noqa: E402
from replay import (  # noqa: E402
    BATCHDRIFT_EXTRA_ARGS,
    DEFAULT_CRITIC,
    EXPECTED_CARD_KWARGS,
    EXPECTED_CARD_SAMPLING,
    PER_SLOT_CTX,
    REQUESTS60_DEFAULT_PATH,
    REQUESTS60_SHA256,
    REQUIRED_ARMS,
    load_and_verify_requests60,
)
from role_set import assemble, callcache, critic, gates  # noqa: E402

AUX_FIELD_NAMES: tuple[str, ...] = ("DECIDING", "ESCALATE", "CONTRASTED")
ALL_LABELLED_LINE_NAMES: tuple[str, ...] = ("VERDICT", "DECIDING", "ESCALATE", "TRAIT", "CONTRASTED", "ANSWER")


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Exact cumulative binomial probability P(X <= k) for X ~ Binomial(n, p)."""

    if p <= 0.0:
        return 1.0 if k >= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, max(0.0, total))


def _binom_sf(k: int, n: int, p: float) -> float:
    """Exact upper tail P(X >= k) for X ~ Binomial(n, p)."""

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, max(0.0, total))


def clopper_pearson_95(k: int, n: int, alpha: float = 0.05) -> dict:
    """Exact Clopper-Pearson two-sided (1 - alpha) confidence interval and one-sided 95% bounds for k of n."""

    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"invalid (k={k}, n={n}) for Clopper-Pearson interval")
    half = alpha / 2.0

    if k == 0:
        low_2 = 0.0
        low_1 = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2.0
            if _binom_sf(k, n, mid) < half:
                lo = mid
            else:
                hi = mid
        low_2 = (lo + hi) / 2.0

        lo1, hi1 = 0.0, 1.0
        for _ in range(70):
            mid = (lo1 + hi1) / 2.0
            if _binom_sf(k, n, mid) < alpha:
                lo1 = mid
            else:
                hi1 = mid
        low_1 = (lo1 + hi1) / 2.0

    if k == n:
        high_2 = 1.0
        high_1 = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2.0
            if _binom_cdf(k, n, mid) > half:
                lo = mid
            else:
                hi = mid
        high_2 = (lo + hi) / 2.0

        lo1, hi1 = 0.0, 1.0
        for _ in range(70):
            mid = (lo1 + hi1) / 2.0
            if _binom_cdf(k, n, mid) > alpha:
                lo1 = mid
            else:
                hi1 = mid
        high_1 = (lo1 + hi1) / 2.0

    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 6),
        "two_sided_95": [round(low_2, 6), round(high_2, 6)],
        "one_sided_95_upper": round(high_1, 6),
        "one_sided_95_lower": round(low_1, 6),
    }


def mcnemar_exact(agree_ref: list[bool], agree_arm: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired key agreement (reference arm vs compared arm)."""

    if len(agree_ref) != len(agree_arm):
        raise ValueError(f"paired lists must have equal length: {len(agree_ref)} != {len(agree_arm)}")
    n11 = sum(1 for a, b in zip(agree_ref, agree_arm) if a and b)
    n10 = sum(1 for a, b in zip(agree_ref, agree_arm) if a and not b)
    n01 = sum(1 for a, b in zip(agree_ref, agree_arm) if not a and b)
    n00 = sum(1 for a, b in zip(agree_ref, agree_arm) if not a and not b)
    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
        degenerate = True
    else:
        k_min = min(n10, n01)
        tail = sum(math.comb(discordant, i) * (0.5**discordant) for i in range(k_min + 1))
        p_val = min(1.0, 2.0 * tail)
        degenerate = False
    return {
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "discordant": discordant,
        "p_value": round(p_val, 6),
        "degenerate": degenerate,
    }


def extract_labelled_lines(text: str) -> dict[str, str]:
    """Extract raw labelled lines (VERDICT, DECIDING, ESCALATE, TRAIT, CONTRASTED, ANSWER) from a critic reply."""

    out: dict[str, str] = {}
    for name in ALL_LABELLED_LINE_NAMES:
        m = re.search(rf"(?im)^\s*\**\s*{name}\s*\**\s*:\s*(.*)$", text)
        if m:
            out[name] = m.group(1).strip()
    return out


def evaluate_group1(req: dict, reply: str, g: gates.Gates) -> dict:
    """Evaluate Group 1 (pipeline decision outcome) for one request using role-set's critic.py functions."""

    check = req.get("check")
    material = req.get("material") or ""
    content_kind = req.get("content_kind")

    if check == "verdict":
        key_verdict = req.get("key_verdict") or "supported"
        key_deciding = list(req.get("key_deciding") or [])
        cv_ev = critic.compare_verdict(
            g,
            reply,
            key_verdict,
            key_deciding,
            material,
            check_evidence=True,
            content_kind=content_kind,
        )
        cv_no_ev = critic.compare_verdict(
            g,
            reply,
            key_verdict,
            key_deciding,
            material,
            check_evidence=False,
            content_kind=content_kind,
        )
        return {
            "check": "verdict",
            "verdict": cv_ev.critic_verdict,
            "deciding_resolved": cv_ev.critic_deciding,
            "verbatim_ok": cv_no_ev.code != "VOID",
            "code_with_evidence": cv_ev.code,
            "ok_with_evidence": cv_ev.ok,
            "code_verdict_only": cv_no_ev.code,
            "ok_verdict_only": cv_no_ev.ok,
            "agrees_with_key": cv_ev.ok,
        }

    if check == "twin_verdict":
        key_verdict = req.get("key_verdict") or "supported"
        tv = critic.compare_twin_verdict(g, reply, key_verdict)
        cv_v = critic.compare_verdict(
            g,
            reply,
            key_verdict,
            [],
            material,
            check_evidence=False,
            content_kind=content_kind,
        )
        return {
            "check": "twin_verdict",
            "verdict": tv.critic_verdict,
            "deciding_resolved": cv_v.critic_deciding,
            "verbatim_ok": cv_v.code != "VOID",
            "code": tv.code,
            "ok": tv.ok,
            "agrees_with_key": tv.ok,
        }

    if check == "flaw":
        key_flaw = req.get("key_flaw") or "none"
        cf = critic.compare_flaw(reply, key_flaw)
        word = reply.strip().split()[0].strip(".,:;'\"*").lower() if reply.strip() else ""
        return {
            "check": "flaw",
            "flaw_word": word,
            "code": cf.code,
            "ok": cf.ok,
            "agrees_with_key": cf.ok,
        }

    if check == "trait":
        key_trait = req.get("key_trait") or req.get("lesson") or "T10"
        ct = critic.compare_trait(reply, key_trait)
        return {
            "check": "trait",
            "trait": critic.trait_named(reply),
            "code": ct.code,
            "ok": ct.ok,
            "agrees_with_key": ct.ok,
        }

    if check == "answer":
        key_answer = req.get("key_answer") or ""
        ca = critic.compare_answer(reply, key_answer)
        lines = critic.ANSWER_LINE.findall(reply)
        closed = assemble.closed_answer(lines[-1]) if lines else None
        return {
            "check": "answer",
            "answer_raw": lines[-1] if lines else None,
            "closed_answer": closed,
            "code": ca.code,
            "ok": ca.ok,
            "agrees_with_key": ca.ok,
        }

    raise ValueError(f"unsupported check type {check!r} at request_index={req.get('request_index')}")


def group1_differs(req: dict, g1_ref: dict, g1_arm: dict, g: gates.Gates) -> tuple[bool, list[str]]:
    """Determine whether Group 1 (pipeline decision outcome) differs between reference and compared arm."""

    check = req.get("check")
    reasons: list[str] = []

    if check == "verdict":
        if g1_ref["verdict"] != g1_arm["verdict"]:
            reasons.append("VERDICT")
        if g1_ref["verbatim_ok"] != g1_arm["verbatim_ok"]:
            reasons.append("DECIDING_VERBATIM")
        if g1_ref["code_with_evidence"] != g1_arm["code_with_evidence"] or g1_ref["code_verdict_only"] != g1_arm["code_verdict_only"]:
            reasons.append("DECIDING_COVERAGE")
        elif (
            g1_ref["verbatim_ok"]
            and g1_arm["verbatim_ok"]
            and g1_ref["verdict"] == g1_arm["verdict"]
            and g1_ref["verdict"] != "cannot_tell"
        ):
            mismatch_fn = getattr(g.module, "critic_mismatch", None)
            material = req.get("material") or ""
            if mismatch_fn is not None and mismatch_fn(
                g1_ref.get("deciding_resolved") or [],
                g1_arm.get("deciding_resolved") or [],
                material,
            ):
                reasons.append("DECIDING_COVERAGE")
        return bool(reasons), reasons

    if check == "twin_verdict":
        if g1_ref["verdict"] != g1_arm["verdict"]:
            reasons.append("VERDICT")
        if g1_ref["code"] != g1_arm["code"]:
            reasons.append("TWIN_VERDICT_CODE")
        if g1_ref["verbatim_ok"] != g1_arm["verbatim_ok"]:
            reasons.append("DECIDING_VERBATIM")
        return bool(reasons), reasons

    if check == "flaw":
        if g1_ref["flaw_word"] != g1_arm["flaw_word"] or g1_ref["code"] != g1_arm["code"]:
            reasons.append("FLAW")
        return bool(reasons), reasons

    if check == "trait":
        if g1_ref["trait"] != g1_arm["trait"] or g1_ref["code"] != g1_arm["code"]:
            reasons.append("TRAIT")
        return bool(reasons), reasons

    if check == "answer":
        c_ref, c_arm = g1_ref.get("closed_answer"), g1_arm.get("closed_answer")
        same = bool(c_ref is not None and c_arm is not None and critic.same_answer(c_ref, c_arm))
        if g1_ref["code"] != g1_arm["code"] or not same:
            reasons.append("ANSWER")
        return bool(reasons), reasons

    raise ValueError(f"unsupported check {check!r}")


def compare_logprobs(
    tokens_ref: list[str],
    top_ref: list[list],
    tokens_arm: list[str],
    top_arm: list[list],
) -> dict:
    """Compare token streams and top-5 logprobs per step (DESIGN-v1.md §4 lines 193-194)."""

    first_diff: int | None = None
    for i, (ta, tb) in enumerate(zip(tokens_ref, tokens_arm)):
        if ta != tb:
            first_diff = i
            break
    if first_diff is None and len(tokens_ref) != len(tokens_arm):
        first_diff = min(len(tokens_ref), len(tokens_arm))

    prefix_len = first_diff if first_diff is not None else min(len(tokens_ref), len(tokens_arm))
    max_abs_diff = 0.0
    for step_idx in range( min(prefix_len, len(top_ref), len(top_arm)) ):
        step_a = {item[0]: float(item[1]) for item in (top_ref[step_idx] or []) if isinstance(item, (list, tuple)) and len(item) >= 2}
        step_b = {item[0]: float(item[1]) for item in (top_arm[step_idx] or []) if isinstance(item, (list, tuple)) and len(item) >= 2}
        shared_toks = set(step_a) & set(step_b)
        for tok in shared_toks:
            diff = abs(step_a[tok] - step_b[tok])
            if diff > max_abs_diff:
                max_abs_diff = diff
    return {
        "tokens_equal": first_diff is None,
        "first_token_diff_index": first_diff,
        "max_abs_logprob_diff": round(max_abs_diff, 6),
    }


def _cached_tokens_of(gen_row: dict) -> int:
    timings = gen_row.get("timings") if isinstance(gen_row.get("timings"), dict) else {}
    usage = gen_row.get("usage") if isinstance(gen_row.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return max(
        int(timings.get("cache_n") or 0),
        int(details.get("cached_tokens") or 0),
        int(usage.get("cached_tokens") or 0),
    )


def load_and_validate_arm(
    *,
    arm_name: str,
    calls_path: Path,
    generations_path: Path,
    requests: list[dict],
    expected_parallel: int | None = None,
    props_slots: dict | None = None,
    extra_args: list[str] | None = None,
    stub_requests_path: Path | None = None,
) -> dict:
    """Load an arm's calls and generations journals, join by call_index == request_index + 1, and detect any known-bad inputs."""

    void_reasons: list[str] = []
    calls_rows = [json.loads(ln) for ln in calls_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    gen_rows = [json.loads(ln) for ln in generations_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    non_result_gens = [g for g in gen_rows if g.get("kind") != "result"]
    if non_result_gens:
        void_reasons.append(f"non_result_generations:{len(non_result_gens)}")

    result_gens = [g for g in gen_rows if g.get("kind") == "result"]
    gen_by_call_index: dict[int, dict] = {}
    duplicate_call_indices: list[int] = []
    invalid_call_indices: list = []
    out_of_order_generations = 0

    for line_idx, g in enumerate(result_gens):
        cidx = g.get("call_index")
        if not isinstance(cidx, int) or isinstance(cidx, bool) or cidx < 1:
            invalid_call_indices.append(cidx)
            continue
        if cidx != line_idx + 1:
            out_of_order_generations += 1
        if cidx in gen_by_call_index:
            duplicate_call_indices.append(cidx)
        gen_by_call_index[cidx] = g

    if invalid_call_indices:
        void_reasons.append(f"invalid_call_index:{invalid_call_indices}")
    if duplicate_call_indices:
        void_reasons.append(f"duplicate_call_index:{duplicate_call_indices}")

    calls_by_req_index: dict[int, dict] = {}
    duplicate_req_indices: list[int] = []
    invalid_req_indices: list = []
    out_of_order_calls = 0

    for line_idx, c in enumerate(calls_rows):
        ridx = c.get("request_index")
        if not isinstance(ridx, int) or isinstance(ridx, bool) or ridx < 0:
            invalid_req_indices.append(ridx)
            continue
        if ridx != line_idx:
            out_of_order_calls += 1
        if ridx in calls_by_req_index:
            duplicate_req_indices.append(ridx)
        calls_by_req_index[ridx] = c

    if invalid_req_indices:
        void_reasons.append(f"invalid_request_index:{invalid_req_indices}")
    if duplicate_req_indices:
        void_reasons.append(f"duplicate_request_index:{duplicate_req_indices}")

    missing_indices = [r["request_index"] for r in requests if r["request_index"] not in calls_by_req_index]
    if missing_indices:
        void_reasons.append(f"missing_request_indices:{missing_indices}")

    missing_call_indices = [r["request_index"] + 1 for r in requests if (r["request_index"] + 1) not in gen_by_call_index]
    if missing_call_indices:
        void_reasons.append(f"missing_generation_call_indices:{missing_call_indices}")

    payload_mismatches: list[int] = []
    seed_errors: list[int] = []
    reused_rows: list[int] = []
    cached_token_rows: list[int] = []
    join_text_mismatches: list[int] = []
    joined_by_index: dict[int, dict] = {}

    for req in requests:
        idx = req["request_index"]
        c_row = calls_by_req_index.get(idx)
        g_row = gen_by_call_index.get(idx + 1)
        if c_row is None or g_row is None:
            continue

        if c_row.get("reused"):
            reused_rows.append(idx)

        seed = c_row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed != req["seed"]:
            seed_errors.append(idx)

        comp = c_row.get("completion") if isinstance(c_row.get("completion"), dict) else {}
        c_msg_sha = c_row.get("messages_sha256") or comp.get("messages_sha256")
        c_pay_sha = c_row.get("payload_sha256")
        if (
            c_row.get("key") != req["key"]
            or c_msg_sha != req["messages_sha256"]
            or (c_pay_sha is not None and c_pay_sha != req["payload_sha256"])
        ):
            payload_mismatches.append(idx)
        elif isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
            if canonical_payload_sha256(req["messages"], seed) != req["payload_sha256"]:
                payload_mismatches.append(idx)

        c_text = comp.get("text") if isinstance(c_row.get("completion"), dict) else str(c_row.get("completion") or "")
        g_text = g_row.get("response") or ""
        if (c_text or "") != g_text:
            join_text_mismatches.append(idx)

        if _cached_tokens_of(g_row) > 0:
            cached_token_rows.append(idx)

        joined_by_index[idx] = {
            "request": req,
            "call": c_row,
            "generation": g_row,
            "text": c_text or "",
            "reasoning_content": g_row.get("reasoning_content") or "",
            "tokens": list(g_row.get("tokens") or []),
            "top": list(g_row.get("top") or []),
        }

    if reused_rows:
        void_reasons.append(f"reused_rows:{reused_rows}")
    if seed_errors:
        void_reasons.append(f"seed_errors:{seed_errors}")
    if payload_mismatches:
        void_reasons.append(f"payload_mismatches:{payload_mismatches}")
    if join_text_mismatches:
        void_reasons.append(f"join_text_mismatches:{join_text_mismatches}")
    if cached_token_rows:
        void_reasons.append(f"cached_prompt_tokens:{cached_token_rows}")

    if props_slots is not None:
        if expected_parallel is not None and props_slots.get("total_slots") != expected_parallel:
            void_reasons.append(f"props_total_slots:{props_slots.get('total_slots')}!={expected_parallel}")
        if props_slots.get("n_ctx_per_slot") != PER_SLOT_CTX:
            void_reasons.append(f"props_n_ctx_per_slot:{props_slots.get('n_ctx_per_slot')}!={PER_SLOT_CTX}")
    if extra_args is not None and list(extra_args) != list(BATCHDRIFT_EXTRA_ARGS):
        void_reasons.append(f"extra_args_mismatch:{extra_args}")

    stub_dropped_seeds: list[int] = []
    stub_sampling_errors: list[dict] = []
    if stub_requests_path is not None and stub_requests_path.is_file():
        stub_bodies = [json.loads(ln) for ln in stub_requests_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(stub_bodies) != len(requests):
            void_reasons.append(f"stub_requests_count:{len(stub_bodies)}!={len(requests)}")
        for idx, (b, req) in enumerate(zip(stub_bodies, requests)):
            s = b.get("seed") if "seed" in b else None
            if s is None or isinstance(s, bool) or not isinstance(s, int) or s != req["seed"]:
                stub_dropped_seeds.append(idx)
            for k, want_v in EXPECTED_CARD_SAMPLING.items():
                if b.get(k) != want_v:
                    stub_sampling_errors.append({"request_index": idx, "field": k, "got": b.get(k), "want": want_v})
            if (b.get("chat_template_kwargs") or {}) != EXPECTED_CARD_KWARGS:
                stub_sampling_errors.append({
                    "request_index": idx,
                    "field": "chat_template_kwargs",
                    "got": b.get("chat_template_kwargs"),
                    "want": EXPECTED_CARD_KWARGS,
                })
        if stub_dropped_seeds:
            void_reasons.append(f"stub_dropped_seeds:{stub_dropped_seeds}")
        if stub_sampling_errors:
            void_reasons.append(f"stub_sampling_errors:{len(stub_sampling_errors)}")

    return {
        "arm": arm_name,
        "valid": len(void_reasons) == 0,
        "void_reasons": void_reasons,
        "joined_count": len(joined_by_index),
        "out_of_order_calls": out_of_order_calls,
        "out_of_order_generations": out_of_order_generations,
        "payload_mismatches": payload_mismatches,
        "seed_errors": seed_errors,
        "stub_dropped_seeds": stub_dropped_seeds,
        "stub_sampling_errors": stub_sampling_errors,
        "reused_rows": reused_rows,
        "cached_token_rows": cached_token_rows,
        "join_text_mismatches": join_text_mismatches,
        "joined_by_index": joined_by_index,
    }


def compare_two_arms(
    ref_arm: dict,
    comp_arm: dict,
    requests: list[dict],
    g: gates.Gates | None = None,
) -> dict:
    """Compare `comp_arm` against `ref_arm` per request and field across all four groups (DESIGN-v1.md §4)."""

    if g is None:
        g = gates.load_gates()

    ref_map = ref_arm["joined_by_index"]
    comp_map = comp_arm["joined_by_index"]

    per_request: list[dict] = []
    group1_drifts = 0
    group2_drifts = 0
    group3_drifts = 0
    group4_drifts = 0
    exact_matches = 0
    verdict_drifts = 0
    flaw_drifts = 0
    aux_field_counts = {k: 0 for k in AUX_FIELD_NAMES}
    agree_ref_list: list[bool] = []
    agree_comp_list: list[bool] = []
    max_logprob_diff_identical = 0.0

    for req in requests:
        idx = req["request_index"]
        if idx not in ref_map or idx not in comp_map:
            continue
        r_item = ref_map[idx]
        c_item = comp_map[idx]

        g1_ref = evaluate_group1(req, r_item["text"], g)
        g1_comp = evaluate_group1(req, c_item["text"], g)
        g1_changed, g1_reasons = group1_differs(req, g1_ref, g1_comp, g)

        lines_ref = extract_labelled_lines(r_item["text"])
        lines_comp = extract_labelled_lines(c_item["text"])
        aux_changed_fields = [
            k for k in AUX_FIELD_NAMES if lines_ref.get(k) != lines_comp.get(k)
        ]
        g2_changed = len(aux_changed_fields) > 0

        g3_changed = r_item["reasoning_content"] != c_item["reasoning_content"]

        lp_info = compare_logprobs(
            r_item["tokens"],
            r_item["top"],
            c_item["tokens"],
            c_item["top"],
        )
        text_changed = r_item["text"] != c_item["text"]
        g4_changed = text_changed or g3_changed or (not lp_info["tokens_equal"])

        if g1_changed:
            group1_drifts += 1
            if "VERDICT" in g1_reasons:
                verdict_drifts += 1
            if "FLAW" in g1_reasons:
                flaw_drifts += 1
        if g2_changed:
            group2_drifts += 1
            for k in aux_changed_fields:
                aux_field_counts[k] += 1
        if g3_changed:
            group3_drifts += 1
        if g4_changed:
            group4_drifts += 1
        else:
            exact_matches += 1
            if lp_info["max_abs_logprob_diff"] > max_logprob_diff_identical:
                max_logprob_diff_identical = lp_info["max_abs_logprob_diff"]

        agree_ref_list.append(bool(g1_ref["agrees_with_key"]))
        agree_comp_list.append(bool(g1_comp["agrees_with_key"]))

        per_request.append({
            "request_index": idx,
            "key": req["key"],
            "slot": req["slot"],
            "check": req["check"],
            "exact_match": not g4_changed,
            "group1_decision_changed": g1_changed,
            "group1_reasons": g1_reasons,
            "group2_auxiliary_changed": g2_changed,
            "group2_fields": aux_changed_fields,
            "group3_reasoning_changed": g3_changed,
            "group4_whole_reply_changed": g4_changed,
            "response_text_changed": text_changed,
            "tokens_equal": lp_info["tokens_equal"],
            "first_token_diff_index": lp_info["first_token_diff_index"],
            "max_abs_logprob_diff": lp_info["max_abs_logprob_diff"],
            "ref_agrees_with_key": bool(g1_ref["agrees_with_key"]),
            "arm_agrees_with_key": bool(g1_comp["agrees_with_key"]),
        })

    n = len(per_request)
    cp_exact = clopper_pearson_95(exact_matches, n) if n > 0 else None
    cp_decision = clopper_pearson_95(group1_drifts, n) if n > 0 else None
    mcnemar = mcnemar_exact(agree_ref_list, agree_comp_list) if n > 0 else None

    return {
        "ref_arm": ref_arm["arm"],
        "arm": comp_arm["arm"],
        "compared": n,
        "exact_matches": exact_matches,
        "group1_decision_drifts": group1_drifts,
        "verdict_drifts": verdict_drifts,
        "flaw_drifts": flaw_drifts,
        "group2_auxiliary_drifts": group2_drifts,
        "group2_field_changes": aux_field_counts,
        "group3_reasoning_drifts": group3_drifts,
        "group4_whole_reply_drifts": group4_drifts,
        "max_abs_logprob_diff_identical_tokens": round(max_logprob_diff_identical, 6),
        "clopper_pearson_exact_match": cp_exact,
        "clopper_pearson_decision_drift": cp_decision,
        "mcnemar_key_agreement": mcnemar,
        "drifts": [r for r in per_request if not r["exact_match"]],
        "per_request": per_request,
    }


def classify_study(
    *,
    loaded_arms: dict[str, dict],
    requests: list[dict],
    g: gates.Gates | None = None,
) -> dict:
    """Run all pre-registered comparisons (A2 vs A1, each arm vs A1, B4r vs B4, A1 vs P1r) and classify each arm."""

    if g is None:
        g = gates.load_gates()

    if "A1" not in loaded_arms:
        raise RuntimeError("REFUSED: reference arm A1 is missing")
    a1 = loaded_arms["A1"]

    # Reference check: A1 against P1r recorded completions in requests-60.jsonl (DESIGN-v1.md §3 step 3, §4 line 235).
    a1_vs_p1r_exact = 0
    a1_vs_p1r_checked = 0
    for req in requests:
        idx = req["request_index"]
        if idx in a1["joined_by_index"] and req.get("p1r_completion_text") is not None:
            a1_vs_p1r_checked += 1
            if a1["joined_by_index"][idx]["text"] == req["p1r_completion_text"]:
                a1_vs_p1r_exact += 1

    vs_a1: dict[str, dict] = {}
    for arm_name, arm_data in loaded_arms.items():
        if arm_name == "A1":
            continue
        vs_a1[arm_name] = compare_two_arms(a1, arm_data, requests, g)

    a2_comp = vs_a1.get("A2")
    a2_valid = bool(loaded_arms.get("A2", {}).get("valid")) and bool(a1.get("valid"))
    a2_clean = bool(
        a2_valid
        and a2_comp is not None
        and a2_comp["compared"] == len(requests)
        and a2_comp["group4_whole_reply_drifts"] == 0
        and a2_comp["group1_decision_drifts"] == 0
        and a2_comp["group2_auxiliary_drifts"] == 0
        and a2_comp["group3_reasoning_drifts"] == 0
    )
    noise_floor_decision = 0 if a2_clean else (a2_comp["group1_decision_drifts"] if a2_comp else 0)
    noise_floor_whole = 0 if a2_clean else (a2_comp["group4_whole_reply_drifts"] if a2_comp else 0)

    b4r_vs_b4 = None
    b4_reproducible = None
    if "B4" in loaded_arms and "B4r" in loaded_arms:
        b4r_vs_b4 = compare_two_arms(loaded_arms["B4"], loaded_arms["B4r"], requests, g)
        b4_reproducible = bool(
            loaded_arms["B4"]["valid"]
            and loaded_arms["B4r"]["valid"]
            and b4r_vs_b4["compared"] == len(requests)
            and b4r_vs_b4["group4_whole_reply_drifts"] == 0
        )

    arm_tiers: dict[str, str] = {"A1": "reference" if a1["valid"] and a1["joined_count"] == len(requests) else "void"}
    for arm_name, arm_data in loaded_arms.items():
        if arm_name == "A1":
            continue
        comp = vs_a1[arm_name]
        if not a1["valid"] or not arm_data["valid"] or comp["compared"] != len(requests):
            arm_tiers[arm_name] = "void"
            continue
        if comp["group1_decision_drifts"] > noise_floor_decision:
            arm_tiers[arm_name] = "not-safe"
            continue
        is_width4 = arm_name in ("B4", "B4r")
        width4_ok = (b4_reproducible is not False) if is_width4 else True
        if a2_clean and comp["group4_whole_reply_drifts"] == 0 and width4_ok:
            arm_tiers[arm_name] = "pinned-safe"
        else:
            arm_tiers[arm_name] = "decision-safe-only"

    arm_validity = {
        name: {
            "valid": d["valid"],
            "void_reasons": d["void_reasons"],
            "joined_count": d["joined_count"],
            "out_of_order_calls": d["out_of_order_calls"],
            "out_of_order_generations": d["out_of_order_generations"],
            "payload_mismatches": d["payload_mismatches"],
            "seed_errors": d["seed_errors"],
            "stub_dropped_seeds": d["stub_dropped_seeds"],
            "reused_rows": d["reused_rows"],
            "cached_token_rows": d["cached_token_rows"],
        }
        for name, d in loaded_arms.items()
    }

    return {
        "requests_count": len(requests),
        "validity_gate_a2": {
            "a2_present": "A2" in loaded_arms,
            "a2_clean": a2_clean,
            "noise_floor_decision_drifts": noise_floor_decision,
            "noise_floor_whole_reply_drifts": noise_floor_whole,
        },
        "a1_vs_p1r_reference": {
            "checked": a1_vs_p1r_checked,
            "exact_matches": a1_vs_p1r_exact,
        },
        "b4_reproducibility": {
            "b4_and_b4r_present": b4r_vs_b4 is not None,
            "reproducible": b4_reproducible,
            "comparison": b4r_vs_b4,
        },
        "arm_validity": arm_validity,
        "arm_tiers": arm_tiers,
        "comparisons_vs_a1": vs_a1,
    }


def compare_study_dir(
    out_dir: Path,
    *,
    requests_path: Path = REQUESTS60_DEFAULT_PATH,
    critic_model: str = DEFAULT_CRITIC,
    expected_requests_sha256: str | None = REQUESTS60_SHA256,
    expected_requests_count: int | None = 60,
) -> dict:
    """Load all arms present in `out_dir` and run `classify_study`."""

    requests, _ = load_and_verify_requests60(
        requests_path,
        expected_sha256=expected_requests_sha256,
        expected_count=expected_requests_count,
    )
    loaded: dict[str, dict] = {}
    for arm_name, parallel in REQUIRED_ARMS:
        c_path = out_dir / f"calls-{arm_name}-{critic_model}.jsonl"
        g_path = out_dir / f"generations-{arm_name}-{critic_model}.jsonl"
        if not c_path.is_file() or not g_path.is_file():
            continue
        stub_req_path = out_dir / f"stub-requests-{arm_name}-{critic_model}.jsonl"
        loaded[arm_name] = load_and_validate_arm(
            arm_name=arm_name,
            calls_path=c_path,
            generations_path=g_path,
            requests=requests,
            expected_parallel=parallel,
            stub_requests_path=stub_req_path if stub_req_path.is_file() else None,
        )
    return classify_study(loaded_arms=loaded, requests=requests)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare batching-drift arms against A1 and validate study gates.")
    ap.add_argument("--requests", type=Path, default=REQUESTS60_DEFAULT_PATH, help="Path to requests-60.jsonl")
    ap.add_argument("--dir", type=Path, default=None, help="Directory holding calls-{arm}-*.jsonl and generations-{arm}-*.jsonl")
    ap.add_argument(
        "--arm",
        action="append",
        default=[],
        help="Explicit arm specification: <ARM>=<calls.jsonl>:<generations.jsonl>",
    )
    ap.add_argument("--critic-model", default=DEFAULT_CRITIC)
    ap.add_argument("--no-pin-check", action="store_true", help="Skip requests-60.jsonl sha256 check (for test fixtures)")
    ap.add_argument("--out", type=Path, default=None, help="Optional file to write the JSON report")
    args = ap.parse_args(argv)

    expect_sha = None if args.no_pin_check else REQUESTS60_SHA256
    expect_cnt = None if args.no_pin_check else 60
    if args.dir is not None:
        report = compare_study_dir(
            args.dir,
            requests_path=args.requests,
            critic_model=args.critic_model,
            expected_requests_sha256=expect_sha,
            expected_requests_count=expect_cnt,
        )
    elif args.arm:
        requests, _ = load_and_verify_requests60(
            args.requests,
            expected_sha256=expect_sha,
            expected_count=expect_cnt,
        )
        arm_parallel = dict(REQUIRED_ARMS)
        loaded: dict[str, dict] = {}
        for spec in args.arm:
            name, rest = spec.split("=", 1)
            sep_idx = rest.find(":", 2) if len(rest) >= 2 and rest[1] == ":" else rest.find(":")
            if sep_idx < 0:
                raise RuntimeError(f"invalid --arm spec {spec!r}; expected <ARM>=<calls.jsonl>:<generations.jsonl>")
            c_str, g_str = rest[:sep_idx], rest[sep_idx + 1 :]
            loaded[name] = load_and_validate_arm(
                arm_name=name,
                calls_path=Path(c_str),
                generations_path=Path(g_str),
                requests=requests,
                expected_parallel=arm_parallel.get(name),
            )
        report = classify_study(loaded_arms=loaded, requests=requests)
    else:
        raise RuntimeError("pass either --dir <batchdrift_out_dir> or one or more --arm <ARM>=<calls>:<generations>")

    output_json = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.write_text(output_json + "\n", encoding="utf-8", newline="\n")
    print(output_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
