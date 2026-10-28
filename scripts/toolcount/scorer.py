"""Scorer and cluster-level statistical analysis for the tool-count cliff study.

Implements Design v1 §§3, 4, 7, 8:
- Reads `upstream_response.choices[0].message.tool_calls` first, falling back to `message.content`.
- Classifies each call into one of 5 primary categories:
  `correct`, `wrong_tool`, `no_call`, `malformed`, `abstention`.
- Computes secondary metrics: argument schema-validity and value-correctness (among correct-tool calls),
  abstention rate on the 30 no-tool items, refusal rate on the 170 answerable items, format failure rate,
  accuracy by position bin (0..5), prompt tokens, output tokens, and wall time.
- Aggregates answerable accuracy to the 85 target-tool clusters (averaging the 2 phrasings per target tool first).
- Runs the paired permutation (sign-flip) test (10^5 Monte Carlo draws, seeded) on the 85 per-target-tool
  mean differences (keeping zero-difference clusters), with Holm step-down adjustment (family alpha = 0.05).
- Runs the paired cluster bootstrap over the 85 target tools for 95% confidence intervals.
- Evaluates pre-registered Decision Rules 1-4 (cliff, ceiling, compass narrowing, length confound, and
  measured rho/MDE floor beside every 'not shown' cell).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from scripts.toolcount import COMPASS_COUNTS, DEFAULT_SEED, STUDY_ID, TOOL_COUNTS
from scripts.toolcount.gen_registry import position_to_bin, write_with_sha256

_TAG_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([a-zA-Z0-9_]+)\{(.*?)\}<tool_call\|>",
    re.DOTALL,
)


def extract_tool_call_from_response(
    upstream_response: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract the first tool call from `upstream_response.choices[0].message.tool_calls`
    with `message.content` as fallback (Design v1 §3, §4).

    Returns a dict with:
      - `status`: `"called"` | `"no_call"` | `"malformed"`
      - `tool_name`: str | None
      - `arguments`: dict[str, Any] | None
      - `raw_arguments`: Any
      - `source`: `"tool_calls"` | `"content_fallback"` | `"none"`
      - `extra_tool_calls_count`: int
      - `error`: str | None
    """
    if not isinstance(upstream_response, dict):
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": None,
            "source": "none",
            "extra_tool_calls_count": 0,
            "error": "missing_or_invalid_upstream_response",
        }

    choices = upstream_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": None,
            "source": "none",
            "extra_tool_calls_count": 0,
            "error": "missing_choices",
        }

    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": None,
            "source": "none",
            "extra_tool_calls_count": 0,
            "error": "missing_message",
        }

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        extra_count = max(0, len(tool_calls) - 1)
        tc0 = tool_calls[0]
        if not isinstance(tc0, dict):
            return {
                "status": "malformed",
                "tool_name": None,
                "arguments": None,
                "raw_arguments": tc0,
                "source": "tool_calls",
                "extra_tool_calls_count": extra_count,
                "error": "non_dict_tool_call",
            }
        fn = tc0.get("function")
        if not isinstance(fn, dict):
            return {
                "status": "malformed",
                "tool_name": None,
                "arguments": None,
                "raw_arguments": tc0,
                "source": "tool_calls",
                "extra_tool_calls_count": extra_count,
                "error": "missing_function_object",
            }
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if not isinstance(name, str) or not name.strip():
            return {
                "status": "malformed",
                "tool_name": None,
                "arguments": None,
                "raw_arguments": raw_args,
                "source": "tool_calls",
                "extra_tool_calls_count": extra_count,
                "error": "empty_or_invalid_tool_name",
            }
        parsed_args, err = _parse_arguments_payload(raw_args)
        if err is not None or parsed_args is None:
            return {
                "status": "malformed",
                "tool_name": name.strip(),
                "arguments": None,
                "raw_arguments": raw_args,
                "source": "tool_calls",
                "extra_tool_calls_count": extra_count,
                "error": err or "invalid_arguments_json",
            }
        return {
            "status": "called",
            "tool_name": name.strip(),
            "arguments": parsed_args,
            "raw_arguments": raw_args,
            "source": "tool_calls",
            "extra_tool_calls_count": extra_count,
            "error": None,
        }

    # Fallback: check message.content
    content = message.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        return {
            "status": "no_call",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": None,
            "source": "none",
            "extra_tool_calls_count": 0,
            "error": None,
        }

    if not isinstance(content, str):
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": content,
            "source": "content_fallback",
            "extra_tool_calls_count": 0,
            "error": "non_string_content",
        }

    text = content.strip()

    # Check `<tool_call>{...}</tool_call>` tag fallback
    tag_match = _TAG_TOOL_CALL_RE.search(text)
    if tag_match:
        try:
            obj = json.loads(tag_match.group(1))
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj["name"].strip():
                args_obj, err = _parse_arguments_payload(obj.get("arguments", {}))
                if err is None and args_obj is not None:
                    return {
                        "status": "called",
                        "tool_name": obj["name"].strip(),
                        "arguments": args_obj,
                        "raw_arguments": obj.get("arguments", {}),
                        "source": "content_fallback",
                        "extra_tool_calls_count": 0,
                        "error": None,
                    }
        except Exception:
            pass
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": text,
            "source": "content_fallback",
            "extra_tool_calls_count": 0,
            "error": "malformed_tool_call_tag",
        }

    # Check Gemma `<|tool_call>call:name{...}<tool_call|>` fallback
    gemma_match = _GEMMA_TOOL_CALL_RE.search(text)
    if gemma_match:
        t_name = gemma_match.group(1).strip()
        inner = "{" + gemma_match.group(2) + "}"
        try:
            args_obj = json.loads(inner)
            if isinstance(args_obj, dict):
                return {
                    "status": "called",
                    "tool_name": t_name,
                    "arguments": args_obj,
                    "raw_arguments": inner,
                    "source": "content_fallback",
                    "extra_tool_calls_count": 0,
                    "error": None,
                }
        except Exception:
            return {
                "status": "malformed",
                "tool_name": t_name,
                "arguments": None,
                "raw_arguments": inner,
                "source": "content_fallback",
                "extra_tool_calls_count": 0,
                "error": "malformed_gemma_tool_call_tag",
            }

    # Check if content is a direct JSON tool call or broken tool-call markup
    if text.startswith("{") or "<tool_call" in text or "<function=" in text or "<|tool_call" in text:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj["name"].strip():
                args_obj, err = _parse_arguments_payload(obj.get("arguments", {}))
                if err is None and args_obj is not None:
                    return {
                        "status": "called",
                        "tool_name": obj["name"].strip(),
                        "arguments": args_obj,
                        "raw_arguments": obj.get("arguments", {}),
                        "source": "content_fallback",
                        "extra_tool_calls_count": 0,
                        "error": None,
                    }
        except Exception:
            pass
        return {
            "status": "malformed",
            "tool_name": None,
            "arguments": None,
            "raw_arguments": text,
            "source": "content_fallback",
            "extra_tool_calls_count": 0,
            "error": "unparseable_content_tool_call",
        }

    # Plain natural-language refusal / abstention with no tool call
    return {
        "status": "no_call",
        "tool_name": None,
        "arguments": None,
        "raw_arguments": None,
        "source": "content_fallback",
        "extra_tool_calls_count": 0,
        "error": None,
    }


def _parse_arguments_payload(raw_args: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Parse tool call arguments from dict or JSON string."""
    if raw_args is None or raw_args == "":
        return {}, None
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception as exc:
            return None, f"json_decode_error: {exc}"
        if not isinstance(parsed, dict):
            return None, "arguments_not_json_object"
        return parsed, None
    return None, f"unexpected_arguments_type:{type(raw_args).__name__}"


def _matches_json_type(val: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(val, str)
    if expected_type == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected_type == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if expected_type == "boolean":
        return isinstance(val, bool)
    if expected_type == "object":
        return isinstance(val, dict)
    if expected_type == "array":
        return isinstance(val, list)
    return True


def check_arguments_grade(
    arguments: dict[str, Any] | None,
    input_schema: dict[str, Any],
    expected_required_args: dict[str, Any],
) -> tuple[bool, bool]:
    """Return `(schema_valid, value_correct)` for a tool call's parsed arguments."""
    if not isinstance(arguments, dict):
        return False, False

    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    additional_ok = input_schema.get("additionalProperties", True)

    for req_key in required:
        if req_key not in arguments:
            return False, False

    if additional_ok is False:
        for key in arguments:
            if key not in props:
                return False, False

    for key, val in arguments.items():
        if key in props:
            expected_t = props[key].get("type")
            if expected_t and not _matches_json_type(val, expected_t):
                return False, False

    # Schema is valid; now check exact value match on required_args
    for req_key, exp_val in expected_required_args.items():
        if arguments.get(req_key) != exp_val:
            return True, False

    return True, True


def score_single_call(
    call_record: dict[str, Any],
    item: dict[str, Any],
    tool_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Grade one call receipt against its item key and tool schema.

    Primary classification is one of:
      `correct` | `wrong_tool` | `no_call` | `malformed` | `abstention`
    """
    upstream = call_record.get("upstream_response")
    extracted = extract_tool_call_from_response(upstream)
    status = extracted["status"]
    called_tool = extracted["tool_name"]
    parsed_args = extracted["arguments"]
    is_abstention = bool(item.get("is_abstention", False))
    expected_tool = item["key"]["tool"]
    expected_args = item["key"].get("required_args") or {}

    if status == "malformed":
        classification = "malformed"
        exact_tool_correct: int | None = None if is_abstention else 0
        abstention_correct: int | None = 0 if is_abstention else None
        refused_on_answerable = 0
        schema_valid = False
        value_correct = False
    elif status == "no_call":
        if is_abstention:
            classification = "abstention"
            exact_tool_correct = None
            abstention_correct = 1
            refused_on_answerable = 0
        else:
            classification = "no_call"
            exact_tool_correct = 0
            abstention_correct = None
            refused_on_answerable = 1
        schema_valid = False
        value_correct = False
    else:
        # status == "called"
        if is_abstention:
            classification = "wrong_tool"
            exact_tool_correct = None
            abstention_correct = 0
            refused_on_answerable = 0
            schema_valid = False
            value_correct = False
        elif called_tool == expected_tool:
            classification = "correct"
            exact_tool_correct = 1
            abstention_correct = None
            refused_on_answerable = 0
            target_schema = tool_by_name[expected_tool]["input_schema"]
            schema_valid, value_correct = check_arguments_grade(
                arguments=parsed_args,
                input_schema=target_schema,
                expected_required_args=expected_args,
            )
        else:
            classification = "wrong_tool"
            exact_tool_correct = 0
            abstention_correct = None
            refused_on_answerable = 0
            schema_valid = False
            value_correct = False

    target_pos = call_record.get("target_position")
    tool_order = call_record.get("tool_list_order") or []
    pos_bin = call_record.get("position_bin")
    if pos_bin is None and target_pos is not None and len(tool_order) > 0:
        pos_bin = position_to_bin(int(target_pos), len(tool_order))

    return {
        "item_id": item["item_id"],
        "cluster_id": item["cluster_id"],
        "phrasing_index": item.get("phrasing_index", 0),
        "is_abstention": is_abstention,
        "half": item.get("half"),
        "arm": call_record.get("arm"),
        "cell": call_record.get("cell"),
        "N": call_record.get("N"),
        "classification": classification,
        "exact_tool_correct": exact_tool_correct,
        "abstention_correct": abstention_correct,
        "refused_on_answerable": refused_on_answerable,
        "format_failure": 1 if classification == "malformed" else 0,
        "schema_valid": schema_valid,
        "value_correct": value_correct,
        "expected_tool": expected_tool,
        "called_tool": called_tool,
        "parsed_arguments": parsed_args,
        "extraction_source": extracted["source"],
        "extra_tool_calls_count": extracted["extra_tool_calls_count"],
        "extraction_error": extracted["error"],
        "target_position": target_pos,
        "position_bin": pos_bin,
        "prompt_tokens": call_record.get("prompt_tokens", 0),
        "output_tokens": call_record.get("output_tokens", 0),
        "wall_time_s": call_record.get("wall_time_s", 0.0),
    }


def compute_cluster_means(
    scored_calls: list[dict[str, Any]],
) -> dict[str, float]:
    """Average the 2 phrasings per target tool first (Design §4: unit is the target tool cluster)."""
    by_cluster: dict[str, list[int]] = {}
    for row in scored_calls:
        if row["is_abstention"] or row["exact_tool_correct"] is None:
            continue
        cid = row["cluster_id"]
        by_cluster.setdefault(cid, []).append(int(row["exact_tool_correct"]))

    return {
        cid: sum(vals) / len(vals)
        for cid, vals in sorted(by_cluster.items())
        if vals
    }


def estimate_intracluster_rho(scored_calls: list[dict[str, Any]]) -> float:
    """Estimate intracluster correlation rho between phrasing 0 and phrasing 1 across target tools."""
    by_cluster: dict[str, dict[int, int]] = {}
    for row in scored_calls:
        if row["is_abstention"] or row["exact_tool_correct"] is None:
            continue
        cid = row["cluster_id"]
        p_idx = int(row.get("phrasing_index", 0))
        by_cluster.setdefault(cid, {})[p_idx] = int(row["exact_tool_correct"])

    pairs = [
        (d[0], d[1])
        for d in by_cluster.values()
        if 0 in d and 1 in d
    ]
    if len(pairs) < 2:
        return 0.0

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


def estimate_paired_diff_rho_and_discordance(
    calls_a: list[dict[str, Any]],
    calls_b: list[dict[str, Any]],
) -> dict[str, float]:
    """Estimate phrasing-level discordance rate and intracluster correlation rho on paired drops (A - B).

    Design §2 & §4 Rule 4: the unclustered floor 10.2 pts is calibrated at 15% discordance
    (scaling by sqrt(discordance / 0.15)), and the cluster floor scales by sqrt(1 + max(0, rho_diff))
    where rho_diff is the correlation between phrasing 0 and phrasing 1 paired differences
    d_{c,p} = y_{c,p}(A) - y_{c,p}(B).
    """
    by_cluster_a: dict[str, dict[int, int]] = {}
    for row in calls_a:
        if row["is_abstention"] or row["exact_tool_correct"] is None:
            continue
        cid = row["cluster_id"]
        p_idx = int(row.get("phrasing_index", 0))
        by_cluster_a.setdefault(cid, {})[p_idx] = int(row["exact_tool_correct"])

    by_cluster_b: dict[str, dict[int, int]] = {}
    for row in calls_b:
        if row["is_abstention"] or row["exact_tool_correct"] is None:
            continue
        cid = row["cluster_id"]
        p_idx = int(row.get("phrasing_index", 0))
        by_cluster_b.setdefault(cid, {})[p_idx] = int(row["exact_tool_correct"])

    common_cids = sorted(set(by_cluster_a.keys()) & set(by_cluster_b.keys()))
    diff_pairs: list[tuple[int, int]] = []
    discordant_items = 0
    total_paired_items = 0

    for cid in common_cids:
        da = by_cluster_a[cid]
        db = by_cluster_b[cid]
        for p_idx in sorted(set(da.keys()) & set(db.keys())):
            total_paired_items += 1
            if da[p_idx] != db[p_idx]:
                discordant_items += 1
        if 0 in da and 1 in da and 0 in db and 1 in db:
            diff_pairs.append((da[0] - db[0], da[1] - db[1]))

    discordance_rate = (
        discordant_items / total_paired_items if total_paired_items > 0 else 0.15
    )

    if len(diff_pairs) < 2:
        rho_diff = 0.0
    else:
        xs = [p[0] for p in diff_pairs]
        ys = [p[1] for p in diff_pairs]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        denom = math.sqrt(var_x * var_y)
        rho_diff = max(-1.0, min(1.0, cov / denom)) if denom > 1e-12 else 0.0

    deff = 1.0 + max(0.0, rho_diff)
    unclustered_floor_pts = (
        10.2 * math.sqrt(discordance_rate / 0.15) if discordance_rate > 0.0 else 0.0
    )
    cluster_floor_pts = round(unclustered_floor_pts * math.sqrt(deff), 2)
    return {
        "discordance_rate": discordance_rate,
        "paired_diff_rho": rho_diff,
        "deff": deff,
        "unclustered_floor_pts": round(unclustered_floor_pts, 2),
        "measured_cluster_floor_pts": cluster_floor_pts,
    }


def paired_sign_flip_test(
    cluster_means_a: dict[str, float],
    cluster_means_b: dict[str, float],
    n_flips: int = 100_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Paired permutation (sign-flip) test on per-target-tool mean differences (A - B).

    Zero-difference clusters remain in the denominator K (Design v1 §4, review ke).
    """
    common_clusters = sorted(set(cluster_means_a.keys()) & set(cluster_means_b.keys()))
    k = len(common_clusters)
    if k == 0:
        raise ValueError("No overlapping clusters between cell A and cell B")

    diffs = [cluster_means_a[c] - cluster_means_b[c] for c in common_clusters]
    obs_mean_diff = sum(diffs) / k
    abs_obs = abs(obs_mean_diff)
    non_zero_diffs = [d for d in diffs if d != 0.0]

    if abs_obs == 0.0 or not non_zero_diffs:
        return {
            "n_clusters": k,
            "n_nonzero_clusters": len(non_zero_diffs),
            "mean_a": sum(cluster_means_a[c] for c in common_clusters) / k,
            "mean_b": sum(cluster_means_b[c] for c in common_clusters) / k,
            "mean_diff": obs_mean_diff,
            "mean_diff_pts": obs_mean_diff * 100.0,
            "p_value": 1.0,
            "n_flips": n_flips,
        }

    rng = random.Random(f"{seed}:signflip:{k}:{round(obs_mean_diff, 8)}")
    m_nz = len(non_zero_diffs)
    threshold = abs_obs * k - 1e-12
    extreme_count = 0

    # Random bits draw for fast 10^5 Monte Carlo sign flips
    for _ in range(n_flips):
        bits = rng.getrandbits(m_nz)
        signed_sum = 0.0
        for idx, d_val in enumerate(non_zero_diffs):
            if (bits >> idx) & 1:
                signed_sum += d_val
            else:
                signed_sum -= d_val
        if abs(signed_sum) >= threshold:
            extreme_count += 1

    p_val = (extreme_count + 1) / (n_flips + 1)
    return {
        "n_clusters": k,
        "n_nonzero_clusters": m_nz,
        "mean_a": sum(cluster_means_a[c] for c in common_clusters) / k,
        "mean_b": sum(cluster_means_b[c] for c in common_clusters) / k,
        "mean_diff": obs_mean_diff,
        "mean_diff_pts": obs_mean_diff * 100.0,
        "p_value": p_val,
        "n_flips": n_flips,
    }


def paired_cluster_bootstrap_ci(
    cluster_means_a: dict[str, float],
    cluster_means_b: dict[str, float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Paired cluster bootstrap over the K target tools for 95% confidence intervals on (A - B)."""
    common_clusters = sorted(set(cluster_means_a.keys()) & set(cluster_means_b.keys()))
    k = len(common_clusters)
    if k == 0:
        raise ValueError("No overlapping clusters for bootstrap")

    diffs = [cluster_means_a[c] - cluster_means_b[c] for c in common_clusters]
    obs_mean = sum(diffs) / k
    if all(d == diffs[0] for d in diffs):
        return {
            "mean_diff": obs_mean,
            "ci_low": obs_mean,
            "ci_high": obs_mean,
            "ci_low_pts": obs_mean * 100.0,
            "ci_high_pts": obs_mean * 100.0,
        }

    rng = random.Random(f"{seed}:bootstrap:{k}:{round(obs_mean, 8)}")
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample_sum = sum(diffs[rng.randrange(k)] for _ in range(k))
        boot_means.append(sample_sum / k)

    boot_means.sort()
    lo_idx = max(0, min(n_boot - 1, int(math.floor((alpha / 2.0) * n_boot))))
    hi_idx = max(0, min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1))
    ci_low = boot_means[lo_idx]
    ci_high = boot_means[hi_idx]
    return {
        "mean_diff": obs_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_pts": ci_low * 100.0,
        "ci_high_pts": ci_high * 100.0,
    }


def holm_adjust(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Apply Holm step-down adjustment to a family of named p-values."""
    m = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    result: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    still_rejecting = True

    for rank_idx, (name, raw_p) in enumerate(sorted_items):
        multiplier = m - rank_idx
        adj_p = min(1.0, max(running_max, raw_p * multiplier))
        running_max = adj_p
        if adj_p > alpha:
            still_rejecting = False
        result[name] = {
            "raw_p": raw_p,
            "holm_p": adj_p,
            "significant": still_rejecting,
            "rank": rank_idx + 1,
            "family_size": m,
            "alpha": alpha,
        }
    return result


def summarize_cell(scored_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute primary and secondary metrics for a single (model, arm, cell) slice."""
    answerable = [r for r in scored_calls if not r["is_abstention"]]
    abstentions = [r for r in scored_calls if r["is_abstention"]]
    correct_calls = [r for r in answerable if r["classification"] == "correct"]

    ans_total = len(answerable)
    abst_total = len(abstentions)
    all_total = len(scored_calls)

    exact_acc = (
        sum(int(r["exact_tool_correct"] or 0) for r in answerable) / ans_total
        if ans_total
        else 0.0
    )
    cluster_means = compute_cluster_means(scored_calls)
    cluster_acc = (
        sum(cluster_means.values()) / len(cluster_means)
        if cluster_means
        else 0.0
    )

    schema_valid_rate = (
        sum(1 for r in correct_calls if r["schema_valid"]) / len(correct_calls)
        if correct_calls
        else 0.0
    )
    value_correct_rate = (
        sum(1 for r in correct_calls if r["value_correct"]) / len(correct_calls)
        if correct_calls
        else 0.0
    )
    abstention_rate = (
        sum(int(r["abstention_correct"] or 0) for r in abstentions) / abst_total
        if abst_total
        else 0.0
    )
    refusal_on_answerable_rate = (
        sum(int(r["refused_on_answerable"] or 0) for r in answerable) / ans_total
        if ans_total
        else 0.0
    )
    format_failure_rate = (
        sum(int(r["format_failure"] or 0) for r in scored_calls) / all_total
        if all_total
        else 0.0
    )

    # Accuracy by position bin (0..5) on answerable items
    by_bin: dict[str, dict[str, Any]] = {}
    for b in range(6):
        bin_rows = [r for r in answerable if r.get("position_bin") == b]
        b_tot = len(bin_rows)
        b_hits = sum(int(r["exact_tool_correct"] or 0) for r in bin_rows)
        by_bin[str(b)] = {
            "count": b_tot,
            "correct": b_hits,
            "accuracy": b_hits / b_tot if b_tot else None,
        }

    counts_by_class: dict[str, int] = {
        "correct": 0,
        "wrong_tool": 0,
        "no_call": 0,
        "malformed": 0,
        "abstention": 0,
    }
    for r in scored_calls:
        c = r["classification"]
        counts_by_class[c] = counts_by_class.get(c, 0) + 1

    mean_prompt_tokens = (
        sum(float(r.get("prompt_tokens") or 0) for r in scored_calls) / all_total
        if all_total
        else 0.0
    )
    mean_output_tokens = (
        sum(float(r.get("output_tokens") or 0) for r in scored_calls) / all_total
        if all_total
        else 0.0
    )
    mean_wall_s = (
        sum(float(r.get("wall_time_s") or 0.0) for r in scored_calls) / all_total
        if all_total
        else 0.0
    )

    rho = estimate_intracluster_rho(scored_calls)
    deff = 1.0 + max(0.0, rho)
    # Design §2: unclustered floor at 15% discordance is 10.2 points; scaled by sqrt(DEFF)
    floor_pts = 10.2 * math.sqrt(deff)

    return {
        "total_calls": all_total,
        "answerable_calls": ans_total,
        "abstention_calls": abst_total,
        "n_clusters": len(cluster_means),
        "exact_tool_accuracy": exact_acc,
        "cluster_mean_accuracy": cluster_acc,
        "classification_counts": counts_by_class,
        "secondary": {
            "schema_valid_rate_given_correct": schema_valid_rate,
            "value_correct_rate_given_correct": value_correct_rate,
            "abstention_rate_on_no_tool_items": abstention_rate,
            "refusal_rate_on_answerable_items": refusal_on_answerable_rate,
            "format_failure_rate": format_failure_rate,
            "accuracy_by_position_bin": by_bin,
            "mean_prompt_tokens": mean_prompt_tokens,
            "mean_output_tokens": mean_output_tokens,
            "mean_wall_time_s": mean_wall_s,
        },
        "intracluster_rho": rho,
        "deff": deff,
        "cluster_floor_pts": round(floor_pts, 2),
        "cluster_means": cluster_means,
    }


def evaluate_model_decisions(
    cells_by_key: dict[str, list[dict[str, Any]]],
    compass_recall_by_n: dict[str, float] | None = None,
    n_flips: int = 100_000,
    n_boot: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate cliff, ceiling, compass recovery, length confound, and floor (Design §4 Rules 1-4)
    for one model.

    Expected keys in `cells_by_key`:
      - `full_5`, `full_10`, `full_20`, `full_40`, `full_70`, `full_100`
      - `compass_10`, `compass_20`, `compass_40`, `compass_70`, `compass_100`
      - `length_control_10`
    """
    summaries: dict[str, dict[str, Any]] = {
        k: summarize_cell(v) for k, v in cells_by_key.items() if v
    }

    # Measured within-cell rho across full-arm cells
    rho_values = [
        summaries[f"full_{n}"]["intracluster_rho"]
        for n in TOOL_COUNTS
        if f"full_{n}" in summaries
    ]
    mean_within_rho = sum(rho_values) / len(rho_values) if rho_values else 0.0

    # 1. Full arm comparisons: N in {10, 20, 40, 70, 100} vs N=5
    full_5_means = summaries["full_5"]["cluster_means"] if "full_5" in summaries else {}
    full_5_calls = cells_by_key.get("full_5", [])
    full_comparisons: dict[str, dict[str, Any]] = {}
    raw_p_full: dict[str, float] = {}
    paired_stats_list: list[dict[str, float]] = []

    for n in COMPASS_COUNTS:
        cell_key = f"full_{n}"
        if cell_key not in summaries or not full_5_means:
            continue
        cell_means = summaries[cell_key]["cluster_means"]
        if not (set(full_5_means.keys()) & set(cell_means.keys())):
            continue
        # Test full_5 - full_n (positive drop means accuracy fell at N)
        perm = paired_sign_flip_test(
            cluster_means_a=full_5_means,
            cluster_means_b=cell_means,
            n_flips=n_flips,
            seed=seed + n,
        )
        boot = paired_cluster_bootstrap_ci(
            cluster_means_a=full_5_means,
            cluster_means_b=cell_means,
            n_boot=n_boot,
            seed=seed + 500 + n,
        )
        paired_stats = estimate_paired_diff_rho_and_discordance(
            calls_a=full_5_calls,
            calls_b=cells_by_key.get(cell_key, []),
        )
        paired_stats_list.append(paired_stats)
        raw_p_full[cell_key] = perm["p_value"]
        full_comparisons[cell_key] = {
            "N": n,
            "acc_n5": perm["mean_a"],
            "acc_n": perm["mean_b"],
            "drop_pts": perm["mean_diff_pts"],
            "permutation": perm,
            "bootstrap_ci_drop": boot,
            "discordance_rate": round(paired_stats["discordance_rate"], 4),
            "paired_diff_rho": round(paired_stats["paired_diff_rho"], 4),
            "measured_cluster_floor_pts": paired_stats["measured_cluster_floor_pts"],
        }

    if paired_stats_list:
        mean_discordance = sum(ps["discordance_rate"] for ps in paired_stats_list) / len(paired_stats_list)
        mean_paired_rho = sum(ps["paired_diff_rho"] for ps in paired_stats_list) / len(paired_stats_list)
        mean_unclustered_floor_pts = (
            10.2 * math.sqrt(mean_discordance / 0.15) if mean_discordance > 0.0 else 0.0
        )
        measured_floor_pts = round(
            mean_unclustered_floor_pts * math.sqrt(1.0 + max(0.0, mean_paired_rho)), 2
        )
    else:
        mean_discordance = 0.15
        mean_paired_rho = mean_within_rho
        mean_unclustered_floor_pts = 10.2
        measured_floor_pts = round(10.2 * math.sqrt(1.0 + max(0.0, mean_within_rho)), 2)

    holm_full = holm_adjust(raw_p_full, alpha=0.05) if raw_p_full else {}
    cliff_n: int | None = None
    prev_n = 5
    ceiling_n = 100

    for n in COMPASS_COUNTS:
        cell_key = f"full_{n}"
        if cell_key not in full_comparisons:
            continue
        comp = full_comparisons[cell_key]
        h_info = holm_full[cell_key]
        comp["holm"] = h_info
        drop_pts = comp["drop_pts"]
        sig = h_info["significant"]

        if sig and drop_pts >= 10.0:
            comp["verdict"] = "cliff_drop"
            comp["floor_note"] = None
            if cliff_n is None:
                cliff_n = n
                ceiling_n = prev_n
        elif drop_pts > 0.0:
            comp["verdict"] = "not shown"
            comp["floor_note"] = (
                f"not shown (drop under the floor {measured_floor_pts:.1f} pts not ruled out)"
            )
        else:
            comp["verdict"] = "no_drop"
            comp["floor_note"] = None
        prev_n = n

    # 2. Compass vs Full at each N in {10, 20, 40, 70, 100} and specifically at cliff_n (or 100 if no cliff)
    # Design §4 Rule 2: the confirmatory Holm family tests compass vs full at each model's cliff N (or 100 if none)
    # across the evaluated models (family size = 1 for a single-model evaluation, M across M models).
    eval_n_for_rule2 = cliff_n if cliff_n is not None else 100
    compass_comparisons: dict[str, dict[str, Any]] = {}
    raw_p_compass: dict[str, float] = {}

    for n in COMPASS_COUNTS:
        f_key = f"full_{n}"
        c_key = f"compass_{n}"
        if f_key not in summaries or c_key not in summaries:
            continue
        if not (set(summaries[c_key]["cluster_means"].keys()) & set(summaries[f_key]["cluster_means"].keys())):
            continue
        perm = paired_sign_flip_test(
            cluster_means_a=summaries[c_key]["cluster_means"],
            cluster_means_b=summaries[f_key]["cluster_means"],
            n_flips=n_flips,
            seed=seed + 1000 + n,
        )
        boot = paired_cluster_bootstrap_ci(
            cluster_means_a=summaries[c_key]["cluster_means"],
            cluster_means_b=summaries[f_key]["cluster_means"],
            n_boot=n_boot,
            seed=seed + 1500 + n,
        )
        raw_p_compass[c_key] = perm["p_value"]
        compass_comparisons[c_key] = {
            "N": n,
            "acc_compass": perm["mean_a"],
            "acc_full": perm["mean_b"],
            "gain_pts": perm["mean_diff_pts"],
            "permutation": perm,
            "bootstrap_ci_gain": boot,
            "recall_at_5": (
                compass_recall_by_n.get(str(n)) if compass_recall_by_n else None
            ),
        }

    exploratory_holm_compass = holm_adjust(raw_p_compass, alpha=0.05) if raw_p_compass else {}
    for c_key, comp in compass_comparisons.items():
        comp["exploratory_within_model_holm"] = exploratory_holm_compass[c_key]

    rule2_key = f"compass_{eval_n_for_rule2}"
    rule2_eval: dict[str, Any] = {"evaluated_at_N": eval_n_for_rule2}
    if rule2_key in compass_comparisons:
        rule2_single_holm = holm_adjust(
            {rule2_key: raw_p_compass[rule2_key]}, alpha=0.05
        )[rule2_key]
        compass_comparisons[rule2_key]["holm"] = rule2_single_holm
        c_eval = compass_comparisons[rule2_key]
        gain = c_eval["gain_pts"]
        sig = rule2_single_holm["significant"]
        recall_100 = compass_recall_by_n.get("100") if compass_recall_by_n else None
        rule2_eval.update(
            {
                "gain_pts": gain,
                "raw_p": rule2_single_holm["raw_p"],
                "holm_p": rule2_single_holm["holm_p"],
                "holm_family_size": rule2_single_holm["family_size"],
                "significant": sig,
                "helps_by_at_least_10pts_significant": bool(sig and gain >= 10.0),
                "within_3pts_or_below": bool(gain <= 3.0),
                "recall_at_5_n100": recall_100,
                "recall_below_90pct_bound": (
                    bool(recall_100 is not None and recall_100 < 0.90)
                ),
            }
        )

    # 3. Rule 3: Length-control cell vs full(N=10) and vs full(N=100) (Holm over the 2, alpha=0.05)
    rule3_eval: dict[str, Any] = {}
    if (
        "length_control_10" in summaries
        and "full_10" in summaries
        and "full_100" in summaries
        and (set(summaries["length_control_10"]["cluster_means"].keys()) & set(summaries["full_10"]["cluster_means"].keys()))
        and (set(summaries["length_control_10"]["cluster_means"].keys()) & set(summaries["full_100"]["cluster_means"].keys()))
    ):
        lc_means = summaries["length_control_10"]["cluster_means"]
        f10_means = summaries["full_10"]["cluster_means"]
        f100_means = summaries["full_100"]["cluster_means"]

        # Drop from full_10 to length_control_10 (positive means length hurt accuracy at 10 tools)
        perm_vs_10 = paired_sign_flip_test(
            cluster_means_a=f10_means,
            cluster_means_b=lc_means,
            n_flips=n_flips,
            seed=seed + 2010,
        )
        boot_vs_10 = paired_cluster_bootstrap_ci(
            cluster_means_a=f10_means,
            cluster_means_b=lc_means,
            n_boot=n_boot,
            seed=seed + 2510,
        )
        # Gap between length_control_10 and full_100
        perm_vs_100 = paired_sign_flip_test(
            cluster_means_a=lc_means,
            cluster_means_b=f100_means,
            n_flips=n_flips,
            seed=seed + 2100,
        )
        boot_vs_100 = paired_cluster_bootstrap_ci(
            cluster_means_a=lc_means,
            cluster_means_b=f100_means,
            n_boot=n_boot,
            seed=seed + 2600,
        )
        holm_r3 = holm_adjust(
            {
                "vs_full_10": perm_vs_10["p_value"],
                "vs_full_100": perm_vs_100["p_value"],
            },
            alpha=0.05,
        )
        sig_below_10 = (
            holm_r3["vs_full_10"]["significant"] and perm_vs_10["mean_diff_pts"] > 0.0
        )
        full_10_to_100_drop = (
            (summaries["full_10"]["cluster_mean_accuracy"] - summaries["full_100"]["cluster_mean_accuracy"])
            * 100.0
        )
        length_drop_pts = perm_vs_10["mean_diff_pts"]
        share_reproduced = (
            length_drop_pts / full_10_to_100_drop
            if abs(full_10_to_100_drop) > 1e-9
            else 0.0
        )
        rule3_eval = {
            "acc_full_10": summaries["full_10"]["cluster_mean_accuracy"],
            "acc_length_control_10": summaries["length_control_10"]["cluster_mean_accuracy"],
            "acc_full_100": summaries["full_100"]["cluster_mean_accuracy"],
            "drop_from_full_10_pts": length_drop_pts,
            "gap_above_full_100_pts": perm_vs_100["mean_diff_pts"],
            "full_10_to_100_drop_pts": full_10_to_100_drop,
            "share_of_full_drop_reproduced_by_length": share_reproduced,
            "vs_full_10": {
                "permutation": perm_vs_10,
                "bootstrap_ci": boot_vs_10,
                "holm": holm_r3["vs_full_10"],
            },
            "vs_full_100": {
                "permutation": perm_vs_100,
                "bootstrap_ci": boot_vs_100,
                "holm": holm_r3["vs_full_100"],
            },
            "length_confound_significant": sig_below_10,
            "verdict": (
                "part of the drop is prompt length, not tool count; do not attribute cliff to count alone"
                if sig_below_10
                else "length alone is not the cause here"
            ),
        }

    #Strip raw cluster_means maps from output summary to keep JSON compact if needed, or retain
    return {
        "rule1_cliff_and_ceiling": {
            "cliff_N": cliff_n,
            "cliff_status": (
                f"cliff at N={cliff_n}" if cliff_n is not None else "no cliff observed to 100"
            ),
            "ceiling_N": ceiling_n,
            "measured_discordance_rate": round(mean_discordance, 4),
            "measured_paired_diff_rho": round(mean_paired_rho, 4),
            "measured_intracluster_rho": round(mean_paired_rho, 4),
            "within_cell_intracluster_rho": round(mean_within_rho, 4),
            "measured_unclustered_floor_pts": round(mean_unclustered_floor_pts, 2),
            "measured_cluster_floor_pts": measured_floor_pts,
            "comparisons_vs_n5": full_comparisons,
        },
        "rule2_compass_narrowing": {
            "summary_at_cliff_or_100": rule2_eval,
            "comparisons_by_n": compass_comparisons,
        },
        "rule3_length_confound": rule3_eval,
        "rule4_floor": {
            "measured_discordance_rate": round(mean_discordance, 4),
            "measured_paired_diff_rho": round(mean_paired_rho, 4),
            "measured_intracluster_rho": round(mean_paired_rho, 4),
            "within_cell_intracluster_rho": round(mean_within_rho, 4),
            "measured_unclustered_floor_pts": round(mean_unclustered_floor_pts, 2),
            "measured_cluster_floor_pts": measured_floor_pts,
            "design_floor_range_pts": [11.4, 12.5],
        },
        "cell_summaries": {
            k: {sk: sv for sk, sv in v.items() if sk != "cluster_means"}
            for k, v in summaries.items()
        },
    }


def evaluate_cross_model_decisions(
    per_model_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Combine per-model Decision Rules 1 & 2 into the studio-level rule decisions (Design §4)."""
    ceilings = [
        rep["rule1_cliff_and_ceiling"]["ceiling_N"]
        for rep in per_model_reports.values()
    ]
    cliffs = [
        rep["rule1_cliff_and_ceiling"]["cliff_N"]
        for rep in per_model_reports.values()
    ]
    any_cliff = any(c is not None for c in cliffs)
    studio_ceiling = min(ceilings) if (ceilings and any_cliff) else 100

    # Rule 2 across models: Holm over the evaluated models at each model's cliff N (or 100)
    raw_p_models: dict[str, float] = {}
    gains_models: dict[str, float] = {}
    for m_name, rep in per_model_reports.items():
        r2 = rep["rule2_compass_narrowing"]["summary_at_cliff_or_100"]
        eval_n = r2["evaluated_at_N"]
        c_key = f"compass_{eval_n}"
        c_comp = rep["rule2_compass_narrowing"]["comparisons_by_n"].get(c_key)
        if c_comp is not None:
            raw_p_models[m_name] = c_comp["permutation"]["p_value"]
            gains_models[m_name] = c_comp["gain_pts"]

    holm_models = holm_adjust(raw_p_models, alpha=0.05) if raw_p_models else {}

    # Synchronize each model's summary_at_cliff_or_100 with the cross-model Rule 2 Holm family
    models_out: dict[str, dict[str, Any]] = {}
    for m_name, rep in per_model_reports.items():
        rep_copy = dict(rep)
        r2_section = dict(rep_copy["rule2_compass_narrowing"])
        r2_summary = dict(r2_section["summary_at_cliff_or_100"])
        comps_by_n = {k: dict(v) for k, v in r2_section["comparisons_by_n"].items()}
        if m_name in holm_models:
            h_entry = holm_models[m_name]
            eval_n = r2_summary["evaluated_at_N"]
            c_key = f"compass_{eval_n}"
            gain = gains_models[m_name]
            sig = h_entry["significant"]
            r2_summary.update(
                {
                    "raw_p": h_entry["raw_p"],
                    "holm_p": h_entry["holm_p"],
                    "holm_family_size": h_entry["family_size"],
                    "significant": sig,
                    "helps_by_at_least_10pts_significant": bool(sig and gain >= 10.0),
                }
            )
            if c_key in comps_by_n:
                comps_by_n[c_key]["holm"] = h_entry
        r2_section["summary_at_cliff_or_100"] = r2_summary
        r2_section["comparisons_by_n"] = comps_by_n
        rep_copy["rule2_compass_narrowing"] = r2_section
        models_out[m_name] = rep_copy

    models_helping_10pts = sum(
        1
        for m_name in raw_p_models
        if holm_models[m_name]["significant"] and gains_models[m_name] >= 10.0
    )
    all_within_3pts = bool(
        gains_models and all(g <= 3.0 for g in gains_models.values())
    )

    if models_helping_10pts >= 2:
        rule2_studio_verdict = "build narrowing in"
    elif all_within_3pts:
        rule2_studio_verdict = "do not build"
    else:
        rule2_studio_verdict = "narrowing helps some models; decide per model"

    return {
        "study_id": STUDY_ID,
        "rule1_studio_ceiling": {
            "studio_limit_N": studio_ceiling,
            "any_model_has_cliff": any_cliff,
            "per_model_ceilings": {
                m: rep["rule1_cliff_and_ceiling"]["ceiling_N"]
                for m, rep in models_out.items()
            },
            "per_model_cliffs": {
                m: rep["rule1_cliff_and_ceiling"]["cliff_N"]
                for m, rep in models_out.items()
            },
        },
        "rule2_studio_narrowing": {
            "verdict": rule2_studio_verdict,
            "models_with_10pt_significant_gain": models_helping_10pts,
            "per_model_gains_pts": gains_models,
            "per_model_holm": holm_models,
        },
        "models": models_out,
    }


def score_calls_file(
    calls_jsonl: Path,
    r0_dir: Path,
    out_path: Path,
    n_flips: int = 100_000,
    n_boot: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score a JSONL receipt file of calls and write the full statistical report."""
    registry = json.loads((r0_dir / "registry.json").read_text(encoding="utf-8"))
    items = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
    compass_path = r0_dir / "compass_lists.json"
    compass_recall: dict[str, float] | None = None
    if compass_path.is_file():
        compass_data = json.loads(compass_path.read_text(encoding="utf-8"))
        compass_recall = {
            n_str: info["recall_at_5"]
            for n_str, info in compass_data.get("recall_at_5_by_n", {}).items()
        }

    tool_by_name = {t["name"]: t for t in registry}
    item_by_id = {it["item_id"]: it for it in items}

    by_model_cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    scored_rows: list[dict[str, Any]] = []

    for line in calls_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        item = item_by_id[rec["item_id"]]
        scored = score_single_call(rec, item, tool_by_name)
        model_id = rec.get("model", "unknown_model")
        scored["model"] = model_id
        scored_rows.append(scored)

        arm = rec.get("arm")
        n_val = rec.get("N")
        if arm == "full":
            cell_key = f"full_{n_val}"
        elif arm == "compass":
            cell_key = f"compass_{n_val}"
        else:
            cell_key = f"length_control_{n_val}"
        by_model_cells.setdefault(model_id, {}).setdefault(cell_key, []).append(scored)

    per_model_reports: dict[str, dict[str, Any]] = {}
    for m_name, cells_map in sorted(by_model_cells.items()):
        per_model_reports[m_name] = evaluate_model_decisions(
            cells_by_key=cells_map,
            compass_recall_by_n=compass_recall,
            n_flips=n_flips,
            n_boot=n_boot,
            seed=seed,
        )

    report = evaluate_cross_model_decisions(per_model_reports)
    report["total_scored_calls"] = len(scored_rows)
    out_sha = write_with_sha256(
        out_path, json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    report["report_sha256"] = out_sha
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score tool-count cliff receipts and run cluster-level tests.")
    parser.add_argument("--calls-jsonl", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path, default=Path("artifacts/toolcount/r0"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/toolcount/score_report.json"))
    parser.add_argument("--n-flips", type=int, default=100_000)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    report = score_calls_file(
        calls_jsonl=args.calls_jsonl,
        r0_dir=args.r0_dir,
        out_path=args.out,
        n_flips=args.n_flips,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(json.dumps({"total_scored_calls": report["total_scored_calls"], "report_sha256": report["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
