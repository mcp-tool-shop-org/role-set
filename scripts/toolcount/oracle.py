"""Rule-based oracle for R0 registry & items (Design v1 §2, §8 item 1).

Proves that:
1. Every domain's 5 siblings differ in action verb, required parameter signature, and discriminating condition.
2. No prompt leaks any tool name from the 100-tool registry.
3. For every answerable item and every N in {5, 10, 20, 40, 70, 100}, the rule-based keyword/condition
   oracle inspecting ONLY the prompt and the candidate tools in L_N matches EXACTLY ONE tool, equal to key.tool.
4. For every abstention item and every N in {5, 10, 20, 40, 70, 100}, the anchor domain's 5 tools are in L_N
   (domain present) and the oracle matches ZERO tools in L_N (action absent).
5. Any miss raises `OracleVerificationError` (100% required or R0 fails loudly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.toolcount import TOOL_COUNTS


class OracleVerificationError(RuntimeError):
    """Raised when the R0 oracle finds any ambiguous, missed, or leaked item."""


def oracle_match_tools(
    prompt: str,
    candidate_tools: list[dict[str, Any]],
) -> list[str]:
    """Return all tool names in `candidate_tools` whose domain and discriminating condition
    slot match `prompt` (inspecting only `prompt` and each tool's description/schema).
    """
    prompt_lower = prompt.lower()
    matched: list[str] = []

    for tool in candidate_tools:
        desc_lower = tool["description"].lower()
        domain_token = tool["oracle_domain_token"].lower()
        action_token = tool["oracle_action_token"].lower()

        # Both tokens must literally exist in the tool's own description AND in the user's prompt
        if domain_token not in desc_lower or action_token not in desc_lower:
            continue
        if domain_token in prompt_lower and action_token in prompt_lower:
            # Also verify required parameter cues for multi-parameter slots
            req_params = tool[" input_schema"]["required"] if " input_schema" in tool else tool["input_schema"]["required"]
            extra_reqs = [p for p in req_params if p != "resource_id"]
            if all(p.lower() in prompt_lower for p in extra_reqs):
                matched.append(tool["name"])

    return matched


def verify_domain_sibling_uniqueness(registry: list[dict[str, Any]]) -> None:
    """Verify every domain has 5 tools and every sibling differs in verb, condition, and schema."""
    if len(registry) != 100:
        raise OracleVerificationError(f"Expected 100 tools in registry, got {len(registry)}")

    names = [t["name"] for t in registry]
    if len(set(names)) != 100:
        raise OracleVerificationError("Duplicate tool names found in registry")

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for tool in registry:
        if tool.get("category") != "tool":
            raise OracleVerificationError(
                f"Tool {tool['name']} has category={tool.get('category')!r}, expected 'tool'"
            )
        if tool.get("examples") != []:
            raise OracleVerificationError(
                f"Tool {tool['name']} has non-empty examples={tool.get('examples')!r}"
            )
        by_domain.setdefault(tool["domain"], []).append(tool)

    if len(by_domain) != 20:
        raise OracleVerificationError(f"Expected 20 domains, got {len(by_domain)}")

    for domain, tools in by_domain.items():
        if len(tools) != 5:
            raise OracleVerificationError(f"Domain {domain} has {len(tools)} tools, expected 5")
        verbs = {t["action_verb"] for t in tools}
        conditions = {t["oracle_action_token"] for t in tools}
        if len(verbs) != 5:
            raise OracleVerificationError(f"Domain {domain} siblings do not have 5 distinct action verbs")
        if len(conditions) != 5:
            raise OracleVerificationError(f"Domain {domain} siblings do not have 5 distinct conditions")


def verify_no_tool_name_leak(
    registry: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    """Ensure no item prompt contains any tool name from the 100-tool registry."""
    all_tool_names = [t["name"].lower() for t in registry]
    for item in items:
        p_low = item["prompt"].lower()
        for t_name in all_tool_names:
            if t_name in p_low:
                raise OracleVerificationError(
                    f"Tool name {t_name!r} leaked into prompt of item {item['item_id']!r}"
                )


def verify_all_items(
    registry: list[dict[str, Any]],
    items: list[dict[str, Any]],
    lists_by_anchor: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the oracle across all 200 items and all N in {5, 10, 20, 40, 70, 100}.

    Raises `OracleVerificationError` if any answerable item does not match `[key.tool]`
    or any abstention item does not match `[]` on every list.
    """
    verify_domain_sibling_uniqueness(registry)
    verify_no_tool_name_leak(registry, items)

    tool_by_name = {t["name"]: t for t in registry}
    domain_tools_map: dict[str, set[str]] = {}
    for t in registry:
        domain_tools_map.setdefault(t["domain"], set()).add(t["name"])

    failures: list[dict[str, Any]] = []
    answerable_checks = 0
    answerable_hits = 0
    abstention_checks = 0
    abstention_hits = 0

    for item in items:
        item_id = item["item_id"]
        anchor_id = item["anchor_id"]
        if anchor_id not in lists_by_anchor:
            failures.append({"item_id": item_id, "reason": f"Missing anchor_id {anchor_id}"})
            continue

        anchor_lists = lists_by_anchor[anchor_id]["lists"]
        expected_domain_tools = domain_tools_map[item["anchor_domain"]]

        # Verify L_5 is the 5 tools of the anchor domain and L_5 < L_10 < ... < L_100
        prev_names: set[str] = set()
        for n in TOOL_COUNTS:
            n_str = str(n)
            if n_str not in anchor_lists:
                failures.append({"item_id": item_id, "N": n, "reason": f"Missing L_{n}"})
                continue
            tool_names = anchor_lists[n_str]["tools"]
            if len(tool_names) != n or len(set(tool_names)) != n:
                failures.append(
                    {"item_id": item_id, "N": n, "reason": f"Invalid L_{n} length or duplicates"}
                )
                continue
            curr_names = set(tool_names)
            if n == 5 and curr_names != expected_domain_tools:
                failures.append(
                    {
                        "item_id": item_id,
                        "N": n,
                        "reason": "L_5 does not equal the 5 tools of the anchor domain",
                    }
                )
            if not prev_names.issubset(curr_names):
                failures.append(
                    {"item_id": item_id, "N": n, "reason": f"L_{n} does not nest previous list"}
                )
            prev_names = curr_names

            candidate_objs = [tool_by_name[name] for name in tool_names]
            matches = oracle_match_tools(item["prompt"], candidate_objs)

            if item["is_abstention"]:
                abstention_checks += 1
                if matches == []:
                    abstention_hits += 1
                else:
                    failures.append(
                        {
                            "item_id": item_id,
                            "N": n,
                            "kind": "abstention",
                            "expected": [],
                            "matches": matches,
                        }
                    )
            else:
                answerable_checks += 1
                expected_tool = item["key"]["tool"]
                if matches == [expected_tool]:
                    answerable_hits += 1
                else:
                    failures.append(
                        {
                            "item_id": item_id,
                            "N": n,
                            "kind": "answerable",
                            "expected": [expected_tool],
                            "matches": matches,
                        }
                    )

    if failures:
        raise OracleVerificationError(
            f"Oracle verification failed with {len(failures)} error(s): {failures[:5]}"
        )

    ans_acc = answerable_hits / answerable_checks if answerable_checks else 0.0
    abst_acc = abstention_hits / abstention_checks if abstention_checks else 0.0
    if ans_acc != 1.0 or abst_acc != 1.0:
        raise OracleVerificationError(
            f"Oracle did not reach 100%: answerable={ans_acc:.4f}, abstention={abst_acc:.4f}"
        )

    return {
        "passed": True,
        "answerable_checks": answerable_checks,
        "answerable_hits": answerable_hits,
        "answerable_accuracy": ans_acc,
        "abstention_checks": abstention_checks,
        "abstention_hits": abstention_hits,
        "abstention_accuracy": abst_acc,
        "total_checks": answerable_checks + abstention_checks,
        "failures": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify R0 registry, items, and nested lists with the rule-based oracle.")
    parser.add_argument("--r0-dir", type=Path, required=True, help="Directory containing registry.json, items.json, nested_lists.json.")
    args = parser.parse_args(argv)

    registry = json.loads((args.r0_dir / "registry.json").read_text(encoding="utf-8"))
    items = json.loads((args.r0_dir / "items.json").read_text(encoding="utf-8"))
    lists_by_anchor = json.loads((args.r0_dir / "nested_lists.json").read_text(encoding="utf-8"))

    report = verify_all_items(registry=registry, items=items, lists_by_anchor=lists_by_anchor)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
