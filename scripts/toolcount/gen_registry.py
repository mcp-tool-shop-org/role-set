"""R0 synthetic 100-tool registry, 200 items, nested lists L_N, and 40-item audit export.

Implements Design v1 §2, §5, §7, §8 item 1:
- 100 tools across 20 domains x 5 tools per domain.
- Constant `category = "tool"` and `examples = []` for every tool (R&D ruling, Design §2).
- 85 target tools (4 per domain + 5th in 5 seeded domains) x 2 phrasings = 170 answerable items.
- 30 abstention items (15% of 200), each anchored on its present domain's 5 tools.
- Nested lists L_5 < L_10 < L_20 < L_40 < L_70 < L_100 per target tool and per abstention item:
  - L_5 = all 5 tools of the anchor domain (sibling-first rule).
  - N-5 added tools from the other 19 domains: 1/3 near_name, 1/3 near_description, 1/3 unrelated
    (remainder in that order), with realised mix recorded per list.
- Deterministic Half A (43 target clusters = 86 answerable + 15 abstention = 101 items)
  and Half B (42 target clusters = 84 answerable + 15 abstention = 99 items).
- Balanced 6-bin target position assignment and seeded per-call list ordering.
- 40-item seeded audit sample exported as a readable Markdown table (blind view + key).
- Every output file written with LF line endings and a `.sha256` sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from scripts.toolcount import (
    DEFAULT_SEED,
    STUDY_ID,
    TOOL_CATEGORY_CONSTANT,
    TOOL_COUNTS,
)

# 20 distinct domains: (domain_id, domain_noun, domain_phrase, sample_resource_prefix)
DOMAINS: tuple[tuple[str, str, str, str], ...] = (
    ("files", "file", "workspace filesystem path", "/srv/work/report_v2.csv"),
    ("git", "commit", "git repository revision", "refs/heads/release-42"),
    ("calendar", "event", "team calendar schedule", "cal-evt-2026-1104"),
    ("database", "table", "relational database table", "billing_Clickstream_v3"),
    ("image", "raster", "bitmap raster canvas", "assets/hero_banner_4k.png"),
    ("audio", "waveform", "multitrack audio stem", "stems/podcast_ep19.wav"),
    ("network", "route", "virtual network subnet route", "10.88.14.0/24"),
    ("build", "target", "compiler build pipeline target", "//services/auth:bin"),
    ("archive", "tarball", "compressed archive volume", "backups/q3_ledger.tar.zst"),
    ("crypto", "keyring", "cryptographic signing keyring", "kr-prod-ed25519-07"),
    ("container", "pod", "container runtime pod spec", "pod-worker-gpu-03"),
    ("email", "mailbox", "outbound mail relay thread", "mbox-ops-alerts-09"),
    ("metrics", "series", "telemetry time-series metric", "rpc.latency.p99"),
    ("queue", "topic", "message broker queue topic", "orders.fulfillment.dlq"),
    ("search", "index", "inverted lexical search index", "idx-docs-en-2026"),
    ("secrets", "vault", "encrypted secret vault entry", "vault://prod/db-token"),
    ("spreadsheet", "sheet", "tabular workbook worksheet", "fin_q4_forecast!B2:G90"),
    ("storage", "bucket", "object storage bucket blob", "s3://artifacts-eu/model.bin"),
    ("testing", "suite", "regression test runner suite", "suite://integration/payments"),
    ("workflow", "dag", "orchestration workflow DAG", "dag_nightly_etl_v5"),
)

# 5 action verbs (one per sibling slot s in 0..4)
SLOT_VERBS: tuple[str, ...] = (
    "inspect",
    "validate",
    "export",
    "synchronize",
    "archive",
)

# 5 qualifier nouns cycled via (d + s) % 5 so cross-domain tools share noun roots too
QUALIFIERS: tuple[str, ...] = (
    "manifest",
    "snapshot",
    "ledger",
    "bundle",
    "channel",
)

# 5 discriminating condition slots (unique per sibling s in 0..4 within every domain)
SLOT_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "slot_index": 0,
        "verb": "inspect",
        "intent_phrase_a": "read the live metadata header",
        "intent_phrase_b": "retrieve the current header summary",
        "condition_phrase": "from the local workspace cache",
        "oracle_action_token": "local workspace cache",
        "param_specs": (
            ("resource_id", "string", True, "Identifier of the target resource to inspect."),
        ),
    },
    {
        "slot_index": 1,
        "verb": "validate",
        "intent_phrase_a": "check structural conformance",
        "intent_phrase_b": "verify policy compliance constraints",
        "condition_phrase": "against strict compliance rulebook level",
        "oracle_action_token": "strict compliance rulebook",
        "param_specs": (
            ("resource_id", "string", True, "Identifier of the target resource to validate."),
            ("rulebook_level", "integer", True, "Numeric strictness level of the compliance rulebook."),
        ),
    },
    {
        "slot_index": 2,
        "verb": "export",
        "intent_phrase_a": "package and serialize contents",
        "intent_phrase_b": "produce a portable serialized extract",
        "condition_phrase": "into a detached signature bundle",
        "oracle_action_token": "detached signature bundle",
        "param_specs": (
            ("resource_id", "string", True, "Identifier of the target resource to export."),
            ("include_checksum", "boolean", True, "Whether to embed a detached SHA-256 checksum."),
        ),
    },
    {
        "slot_index": 3,
        "verb": "synchronize",
        "intent_phrase_a": "reconcile state deltas",
        "intent_phrase_b": "mirror incremental changes",
        "condition_phrase": "over the authenticated replication channel",
        "oracle_action_token": "authenticated replication channel",
        "param_specs": (
            ("resource_id", "string", True, "Identifier of the target resource to synchronize."),
            ("peer_endpoint", "string", True, "Authenticated replication peer endpoint URI."),
            ("timeout_seconds", "integer", False, "Optional replication timeout in seconds."),
        ),
    },
    {
        "slot_index": 4,
        "verb": "archive",
        "intent_phrase_a": "seal historical revisions",
        "intent_phrase_b": "freeze completed records",
        "condition_phrase": "into immutable cold retention with epoch lock",
        "oracle_action_token": "immutable cold retention",
        "param_specs": (
            ("resource_id", "string", True, "Identifier of the target resource to archive."),
            ("retention_days", "integer", True, "Number of days to lock in immutable cold retention."),
            ("seal_immutable", "boolean", True, "Whether to enforce the immutable epoch lock."),
            ("notes_tag", "string", False, "Optional audit annotation tag."),
        ),
    },
)

# 5 shared cross-domain effect clauses assigned by (d + 2 * s) % 5 for near_description distractors
EFFECT_CLAUSES: tuple[str, ...] = (
    "emitting a deterministic audit digest for downstream verifiers",
    "preserving idempotent cursor checkpoints across retries",
    "enforcing zero-copy buffer isolation during execution",
    "recording monotonic lineage timestamps in the status log",
    "applying schema-locked field normalization before completion",
)

# Unsupported actions for abstention items (never appear in any tool description or schema)
ABSTENTION_ACTIONS: tuple[tuple[str, str], ...] = (
    (
        "permanently shred and degauss the physical storage platters",
        "physical platter degaussing",
    ),
    (
        "notarize a wet-ink statutory affidavit with the county clerk",
        "statutory county notarization",
    ),
    (
        "flash custom microcontroller voltage firmware over JTAG",
        "microcontroller JTAG firmware flashing",
    ),
)


def sha256_bytes(data: bytes) -> str:
    """Return hex sha256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def write_with_sha256(path: Path, text: str) -> str:
    """Write text with LF line endings and create a `<path>.sha256` sidecar."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = normalized.encode("utf-8")
    digest = sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_bytes(f"{digest}  {path.name}\n".encode("utf-8"))
    return digest


def build_registry() -> list[dict[str, Any]]:
    """Construct the 100-tool synthetic registry (20 domains x 5 tools each)."""
    tools: list[dict[str, Any]] = []
    for d_idx, (domain_id, domain_noun, domain_phrase, _sample_res) in enumerate(DOMAINS):
        for s_idx, slot in enumerate(SLOT_CONDITIONS):
            verb = slot["verb"]
            qualifier = QUALIFIERS[(d_idx + s_idx) % len(QUALIFIERS)]
            tool_name = f"{verb}_{domain_noun}_{qualifier}"
            effect_idx = (d_idx + 2 * s_idx) % len(EFFECT_CLAUSES)
            effect_clause = EFFECT_CLAUSES[effect_idx]
            condition_phrase = slot["condition_phrase"]

            description = (
                f"{verb.capitalize()}s the {domain_phrase} ({domain_id} domain) "
                f"{condition_phrase}, {effect_clause}."
            )

            properties: dict[str, dict[str, Any]] = {}
            parameters_summary: dict[str, str] = {}
            required_list: list[str] = []
            for p_name, p_type, is_req, p_desc in slot["param_specs"]:
                full_desc = f"{p_desc} ({domain_phrase})"
                properties[p_name] = {
                    "type": p_type,
                    "description": full_desc,
                }
                parameters_summary[p_name] = full_desc
                if is_req:
                    required_list.append(p_name)

            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required_list,
                "additionalProperties": False,
            }

            tools.append(
                {
                    "tool_id": f"tool_{d_idx:02d}_{s_idx:02d}",
                    "name": tool_name,
                    "domain": domain_id,
                    "domain_index": d_idx,
                    "slot_index": s_idx,
                    "domain_noun": domain_noun,
                    "domain_phrase": domain_phrase,
                    "action_verb": verb,
                    "qualifier_noun": qualifier,
                    "condition_phrase": condition_phrase,
                    "oracle_domain_token": domain_phrase,
                    "oracle_action_token": slot["oracle_action_token"],
                    "effect_family": effect_idx,
                    "effect_clause": effect_clause,
                    "category": TOOL_CATEGORY_CONSTANT,
                    "examples": [],
                    "description": description,
                    "parameters": parameters_summary,
                    "required_params": required_list,
                    "input_schema": input_schema,
                }
            )
    return tools


def classify_cross_domain_distractors(
    anchor_domain_idx: int,
    anchor_slot_idx: int,
    registry: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Partition the 95 cross-domain tools into near_name, near_description, unrelated.

    - near_name: cross-domain tools sharing the anchor's action_verb or qualifier_noun in their name.
    - near_description: remaining cross-domain tools sharing the anchor's effect_clause (or adjacent
      operational effect family) in their description while differing in domain and condition.
    - unrelated: all other cross-domain tools.
    Each pool is ordered deterministically by a seeded RNG keyed on (seed, anchor_domain_idx, anchor_slot_idx)
    and sized so that up to N=70 (22 near_name, 22 near_description, 21 unrelated) every 1/3 quota is met.
    """
    anchor_verb = SLOT_VERBS[anchor_slot_idx]
    anchor_qualifier = QUALIFIERS[(anchor_domain_idx + anchor_slot_idx) % len(QUALIFIERS)]
    anchor_effect = (anchor_domain_idx + 2 * anchor_slot_idx) % len(EFFECT_CLAUSES)
    secondary_effect = (anchor_effect + 1) % len(EFFECT_CLAUSES)

    cross_tools = [t for t in registry if t["domain_index"] != anchor_domain_idx]
    rng = random.Random(f"{seed}:distractors:{anchor_domain_idx}:{anchor_slot_idx}")

    # Tools sharing verb (19) or qualifier noun (16 across other slots) = 35 candidates
    near_name_candidates = [
        t["name"]
        for t in cross_tools
        if t["action_verb"] == anchor_verb or t["qualifier_noun"] == anchor_qualifier
    ]
    rng.shuffle(near_name_candidates)
    # Keep 25 near_name tools (all 19 same-verb + 6 same-qualifier) so 70 remain for near_description & unrelated
    same_verb = [
        t["name"] for t in cross_tools if t["action_verb"] == anchor_verb
    ]
    same_qual_only = [
        t["name"]
        for t in cross_tools
        if t["action_verb"] != anchor_verb and t["qualifier_noun"] == anchor_qualifier
    ]
    rng.shuffle(same_verb)
    rng.shuffle(same_qual_only)
    near_name = same_verb + same_qual_only[:6]  # 19 + 6 = 25 tools
    near_name_set = set(near_name)

    remaining = [t for t in cross_tools if t["name"] not in near_name_set]
    primary_desc = [t["name"] for t in remaining if t["effect_family"] == anchor_effect]
    secondary_desc = [t["name"] for t in remaining if t["effect_family"] == secondary_effect]
    rng.shuffle(primary_desc)
    rng.shuffle(secondary_desc)
    near_description = (primary_desc + secondary_desc)[:25]
    near_desc_set = set(near_description)

    unrelated = [t["name"] for t in remaining if t["name"] not in near_desc_set]
    rng.shuffle(unrelated)

    assert len(near_name) == 25
    assert len(near_description) == 25
    assert len(unrelated) == 45
    assert len(near_name_set | near_desc_set | set(unrelated)) == 95

    return {
        "near_name": near_name,
        "near_description": near_description,
        "unrelated": unrelated,
    }


def third_quotas(added_count: int) -> tuple[int, int, int]:
    """Return (near_name, near_description, unrelated) quotas for N-5 added tools.

    Design §2: 'one third each of three kinds (rounded, remainder in the order listed)'.
    """
    base, rem = divmod(added_count, 3)
    q_name = base + (1 if rem >= 1 else 0)
    q_desc = base + (1 if rem >= 2 else 0)
    q_unrel = base
    return q_name, q_desc, q_unrel


def build_nested_list_for_anchor(
    anchor_domain_idx: int,
    anchor_slot_idx: int,
    registry: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build nested lists L_5 < L_10 < L_20 < L_40 < L_70 < L_100 for one anchor."""
    in_domain = [t["name"] for t in registry if t["domain_index"] == anchor_domain_idx]
    assert len(in_domain) == 5

    pools = classify_cross_domain_distractors(
        anchor_domain_idx=anchor_domain_idx,
        anchor_slot_idx=anchor_slot_idx,
        registry=registry,
        seed=seed,
    )

    lists_by_n: dict[str, dict[str, Any]] = {}
    prev_set: set[str] = set()

    for n in TOOL_COUNTS:
        added_needed = n - 5
        if added_needed == 0:
            selected_name: list[str] = []
            selected_desc: list[str] = []
            selected_unrel: list[str] = []
            q_name, q_desc, q_unrel = 0, 0, 0
        elif n == 100:
            q_name, q_desc, q_unrel = third_quotas(added_needed)
            selected_name = list(pools["near_name"])
            selected_desc = list(pools["near_description"])
            selected_unrel = list(pools["unrelated"])
        else:
            q_name, q_desc, q_unrel = third_quotas(added_needed)
            selected_name = list(pools["near_name"][:q_name])
            selected_desc = list(pools["near_description"][:q_desc])
            selected_unrel = list(pools["unrelated"][:q_unrel])

        combined = in_domain + selected_name + selected_desc + selected_unrel
        assert len(combined) == n, f"Expected {n} tools, got {len(combined)}"
        curr_set = set(combined)
        assert len(curr_set) == n, f"Duplicate tools in L_{n}"
        assert prev_set.issubset(curr_set), f"L_{n} does not nest previous list"
        prev_set = curr_set

        lists_by_n[str(n)] = {
            "N": n,
            "tools": combined,
            "target_mix_added": {
                "near_name": q_name,
                "near_description": q_desc,
                "unrelated": q_unrel,
            },
            "realised_mix": {
                "in_domain": len(in_domain),
                "near_name": len(selected_name),
                "near_description": len(selected_desc),
                "unrelated": len(selected_unrel),
            },
        }

    return {
        "anchor_domain": DOMAINS[anchor_domain_idx][0],
        "anchor_domain_index": anchor_domain_idx,
        "anchor_slot_index": anchor_slot_idx,
        "lists": lists_by_n,
    }


def _build_required_args(
    domain_idx: int,
    slot_idx: int,
    phrasing_idx: int,
) -> dict[str, Any]:
    """Deterministically draw typed argument values from the pool before writing the prompt."""
    _domain_id, domain_noun, _domain_phrase, sample_res = DOMAINS[domain_idx]
    suffix = "alpha" if phrasing_idx == 0 else "beta"
    res_value = f"{sample_res}#{domain_noun}-{suffix}"

    if slot_idx == 0:
        return {"resource_id": res_value}
    if slot_idx == 1:
        level = 2 if phrasing_idx == 0 else 4
        return {"resource_id": res_value, "rulebook_level": level}
    if slot_idx == 2:
        flag = True if phrasing_idx == 0 else False
        return {"resource_id": res_value, "include_checksum": flag}
    if slot_idx == 3:
        peer = f"tls://replica-0{phrasing_idx + 1}.{domain_noun}.internal:8443"
        return {"resource_id": res_value, "peer_endpoint": peer}
    if slot_idx == 4:
        days = 90 if phrasing_idx == 0 else 365
        seal = True if phrasing_idx == 0 else False
        return {
            "resource_id": res_value,
            "retention_days": days,
            "seal_immutable": seal,
        }
    raise ValueError(f"Unexpected slot_idx: {slot_idx}")


def _render_answerable_prompt(
    tool: dict[str, Any],
    required_args: dict[str, Any],
    phrasing_idx: int,
) -> str:
    """Render user prompt stating intent, discriminating slot, and required args without tool name."""
    slot = SLOT_CONDITIONS[tool["slot_index"]]
    domain_phrase = tool["domain_phrase"]
    intent = slot["intent_phrase_a"] if phrasing_idx == 0 else slot["intent_phrase_b"]
    cond = slot["condition_phrase"]
    res_id = required_args["resource_id"]

    s_idx = tool["slot_index"]
    if s_idx == 0:
        arg_clause = f"for resource `{res_id}`"
    elif s_idx == 1:
        lvl = required_args["rulebook_level"]
        arg_clause = f"for resource `{res_id}` with rulebook_level set to {lvl}"
    elif s_idx == 2:
        chk = str(required_args["include_checksum"]).lower()
        arg_clause = f"for resource `{res_id}` with include_checksum set to {chk}"
    elif s_idx == 3:
        peer = required_args["peer_endpoint"]
        arg_clause = f"for resource `{res_id}` targeting peer_endpoint `{peer}`"
    else:
        days = required_args["retention_days"]
        seal = str(required_args["seal_immutable"]).lower()
        arg_clause = (
            f"for resource `{res_id}` with retention_days set to {days} "
            f"and seal_immutable set to {seal}"
        )

    if phrasing_idx == 0:
        prompt = (
            f"For the {domain_phrase}, please {intent} {cond} {arg_clause}."
        )
    else:
        prompt = (
            f"We need to {intent} on the {domain_phrase} {cond} {arg_clause}."
        )

    # Guarantee tool name never appears in prompt
    assert tool["name"].lower() not in prompt.lower(), (
        f"Tool name {tool['name']} leaked into prompt: {prompt}"
    )
    return prompt


def select_target_tools_and_items(
    registry: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Generate the 85 target-tool clusters (170 answerable items) + 30 abstention items
    and their per-anchor nested lists L_N.
    """
    rng = random.Random(f"{seed}:targets")

    # Select 5 domains to contribute all 5 tools; the other 15 domains contribute 4 tools (20*4 + 5 = 85)
    all_domain_indices = list(range(len(DOMAINS)))
    five_tool_domains = set(rng.sample(all_domain_indices, 5))

    target_tools: list[dict[str, Any]] = []
    for d_idx in all_domain_indices:
        domain_tools = [t for t in registry if t["domain_index"] == d_idx]
        if d_idx in five_tool_domains:
            chosen = domain_tools
        else:
            # Deterministically drop one slot per 4-tool domain
            drop_slot = rng.randrange(5)
            chosen = [t for t in domain_tools if t["slot_index"] != drop_slot]
        target_tools.extend(chosen)

    assert len(target_tools) == 85

    # Split 85 target-tool clusters into Half A (43) and Half B (42)
    cluster_indices = list(range(85))
    rng_split = random.Random(f"{seed}:half_split")
    rng_split.shuffle(cluster_indices)
    half_a_clusters = set(cluster_indices[:43])

    # Balanced position bins (0..5) across the 170 answerable items:
    # Pair each cluster with (bin_b, (bin_b + 3) % 6) so the 2 phrasings fall in opposite halves of the list
    bin_rng = random.Random(f"{seed}:position_bins")
    cluster_base_bins = [i % 6 for i in range(85)]
    bin_rng.shuffle(cluster_base_bins)

    items: list[dict[str, Any]] = []
    lists_by_anchor: dict[str, dict[str, Any]] = {}

    for c_idx, tool in enumerate(target_tools):
        cluster_id = f"cluster_{c_idx:02d}"
        half = "A" if c_idx in half_a_clusters else "B"
        anchor_id = tool["name"]
        lists_by_anchor[anchor_id] = build_nested_list_for_anchor(
            anchor_domain_idx=tool["domain_index"],
            anchor_slot_idx=tool["slot_index"],
            registry=registry,
            seed=seed,
        )
        base_bin = cluster_base_bins[c_idx]
        for p_idx in (0, 1):
            req_args = _build_required_args(
                domain_idx=tool["domain_index"],
                slot_idx=tool["slot_index"],
                phrasing_idx=p_idx,
            )
            prompt = _render_answerable_prompt(tool, req_args, p_idx)
            pos_bin = base_bin if p_idx == 0 else (base_bin + 3) % 6
            item_id = f"ans_{c_idx:02d}_p{p_idx}"
            items.append(
                {
                    "item_id": item_id,
                    "kind": "answerable",
                    "is_abstention": False,
                    "cluster_id": cluster_id,
                    "phrasing_index": p_idx,
                    "half": half,
                    "anchor_id": anchor_id,
                    "anchor_domain": tool["domain"],
                    "anchor_domain_index": tool["domain_index"],
                    "anchor_slot_index": tool["slot_index"],
                    "position_bin": pos_bin,
                    "key": {
                        "tool": tool["name"],
                        "required_args": req_args,
                    },
                    "prompt": prompt,
                }
            )

    # 30 abstention items: 20 across all 20 domains + 10 across 10 seeded domains, split 15/15 between Half A and Half B
    abst_domains_order = list(range(len(DOMAINS))) + rng.sample(list(range(len(DOMAINS))), 10)
    assert len(abst_domains_order) == 30
    abst_half_indices = list(range(30))
    rng_split.shuffle(abst_half_indices)
    half_a_abst = set(abst_half_indices[:15])

    for a_idx, d_idx in enumerate(abst_domains_order):
        domain_id, domain_noun, domain_phrase, sample_res = DOMAINS[d_idx]
        act_phrase, act_token = ABSTENTION_ACTIONS[a_idx % len(ABSTENTION_ACTIONS)]
        anchor_id = f"abst_domain_{domain_id}_{a_idx:02d}"
        half = "A" if a_idx in half_a_abst else "B"
        # Nested lists anchored on the present domain's 5 tools (review ka must-fix 1)
        lists_by_anchor[anchor_id] = build_nested_list_for_anchor(
            anchor_domain_idx=d_idx,
            anchor_slot_idx=a_idx % 5,
            registry=registry,
            seed=seed + 1000 + a_idx,
        )
        prompt = (
            f"For the {domain_phrase} `{sample_res}#{domain_noun}-abst-{a_idx:02d}`, "
            f"please {act_phrase} immediately."
        )
        items.append(
            {
                "item_id": f"abst_{a_idx:02d}",
                "kind": "abstention",
                "is_abstention": True,
                "cluster_id": f"abst_cluster_{a_idx:02d}",
                "phrasing_index": 0,
                "half": half,
                "anchor_id": anchor_id,
                "anchor_domain": domain_id,
                "anchor_domain_index": d_idx,
                "anchor_slot_index": None,
                "unsupported_action_token": act_token,
                "position_bin": None,
                "key": {
                    "tool": None,
                    "required_args": {},
                },
                "prompt": prompt,
            }
        )

    return items, lists_by_anchor


def position_to_bin(position: int, list_len: int) -> int:
    """Map a 0-based tool index in a list of length `list_len` to one of 6 bins (0..5)."""
    if list_len <= 0:
        raise ValueError("list_len must be positive")
    return min(5, (position * 6) // list_len)


def order_tools_for_call(
    tool_names: list[str],
    target_tool: str | None,
    position_bin: int | None,
    seed: int,
    call_key: str,
) -> dict[str, Any]:
    """Deterministically shuffle `tool_names` for a single call, placing `target_tool`
    (if present in `tool_names` and `position_bin` is not None) inside `position_bin` (0..5).
    """
    m = len(tool_names)
    rng = random.Random(f"{seed}:order:{call_key}")

    if target_tool is None or target_tool not in tool_names or position_bin is None:
        shuffled = list(tool_names)
        rng.shuffle(shuffled)
        target_pos = shuffled.index(target_tool) if (target_tool and target_tool in shuffled) else None
        realised_bin = position_to_bin(target_pos, m) if target_pos is not None else None
        return {
            "ordered_tools": shuffled,
            "target_position": target_pos,
            "position_bin": realised_bin,
            "assigned_position_bin": position_bin,
        }

    others = [t for t in tool_names if t != target_tool]
    rng.shuffle(others)

    if m >= 6:
        candidate_positions = [p for p in range(m) if position_to_bin(p, m) == position_bin]
        target_pos = rng.choice(candidate_positions)
        realised_bin = position_to_bin(target_pos, m)
    else:
        # For m == 5, map bins 0..5 across positions 0..4 and preserve assigned bin label
        target_pos = min(m - 1, (position_bin * m) // 6)
        realised_bin = position_bin

    ordered = others[:target_pos] + [target_tool] + others[target_pos:]
    assert len(ordered) == m
    assert ordered[target_pos] == target_tool

    return {
        "ordered_tools": ordered,
        "target_position": target_pos,
        "position_bin": realised_bin,
        "assigned_position_bin": position_bin,
    }


def build_audit_sample_40(
    registry: list[dict[str, Any]],
    items: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], str]:
    """Select a seeded 40-item sample (34 answerable + 6 abstention) and format a
    readable Markdown audit document (blind section first, oracle key second).
    """
    rng = random.Random(f"{seed}:audit40")
    answerable = [it for it in items if not it["is_abstention"]]
    abstentions = [it for it in items if it["is_abstention"]]

    sampled_ans = rng.sample(answerable, 34)
    sampled_abst = rng.sample(abstentions, 6)
    combined = sampled_ans + sampled_abst
    rng.shuffle(combined)

    tools_by_domain: dict[str, list[dict[str, Any]]] = {}
    for t in registry:
        tools_by_domain.setdefault(t["domain"], []).append(t)

    audit_rows: list[dict[str, Any]] = []
    md_lines: list[str] = [
        "# R0 40-Item Blind Audit Sample (Seeded)",
        "",
        f"- **Study:** `{STUDY_ID}`",
        f"- **Seed:** `{seed}`",
        "- **Sample size:** 40 items (34 answerable, 6 abstention; stratified 20% of 200 items)",
        "- **Instructions for blind human audit:** Read Part 1 (Blind Table) and record which tool (if any) from the anchor domain's `L_5` list should be called, then compare against Part 2 (Oracle & Key Reconciliation).",
        "",
        "## Part 1: Blind Audit Table",
        "",
        "| # | Item ID | Anchor Domain | Prompt | Candidate `L_5` Tools (`name`: required params) |",
        "|---|---|---|---|---|",
    ]

    for idx, it in enumerate(combined, start=1):
        dom = it["anchor_domain"]
        dom_tools = tools_by_domain[dom]
        cand_strs = [
            f"`{t['name']}` ({', '.join(t['required_params'])})"
            for t in dom_tools
        ]
        cand_col = "; ".join(cand_strs)
        prompt_clean = it["prompt"].replace("|", "\\|")
        md_lines.append(
            f"| {idx} | `{it['item_id']}` | `{dom}` | {prompt_clean} | {cand_col} |"
        )
        audit_rows.append(
            {
                "audit_index": idx,
                "item_id": it["item_id"],
                "anchor_domain": dom,
                "prompt": it["prompt"],
                "candidate_l5_tools": [t["name"] for t in dom_tools],
                "expected_tool": it["key"]["tool"],
                "expected_required_args": it["key"]["required_args"],
                "is_abstention": it["is_abstention"],
            }
        )

    md_lines.extend(
        [
            "",
            "## Part 2: Oracle & Ground-Truth Key Reconciliation",
            "",
            "| # | Item ID | Kind | Expected Tool (`key.tool`) | Required Args (`key.required_args`) |",
            "|---|---|---|---|---|",
        ]
    )

    for row in audit_rows:
        kind = "abstention" if row["is_abstention"] else "answerable"
        exp_tool = f"`{row['expected_tool']}`" if row["expected_tool"] else "`None` (no call)"
        args_json = json.dumps(row["expected_required_args"], sort_keys=True)
        md_lines.append(
            f"| {row['audit_index']} | `{row['item_id']}` | {kind} | {exp_tool} | `{args_json}` |"
        )

    md_text = "\n".join(md_lines) + "\n"
    return audit_rows, md_text


def generate_all(out_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run full R0 generation, verify 100% oracle pass on all lists, and write pinned artifacts."""
    from scripts.toolcount.oracle import verify_all_items

    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    items, lists_by_anchor = select_target_tools_and_items(registry, seed=seed)

    # Run the rule-based oracle across every item and every N in {5, 10, 20, 40, 70, 100}.
    # Fails loudly if accuracy is not 100.0%.
    oracle_report = verify_all_items(
        registry=registry,
        items=items,
        lists_by_anchor=lists_by_anchor,
    )

    audit_rows, audit_md = build_audit_sample_40(registry, items, seed=seed)

    registry_path = out_dir / "registry.json"
    items_path = out_dir / "items.json"
    lists_path = out_dir / "nested_lists.json"
    audit_md_path = out_dir / "audit_sample_40.md"
    audit_json_path = out_dir / "audit_sample_40.json"
    oracle_path = out_dir / "oracle_report.json"
    manifest_path = out_dir / "r0_manifest.json"

    reg_sha = write_with_sha256(
        registry_path, json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    items_sha = write_with_sha256(
        items_path, json.dumps(items, indent=2, sort_keys=True) + "\n"
    )
    lists_sha = write_with_sha256(
        lists_path, json.dumps(lists_by_anchor, indent=2, sort_keys=True) + "\n"
    )
    audit_md_sha = write_with_sha256(audit_md_path, audit_md)
    audit_json_sha = write_with_sha256(
        audit_json_path, json.dumps(audit_rows, indent=2, sort_keys=True) + "\n"
    )
    oracle_sha = write_with_sha256(
        oracle_path, json.dumps(oracle_report, indent=2, sort_keys=True) + "\n"
    )

    half_a_items = [it for it in items if it["half"] == "A"]
    half_b_items = [it for it in items if it["half"] == "B"]

    manifest = {
        "study_id": STUDY_ID,
        "seed": seed,
        "tool_count": len(registry),
        "domain_count": len(DOMAINS),
        "target_tool_clusters": 85,
        "answerable_items": sum(1 for it in items if not it["is_abstention"]),
        "abstention_items": sum(1 for it in items if it["is_abstention"]),
        "total_items": len(items),
        "tool_category_constant": TOOL_CATEGORY_CONSTANT,
        "tool_examples_constant": [],
        "half_split": {
            "A": {
                "total_items": len(half_a_items),
                "answerable_items": sum(1 for it in half_a_items if not it["is_abstention"]),
                "abstention_items": sum(1 for it in half_a_items if it["is_abstention"]),
                "calls_across_12_cells": len(half_a_items) * 12,
            },
            "B": {
                "total_items": len(half_b_items),
                "answerable_items": sum(1 for it in half_b_items if not it["is_abstention"]),
                "abstention_items": sum(1 for it in half_b_items if it["is_abstention"]),
                "calls_across_12_cells": len(half_b_items) * 12,
            },
        },
        "oracle_summary": {
            "passed": oracle_report["passed"],
            "answerable_accuracy": oracle_report["answerable_accuracy"],
            "abstention_accuracy": oracle_report["abstention_accuracy"],
            "total_checks": oracle_report["total_checks"],
        },
        "files": {
            "registry.json": reg_sha,
            "items.json": items_sha,
            "nested_lists.json": lists_sha,
            "audit_sample_40.md": audit_md_sha,
            "audit_sample_40.json": audit_json_sha,
            "oracle_report.json": oracle_sha,
        },
    }
    manifest_sha = write_with_sha256(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate R0 registry, items, nested lists, and 40-item audit table.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/toolcount/r0"),
        help="Output directory for generated R0 files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed.",
    )
    args = parser.parse_args(argv)
    manifest = generate_all(out_dir=args.out_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
