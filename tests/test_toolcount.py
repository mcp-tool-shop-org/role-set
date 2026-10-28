"""Unit and integration tests for the tool-count cliff study (scripts/toolcount/* and pin_window integration)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.colab import pin_window as pw  # noqa: E402
from scripts.toolcount import (
    COMPASS_COUNTS,
    DEFAULT_SEED,
    LLAMA_CPP_COMMIT,
    LLAMA_SERVER_SHA256,
    RND_PROXY_COMMIT,
    STUDY_ID,
    SYSTEM_PROMPT,
    TOOL_CATEGORY_CONSTANT,
    TOOL_COMPASS_COMMIT,
    TOOL_COUNTS,
)
from scripts.toolcount.compass_lists import (
    FakeEmbedder,
    ToolDefinitionCompat,
    build_compass_lists,
    build_tool_definition,
    cosine_similarity,
    main as compass_main,
    resolve_tool_definition_class,
    write_compass_artifacts,
)
from scripts.toolcount.gen_registry import (
    DOMAINS,
    build_audit_sample_40,
    build_nested_list_for_anchor,
    build_registry,
    classify_cross_domain_distractors,
    generate_all,
    main as gen_main,
    order_tools_for_call,
    position_to_bin,
    select_target_tools_and_items,
    third_quotas,
)
from scripts.toolcount.oracle import (
    OracleVerificationError,
    main as oracle_main,
    oracle_match_tools,
    verify_all_items,
    verify_domain_sibling_uniqueness,
    verify_no_tool_name_leak,
)
from scripts.toolcount.runner import (
    PINNED_MODEL_SPECS,
    SmokeAssertError,
    StubToolCountServer,
    build_call_schedule,
    build_inert_padding_block,
    build_r1_stub_schedule,
    build_study_config,
    calibrate_length_control_padding,
    compute_first_50_repricing,
    count_thinking_tokens,
    format_openai_tool,
    load_resume_journal,
    main as runner_main,
    run_model_window,
    run_preloop_smoke_assert,
    run_r1_stub_verification,
    write_study_config,
)
from scripts.toolcount.scorer import (
    check_arguments_grade,
    compute_cluster_means,
    estimate_intracluster_rho,
    evaluate_cross_model_decisions,
    evaluate_model_decisions,
    extract_tool_call_from_response,
    holm_adjust,
    main as scorer_main,
    paired_cluster_bootstrap_ci,
    paired_sign_flip_test,
    score_calls_file,
    score_single_call,
    summarize_cell,
)


@pytest.fixture(scope="module")
def r0_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate the R0 bundle and compass lists once per test module."""
    r0_dir = tmp_path_factory.mktemp("r0_bundle")
    manifest = generate_all(out_dir=r0_dir, seed=DEFAULT_SEED)
    compass_manifest = write_compass_artifacts(
        r0_dir=r0_dir,
        out_dir=r0_dir,
        seed=DEFAULT_SEED,
        embedder_mode="fake",
    )
    registry = json.loads((r0_dir / "registry.json").read_text(encoding="utf-8"))
    items = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
    lists_by_anchor = json.loads((r0_dir / "nested_lists.json").read_text(encoding="utf-8"))
    compass_data = json.loads((r0_dir / "compass_lists.json").read_text(encoding="utf-8"))
    return {
        "r0_dir": r0_dir,
        "manifest": manifest,
        "compass_manifest": compass_manifest,
        "registry": registry,
        "items": items,
        "lists_by_anchor": lists_by_anchor,
        "compass_data": compass_data,
    }


def test_r0_registry_structure_and_category_examples_ruling(r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    assert len(registry) == 100
    domains = {t["domain"] for t in registry}
    assert len(domains) == 20
    for t in registry:
        assert t["category"] == TOOL_CATEGORY_CONSTANT == "tool"
        assert t["examples"] == []
        props = t["input_schema"]["properties"]
        assert 1 <= len(props) <= 4
        assert len(t["required_params"]) >= 1
        assert t["input_schema"]["additionalProperties"] is False


def test_r0_items_clusters_halves_and_nested_lists(r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    items = r0_bundle["items"]
    lists_by_anchor = r0_bundle["lists_by_anchor"]

    assert len(items) == 200
    ans = [it for it in items if not it["is_abstention"]]
    abst = [it for it in items if it["is_abstention"]]
    assert len(ans) == 170
    assert len(abst) == 30

    clusters = {it["cluster_id"] for it in ans}
    assert len(clusters) == 85
    for cid in clusters:
        c_items = [it for it in ans if it["cluster_id"] == cid]
        assert len(c_items) == 2
        assert {it["phrasing_index"] for it in c_items} == {0, 1}
        assert len({it["half"] for it in c_items}) == 1
        # The 2 phrasings are placed in opposite position bins ((b + 3) % 6)
        b0, b1 = c_items[0]["position_bin"], c_items[1]["position_bin"]
        assert (b0 + 3) % 6 == b1 or (b1 + 3) % 6 == b0

    half_a = [it for it in items if it["half"] == "A"]
    half_b = [it for it in items if it["half"] == "B"]
    assert len(half_a) == 101  # 43 clusters (86 answerable) + 15 abstention
    assert len(half_b) == 99   # 42 clusters (84 answerable) + 15 abstention
    assert sum(1 for it in half_a if not it["is_abstention"]) == 86
    assert sum(1 for it in half_b if not it["is_abstention"]) == 84
    assert sum(1 for it in half_a if it["is_abstention"]) == 15
    assert sum(1 for it in half_b if it["is_abstention"]) == 15

    # Verify sibling-first rule and 1/3 distractor mix on every anchor's nested lists
    tool_by_name = {t["name"]: t for t in registry}
    for _anchor_id, anchor_info in lists_by_anchor.items():
        dom = anchor_info["anchor_domain"]
        lists_map = anchor_info["lists"]
        l5_names = lists_map["5"]["tools"]
        assert len(l5_names) == 5
        assert all(tool_by_name[n]["domain"] == dom for n in l5_names)

        prev: set[str] = set()
        for n in TOOL_COUNTS:
            entry = lists_map[str(n)]
            curr = set(entry["tools"])
            assert len(curr) == n
            assert prev.issubset(curr)
            prev = curr
            mix = entry["realised_mix"]
            assert mix["in_domain"] == 5
            if n < 100:
                q_name, q_desc, q_unrel = third_quotas(n - 5)
                assert mix["near_name"] == q_name
                assert mix["near_description"] == q_desc
                assert mix["unrelated"] == q_unrel
            else:
                assert mix["near_name"] + mix["near_description"] + mix["unrelated"] == 95


def test_r0_audit_sample_40_and_sha256_sidecars(r0_bundle: dict[str, Any]) -> None:
    r0_dir: Path = r0_bundle["r0_dir"]
    audit_rows = json.loads((r0_dir / "audit_sample_40.json").read_text(encoding="utf-8"))
    audit_md = (r0_dir / "audit_sample_40.md").read_text(encoding="utf-8")

    assert len(audit_rows) == 40
    assert sum(1 for r in audit_rows if not r["is_abstention"]) == 34
    assert sum(1 for r in audit_rows if r["is_abstention"]) == 6
    assert "## Part 1: Blind Audit Table" in audit_md
    assert "## Part 2: Oracle & Ground-Truth Key Reconciliation" in audit_md

    for fname in (
        "registry.json",
        "items.json",
        "nested_lists.json",
        "audit_sample_40.md",
        "audit_sample_40.json",
        "oracle_report.json",
        "r0_manifest.json",
        "compass_lists.json",
        "compass_manifest.json",
    ):
        fpath = r0_dir / fname
        sidecar = r0_dir / f"{fname}.sha256"
        assert fpath.is_file() and sidecar.is_file()
        raw = fpath.read_bytes()
        assert b"\r" not in raw
        expected_sha = hashlib.sha256(raw).hexdigest()
        assert sidecar.read_text(encoding="utf-8").strip().startswith(expected_sha)


def test_order_tools_for_call_and_position_bins(r0_bundle: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="list_len must be positive"):
        position_to_bin(0, 0)

    names_20 = [f"tool_{i:02d}" for i in range(20)]
    for b in range(6):
        res = order_tools_for_call(
            tool_names=names_20,
            target_tool="tool_07",
            position_bin=b,
            seed=DEFAULT_SEED,
            call_key=f"test_bin_{b}",
        )
        assert len(res["ordered_tools"]) == 20
        pos = res["target_position"]
        assert pos is not None
        assert res["ordered_tools"][pos] == "tool_07"
        assert position_to_bin(pos, 20) == b
        assert res["position_bin"] == b

    # M == 5 case
    names_5 = [f"tool_{i:02d}" for i in range(5)]
    for b in range(6):
        res5 = order_tools_for_call(
            tool_names=names_5,
            target_tool="tool_02",
            position_bin=b,
            seed=DEFAULT_SEED,
            call_key=f"test5_bin_{b}",
        )
        assert res5["ordered_tools"][res5["target_position"]] == "tool_02"
        assert res5["position_bin"] == b

    # Abstention / missing target case
    res_abst = order_tools_for_call(
        tool_names=names_5,
        target_tool=None,
        position_bin=None,
        seed=DEFAULT_SEED,
        call_key="test_abst",
    )
    assert res_abst["target_position"] is None
    assert res_abst["position_bin"] is None


def test_oracle_100pct_and_loud_failures(r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    items = r0_bundle["items"]
    lists_by_anchor = r0_bundle["lists_by_anchor"]

    report = verify_all_items(registry, items, lists_by_anchor)
    assert report["passed"] is True
    assert report["answerable_accuracy"] == 1.0
    assert report["abstention_accuracy"] == 1.0
    assert report["total_checks"] == 200 * 6

    # 1. Leak detection
    bad_items_leak = [dict(items[0], prompt=f"Please call {registry[0]['name']} right now.")]
    with pytest.raises(OracleVerificationError, match="leaked into prompt"):
        verify_no_tool_name_leak(registry, bad_items_leak)

    # 2. Wrong key on answerable item
    bad_items_key = [dict(items[0], key={"tool": "nonexistent_tool", "required_args": {}})]
    with pytest.raises(OracleVerificationError, match="Oracle verification failed"):
        verify_all_items(registry, bad_items_key, lists_by_anchor)

    # 3. Abstention item whose prompt matches a real tool
    abst_item = next(it for it in items if it["is_abstention"])
    bad_abst = [dict(abst_item, prompt=items[0]["prompt"], anchor_id=items[0]["anchor_id"], anchor_domain=items[0]["anchor_domain"])]
    with pytest.raises(OracleVerificationError, match="Oracle verification failed"):
        verify_all_items(registry, bad_abst, lists_by_anchor)

    # 4. Bad category or examples on a tool
    bad_reg_cat = [dict(registry[0], category="files")] + registry[1:]
    with pytest.raises(OracleVerificationError, match="category"):
        verify_domain_sibling_uniqueness(bad_reg_cat)

    bad_reg_ex = [dict(registry[0], examples=["example 1"])] + registry[1:]
    with pytest.raises(OracleVerificationError, match="examples"):
        verify_domain_sibling_uniqueness(bad_reg_ex)


def test_compass_lists_min_confidence_zero_and_embedding_text(r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    items = r0_bundle["items"]
    lists_by_anchor = r0_bundle["lists_by_anchor"]
    compass_data = r0_bundle["compass_data"]

    assert compass_data["min_confidence"] == 0.0
    assert compass_data["tool_compass_commit"] == TOOL_COMPASS_COMMIT == "6f5a233"
    assert compass_data["tool_category_constant"] == "tool"
    assert compass_data["tool_examples_constant"] == []

    # Verify ToolDefinition.embedding_text() format
    tdef = build_tool_definition(registry[0], cls=ToolDefinitionCompat)
    emb_text = tdef.embedding_text()
    assert f"Tool: {registry[0]['name']}" in emb_text
    assert "Category: tool" in emb_text
    assert f"Description: {registry[0]['description']}" in emb_text
    assert "Parameters: resource_id" in emb_text
    assert "Examples:" not in emb_text
    assert "additionalProperties" not in emb_text

    # Verify ToolDefinition with examples branch for coverage
    tdef_ex = ToolDefinitionCompat(
        name="t", description="d", category="tool", server="s", parameters={"p": {}}, examples=["ex1"]
    )
    assert 'Examples: "ex1"' in tdef_ex.embedding_text()

    # Every (item, N) has 5 tools and 5 raw scores from that item's own L_N
    for item in items:
        iid = item["item_id"]
        anchor_lists = lists_by_anchor[item["anchor_id"]]["lists"]
        c_item = compass_data["items"][iid]["lists"]
        for n in COMPASS_COUNTS:
            n_str = str(n)
            rec = c_item[n_str]
            assert len(rec["top5_tools"]) == 5
            assert len(rec["top5_scores"]) == 5
            assert rec["min_confidence"] == 0.0
            assert set(rec["top5_tools"]).issubset(set(anchor_lists[n_str]["tools"]))

    # Refuses non-zero min_confidence
    with pytest.raises(ValueError, match="min_confidence=0.0"):
        build_compass_lists(registry, items[:2], lists_by_anchor, min_confidence=0.3)

    # Refuses real Ollama embedding without explicit --allow-gpu grant
    with pytest.raises(RuntimeError, match="Publisher grant"):
        write_compass_artifacts(
            r0_dir=r0_bundle["r0_dir"],
            out_dir=r0_bundle["r0_dir"],
            embedder_mode="ollama",
            allow_gpu=False,
        )


def test_scorer_classifications_and_fallbacks(r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    items = r0_bundle["items"]
    tool_by_name = {t["name"]: t for t in registry}

    ans_item = next(it for it in items if not it["is_abstention"])
    abst_item = next(it for it in items if it["is_abstention"])
    target_tool = ans_item["key"]["tool"]
    target_args = ans_item["key"]["required_args"]

    # 1. Native tool_calls correct + value_correct + extra tool call ignored/logged
    rec_correct = {
        "item_id": ans_item["item_id"],
        "arm": "full",
        "cell": "full_5",
        "N": 5,
        "tool_list_order": [target_tool, "a", "b", "c", "d"],
        "target_position": 0,
        "position_bin": 0,
        "upstream_response": {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": target_tool,
                                    "arguments": json.dumps(target_args),
                                },
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "ignored_second_tool", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ]
        },
        "prompt_tokens": 500,
        "output_tokens": 30,
        "wall_time_s": 0.25,
    }
    s_ok = score_single_call(rec_correct, ans_item, tool_by_name)
    assert s_ok["classification"] == "correct"
    assert s_ok["exact_tool_correct"] == 1
    assert s_ok["schema_valid"] is True
    assert s_ok["value_correct"] is True
    assert s_ok["extra_tool_calls_count"] == 1

    # 2. Correct tool but wrong argument value (schema_valid=True, value_correct=False)
    wrong_val_args = dict(target_args, resource_id="wrong_resource_value")
    rec_wrong_val = dict(
        rec_correct,
        upstream_response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": target_tool,
                                    "arguments": json.dumps(wrong_val_args),
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )
    s_wv = score_single_call(rec_wrong_val, ans_item, tool_by_name)
    assert s_wv["classification"] == "correct"
    assert s_wv["schema_valid"] is True
    assert s_wv["value_correct"] is False

    # 3. Wrong tool call
    rec_wrong_tool = dict(
        rec_correct,
        upstream_response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "other_wrong_tool", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        },
    )
    s_wt = score_single_call(rec_wrong_tool, ans_item, tool_by_name)
    assert s_wt["classification"] == "wrong_tool"
    assert s_wt["exact_tool_correct"] == 0

    # 4. No call on answerable item (refusal)
    rec_nocall = dict(
        rec_correct,
        upstream_response={
            "choices": [{"message": {"content": "I cannot find an applicable tool.", "tool_calls": []}}]
        },
    )
    s_nc = score_single_call(rec_nocall, ans_item, tool_by_name)
    assert s_nc["classification"] == "no_call"
    assert s_nc["exact_tool_correct"] == 0
    assert s_nc["refused_on_answerable"] == 1

    # 5. Malformed call (broken JSON arguments)
    rec_malformed = dict(
        rec_correct,
        upstream_response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": target_tool, "arguments": "{broken_json"}}
                        ]
                    }
                }
            ]
        },
    )
    s_mal = score_single_call(rec_malformed, ans_item, tool_by_name)
    assert s_mal["classification"] == "malformed"
    assert s_mal["format_failure"] == 1
    assert s_mal["exact_tool_correct"] == 0

    # 6. Abstention item: correct abstention vs false tool call
    s_abst_ok = score_single_call(rec_nocall, abst_item, tool_by_name)
    assert s_abst_ok["classification"] == "abstention"
    assert s_abst_ok["abstention_correct"] == 1
    assert s_abst_ok["exact_tool_correct"] is None

    s_abst_false = score_single_call(rec_wrong_tool, abst_item, tool_by_name)
    assert s_abst_false["classification"] == "wrong_tool"
    assert s_abst_false["abstention_correct"] == 0

    # 7. Content fallbacks (<tool_call>, <|tool_call>call:...|>, JSON content)
    ext_tag = extract_tool_call_from_response(
        {
            "choices": [
                {
                    "message": {
                        "content": f'<tool_call>{{"name": "{target_tool}", "arguments": {{"resource_id": "x"}}}}</tool_call>',
                        "tool_calls": [],
                    }
                }
            ]
        }
    )
    assert ext_tag["status"] == "called" and ext_tag["tool_name"] == target_tool
    assert ext_tag["source"] == "content_fallback"

    ext_gemma = extract_tool_call_from_response(
        {
            "choices": [
                {
                    "message": {
                        "content": f'<|tool_call>call:{target_tool}{{"resource_id": "x"}}<tool_call|>',
                        "tool_calls": [],
                    }
                }
            ]
        }
    )
    assert ext_gemma["status"] == "called" and ext_gemma["tool_name"] == target_tool

    ext_json = extract_tool_call_from_response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"name": target_tool, "arguments": {"resource_id": "x"}}),
                        "tool_calls": None,
                    }
                }
            ]
        }
    )
    assert ext_json["status"] == "called" and ext_json["tool_name"] == target_tool


def test_cluster_sign_flip_holm_bootstrap_and_decision_rules() -> None:
    # 85 clusters: simulate a significant ~20-point cliff at N=40, compass recovery at N=40, and no length confound
    clusters_n5 = {f"cluster_{i:02d}": 1.0 if i < 78 else 0.0 for i in range(85)}
    clusters_n10 = {f"cluster_{i:02d}": 1.0 if i < 76 else 0.0 for i in range(85)}
    clusters_n40 = {f"cluster_{i:02d}": 1.0 if i < 58 else 0.0 for i in range(85)}

    perm_small = paired_sign_flip_test(clusters_n5, clusters_n10, n_flips=5000, seed=DEFAULT_SEED)
    assert 0.0 < perm_small["mean_diff_pts"] < 5.0
    assert perm_small["p_value"] > 0.05

    perm_cliff = paired_sign_flip_test(clusters_n5, clusters_n40, n_flips=5000, seed=DEFAULT_SEED)
    assert perm_cliff["mean_diff_pts"] > 20.0
    assert perm_cliff["p_value"] < 0.001

    boot_cliff = paired_cluster_bootstrap_ci(clusters_n5, clusters_n40, n_boot=1000, seed=DEFAULT_SEED)
    assert boot_cliff["ci_low_pts"] > 10.0
    assert boot_cliff["ci_high_pts"] > boot_cliff["ci_low_pts"]

    holm = holm_adjust({"n10": perm_small["p_value"], "n40": perm_cliff["p_value"]}, alpha=0.05)
    assert holm["n40"]["significant"] is True
    assert holm["n10"]["significant"] is False

    # Build synthetic scored calls across all 12 cells to test evaluate_model_decisions & evaluate_cross_model_decisions
    def _make_cell_rows(cell_name: str, arm: str, n_val: int, correct_clusters: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for c_idx in range(85):
            hit = 1 if c_idx < correct_clusters else 0
            for p_idx in (0, 1):
                rows.append(
                    {
                        "item_id": f"ans_{c_idx:02d}_p{p_idx}",
                        "cluster_id": f"cluster_{c_idx:02d}",
                        "phrasing_index": p_idx,
                        "is_abstention": False,
                        "half": "A" if c_idx < 43 else "B",
                        "arm": arm,
                        "cell": cell_name,
                        "N": n_val,
                        "classification": "correct" if hit else "wrong_tool",
                        "exact_tool_correct": hit,
                        "abstention_correct": None,
                        "refused_on_answerable": 0,
                        "format_failure": 0,
                        "schema_valid": bool(hit),
                        "value_correct": bool(hit),
                        "position_bin": (c_idx + p_idx * 3) % 6,
                        "prompt_tokens": n_val * 130 + 400,
                        "output_tokens": 35,
                        "wall_time_s": 0.4,
                    }
                )
        for a_idx in range(30):
            rows.append(
                {
                    "item_id": f"abst_{a_idx:02d}",
                    "cluster_id": f"abst_cluster_{a_idx:02d}",
                    "phrasing_index": 0,
                    "is_abstention": True,
                    "half": "A" if a_idx < 15 else "B",
                    "arm": arm,
                    "cell": cell_name,
                    "N": n_val,
                    "classification": "abstention",
                    "exact_tool_correct": None,
                    "abstention_correct": 1,
                    "refused_on_answerable": 0,
                    "format_failure": 0,
                    "schema_valid": False,
                    "value_correct": False,
                    "position_bin": None,
                    "prompt_tokens": n_val * 130 + 400,
                    "output_tokens": 12,
                    "wall_time_s": 0.2,
                }
            )
        return rows

    cells_map = {
        "full_5": _make_cell_rows("full_5", "full", 5, 80),
        "full_10": _make_cell_rows("full_10", "full", 10, 77),  # non-significant small drop -> 'not shown' with floor note
        "full_20": _make_cell_rows("full_20", "full", 20, 76),
        "full_40": _make_cell_rows("full_40", "full", 40, 55),  # >25 pt drop -> cliff at N=40, ceiling at N=20
        "full_70": _make_cell_rows("full_70", "full", 70, 50),
        "full_100": _make_cell_rows("full_100", "full", 100, 45),
        "compass_10": _make_cell_rows("compass_10", "compass", 10, 78),
        "compass_20": _make_cell_rows("compass_20", "compass", 20, 78),
        "compass_40": _make_cell_rows("compass_40", "compass", 40, 76),  # >20 pt recovery over full_40
        "compass_70": _make_cell_rows("compass_70", "compass", 70, 75),
        "compass_100": _make_cell_rows("compass_100", "compass", 100, 74),
        "length_control_10": _make_cell_rows("length_control_10", "length-control", 10, 76),
    }

    model_rep = evaluate_model_decisions(
        cells_by_key=cells_map,
        compass_recall_by_n={"10": 0.98, "20": 0.96, "40": 0.95, "70": 0.92, "100": 0.88},
        n_flips=2000,
        n_boot=200,
        seed=DEFAULT_SEED,
    )
    r1 = model_rep["rule1_cliff_and_ceiling"]
    assert r1["cliff_N"] == 40
    assert r1["ceiling_N"] == 20
    # Rule 4: 'not shown' cells carry the floor note
    assert r1["comparisons_vs_n5"]["full_10"]["verdict"] == "not shown"
    assert "drop under the floor" in r1["comparisons_vs_n5"]["full_10"]["floor_note"]

    r2 = model_rep["rule2_compass_narrowing"]["summary_at_cliff_or_100"]
    assert r2["evaluated_at_N"] == 40
    assert r2["helps_by_at_least_10pts_significant"] is True
    assert r2["recall_below_90pct_bound"] is True

    r3 = model_rep["rule3_length_confound"]
    assert r3["length_confound_significant"] is False
    assert r3["verdict"] == "length alone is not the cause here"

    cross = evaluate_cross_model_decisions(
        {"granite-4.2-8b": model_rep, "qwen3.8-27b": model_rep, "gemma-4-31b-it": model_rep}
    )
    assert cross["rule1_studio_ceiling"]["studio_limit_N"] == 20
    assert cross["rule2_studio_narrowing"]["verdict"] == "build narrowing in"


def test_runner_smoke_assert_and_r1_stub_all_9_branches(tmp_path: Path) -> None:
    # 1. Verify smoke assert fails loudly when thinking_tokens > 0 or output_tokens >= 80
    with StubToolCountServer(fail_smoke_thinking=True) as bad_srv:
        with pytest.raises(SmokeAssertError, match="thinking_tokens=180"):
            run_preloop_smoke_assert(
                endpoint_url=bad_srv.base_url,
                model_name="granite-4.2-8b",
                sample_item={"prompt": "test prompt"},
                sample_tools=build_registry()[:5],
                seed=DEFAULT_SEED,
            )

    # 2. Run the full R1 CPU stub verification reaching all 9 required branches
    r1_report = run_r1_stub_verification(work_dir=tmp_path / "r1_stub", seed=DEFAULT_SEED)
    assert r1_report["r1_passed"] is True
    branches = r1_report["branches_reached"]
    assert branches == {
        "correct": True,
        "wrong_tool": True,
        "no_call": True,
        "malformed": True,
        "abstention": True,
        "length_control_cell": True,
        "compass_cell": True,
        "resume": True,
        "tokenizer_stop": True,
    }
    assert r1_report["tokenizer_stop_receipt"]["measurement_rows_written"] == 0


def test_cli_entrypoints_and_launch_w1_script(tmp_path: Path) -> None:
    r0_out = tmp_path / "cli_r0"
    assert gen_main(["--out-dir", str(r0_out), "--seed", str(DEFAULT_SEED)]) == 0
    assert oracle_main(["--r0-dir", str(r0_out)]) == 0
    assert compass_main(["--r0-dir", str(r0_out), "--out-dir", str(r0_out), "--embedder", "fake"]) == 0

    win_out = tmp_path / "cli_win"
    with StubToolCountServer() as srv:
        assert (
            runner_main(
                [
                    "--mode",
                    "window",
                    "--endpoint-url",
                    srv.base_url,
                    "--model",
                    "granite-4.2-8b",
                    "--r0-dir",
                    str(r0_out),
                    "--out-dir",
                    str(win_out),
                    "--half",
                    "A",
                    "--max-calls",
                    "12",
                ]
            )
            == 0
        )

    score_out = tmp_path / "cli_score.json"
    assert (
        scorer_main(
            [
                "--calls-jsonl",
                str(win_out / "calls.jsonl"),
                "--r0-dir",
                str(r0_out),
                "--out",
                str(score_out),
                "--n-flips",
                "500",
                "--n-boot",
                "100",
            ]
        )
        == 0
    )
    assert score_out.is_file()

    # Check launch-w1.sh contents, LF line endings, and C2-era guards
    launch_sh = Path(__file__).resolve().parents[1] / "scripts" / "toolcount" / "launch-w1.sh"
    raw = launch_sh.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert 'PIN_CAP_UNITS="${PIN_CAP_UNITS:-13}"' in text
    assert 'PIN_WALL_S="${PIN_WALL_S:-9000}"' in text
    assert "LIVE WALL EXCEEDED" in text
    assert "LIVE CAP CEILING REACHED" in text
    assert "DEAD HARNESS" in text
    assert "PIN_TOKENIZER_HARD_STOP=\"1\"" in text


def test_edge_cases_across_all_modules(tmp_path: Path, r0_bundle: dict[str, Any]) -> None:
    registry = r0_bundle["registry"]
    items = r0_bundle["items"]
    lists_by_anchor = r0_bundle["lists_by_anchor"]

    # 1. FakeEmbedder & cosine_similarity & resolve_tool_definition_class edge cases
    with pytest.raises(ValueError, match="dim must be positive"):
        FakeEmbedder(dim=0)
    emb = FakeEmbedder(dim=16)
    v_empty = emb.embed_text("!!!")
    assert v_empty[0] == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0], [1.0, 0.0])

    fake_tc_dir = tmp_path / "fake_tc"
    fake_tc_dir.mkdir()
    (fake_tc_dir / "tool_manifest.py").write_text(
        "class ToolDefinition:\n    pass\n", encoding="utf-8"
    )
    cls_loaded = resolve_tool_definition_class(fake_tc_dir)
    assert cls_loaded.__name__ == "ToolDefinition"

    # 2. Oracle validation error branches
    with pytest.raises(OracleVerificationError, match="Expected 100 tools"):
        verify_domain_sibling_uniqueness(registry[:10])
    dup_reg = [dict(registry[0], name=registry[1]["name"])] + registry[1:]
    with pytest.raises(OracleVerificationError, match="Duplicate tool names"):
        verify_domain_sibling_uniqueness(dup_reg)
    bad_dom_reg = [dict(registry[0], domain="extra_domain")] + registry[1:]
    with pytest.raises(OracleVerificationError, match="Expected 20 domains"):
        verify_domain_sibling_uniqueness(bad_dom_reg)
    skew_dom_reg = [dict(registry[0], domain=registry[5]["domain"])] + registry[1:5] + [dict(registry[5], domain=registry[0]["domain"])] + registry[6:]
    # skew_dom_reg with 6 in one domain and 4 in another:
    skew_reg2 = [dict(registry[0], domain=registry[5]["domain"])] + registry[1:]
    with pytest.raises(OracleVerificationError, match="expected 5"):
        verify_domain_sibling_uniqueness(skew_reg2)
    dup_verb_reg = [dict(registry[0], action_verb=registry[1]["action_verb"])] + registry[1:]
    with pytest.raises(OracleVerificationError, match="5 distinct action verbs"):
        verify_domain_sibling_uniqueness(dup_verb_reg)
    dup_cond_reg = [dict(registry[0], oracle_action_token=registry[1]["oracle_action_token"])] + registry[1:]
    with pytest.raises(OracleVerificationError, match="5 distinct conditions"):
        verify_domain_sibling_uniqueness(dup_cond_reg)

    # verify_all_items structural failure checks
    with pytest.raises(OracleVerificationError, match="Missing anchor_id"):
        verify_all_items(registry, [dict(items[0], anchor_id="missing_anchor")], lists_by_anchor)
    bad_lists = dict(lists_by_anchor)
    anchor0 = items[0]["anchor_id"]
    bad_lists[anchor0] = {"lists": {k: v for k, v in lists_by_anchor[anchor0]["lists"].items() if k != "10"}}
    with pytest.raises(OracleVerificationError, match="Missing L_10"):
        verify_all_items(registry, [items[0]], bad_lists)
    bad_lists2 = dict(lists_by_anchor)
    bad_lists2[anchor0] = {
        "lists": {
            **lists_by_anchor[anchor0]["lists"],
            "5": {"tools": lists_by_anchor[anchor0]["lists"]["5"]["tools"][:4]},
        }
    }
    with pytest.raises(OracleVerificationError, match="Invalid L_5 length"):
        verify_all_items(registry, [items[0]], bad_lists2)

    # 3. Scorer extract_tool_call_from_response & check_arguments_grade edge cases
    assert extract_tool_call_from_response(None)["status"] == "malformed"
    assert extract_tool_call_from_response({})["status"] == "malformed"
    assert extract_tool_call_from_response({"choices": [{}]})["status"] == "malformed"
    assert extract_tool_call_from_response({"choices": [{"message": {"tool_calls": ["bad"]}}]})["status"] == "malformed"
    assert extract_tool_call_from_response({"choices": [{"message": {"tool_calls": [{}]}}]})["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "", "arguments": "{}"}}]}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "t", "arguments": "[1,2]"}}]}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"tool_calls": [{"function": {"name": "t", "arguments": 123}}]}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"content": 123, "tool_calls": []}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"content": "<tool_call>{bad}</tool_call>", "tool_calls": []}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"content": "<|tool_call>call:foo{bad}<tool_call|>", "tool_calls": []}}]}
    )["status"] == "malformed"
    assert extract_tool_call_from_response(
        {"choices": [{"message": {"content": "{bad_json", "tool_calls": []}}]}
    )["status"] == "malformed"

    schema_sample = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "o": {"type": "object"},
            "a": {"type": "array"},
        },
        "required": ["s", "i"],
        "additionalProperties": False,
    }
    assert check_arguments_grade(None, schema_sample, {}) == (False, False)
    assert check_arguments_grade({"s": "ok"}, schema_sample, {}) == (False, False)
    assert check_arguments_grade({"s": "ok", "i": 1, "extra": 2}, schema_sample, {}) == (False, False)
    assert check_arguments_grade({"s": "ok", "i": True}, schema_sample, {}) == (False, False)
    assert check_arguments_grade(
        {"s": "ok", "i": 1, "n": 1.5, "b": False, "o": {}, "a": []},
        schema_sample,
        {"s": "ok", "i": 1},
    ) == (True, True)

    # Zero-difference sign-flip & bootstrap
    zero_a = {"c1": 1.0, "c2": 0.5}
    assert paired_sign_flip_test(zero_a, zero_a)["p_value"] == 1.0
    assert paired_cluster_bootstrap_ci(zero_a, zero_a)["ci_low"] == 0.0
    with pytest.raises(ValueError, match="No overlapping"):
        paired_sign_flip_test({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="No overlapping"):
        paired_cluster_bootstrap_ci({"a": 1.0}, {"b": 1.0})

    # 4. Runner helpers (count_thinking_tokens, build_call_schedule, study_config, r1-stub CLI)
    assert count_thinking_tokens("not_a_dict") == 0  # type: ignore[arg-type]
    assert count_thinking_tokens({"thinking_tokens": 12}) == 12
    assert count_thinking_tokens({"choices": [{"message": {"reasoning_content": "step one two"}}]}) == 3
    assert count_thinking_tokens({"choices": [{"message": {"content": "<think>a b c d</think>done"}}]}) == 4
    assert build_inert_padding_block(0) == ""
    assert compute_first_50_repricing([])["sample_size"] == 0
    assert load_resume_journal(None) == {}
    assert load_resume_journal(tmp_path / "nonexistent") == {}
    with pytest.raises(ValueError, match="half must be"):
        build_call_schedule(items, half="INVALID")
    cfg = build_study_config(extra_models=["custom-model-7b"])
    assert "custom-model-7b" in cfg["models"]

    assert runner_main(["--mode", "r1-stub", "--out-dir", str(tmp_path / "cli_r1")]) == 0


def test_fix1_compass_ollama_embedder_prefixes_and_metadata(
    tmp_path: Path, r0_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix 1: Ollama nomic-embed-text embedder uses /api/embed with search_document:/search_query: prefixes
    and records embedder_kind, embedder_model, and embedder_endpoint in compass_lists.json.
    Tested against a local fake HTTP endpoint (never the rig's Ollama).
    """
    import http.server
    import threading
    from scripts.toolcount.compass_lists import FakeEmbedder, write_compass_artifacts

    recorded_requests: list[dict[str, Any]] = []
    cpu_fake = FakeEmbedder(dim=16, seed=DEFAULT_SEED)

    class FakeOllamaEmbedHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded_requests.append({"path": self.path, "body": body})
            raw_input = body.get("input", "")
            texts = [raw_input] if isinstance(raw_input, str) else list(raw_input)
            embeddings = [cpu_fake.embed_text(t) for t in texts]
            resp = json.dumps({"model": body.get("model", ""), "embeddings": embeddings}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    r0_mini = tmp_path / "r0_mini"
    r0_mini.mkdir(parents=True, exist_ok=True)
    registry = r0_bundle["registry"][:10]
    item0 = dict(r0_bundle["items"][0])
    item0["key"] = dict(item0["key"])
    item0["key"]["tool"] = registry[0]["name"]
    tool_names_10 = [t["name"] for t in registry]
    nested = {
        item0["anchor_id"]: {
            "anchor_id": item0["anchor_id"],
            "lists": {
                str(n): {"N": n, "tools": (tool_names_10 * ((n // 10) + 1))[:n]}
                for n in COMPASS_COUNTS
            },
        }
    }
    (r0_mini / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (r0_mini / "items.json").write_text(json.dumps([item0]), encoding="utf-8")
    (r0_mini / "nested_lists.json").write_text(json.dumps(nested), encoding="utf-8")

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllamaEmbedHandler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out_ollama = tmp_path / "out_ollama"
        write_compass_artifacts(
            r0_dir=r0_mini,
            out_dir=out_ollama,
            seed=DEFAULT_SEED,
            embedder_mode="ollama",
            allow_gpu=True,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert len(recorded_requests) > 0, "Expected OllamaEmbedder to call /api/embed on the fake HTTP server"
    assert all(req["path"] == "/api/embed" for req in recorded_requests)
    assert all(req["body"]["model"] == "nomic-embed-text" for req in recorded_requests)

    inputs: list[str] = []
    for req in recorded_requests:
        inp = req["body"]["input"]
        if isinstance(inp, str):
            inputs.append(inp)
        else:
            inputs.extend(inp)

    doc_inputs = [s for s in inputs if s.startswith("search_document: ")]
    query_inputs = [s for s in inputs if s.startswith("search_query: ")]
    assert len(doc_inputs) == 10, f"Expected 10 tool document embeddings with 'search_document: ', got {len(doc_inputs)}"
    assert len(query_inputs) >= 1, f"Expected query embeddings with 'search_query: ', got {len(query_inputs)}"

    compass_payload = json.loads((out_ollama / "compass_lists.json").read_text(encoding="utf-8"))
    assert compass_payload["embedder_kind"] == "ollama"
    assert compass_payload["embedder_model"] == "nomic-embed-text"
    assert compass_payload["embedder_endpoint"] == f"{base_url}/api/embed"

    # Also verify default fake embedder records kind, model, and endpoint
    out_fake = tmp_path / "out_fake"
    write_compass_artifacts(r0_dir=r0_mini, out_dir=out_fake, seed=DEFAULT_SEED)
    fake_payload = json.loads((out_fake / "compass_lists.json").read_text(encoding="utf-8"))
    assert fake_payload["embedder_kind"] == "fake"
    assert fake_payload["embedder_model"] == "fake-hash-96d"
    assert fake_payload["embedder_endpoint"] == "none"

    # Also verify tool_compass_dir SyncEmbedder delegation when embedder.py is present
    fake_tc = tmp_path / "fake_tc"
    fake_tc.mkdir(parents=True, exist_ok=True)
    (fake_tc / "embedder.py").write_text(
        "class SyncEmbedder:\n"
        "    def __init__(self, **kw):\n"
        "        self.kw = kw\n"
        "    def embed(self, text):\n"
        "        return [1.0, 0.0, 0.0, 0.0]\n"
        "    def embed_query(self, query):\n"
        "        return [0.8, 0.6, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    from scripts.toolcount.compass_lists import OllamaEmbedder

    tc_emb = OllamaEmbedder(base_url=base_url, tool_compass_dir=fake_tc)
    assert tc_emb.embed_document("doc") == [1.0, 0.0, 0.0, 0.0]
    assert tc_emb.embed_query("q") == [0.8, 0.6, 0.0, 0.0]
    assert tc_emb.embed_text("doc") == [1.0, 0.0, 0.0, 0.0]


def test_fix2_calibrate_length_control_padding_both_directions(r0_bundle: dict[str, Any]) -> None:
    """Fix 2: calibrate_length_control_padding adjusts both when calibrated_tokens > tokens_100
    and when calibrated_tokens < tokens_100, using (tokens_100 - tokens_10) / max(1, calibrated_tokens - tokens_10),
    keeping calibrated_prompt_tokens matched to target_100_tool_tokens and well below N_CTX=16384.
    """
    import http.server
    import threading

    registry_by_name = {t["name"]: t for t in r0_bundle["registry"]}
    first_item = r0_bundle["items"][0]
    anchor_lists = r0_bundle["lists_by_anchor"][first_item["anchor_id"]]["lists"]
    l10 = [registry_by_name[n] for n in anchor_lists["10"]["tools"]]
    l100 = [registry_by_name[n] for n in anchor_lists["100"]["tools"]]

    # Direction A: initial padding overshoots target_100_tool_tokens (calibrated_tokens > tokens_100)
    cal_over = calibrate_length_control_padding(
        server_url=None,
        l10_tools=l10,
        l100_tools=l100,
        sample_prompt=first_item["prompt"],
    )
    assert cal_over["calibrated_prompt_tokens"] < 16384, (
        f"Padded L_10 overflowed N_CTX=16384: {cal_over['calibrated_prompt_tokens']}"
    )
    assert abs(cal_over["calibrated_prompt_tokens"] - cal_over["target_100_tool_tokens"]) <= 64, (
        f"Expected calibrated_prompt_tokens ~ {cal_over['target_100_tool_tokens']}, "
        f"got {cal_over['calibrated_prompt_tokens']}"
    )

    # Direction B: initial padding undershoots target_100_tool_tokens (calibrated_tokens < tokens_100)
    class CompactFillerTokenizeHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            content = body.get("content", "")
            # Simulate a tokenizer where non-filler text is 1 token per 2 chars (large tokens_10 and tokens_100)
            # and each filler line is only 10 tokens (< 18 assumed by build_inert_padding_block), so initial
            # padding undershoots tokens_100.
            if "--- BEGIN INERT LENGTH-CONTROL PADDING BLOCK ---" in content:
                before, after_begin = content.split("--- BEGIN INERT LENGTH-CONTROL PADDING BLOCK ---", 1)
                pad_inner, after = after_begin.split("--- END INERT LENGTH-CONTROL PADDING BLOCK ---", 1)
                base_toks = (len(before) + len(after)) // 2
                n_lines = sum(1 for ln in pad_inner.splitlines() if "REFERENCE_FILLER_LINE:" in ln)
                total_toks = base_toks + n_lines * 10
            else:
                total_toks = len(content) // 2
            resp = json.dumps({"tokens": [1] * total_toks}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), CompactFillerTokenizeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cal_under = calibrate_length_control_padding(
            server_url=f"http://127.0.0.1:{port}",
            l10_tools=l10,
            l100_tools=l100,
            sample_prompt=first_item["prompt"],
        )
    finally:
        server.shutdown()
        server.server_close()

    assert abs(cal_under["calibrated_prompt_tokens"] - cal_under["target_100_tool_tokens"]) <= 64, (
        f"Undershoot branch failed to converge to {cal_under['target_100_tool_tokens']}: "
        f"got {cal_under['calibrated_prompt_tokens']}"
    )


def test_fix3_toolcount_window_clones_roleset_before_import_even_in_stub(
    tmp_path: Path, r0_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix 3: toolcount_window runs role_set_env and install_rnd BEFORE importing scripts.toolcount
    and does NOT skip them when stub_mode() is True.
    """
    repo_root_str = str(Path(__file__).resolve().parents[1])
    events: list[str] = []

    def fake_role_set_env(commit: str) -> dict[str, Any]:
        events.append(f"role_set_env:{commit}")
        return {"commit": commit, "src": repo_root_str, "role_set": "role-set", "python": "3.15.0"}

    def fake_install_rnd(commit: str) -> str:
        events.append(f"install_rnd:{commit}")
        return commit

    work_dir = tmp_path / "work"
    content_dir = tmp_path / "content"
    r0_dest = work_dir / "toolcount" / "r0"
    r0_dest.mkdir(parents=True, exist_ok=True)
    for fn in ("registry.json", "items.json", "nested_lists.json", "compass_lists.json"):
        (r0_dest / fn).write_text((r0_bundle["r0_dir"] / fn).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(pw, "WORK", work_dir)
    monkeypatch.setattr(pw, "CONTENT", content_dir)
    monkeypatch.setattr(pw, "_repo_root", lambda: content_dir / "role-set")
    monkeypatch.setattr(pw, "role_set_env", fake_role_set_env)
    monkeypatch.setattr(pw, "install_rnd", fake_install_rnd)
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "deadbeef12345678")
    monkeypatch.setenv("PIN_RND_COMMIT", "6a8f1b5")
    monkeypatch.setenv("PIN_MODELS", "granite-4.2-8b")

    # Simulate fresh Colab VM: remove repo_root_str from sys.path and unload scripts.toolcount*
    saved_sys_path = list(sys.path)
    saved_modules = {k: v for k, v in sys.modules.items() if k == "scripts" or k.startswith("scripts.")}
    try:
        sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(repo_root_str).resolve()]
        for k in list(saved_modules):
            sys.modules.pop(k, None)

        res: dict[str, Any] = {}
        pw.toolcount_window({"commit": "c0", "binary_sha256": "s0"}, [], res)
    finally:
        sys.path[:] = saved_sys_path
        sys.modules.update(saved_modules)

    assert "role_set_env:deadbeef12345678" in events, (
        f"Expected role_set_env to run in stub mode before imports, got events={events}"
    )
    assert "install_rnd:6a8f1b5" in events, (
        f"Expected install_rnd to run in stub mode before imports, got events={events}"
    )
    assert res.get("role_set", {}).get("commit") == "deadbeef12345678"
    assert res.get("rnd") == "6a8f1b5"


def test_fix4_heartbeats_deadline_partial_tar_and_vram_library_sha(
    tmp_path: Path, r0_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fix 4:
    1) run_model_window emits HEARTBEAT lines to stdout (job.log).
    2) Per-call receipts record real peak_vram_mb (from sample_smi_every_2s / vram_fn) and library_sha (from built['libs']).
    3) Checking PIN_DEADLINE between calls cuts the window cleanly and still tars partial calls.jsonl/generations.jsonl.
    """
    import os
    import time

    r0_dir = r0_bundle["r0_dir"]
    items = r0_bundle["items"]
    stub_sched = build_r1_stub_schedule(items)[:4]

    # Part A: Verify heartbeats, library_sha from built['libs'], and live peak_vram_mb via toolcount_window
    work_dir = tmp_path / "work_a"
    content_dir = tmp_path / "content_a"
    r0_dest = work_dir / "toolcount" / "r0"
    r0_dest.mkdir(parents=True, exist_ok=True)
    for fn in ("registry.json", "items.json", "nested_lists.json", "compass_lists.json"):
        (r0_dest / fn).write_text((r0_dir / fn).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(pw, "WORK", work_dir)
    monkeypatch.setattr(pw, "CONTENT", content_dir)
    monkeypatch.setattr(pw, "JOB_LOG", work_dir / "job.log")
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_MODELS", "granite-4.2-8b")
    monkeypatch.setenv("PIN_HEARTBEAT_S", "0")
    monkeypatch.setattr(
        pw,
        "_query_smi_csv",
        lambda: "2026/10/28 00:00:00.000, 14320 MiB, 85 %, 52, 180.00 W\n",
    )

    built_libs = [
        {"name": "libcublas.so.13", "sha256": "aa" * 32, "bytes": 1234},
        {"name": "libcudart.so.13", "sha256": "bb" * 32, "bytes": 5678},
    ]
    built_info = {
        "commit": LLAMA_CPP_COMMIT,
        "sha256": LLAMA_SERVER_SHA256,
        "libs": built_libs,
    }
    res_a: dict[str, Any] = {}
    pw.toolcount_window(built_info, [], res_a)

    captured = capsys.readouterr().out
    assert "HEARTBEAT toolcount" in captured, "Expected run_model_window to emit HEARTBEAT progress lines to stdout"

    calls_lines = (work_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(calls_lines) > 0
    first_receipt = json.loads(calls_lines[0])
    assert first_receipt["peak_vram_mb"] == 14320, (
        f"Expected peak_vram_mb=14320 from sample_smi_every_2s, got {first_receipt['peak_vram_mb']}"
    )
    assert first_receipt["library_sha"] != LLAMA_SERVER_SHA256, (
        "Expected library_sha to be derived from built['libs'], not equal to server_sha256"
    )

    # Part B: Verify PIN_DEADLINE mid-run cuts between calls and tars partial calls.jsonl/generations.jsonl
    work_b = tmp_path / "work_b"
    content_b = tmp_path / "content_b"
    r0_dest_b = work_b / "toolcount" / "r0"
    r0_dest_b.mkdir(parents=True, exist_ok=True)
    for fn in ("registry.json", "items.json", "nested_lists.json", "compass_lists.json"):
        (r0_dest_b / fn).write_text((r0_dir / fn).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(pw, "WORK", work_b)
    monkeypatch.setattr(pw, "CONTENT", content_b)
    monkeypatch.setattr(pw, "JOB_LOG", work_b / "job.log")

    # Expire deadline after 2 calls inside run_model_window
    orig_send = sys.modules["scripts.toolcount.runner"].send_chat_completion
    call_counter = {"n": 0}

    def counting_send(*args: Any, **kwargs: Any) -> Any:
        call_counter["n"] += 1
        # call 1 is smoke assert; calls 2 and 3 are measurement calls 1 and 2
        if call_counter["n"] >= 3:
            os.environ["PIN_DEADLINE"] = str(time.time() - 10.0)
        return orig_send(*args, **kwargs)

    monkeypatch.setattr("scripts.toolcount.runner.send_chat_completion", counting_send)
    monkeypatch.setenv("PIN_DEADLINE", str(time.time() + 3600.0))

    res_b: dict[str, Any] = {}
    with pytest.raises(pw.DeadlineReached):
        pw.toolcount_window(built_info, [], res_b)

    assert (work_b / "calls.jsonl").is_file(), "Expected partial calls.jsonl to be copied on DeadlineReached"
    assert (work_b / "toolcount-out.tar").is_file(), "Expected toolcount-out.tar to be written on DeadlineReached"
    partial_rows = (work_b / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert 1 <= len(partial_rows) < len(stub_sched) + 5





