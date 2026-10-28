"""Colab window runner for the tool-count cliff study (Design v1 §§2, 3, 5, 7, 8).

Features:
- Generates and pins the study-specific proxy/server config (`temperature: 0.0`, fixed `seed`,
  `cache_prompt: false`, `id_slot: 0`, `chat_template_kwargs: {"enable_thinking": false}`),
  never `card-values.json`.
- Runs the three arms (`full` at N in {5,10,20,40,70,100}, `compass` top-5 at N in {10,20,40,70,100},
  and `length-control` at L_10 with one inert padding block in the system message outside all 10 tool
  schemas to match the model's 100-tool prompt token length).
- Executes the pre-loop smoke assertion (`thinking_tokens == 0` and `output_tokens < 80`) before
  each model's measurement loop starts; stops before any measurement row is written if violated.
- Supports Half A (43 target clusters + 15 abstentions = 1,212 calls), Half B (42 target clusters +
  15 abstentions = 1,188 calls), or ALL (2,400 calls), interleaved in a seeded order.
- Computes W2/W4 first-50-call repricing telemetry for W3/W5.
- Checkpoints every call to `calls.jsonl` and `generations.jsonl` (top-5 logprobs) and resumes
  from `PIN_RESUME_CALLS_DIR` without re-running completed calls.
- Includes `StubToolCountServer` so the CPU stub rung R1 (`PIN_STUB=1 PIN_GPU=CPU PIN_SKIP_BINARY=1`)
  reaches every branch on the exact HTTP code path: correct call, wrong call, no-call, malformed call,
  abstention item, length-control cell, compass cell, tokenizer-stop path, and resume.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import random
import socketserver
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.toolcount import (
    COMPASS_COUNTS,
    DEFAULT_SEED,
    LLAMA_CPP_COMMIT,
    LLAMA_SERVER_SHA256,
    RND_PROXY_COMMIT,
    STUDY_ID,
    SYSTEM_PROMPT,
    TOOL_COMPASS_COMMIT,
    TOOL_COUNTS,
)
from scripts.toolcount.compass_lists import build_compass_lists
from scripts.toolcount.gen_registry import (
    build_registry,
    order_tools_for_call,
    select_target_tools_and_items,
     sha256_bytes,
    write_with_sha256,
)
from scripts.toolcount.scorer import (
    evaluate_model_decisions,
    extract_tool_call_from_response,
    score_single_call,
)

# Pinned model metadata from role-set `writers.json` and `pin_window.py` (Design v1 §3)
PINNED_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "granite-4.2-8b": {
        "served_as": "granite-4.2-8b",
        "repo": "ibm-granite/granite-4.2-8b",
        "revision": "a7042144125ce5ae64f89610e8289647fc000b9a",
        "quant": "Q8_0",
        "gguf_file": "granite-4.2-8b-Q8_0.gguf",
        "gguf_sha256": "to pin",
        "tokenizer_check": True,
    },
    "qwen3.8-27b": {
        "served_as": "qwen3.8-27b",
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "quant": "Q8_0",
        "gguf_file": "qwen3.8-27b-Q8_0.gguf",
        "gguf_sha256": "2ab268e0bd3f8e5daf1280e2243ac19abd5e3bb34e8444f5cfff5147fab977d4",
        "tokenizer_check": False,
    },
    "gemma-4-31b-it": {
        "served_as": "gemma-4-31b-it",
        "repo": "google/gemma-4-31B-it",
        "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "quant": "Q8_0",
        "gguf_file": "gemma-4-31b-it-Q8_0.gguf",
        "gguf_sha256": "e357a45071c2cbbc44b97506dfadc34bac0d5f6592f1d500edfaa9647f01e7bf",
        "tokenizer_check": False,
    },
}

MAX_OUTPUT_TOKENS = 256
N_CTX = 16384


class SmokeAssertError(RuntimeError):
    """Raised when the pre-loop smoke assertion (`thinking_tokens == 0`, `output_tokens < 80`) fails."""


def build_study_config(
    seed: int = DEFAULT_SEED,
    extra_models: list[str] | None = None,
) -> dict[str, Any]:
    """Build the study-pinned proxy/sampling config (never `card-values.json`, Design §3)."""
    model_names = list(PINNED_MODEL_SPECS.keys())
    for m in extra_models or []:
        if m not in model_names:
            model_names.append(m)

    models_cfg: dict[str, Any] = {}
    for m in model_names:
        spec = PINNED_MODEL_SPECS.get(
            m,
            {
                "served_as": m,
                "repo": m,
                "revision": "to pin",
                "quant": "Q8_0",
                "gguf_sha256": "to pin",
            },
        )
        models_cfg[m] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "quant": spec["quant"],
            "gguf_sha256": spec["gguf_sha256"],
            "n_ctx": N_CTX,
            "n_parallel": 1,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "sampling": {
                "temperature": 0.0,
                "seed": seed,
                "cache_prompt": False,
                "id_slot": 0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
            },
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

    return {
        "study_id": STUDY_ID,
        "rnd_proxy_commit": RND_PROXY_COMMIT,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "llama_server_sha256": LLAMA_SERVER_SHA256,
        "seed": seed,
        "n_ctx": N_CTX,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "models": models_cfg,
    }


def write_study_config(out_path: Path, seed: int = DEFAULT_SEED, extra_models: list[str] | None = None) -> tuple[dict[str, Any], str]:
    """Write `study_config.json` + `.sha256` and return `(config_dict, sha256)`."""
    cfg = build_study_config(seed=seed, extra_models=extra_models)
    sha = write_with_sha256(out_path, json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return cfg, sha


def format_openai_tool(tool_entry: dict[str, Any]) -> dict[str, Any]:
    """Format a registry tool into the OpenAI `tools` entry passed to llama-server `--jinja`."""
    return {
        "type": "function",
        "function": {
            "name": tool_entry["name"],
            "description": tool_entry["description"],
            "parameters": tool_entry["input_schema"],
        },
    }


def build_inert_padding_block(target_pad_tokens: int) -> str:
    """Build a single inert padding block to place in the system message outside all tool schemas.

    Design v1 §2: '10 tools (L_10), with inert text added in one block outside all 10 tool schemas
    (in the system message; no description or parameter is padded) to bring the prompt to the token
    length of the 100-tool prompt'.
    """
    if target_pad_tokens <= 0:
        return ""
    unit_line = (
        "REFERENCE_FILLER_LINE: neutral background calibration note for context window "
        "length normalization across fixed evaluation cells."
    )
    # ~18 tokens per line; assemble exact token count or character proxy
    repeats = max(1, target_pad_tokens // 18)
    lines = [f"[{idx:04d}] {unit_line}" for idx in range(repeats)]
    block = (
        "\n\n--- BEGIN INERT LENGTH-CONTROL PADDING BLOCK ---\n"
        + "\n".join(lines)
        + "\n--- END INERT LENGTH-CONTROL PADDING BLOCK ---"
    )
    return block


def calibrate_length_control_padding(
    server_url: str | None,
    l10_tools: list[dict[str, Any]],
    l100_tools: list[dict[str, Any]],
    sample_prompt: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Calibrate the single-block system-message padding so L_10 + padding matches L_100 token count."""
    l10_payload = json.dumps([format_openai_tool(t) for t in l10_tools], sort_keys=True)
    l100_payload = json.dumps([format_openai_tool(t) for t in l100_tools], sort_keys=True)

    def _count_tokens(text: str) -> int:
        if server_url:
            try:
                req = urllib.request.Request(
                    f"{server_url.rstrip('/')}/tokenize",
                    data=json.dumps({"content": text}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    toks = body.get("tokens")
                    if isinstance(toks, list):
                        return len(toks)
            except Exception:
                pass
        # Deterministic fallback token estimate (~4 chars per token)
        return max(1, len(text) // 4)

    base_10_text = f"{SYSTEM_PROMPT}\n{l10_payload}\n{sample_prompt}"
    base_100_text = f"{SYSTEM_PROMPT}\n{l100_payload}\n{sample_prompt}"
    tokens_10 = _count_tokens(base_10_text)
    tokens_100 = _count_tokens(base_100_text)
    needed_pad_tokens = max(64, tokens_100 - tokens_10)

    padding_block = build_inert_padding_block(needed_pad_tokens)
    padded_10_text = f"{SYSTEM_PROMPT}{padding_block}\n{l10_payload}\n{sample_prompt}"
    calibrated_tokens = _count_tokens(padded_10_text)

    # Fine-tune if server /tokenize is active and gap > 32 tokens
    if calibrated_tokens < tokens_100 and (tokens_100 - calibrated_tokens) > 16:
        extra_Ratio = tokens_100 / max(1, calibrated_tokens - tokens_10)
        adjusted = int(needed_pad_tokens * extra_Ratio)
        padding_block = build_inert_padding_block(adjusted)
        padded_10_text = f"{SYSTEM_PROMPT}{padding_block}\n{l10_payload}\n{sample_prompt}"
        calibrated_tokens = _count_tokens(padded_10_text)

    pad_sha = sha256_bytes(padding_block.encode("utf-8"))
    return {
        "padding_block": padding_block,
        "padding_chars": len(padding_block),
        "padding_block_sha256": pad_sha,
        "unpadded_l10_tokens": tokens_10,
        "target_100_tool_tokens": tokens_100,
        "calibrated_prompt_tokens": calibrated_tokens,
    }


def build_call_schedule(
    items: list[dict[str, Any]],
    half: str = "ALL",
    seed: int = DEFAULT_SEED,
    max_calls: int | None = None,
) -> list[dict[str, Any]]:
    """Build the interleaved call schedule across the 12 cells for `half` ('ALL', 'A', or 'B').

    Design §5 & §7:
    - 12 cells per item:
      - `full`: N in {5, 10, 20, 40, 70, 100} (6 cells)
      - `compass`: N in {10, 20, 40, 70, 100} (5 cells)
      - `length-control`: N = 10 with 100-tool token length padding (1 cell)
    - Interleaved in a seeded order so a slow-VM spell cannot align with one count N,
      and the first 50 calls cover all arms/counts for W2/W4 repricing.
    """
    half_upper = half.upper()
    if half_upper not in ("ALL", "A", "B"):
        raise ValueError(f"half must be 'ALL', 'A', or 'B', got {half!r}")

    filtered_items = [
        it for it in items if half_upper == "ALL" or it.get("half") == half_upper
    ]

    cells_spec: list[tuple[str, str, int]] = [
        ("full", f"full_{n}", n) for n in TOOL_COUNTS
    ] + [
        ("compass", f"compass_{n}", n) for n in COMPASS_COUNTS
    ] + [
        ("length-control", "length_control_10", 10),
    ]
    assert len(cells_spec) == 12

    schedule: list[dict[str, Any]] = []
    for it in filtered_items:
        for arm, cell_name, n_val in cells_spec:
            schedule.append(
                {
                    "item_id": it["item_id"],
                    "arm": arm,
                    "cell": cell_name,
                    "N": n_val,
                }
            )

    rng = random.Random(f"{seed}:schedule:{half_upper}")
    rng.shuffle(schedule)

    if max_calls is not None and max_calls > 0:
        schedule = schedule[:max_calls]

    return schedule


def count_thinking_tokens(upstream_response: dict[str, Any]) -> int:
    """Extract thinking token count from upstream response (`usage` or `reasoning_content` or `<think>`)."""
    if not isinstance(upstream_response, dict):
        return 0
    if "thinking_tokens" in upstream_response and isinstance(upstream_response["thinking_tokens"], int):
        return int(upstream_response["thinking_tokens"])

    usage = upstream_response.get("usage")
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
            return int(details["reasoning_tokens"])

    choices = upstream_response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(msg, dict):
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                return max(1, len(rc.strip().split()))
            content = msg.get("content")
            if isinstance(content, str) and "<think>" in content and "</think>" in content:
                start = content.find("<think>") + len("<think>")
                end = content.find("</think>", start)
                if end > start:
                    think_body = content[start:end].strip()
                    if think_body:
                        return max(1, len(think_body.split()))
    return 0


def send_chat_completion(
    endpoint_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    timeout: int = 60,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Send one batch-1 tool-calling request to `/v1/chat/completions` and return `(response_dict, wall_s)`."""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": False,
        "id_slot": 0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if extra_body:
        payload.update(extra_body)

    url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()
    wall_s = time.monotonic() - t0
    body = json.loads(raw_bytes.decode("utf-8"))
    return body, wall_s


def run_preloop_smoke_assert(
    endpoint_url: str,
    model_name: str,
    sample_item: dict[str, Any],
    sample_tools: list[dict[str, Any]],
    seed: int = DEFAULT_SEED,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run the pre-loop smoke assertion before any measurement row is written (Design §3, §8 item 2).

    Asserts `thinking_tokens == 0` and `output_tokens < 80`.
    Raises `SmokeAssertError` if violated.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample_item["prompt"]},
    ]
    openai_tools = [format_openai_tool(t) for t in sample_tools]
    body, wall_s = send_chat_completion(
        endpoint_url=endpoint_url,
        model_name=model_name,
        messages=messages,
        tools=openai_tools,
        seed=seed,
        timeout=timeout,
        extra_body={"_smoke_check": True},
    )

    upstream = body.get("upstream_response") if isinstance(body.get("upstream_response"), dict) else body
    usage = upstream.get("usage") if isinstance(upstream, dict) else {}
    if not isinstance(usage, dict):
        usage = {}

    output_tokens = int(usage.get("completion_tokens", body.get("output_tokens", 0)) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", body.get("prompt_tokens", 0)) or 0)
    thinking_tokens = count_thinking_tokens(upstream)

    passed = (thinking_tokens == 0) and (0 < output_tokens < 80)
    report = {
        "passed": passed,
        "model": model_name,
        "thinking_tokens": thinking_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "wall_time_s": round(wall_s, 4),
    }
    if not passed:
        raise SmokeAssertError(
            f"Pre-loop smoke assert failed for {model_name}: "
            f"thinking_tokens={thinking_tokens} (expected 0), "
            f"output_tokens={output_tokens} (expected 1..79)"
        )
    return report


def compute_first_50_repricing(
    call_records: list[dict[str, Any]],
    total_planned_calls: int = 1200,
    load_hours_p90: float = 0.60,
    rate_units_per_hour: float = 5.30,
    cap_units: float = 16.0,
    wall_ceiling_units: float = 15.9,
) -> dict[str, Any]:
    """Compute W2/W4 first-50-call repricing estimate for W3/W5 (Design v1 §5)."""
    sample = call_records[:50]
    if not sample:
        return {"sample_size": 0, "within_ceiling": True}

    total_wall_s = sum(float(r.get("wall_time_s") or 0.0) for r in sample)
    total_prompt_tok = sum(int(r.get("prompt_tokens") or 0) for r in sample)
    total_out_tok = sum(int(r.get("output_tokens") or 0) for r in sample)
    mean_wall_s = total_wall_s / len(sample)

    # Project to full half (1,200 calls) with 1.25x p90 safety factor + load_hours_p90
    projected_call_hours_central = (mean_wall_s * total_planned_calls) / 3600.0
    projected_call_hours_p90 = projected_call_hours_central * 1.25
    total_hours_p90 = projected_call_hours_p90 + load_hours_p90
    projected_units_p90 = total_hours_p90 * rate_units_per_hour
    binding_ceiling = min(cap_units, wall_ceiling_units)

    return {
        "sample_size": len(sample),
        "mean_wall_time_s": round(mean_wall_s, 4),
        "sample_prompt_tokens": total_prompt_tok,
        "sample_output_tokens": total_out_tok,
        "projected_half_hours_p90": round(total_hours_p90, 4),
        "projected_half_units_p90": round(projected_units_p90, 2),
        "binding_ceiling_units": binding_ceiling,
        "within_ceiling": projected_units_p90 <= binding_ceiling,
    }


def load_resume_journal(resume_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Load already-completed calls keyed by `call_id` from `resume_dir/calls.jsonl`."""
    if resume_dir is None:
        return {}
    calls_file = resume_dir / "calls.jsonl" if resume_dir.is_dir() else resume_dir
    if not calls_file.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in calls_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cid = row.get("call_id")
            if isinstance(cid, str):
                completed[cid] = row
        except Exception:
            continue
    return completed


def run_model_window(
    endpoint_url: str,
    model_name: str,
    r0_dir: Path,
    out_dir: Path,
    half: str = "ALL",
    seed: int = DEFAULT_SEED,
    resume_calls_dir: Path | None = None,
    max_calls: int | None = None,
    schedule_override: list[dict[str, Any]] | None = None,
    model_meta: dict[str, Any] | None = None,
    machine_line: str = "CPU-stub",
    peak_vram_mb: int = 0,
    timeout: int = 60,
) -> dict[str, Any]:
    """Execute the tool-count window for `model_name`, writing `calls.jsonl`, `generations.jsonl`,
    and `window_summary.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads((r0_dir / "registry.json").read_text(encoding="utf-8"))
    items = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
    lists_by_anchor = json.loads((r0_dir / "nested_lists.json").read_text(encoding="utf-8"))
    compass_data = json.loads((r0_dir / "compass_lists.json").read_text(encoding="utf-8"))

    tool_by_name = {t["name"]: t for t in registry}
    item_by_id = {it["item_id"]: it for it in items}

    study_cfg_path = out_dir / "study_config.json"
    _cfg_dict, study_cfg_sha = write_study_config(
        study_cfg_path, seed=seed, extra_models=[model_name]
    )
    sys_prompt_sha = sha256_bytes(SYSTEM_PROMPT.encode("utf-8"))

    meta = dict(PINNED_MODEL_SPECS.get(model_name, {}))
    if model_meta:
        meta.update(model_meta)
    model_digest = meta.get("revision", "to pin")
    quant = meta.get("quant", "Q8_0")
    gguf_sha256 = meta.get("gguf_sha256", "to pin")
    server_build = meta.get("server_build", LLAMA_CPP_COMMIT)
    server_sha256 = meta.get("server_sha256", LLAMA_SERVER_SHA256)
    library_sha = meta.get("library_sha", server_sha256)
    template_sha256 = meta.get(
        "template_sha256",
        sha256_bytes(f"jinja:{model_name}:{model_digest}".encode("utf-8")),
    )

    # 1. Pre-loop smoke assertion BEFORE any measurement row is written
    first_ans_item = next(it for it in items if not it["is_abstention"])
    first_l5_names = lists_by_anchor[first_ans_item["anchor_id"]]["lists"]["5"]["tools"]
    first_l5_tools = [tool_by_name[n] for n in first_l5_names]

    smoke_report = run_preloop_smoke_assert(
        endpoint_url=endpoint_url,
        model_name=model_name,
        sample_item=first_ans_item,
        sample_tools=first_l5_tools,
        seed=seed,
        timeout=timeout,
    )

    # 2. Calibrate length-control single-block system-message padding for this model
    first_l10_names = lists_by_anchor[first_ans_item["anchor_id"]]["lists"]["10"]["tools"]
    first_l100_names = lists_by_anchor[first_ans_item["anchor_id"]]["lists"]["100"]["tools"]
    lc_calibration = calibrate_length_control_padding(
        server_url=endpoint_url,
        l10_tools=[tool_by_name[n] for n in first_l10_names],
        l100_tools=[tool_by_name[n] for n in first_l100_names],
        sample_prompt=first_ans_item["prompt"],
        timeout=timeout,
    )
    padded_system_prompt = SYSTEM_PROMPT + lc_calibration["padding_block"]
    padded_sys_sha = sha256_bytes(padded_system_prompt.encode("utf-8"))

    # 3. Build schedule and load resume journal
    schedule = (
        schedule_override
        if schedule_override is not None
        else build_call_schedule(items=items, half=half, seed=seed, max_calls=max_calls)
    )
    reused_map = load_resume_journal(resume_calls_dir)

    calls_path = out_dir / "calls.jsonl"
    generations_path = out_dir / "generations.jsonl"
    all_records: list[dict[str, Any]] = []
    reused_count = 0
    fresh_count = 0

    with (
        calls_path.open("w", encoding="utf-8", newline="\n") as calls_f,
        generations_path.open("w", encoding="utf-8", newline="\n") as gen_f,
    ):
        for step_idx, sched_entry in enumerate(schedule):
            item_id = sched_entry["item_id"]
            arm = sched_entry["arm"]
            cell = sched_entry["cell"]
            n_val = int(sched_entry["N"])
            call_id = f"{model_name}:{item_id}:{cell}"

            if call_id in reused_map:
                rec = dict(reused_map[call_id])
                rec["reused_from_journal"] = True
                calls_f.write(json.dumps(rec, sort_keys=True) + "\n")
                calls_f.flush()
                gen_entry = {
                    "call_id": call_id,
                    "item_id": item_id,
                    "model": model_name,
                    "cell": cell,
                    "reused_from_journal": True,
                    "top_logprobs": rec.get("top_logprobs", []),
                }
                gen_f.write(json.dumps(gen_entry, sort_keys=True) + "\n")
                gen_f.flush()
                all_records.append(rec)
                reused_count += 1
                continue

            item = item_by_id[item_id]
            anchor_id = item["anchor_id"]
            expected_tool = item["key"]["tool"]
            pos_bin = item.get("position_bin")

            compass_meta: dict[str, Any] | None = None
            length_control_meta: dict[str, Any] | None = None

            if arm == "full":
                raw_tool_names = lists_by_anchor[anchor_id]["lists"][str(n_val)]["tools"]
                sys_msg = SYSTEM_PROMPT
                active_sys_sha = sys_prompt_sha
            elif arm == "compass":
                c_info = compass_data["items"][item_id]["lists"][str(n_val)]
                raw_tool_names = list(c_info["top5_tools"])
                sys_msg = SYSTEM_PROMPT
                active_sys_sha = sys_prompt_sha
                compass_meta = {
                    "top5_tools": list(c_info["top5_tools"]),
                    "top5_scores": list(c_info["top5_scores"]),
                    "min_confidence": c_info["min_confidence"],
                    "tool_compass_commit": c_info["tool_compass_commit"],
                    "target_in_top5": c_info["target_in_top5"],
                }
            elif arm == "length-control":
                raw_tool_names = lists_by_anchor[anchor_id]["lists"]["10"]["tools"]
                sys_msg = padded_system_prompt
                active_sys_sha = padded_sys_sha
                length_control_meta = {
                    "padding_chars": lc_calibration["padding_chars"],
                    "padding_block_sha256": lc_calibration["padding_block_sha256"],
                    "unpadded_l10_tokens": lc_calibration["unpadded_l10_tokens"],
                    "target_100_tool_tokens": lc_calibration["target_100_tool_tokens"],
                    "calibrated_prompt_tokens": lc_calibration["calibrated_prompt_tokens"],
                }
            else:
                raise ValueError(f"Unknown arm: {arm}")

            ordering = order_tools_for_call(
                tool_names=raw_tool_names,
                target_tool=expected_tool,
                position_bin=pos_bin,
                seed=seed,
                call_key=call_id,
            )
            ordered_names = ordering["ordered_tools"]
            openai_tools = [format_openai_tool(tool_by_name[name]) for name in ordered_names]

            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": item["prompt"]},
            ]
            extra_body: dict[str, Any] = {
                "_call_id": call_id,
                "_item_id": item_id,
                "_arm": arm,
                "_cell": cell,
                "_N": n_val,
                "_is_abstention": item["is_abstention"],
                "_expected_tool": expected_tool,
                "_expected_args": item["key"]["required_args"],
                "_step_idx": step_idx,
            }
            if sched_entry.get("_stub_force_branch"):
                extra_body["_stub_force_branch"] = sched_entry["_stub_force_branch"]

            resp_body, wall_s = send_chat_completion(
                endpoint_url=endpoint_url,
                model_name=model_name,
                messages=messages,
                tools=openai_tools,
                seed=seed,
                timeout=timeout,
                extra_body=extra_body,
            )

            upstream = (
                resp_body.get("upstream_response")
                if isinstance(resp_body.get("upstream_response"), dict)
                else resp_body
            )
            usage = upstream.get("usage") if isinstance(upstream, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            prompt_tokens = int(usage.get("prompt_tokens", resp_body.get("prompt_tokens", 0)) or 0)
            output_tokens = int(usage.get("completion_tokens", resp_body.get("output_tokens", 0)) or 0)
            thinking_tokens = count_thinking_tokens(upstream)

            choices = upstream.get("choices") if isinstance(upstream, dict) else []
            finish_reason = (
                choices[0].get("finish_reason")
                if (isinstance(choices, list) and choices and isinstance(choices[0], dict))
                else None
            )
            truncated = bool(finish_reason == "length" or output_tokens >= MAX_OUTPUT_TOKENS)
            logprobs_data = (
                choices[0].get("logprobs")
                if (isinstance(choices, list) and choices and isinstance(choices[0], dict))
                else None
            )

            parsed_call = extract_tool_call_from_response(upstream)

            prelim_record = {
                "call_id": call_id,
                "item_id": item_id,
                "arm": arm,
                "cell": cell,
                "N": n_val,
                "tool_list_order": ordered_names,
                "target_position": ordering["target_position"],
                "position_bin": ordering["position_bin"],
                "upstream_response": upstream,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "wall_time_s": round(wall_s, 6),
            }
            scored = score_single_call(prelim_record, item, tool_by_name)

            receipt: dict[str, Any] = {
                "call_id": call_id,
                "item_id": item_id,
                "arm": arm,
                "cell": cell,
                "N": n_val,
                "tool_list_order": ordered_names,
                "target_position": ordering["target_position"],
                "position_bin": ordering["position_bin"],
                "anchor_id": anchor_id,
                "anchor_domain": item["anchor_domain"],
                "cluster_id": item["cluster_id"],
                "half": item["half"],
                "is_abstention": item["is_abstention"],
                "model": model_name,
                "model_digest": model_digest,
                "quant": quant,
                "gguf_sha256": gguf_sha256,
                "server_build": server_build,
                "server_sha256": server_sha256,
                "library_sha": library_sha,
                "study_config_sha256": study_cfg_sha,
                "template_sha256": template_sha256,
                "system_prompt_sha256": active_sys_sha,
                "seed": seed,
                "upstream_response": upstream,
                "parsed_call": {
                    "status": parsed_call["status"],
                    "tool_name": parsed_call["tool_name"],
                    "arguments": parsed_call["arguments"],
                    "source": parsed_call["source"],
                    "extra_tool_calls_count": parsed_call["extra_tool_calls_count"],
                    "error": parsed_call["error"],
                },
                "grade": {
                    "classification": scored["classification"],
                    "exact_tool_correct": scored["exact_tool_correct"],
                    "abstention_correct": scored["abstention_correct"],
                    "schema_valid": scored["schema_valid"],
                    "value_correct": scored["value_correct"],
                    "refused_on_answerable": scored["refused_on_answerable"],
                    "format_failure": scored["format_failure"],
                },
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "truncated": truncated,
                "wall_time_s": round(wall_s, 6),
                "peak_vram_mb": peak_vram_mb,
                "machine_line": machine_line,
                "compass_meta": compass_meta,
                "length_control_meta": length_control_meta,
                "reused_from_journal": False,
            }
            calls_f.write(json.dumps(receipt, sort_keys=True) + "\n")
            calls_f.flush()

            gen_record = {
                "call_id": call_id,
                "item_id": item_id,
                "model": model_name,
                "arm": arm,
                "cell": cell,
                "N": n_val,
                "seed": seed,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "top_logprobs": logprobs_data,
                "reused_from_journal": False,
            }
            gen_f.write(json.dumps(gen_record, sort_keys=True) + "\n")
            gen_f.flush()

            all_records.append(receipt)
            fresh_count += 1

    repricing_50 = compute_first_50_repricing(all_records)
    class_counts: dict[str, int] = {
        "correct": 0,
        "wrong_tool": 0,
        "no_call": 0,
        "malformed": 0,
        "abstention": 0,
    }
    arms_reached: set[str] = set()
    for r in all_records:
        c = r["grade"]["classification"]
        class_counts[c] = class_counts.get(c, 0) + 1
        arms_reached.add(r["arm"])

    summary = {
        "study_id": STUDY_ID,
        "model": model_name,
        "half": half,
        "seed": seed,
        "total_calls": len(all_records),
        "fresh_calls": fresh_count,
        "reused_calls": reused_count,
        "smoke_assert": smoke_report,
        "length_control_calibration": {
            k: v for k, v in lc_calibration.items() if k != "padding_block"
        },
        "first_50_repricing": repricing_50,
        "classification_counts": class_counts,
        "arms_reached": sorted(arms_reached),
        "study_config_sha256": study_cfg_sha,
    }
    summary_sha = write_with_sha256(
        out_dir / "window_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    summary["window_summary_sha256"] = summary_sha
    return summary


class _ReusableTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class StubToolCountServer:
    """In-process HTTP stub server for R1 CPU stub runs and unit tests.

    Handles `/health`, `/props`, `/tokenize`, `/detokenize`, and `/v1/chat/completions`,
    deterministically exercising every branch required by Card kh item 9:
      - `correct` call
      - `wrong_tool` call
      - `no_call` on an answerable item
      - `malformed` call
      - `abstention` on an abstention item
      - `length-control` cell
      - `compass` cell
      - pre-loop smoke assertion (`thinking_tokens == 0`, `output_tokens < 80`)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        fail_smoke_thinking: bool = False,
        force_tokenizer_mismatch: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.fail_smoke_thinking = fail_smoke_thinking
        self.force_tokenizer_mismatch = force_tokenizer_mismatch
        self.requests_log: list[dict[str, Any]] = []
        self._server: _ReusableTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("StubToolCountServer is not started")
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                if self.path.startswith("/health"):
                    self._send_json(200, {"status": "ok"})
                elif self.path.startswith("/props"):
                    self._send_json(
                        200,
                        {
                            "default_generation_settings": {
                                "n_ctx": N_CTX,
                            },
                            "total_slots": 1,
                        },
                    )
                else:
                    self._send_json(404, {"error": "not_found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    body = json.loads(raw_body.decode("utf-8"))
                except Exception:
                    body = {}

                if self.path.startswith("/tokenize"):
                    content = str(body.get("content", ""))
                    if parent.force_tokenizer_mismatch:
                        tokens = [999999, 888888]
                    else:
                        tokens = [
                            100 + (b % 5000) for b in content.encode("utf-8")
                        ]
                    self._send_json(200, {"tokens": tokens})
                    return

                if self.path.startswith("/detokenize"):
                    tokens = body.get("tokens") or []
                    self._send_json(200, {"content": "x" * len(tokens)})
                    return

                if self.path.startswith("/v1/chat/completions"):
                    parent.requests_log.append(body)
                    resp = parent._build_chat_response(body)
                    self._send_json(200, resp)
                    return

                self._send_json(404, {"error": "not_found"})

        self._server = _ReusableTCPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> StubToolCountServer:
        self.start()
        return self

    def __exit__(self, * exc_info: Any) -> None:
        self.stop()

    def _build_chat_response(self, body: dict[str, Any]) -> dict[str, Any]:
        tools = body.get("tools") or []
        messages = body.get("messages") or []
        raw_text = json.dumps({"messages": messages, "tools": tools})
        prompt_tokens = max(32, len(raw_text) // 4)

        if body.get("_smoke_check"):
            if self.fail_smoke_thinking:
                return {
                    "id": "chatcmpl-stub-smoke-fail",
                    "object": "chat.completion",
                    "model": body.get("model", "stub"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "<think>long reasoning trace</think>",
                                "reasoning_content": "thinking tokens active",
                                "tool_calls": [],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 256,
                        "total_tokens": prompt_tokens + 256,
                        "completion_tokens_details": {"reasoning_tokens": 180},
                    },
                }
            first_tool_name = (
                tools[0]["function"]["name"]
                if tools
                else "inspect_file_manifest"
            )
            return {
                "id": "chatcmpl-stub-smoke-ok",
                "object": "chat.completion",
                "model": body.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_smoke_0",
                                    "type": "function",
                                    "function": {
                                        "name": first_tool_name,
                                        "arguments": json.dumps({"resource_id": "smoke"}),
                                    },
                                }
                            ],
                        },
                        "logprobs": {
                            "content": [
                                {
                                    "token": first_tool_name,
                                    "logprob": -0.01,
                                    "top_logprobs": [
                                        {"token": first_tool_name, "logprob": -0.01},
                                        {"token": "other", "logprob": -4.60},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 24,
                    "total_tokens": prompt_tokens + 24,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }

        forced = body.get("_stub_force_branch")
        is_abst = bool(body.get("_is_abstention", False))
        expected_tool = body.get("_expected_tool")
        expected_args = body.get("_expected_args") or {}
        step_idx = int(body.get("_step_idx", 0) or 0)

        offered_names = [
            t.get("function", {}).get("name")
            for t in tools
            if isinstance(t, dict)
        ]

        if forced is None:
            if is_abst:
                forced = "abstention"
            else:
                # Deterministically cycle branches if not explicitly tagged
                mod = step_idx % 10
                if mod == 7:
                    forced = "wrong_tool"
                elif mod == 8:
                    forced = "no_call"
                elif mod == 9:
                    forced = "malformed"
                else:
                    forced = "correct"

        if forced == "abstention" or forced == "no_call":
            message_obj: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
            }
            finish_reason = "stop"
            out_tokens = 12
        elif forced == "malformed":
            bad_name = offered_names[0] if offered_names else "broken_tool"
            message_obj = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_malformed_{step_idx}",
                        "type": "function",
                        "function": {
                            "name": bad_name,
                            "arguments": "{unclosed_json_arguments: true",
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
            out_tokens = 28
        elif forced == "wrong_tool":
            distractors = [n for n in offered_names if n and n != expected_tool]
            picked = distractors[0] if distractors else "nonexistent_tool"
            message_obj = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_wrong_{step_idx}",
                        "type": "function",
                        "function": {
                            "name": picked,
                            "arguments": json.dumps({"resource_id": "wrong"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
            out_tokens = 32
        else:
            # forced == "correct"
            picked = (
                expected_tool
                if (expected_tool and expected_tool in offered_names)
                else (offered_names[0] if offered_names else expected_tool or "inspect_file_manifest")
            )
            message_obj = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_ok_{step_idx}",
                        "type": "function",
                        "function": {
                            "name": picked,
                            "arguments": json.dumps(expected_args, sort_keys=True),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
            out_tokens = 36

        top5_lp = [
            {"token": "tok_0", "logprob": -0.02},
            {"token": "tok_1", "logprob": -3.95},
            {"token": "tok_2", "logprob": -5.10},
            {"token": "tok_3", "logprob": -6.20},
            {"token": "tok_4", "logprob": -7.05},
        ]
        return {
            "id": f"chatcmpl-stub-{step_idx}",
            "object": "chat.completion",
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message_obj,
                    "logprobs": {
                        "content": [
                            {
                                "token": "tok_0",
                                "logprob": -0.02,
                                "top_logprobs": top5_lp,
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": prompt_tokens + out_tokens,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }


def build_r1_stub_schedule(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact deterministic schedule for R1 CPU stub that explicitly reaches every branch:
    - correct call (`full`)
    - wrong_tool call (`full`)
    - no_call on answerable item (`full`)
    - malformed call (`full`)
    - abstention item (`full`)
    - length-control cell (`length-control`)
    - compass cell (`compass`)
    """
    ans_items = [it for it in items if not it["is_abstention"]]
    abst_items = [it for it in items if it["is_abstention"]]

    return [
        {
            "item_id": ans_items[0]["item_id"],
            "arm": "full",
            "cell": "full_5",
            "N": 5,
            "_stub_force_branch": "correct",
        },
        {
            "item_id": ans_items[1]["item_id"],
            "arm": "full",
            "cell": "full_10",
            "N": 10,
            "_stub_force_branch": "wrong_tool",
        },
        {
            "item_id": ans_items[2]["item_id"],
            "arm": "full",
            "cell": "full_20",
            "N": 20,
            "_stub_force_branch": "no_call",
        },
        {
            "item_id": ans_items[3]["item_id"],
            "arm": "full",
            "cell": "full_40",
            "N": 40,
            "_stub_force_branch": "malformed",
        },
        {
            "item_id": abst_items[0]["item_id"],
            "arm": "full",
            "cell": "full_5",
            "N": 5,
            "_stub_force_branch": "abstention",
        },
        {
            "item_id": ans_items[4]["item_id"],
            "arm": "length-control",
            "cell": "length_control_10",
            "N": 10,
            "_stub_force_branch": "correct",
        },
        {
            "item_id": ans_items[5]["item_id"],
            "arm": "compass",
            "cell": "compass_40",
            "N": 40,
            "_stub_force_branch": "correct",
        },
    ]


def run_r1_stub_verification(
    work_dir: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute the full CPU stub rung R1 (`PIN_STUB=1 PIN_GPU=CPU PIN_SKIP_BINARY=1`) reaching
    every required branch:
      1. `correct` call
      2. `wrong_tool` call
      3. `no_call`
      4. `malformed` call
      5. `abstention` item
      6. `length-control` cell
      7. `compass` cell
      8. `tokenizer_stop` path (forced mismatch stops before any measurement row is written)
      9. `resume` from journal (`PIN_RESUME_CALLS_DIR`)
    """
    from scripts.toolcount.gen_registry import generate_all
    from scripts.toolcount.compass_lists import write_compass_artifacts

    r0_dir = work_dir / "r0"
    generate_all(out_dir=r0_dir, seed=seed)
    write_compass_artifacts(r0_dir=r0_dir, out_dir=r0_dir, seed=seed, embedder_mode="fake")

    items = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
    schedule = build_r1_stub_schedule(items)

    # Pass 1: run first 5 items of the schedule
    run1_dir = work_dir / "run_initial"
    with StubToolCountServer() as srv:
        summary_1 = run_model_window(
            endpoint_url=srv.base_url,
            model_name="granite-4.2-8b",
            r0_dir=r0_dir,
            out_dir=run1_dir,
            half="ALL",
            seed=seed,
            schedule_override=schedule[:5],
            machine_line="CPU-stub-R1",
        )

    # Pass 2: resume from run1_dir and complete all 7 items (reuses 5, runs 2 fresh: length-control & compass)
    run2_dir = work_dir / "run_resumed"
    with StubToolCountServer() as srv:
        summary_2 = run_model_window(
            endpoint_url=srv.base_url,
            model_name="granite-4.2-8b",
            r0_dir=r0_dir,
            out_dir=run2_dir,
            half="ALL",
            seed=seed,
            resume_calls_dir=run1_dir,
            schedule_override=schedule,
            machine_line="CPU-stub-R1",
        )

    # Pass 3: verify tokenizer-stop path via pin_window with forced mismatch
    from scripts.colab import pin_window

    tok_stop_dir = work_dir / "tokenizer_stop_check"
    tok_stop_dir.mkdir(parents=True, exist_ok=True)
    prev_env = {k: os.environ.get(k) for k in (
        "PIN_STUB",
        "PIN_GPU",
        "PIN_SKIP_BINARY",
        "PIN_ROLE",
        "PIN_MODELS",
        "PIN_OUT_DIR",
        "PIN_TOKENIZER_HARD_STOP",
        "PIN_FORCE_TOKENIZER_MISMATCH",
    )}
    try:
        os.environ["PIN_STUB"] = "1"
        os.environ["PIN_GPU"] = "CPU"
        os.environ["PIN_SKIP_BINARY"] = "1"
        os.environ["PIN_ROLE"] = "toolcount"
        os.environ["PIN_MODELS"] = "granite-4.2-8b"
        os.environ["PIN_OUT_DIR"] = str(tok_stop_dir)
        os.environ["PIN_TOKENIZER_HARD_STOP"] = "1"
        os.environ["PIN_FORCE_TOKENIZER_MISMATCH"] = "1"
        tok_exit_code = pin_window.main()
    finally:
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    tok_receipt = json.loads((tok_stop_dir / "pin-result.json").read_text(encoding="utf-8"))
    tok_calls_file = tok_stop_dir / "calls.jsonl"
    tok_rows_written = (
        len(tok_calls_file.read_text(encoding="utf-8").splitlines())
        if tok_calls_file.exists()
        else 0
    )

    branches = {
        "correct": summary_2["classification_counts"].get("correct", 0) >= 1,
        "wrong_tool": summary_2["classification_counts"].get("wrong_tool", 0) >= 1,
        "no_call": summary_2["classification_counts"].get("no_call", 0) >= 1,
        "malformed": summary_2["classification_counts"].get("malformed", 0) >= 1,
        "abstention": summary_2["classification_counts"].get("abstention", 0) >= 1,
        "length_control_cell": "length-control" in summary_2["arms_reached"],
        "compass_cell": "compass" in summary_2["arms_reached"],
        "resume": summary_2["reused_calls"] == 5 and summary_2["fresh_calls"] == 2,
        "tokenizer_stop": (
            tok_exit_code != 0
            and tok_receipt.get("ok") is False
            and tok_rows_written == 0
            and bool(tok_receipt.get("stopped_reason"))
        ),
    }
    if not all(branches.values()):
        raise RuntimeError(f"R1 stub verification missed branches: {branches}")

    report = {
        "study_id": STUDY_ID,
        "r1_passed": True,
        "branches_reached": branches,
        "initial_run": summary_1,
        "resumed_run": summary_2,
        "tokenizer_stop_receipt": {
            "exit_code": tok_exit_code,
            "ok": tok_receipt.get("ok"),
            "stopped_reason": tok_receipt.get("stopped_reason"),
            "measurement_rows_written": tok_rows_written,
        },
    }
    write_with_sha256(
        work_dir / "r1_stub_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tool-count cliff study window or R1 CPU stub.")
    parser.add_argument("--mode", choices=("window", "r1-stub"), default="window")
    parser.add_argument("--endpoint-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--model", type=str, default="granite-4.2-8b")
    parser.add_argument("--r0-dir", type=Path, default=Path("artifacts/toolcount/r0"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/toolcount/window"))
    parser.add_argument("--half", choices=("ALL", "A", "B"), default="ALL")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume-calls-dir", type=Path, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args(argv)

    if args.mode == "r1-stub":
        rep = run_r1_stub_verification(work_dir=args.out_dir, seed=args.seed)
        print(json.dumps({"r1_passed": rep["r1_passed"], "branches_reached": rep["branches_reached"]}, indent=2))
        return 0

    resume_dir = args.resume_calls_dir
    if resume_dir is None and os.environ.get("PIN_RESUME_CALLS_DIR"):
        resume_dir = Path(os.environ["PIN_RESUME_CALLS_DIR"])

    summary = run_model_window(
        endpoint_url=args.endpoint_url,
        model_name=args.model,
        r0_dir=args.r0_dir,
        out_dir=args.out_dir,
        half=args.half,
        seed=args.seed,
        resume_calls_dir=resume_dir,
        max_calls=args.max_calls,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
