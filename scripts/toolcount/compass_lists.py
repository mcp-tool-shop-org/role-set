"""Compass top-5 retrieval lists per (item, N) for N in {10, 20, 40, 70, 100}.

Implements Design v1 §2 ('Compass retrieval'):
- Builds each registry tool as a tool-compass `ToolDefinition` with constant `category="tool"`,
  `examples=[]`, `parameters=dict(input_schema["properties"])`, `raw_schema=input_schema`.
- Embeds `ToolDefinition.embedding_text()` (name, category, description, parameter names; never raw_schema).
- Retrieves top-5 for each (item, N) strictly from that item's own L_N with `min_confidence=0.0`.
- Records all 5 raw cosine scores per (item, N), `min_confidence=0.0`, `tool_compass_commit="6f5a233"`,
  and reports `recall@5` per N on the 170 answerable items.
- Tested with a deterministic CPU `FakeEmbedder` (no GPU, no Ollama, no downloads). Real Ollama
  embedding is gated behind an explicit `--allow-gpu` flag for R&D after Publisher grant.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from scripts.toolcount import (
    COMPASS_COUNTS,
    DEFAULT_SEED,
    STUDY_ID,
    TOOL_CATEGORY_CONSTANT,
    TOOL_COMPASS_COMMIT,
)
from scripts.toolcount.gen_registry import write_with_sha256

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DOCUMENT_PREFIX = "search_document: "
OLLAMA_EMBED_PATH = "/api/embed"


@dataclasses.dataclass
class ToolDefinitionCompat:
    """Compatible mirror of tool-compass `ToolDefinition` (`tool_manifest.py:23-64` at `6f5a233`)."""

    name: str
    description: str
    category: str
    server: str
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    examples: list[str] = dataclasses.field(default_factory=list)
    raw_schema: dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def embedding_text(self) -> str:
        """Exact match to `tool_manifest.py:46-64` at commit `6f5a233`."""
        parts = [
            f"Tool: {self.name}",
            f"Category: {self.category}",
            f"Description: {self.description}",
        ]
        if self.examples:
            examples_str = ", ".join(f'"{ex}"' for ex in self.examples[:4])
            parts.append(f"Examples: {examples_str}")
        if self.parameters:
            param_names = ", ".join(self.parameters.keys())
            parts.append(f"Parameters: {param_names}")
        return " | ".join(parts)


def resolve_tool_definition_class(tool_compass_dir: Path | None = None) -> type[Any]:
    """Load `ToolDefinition` from tool-compass if available, else return `ToolDefinitionCompat`."""
    if tool_compass_dir is not None and (tool_compass_dir / "tool_manifest.py").is_file():
        tc_str = str(tool_compass_dir.resolve())
        if tc_str not in sys.path:
            sys.path.insert(0, tc_str)
        try:
            import tool_manifest  # type: ignore[import-not-found]

            return tool_manifest.ToolDefinition
        except Exception:
            return ToolDefinitionCompat
    return ToolDefinitionCompat


def build_tool_definition(
    tool_entry: dict[str, Any],
    cls: type[Any] = ToolDefinitionCompat,
) -> Any:
    """Build a `ToolDefinition` with `category='tool'` and `examples=[]` (R&D ruling, Design §2)."""
    props = dict(tool_entry["input_schema"].get("properties", {}))
    return cls(
        name=tool_entry["name"],
        description=tool_entry["description"],
        category=TOOL_CATEGORY_CONSTANT,
        server="toolcount_registry",
        parameters=props,
        examples=[],
        raw_schema=tool_entry["input_schema"],
    )


class EmbedderProtocol(Protocol):
    """Protocol for query and document embedding."""

    def embed_text(self, text: str) -> list[float]: ...


_WORD_RE = re.compile(r"[a-z0-9_]+")


class FakeEmbedder:
    """Deterministic CPU-only embedder for testing and R1 stub (no GPU, no network, no Ollama).

    Produces L2-normalized vectors from token/bigram hashes so tools sharing domain and condition
    phrases with the prompt have high cosine similarity, without touching any external service.
    """

    kind: str = "fake"
    model: str = "fake-hash-96d"
    endpoint: str = "none"

    def __init__(self, dim: int = 96, seed: int = DEFAULT_SEED) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.seed = seed
        self.model = f"fake-hash-{dim}d"

    def embed_text(self, text: str) -> list[float]:
        tokens = _WORD_RE.findall(text.lower())
        vec = [0.0] * self.dim
        # Unigrams + bigrams for phrase sensitivity
        features = list(tokens) + [
            f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
        ]
        if not features:
            vec[0] = 1.0
            return vec

        for feat in features:
            digest = hashlib.sha256(f"{self.seed}:{feat}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.5
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_document(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)


def _l2_normalize(raw: list[float]) -> list[float]:
    if not raw:
        raise RuntimeError("Empty embedding vector returned by provider")
    norm = math.sqrt(sum(float(v) * float(v) for v in raw))
    if norm == 0.0:
        raise RuntimeError("Zero-norm embedding vector returned by provider")
    return [float(v) / norm for v in raw]


class OllamaEmbedder:
    """Real Ollama `nomic-embed-text` embedder (`tool-compass/embedder.py:45`, `/api/embed`).

    Applies nomic task prefixes (`search_document: ` for tool definitions, `search_query: ` for
    user prompts) and L2-normalizes returned vectors. Delegates to `tool-compass`'s `SyncEmbedder`
    when `tool_compass_dir` is provided and its dependencies are installed, or POSTs directly to
    `{base_url}/api/embed`.
    """

    kind: str = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        model: str = EMBEDDING_MODEL,
        tool_compass_dir: Path | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        resolved_base = (base_url or os.environ.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).rstrip("/")
        self.base_url = resolved_base
        self.model = model
        self.endpoint = f"{resolved_base}{OLLAMA_EMBED_PATH}"
        self.timeout_s = timeout_s
        self._sync_embedder: Any = None
        if tool_compass_dir is not None and (tool_compass_dir / "embedder.py").is_file():
            tc_str = str(tool_compass_dir.resolve())
            if tc_str not in sys.path:
                sys.path.insert(0, tc_str)
            try:
                import embedder as tc_embedder  # type: ignore[import-not-found]

                self._sync_embedder = tc_embedder.SyncEmbedder(
                    base_url=self.base_url,
                    model=self.model,
                    provider="ollama",
                )
            except Exception:
                self._sync_embedder = None

    def _post_embed(self, prefixed_text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": prefixed_text}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "embeddings" in data and data["embeddings"]:
            raw = [float(x) for x in data["embeddings"][0]]
        elif "embedding" in data and data["embedding"]:
            raw = [float(x) for x in data["embedding"]]
        else:
            raise RuntimeError(f"Unexpected Ollama /api/embed response keys: {sorted(data.keys())}")
        return _l2_normalize(raw)

    def embed_document(self, text: str) -> list[float]:
        if self._sync_embedder is not None:
            vec = self._sync_embedder.embed(text)
            return _l2_normalize([float(x) for x in vec])
        return self._post_embed(f"{NOMIC_DOCUMENT_PREFIX}{text}")

    def embed_query(self, query: str) -> list[float]:
        if self._sync_embedder is not None:
            vec = self._sync_embedder.embed_query(query)
            return _l2_normalize([float(x) for x in vec])
        return self._post_embed(f"{NOMIC_QUERY_PREFIX}{query}")

    def embed_text(self, text: str) -> list[float]:
        return self.embed_document(text)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimension mismatch")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve_top5_for_item_list(
    query: str,
    candidate_tool_names: list[str],
    tool_vectors: dict[str, list[float]],
    embedder: EmbedderProtocol,
    min_confidence: float = 0.0,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Retrieve top-k tools from `candidate_tool_names` by cosine similarity with `min_confidence=0.0`."""
    embed_q = getattr(embedder, "embed_query", None)
    q_vec = embed_q(query) if callable(embed_q) else embedder.embed_text(query)
    scored: list[tuple[str, float]] = []
    for name in candidate_tool_names:
        t_vec = tool_vectors[name]
        score = cosine_similarity(q_vec, t_vec)
        if score >= min_confidence or min_confidence <= 0.0:
            scored.append((name, round(score, 6)))

    # Sort descending by score, tie-break deterministically by tool name
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k]


def build_compass_lists(
    registry: list[dict[str, Any]],
    items: list[dict[str, Any]],
    lists_by_anchor: dict[str, dict[str, Any]],
    embedder: EmbedderProtocol | None = None,
    embedder_label: str = "fake_embedder_cpu",
    min_confidence: float = 0.0,
    tool_compass_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute top-5 compass lists for every (item, N) with N in {10, 20, 40, 70, 100}."""
    if min_confidence != 0.0:
        raise ValueError("Design v1 §2 requires min_confidence=0.0 for compass retrieval")

    active_embedder: EmbedderProtocol = embedder if embedder is not None else FakeEmbedder(seed=seed)
    td_cls = resolve_tool_definition_class(tool_compass_dir)

    tool_defs: dict[str, Any] = {}
    embedding_texts: dict[str, str] = {}
    tool_vectors: dict[str, list[float]] = {}
    embed_doc = getattr(active_embedder, "embed_document", None)

    for t in registry:
        t_def = build_tool_definition(t, cls=td_cls)
        if t_def.category != TOOL_CATEGORY_CONSTANT or t_def.examples != []:
            raise ValueError(
                f"Invalid ToolDefinition category/examples on {t['name']}: "
                f"category={t_def.category!r}, examples={t_def.examples!r}"
            )
        emb_text = t_def.embedding_text()
        # Verify raw_schema is NOT in embedding_text (Design §2)
        assert "additionalProperties" not in emb_text
        tool_defs[t["name"]] = t_def
        embedding_texts[t["name"]] = emb_text
        tool_vectors[t["name"]] = (
            embed_doc(emb_text) if callable(embed_doc) else active_embedder.embed_text(emb_text)
        )

    by_item: dict[str, dict[str, Any]] = {}
    recall_counts_by_n: dict[str, dict[str, int]] = {
        str(n): {"answerable_total": 0, "answerable_hits": 0}
        for n in COMPASS_COUNTS
    }

    for item in items:
        item_id = item["item_id"]
        anchor_id = item["anchor_id"]
        anchor_lists = lists_by_anchor[anchor_id]["lists"]
        expected_tool = item["key"]["tool"]
        is_abst = bool(item["is_abstention"])

        per_n_records: dict[str, Any] = {}
        for n in COMPASS_COUNTS:
            n_str = str(n)
            candidate_names = anchor_lists[n_str]["tools"]
            assert len(candidate_names) == n

            top5 = retrieve_top5_for_item_list(
                query=item["prompt"],
                candidate_tool_names=candidate_names,
                tool_vectors=tool_vectors,
                embedder=active_embedder,
                min_confidence=min_confidence,
                top_k=5,
            )
            assert len(top5) == 5, f"Expected 5 tools from compass at N={n}, got {len(top5)}"
            top5_tools = [pair[0] for pair in top5]
            top5_scores = [pair[1] for pair in top5]

            # Every returned tool must come strictly from that item's own L_N
            assert set(top5_tools).issubset(set(candidate_names))

            if not is_abst:
                hit = expected_tool in top5_tools
                recall_counts_by_n[n_str]["answerable_total"] += 1
                if hit:
                    recall_counts_by_n[n_str]["answerable_hits"] += 1
            else:
                hit = None

            per_n_records[n_str] = {
                "N": n,
                "top5_tools": top5_tools,
                "top5_scores": top5_scores,
                "target_in_top5": hit,
                "min_confidence": min_confidence,
                "tool_compass_commit": TOOL_COMPASS_COMMIT,
                "entry_point": "gateway.compass(min_confidence=0.0) / CompassIndex.search(top_k=5, min_score=0.0)",
            }

        by_item[item_id] = {
            "item_id": item_id,
            "anchor_id": anchor_id,
            "is_abstention": is_abst,
            "expected_tool": expected_tool,
            "lists": per_n_records,
        }

    recall_at_5_by_n: dict[str, dict[str, Any]] = {}
    for n in COMPASS_COUNTS:
        n_str = str(n)
        tot = recall_counts_by_n[n_str]["answerable_total"]
        hits = recall_counts_by_n[n_str]["answerable_hits"]
        rate = hits / tot if tot else 0.0
        recall_at_5_by_n[n_str] = {
            "N": n,
            "answerable_total": tot,
            "answerable_hits": hits,
            "recall_at_5": round(rate, 6),
            "below_90pct_bound_flag": rate < 0.90 if n == 100 else False,
        }

    embedder_kind = str(
        getattr(
            active_embedder,
            "kind",
            "fake" if embedder_label == "fake_embedder_cpu" else embedder_label,
        )
    )
    embedder_model = str(
        getattr(
            active_embedder,
            "model",
            "fake-hash-96d" if embedder_kind == "fake" else EMBEDDING_MODEL,
        )
    )
    embedder_endpoint = str(
        getattr(
            active_embedder,
            "endpoint",
            "none" if embedder_kind == "fake" else f"{OLLAMA_BASE_URL}{OLLAMA_EMBED_PATH}",
        )
    )

    return {
        "study_id": STUDY_ID,
        "seed": seed,
        "tool_compass_commit": TOOL_COMPASS_COMMIT,
        "entry_point": "gateway.compass(min_confidence=0.0) / CompassIndex.search(top_k=5, min_score=0.0)",
        "embedder": embedder_label,
        "embedder_kind": embedder_kind,
        "embedder_model": embedder_model,
        "embedder_endpoint": embedder_endpoint,
        "min_confidence": min_confidence,
        "tool_category_constant": TOOL_CATEGORY_CONSTANT,
        "tool_examples_constant": [],
        "sample_embedding_text": embedding_texts[registry[0]["name"]],
        "recall_at_5_by_n": recall_at_5_by_n,
        "items": by_item,
    }


def write_compass_artifacts(
    r0_dir: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    embedder_mode: str = "fake",
    allow_gpu: bool = False,
    tool_compass_dir: Path | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Load R0 files, build compass top-5 lists, and write pinned JSON + sha256 sidecars."""
    if embedder_mode != "fake" and not allow_gpu:
        raise RuntimeError(
            "Real Ollama/GPU embedding is forbidden without an explicit Publisher grant (--allow-gpu)."
        )

    registry = json.loads((r0_dir / "registry.json").read_text(encoding="utf-8"))
    items = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
    lists_by_anchor = json.loads((r0_dir / "nested_lists.json").read_text(encoding="utf-8"))

    if embedder_mode == "fake":
        embedder: EmbedderProtocol = FakeEmbedder(seed=seed)
    elif embedder_mode == "ollama":
        embedder = OllamaEmbedder(
            base_url=ollama_base_url,
            model=ollama_model,
            tool_compass_dir=tool_compass_dir,
        )
    else:
        raise ValueError(f"Unsupported embedder_mode: {embedder_mode!r}")

    data = build_compass_lists(
        registry=registry,
        items=items,
        lists_by_anchor=lists_by_anchor,
        embedder=embedder,
        embedder_label="fake_embedder_cpu" if embedder_mode == "fake" else embedder_mode,
        min_confidence=0.0,
        tool_compass_dir=tool_compass_dir,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    compass_path = out_dir / "compass_lists.json"
    compass_sha = write_with_sha256(
        compass_path, json.dumps(data, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "study_id": STUDY_ID,
        "seed": seed,
        "tool_compass_commit": TOOL_COMPASS_COMMIT,
        "embedder": data["embedder"],
        "embedder_kind": data["embedder_kind"],
        "embedder_model": data["embedder_model"],
        "embedder_endpoint": data["embedder_endpoint"],
        "min_confidence": data["min_confidence"],
        "tool_category_constant": data["tool_category_constant"],
        "tool_examples_constant": data["tool_examples_constant"],
        "recall_at_5_by_n": data["recall_at_5_by_n"],
        "compass_lists_sha256": compass_sha,
    }
    manifest_path = out_dir / "compass_manifest.json"
    manifest_sha = write_with_sha256(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["compass_manifest_sha256"] = manifest_sha
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pinned compass top-5 lists for each (item, N).")
    parser.add_argument("--r0-dir", type=Path, default=Path("artifacts/toolcount/r0"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/toolcount/r0"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--embedder", choices=("fake", "ollama"), default="fake")
    parser.add_argument("--allow-gpu", action="store_true", help="Explicit grant flag for real Ollama embedding.")
    parser.add_argument("--tool-compass-dir", type=Path, default=None)
    parser.add_argument("--ollama-url", type=str, default=None)
    parser.add_argument("--ollama-model", type=str, default=EMBEDDING_MODEL)
    args = parser.parse_args(argv)

    manifest = write_compass_artifacts(
        r0_dir=args.r0_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        embedder_mode=args.embedder,
        allow_gpu=args.allow_gpu,
        tool_compass_dir=args.tool_compass_dir,
        ollama_base_url=args.ollama_url,
        ollama_model=args.ollama_model,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
