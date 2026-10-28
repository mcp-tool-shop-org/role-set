"""Tool-count cliff study package (R0 generator, oracle, compass lists, runner, scorer)."""

from __future__ import annotations

STUDY_ID = "tool-count-cliff-v1"
DEFAULT_SEED = 20261028
TOOL_COUNTS = (5, 10, 20, 40, 70, 100)
COMPASS_COUNTS = (10, 20, 40, 70, 100)
ARMS = ("full", "compass", "length-control")

TOOL_CATEGORY_CONSTANT = "tool"
TOOL_EXAMPLES_CONSTANT: tuple[str, ...] = ()

TOOL_COMPASS_COMMIT = "6f5a233"
RND_PROXY_COMMIT = "8400846"
LLAMA_CPP_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
LLAMA_SERVER_SHA256 = "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"

SYSTEM_PROMPT = "Call exactly one tool if one applies; call none if none applies"

__all__ = [
    "ARMS",
    "COMPASS_COUNTS",
    "DEFAULT_SEED",
    "LLAMA_CPP_COMMIT",
    "LLAMA_SERVER_SHA256",
    "RND_PROXY_COMMIT",
    "STUDY_ID",
    "SYSTEM_PROMPT",
    "TOOL_CATEGORY_CONSTANT",
    "TOOL_COMPASS_COMMIT",
    "TOOL_COUNTS",
    "TOOL_EXAMPLES_CONSTANT",
]
