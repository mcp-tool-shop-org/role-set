"""Standard controls for an Advocate plan. No model is called.

Shuffled labels sit at the chance of a permutation of the same multiset.
The brief-only probe reads the prose alone. It holds out each slot's positive
and negative together, and reports accuracy beside a 200-permutation null.
On a pilot of 26 texts that probe is a smoke test of the machinery, not a verdict.
Writers are split evenly across the batch.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter


def expected_shuffle_match(keys: list[str]) -> float:
    """The chance a random permutation of these labels lands on the true label."""

    n = len(keys)
    if n < 2:
        return 1.0
    return sum(count * (count - 1) for count in Counter(keys).values()) / (n * (n - 1))


def shuffled_label_rate(keys: list[str], trials: int = 40, seed: int = 0) -> float:
    rng = random.Random(seed)
    rates = []
    for _ in range(trials):
        guess = list(keys)
        rng.shuffle(guess)
        rates.append(sum(a == b for a, b in zip(keys, guess)) / len(keys))
    return sum(rates) / len(rates)


def writer_split(families: list[str]) -> dict[str, int]:
    return dict(Counter(families))


def _probe_tokens(prose: str) -> set[str]:
    words = re.findall(r"[a-z0-9$]+", prose.lower())
    bucket = "len-short" if len(words) < 60 else "len-long"
    return set(words) | {bucket}


def _predict_class(prose: str, train: list[tuple[str, bool]]) -> bool:
    """Naive Bayes on word presence, trained on every text except the one being judged."""

    counts = {True: 0, False: 0}
    docs: dict[bool, list[set[str]]] = {True: [], False: []}
    vocab: set[str] = set()
    for text, label in train:
        bag = _probe_tokens(text)
        docs[label].append(bag)
        vocab |= bag
        counts[label] += 1
    if counts[True] == 0 or counts[False] == 0:
        return counts[True] >= counts[False]
    vocab_size = max(len(vocab), 1)
    best_label = True
    best_score = -1e300
    observed = _probe_tokens(prose)
    total = len(train)
    for label in (True, False):
        score = math.log(counts[label] / total)
        seen: Counter[str] = Counter()
        for bag in docs[label]:
            seen.update(bag)
        for token in observed:
            score += math.log((seen[token] + 1) / (counts[label] + vocab_size))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _grouped_accuracy(items: list[tuple[str, bool]], groups: list[str]) -> float:
    """Score every text with its slot held out, so a pair never trains on itself."""

    correct = 0
    for slot in dict.fromkeys(groups):
        train = [item for item, group in zip(items, groups) if group != slot]
        for (prose, label), group in zip(items, groups):
            if group != slot:
                continue
            correct += int(_predict_class(prose, train) == label)
    return correct / len(items)


def brief_only_probe(
    items: list[tuple[str, bool]],
    groups: list[str] | None = None,
    permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Predict positive or negative from the prose alone.

    `items` are (prose, is_positive). `groups` names the slot of each text, and a
    slot's positive and negative are held out together. The null shuffles labels.
    """

    if len(items) < 2:
        raise ValueError("the probe needs at least two texts")
    if groups is None:
        groups = [str(index) for index in range(len(items))]
    if len(groups) != len(items):
        raise ValueError("the probe groups do not match the texts")
    accuracy = _grouped_accuracy(items, groups)
    rng = random.Random(seed)
    scores = []
    for _ in range(permutations):
        labels = [label for _prose, label in items]
        rng.shuffle(labels)
        shuffled = [(prose, label) for (prose, _label), label in zip(items, labels)]
        scores.append(_grouped_accuracy(shuffled, groups))
    ordered = sorted(scores)
    mean = sum(scores) / len(scores)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    message = (
        f"brief-only probe accuracy {accuracy:.2f} beside a {permutations}-permutation null "
        f"(mean {mean:.2f}, 95th percentile {p95:.2f}). "
        f"On {len(items)} texts this is a smoke test of the machinery, not a verdict."
    )
    return {
        "accuracy": accuracy,
        "null_mean": mean,
        "null_p95": p95,
        "n": len(items),
        "message": message,
    }
