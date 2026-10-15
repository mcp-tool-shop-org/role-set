"""Standard controls for an Advocate plan. No model is called.

Shuffled labels sit at the chance of a permutation of the same multiset.
The brief-only probe reads the prose alone and predicts positive or negative
from length and a leave-one-out bag of words. On a pilot of 26 texts that
probe is a smoke test of the machinery, not a verdict.
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


def brief_only_rate(keys: list[str], trials: int = 40, seed: int = 1) -> float:
    """Guesses a label uniformly and ignores the prose. Chance is one over the label count."""

    labels = sorted(set(keys))
    rng = random.Random(seed)
    rates = []
    for _ in range(trials):
        guesses = [rng.choice(labels) for _ in keys]
        rates.append(sum(guess == key for guess, key in zip(guesses, keys)) / len(keys))
    return sum(rates) / len(rates)


def chance_uniform(keys: list[str]) -> float:
    return 1 / len(set(keys))


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


def brief_only_probe(items: list[tuple[str, bool]]) -> dict:
    """Predict positive or negative from the prose alone. `items` are (prose, is_positive)."""

    if len(items) < 2:
        raise ValueError("the probe needs at least two texts")
    correct = 0
    for index, (prose, label) in enumerate(items):
        train = items[:index] + items[index + 1:]
        correct += int(_predict_class(prose, train) == label)
    accuracy = correct / len(items)
    positive = sum(1 for _prose, label in items if label) / len(items)
    chance = max(positive, 1 - positive)
    message = (
        f"brief-only probe accuracy {accuracy:.2f} beside chance {chance:.2f}. "
        f"On {len(items)} texts this is a smoke test of the machinery, not a verdict."
    )
    return {"accuracy": accuracy, "chance": chance, "n": len(items), "message": message}
