"""Standard controls for an Advocate plan. No model is called.

Shuffled labels sit at the chance of a permutation of the same multiset.
A brief-only probe that cannot see the key sits at one over the number of labels.
Writers are split evenly across the batch.
"""

from __future__ import annotations

import random
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
